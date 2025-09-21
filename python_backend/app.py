# Add these imports at the top of app.py
import os
import json
import signal
import sys
from typing import Dict, Any, List
from functools import lru_cache
import gc

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import folium
from pymongo import MongoClient

# Add these configurations after imports
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client to reuse connections
_mongo_client = None

def get_mongo_client():
    """
    Build a MongoClient using MONGO_URI from environment if provided,
    otherwise fallback to the project default without trailing slash.
    Reuse connection to avoid connection overhead.
    """
    global _mongo_client
    if _mongo_client is None:
        uri = os.environ.get("MONGO_URI") or "mongodb+srv://setudeyindia_db_user:G8z0myA0G7D9Hiwt@cluster0.pw3lt0s.mongodb.net/water"
        if uri.endswith('/'):
            uri = uri[:-1]
        _mongo_client = MongoClient(uri, maxPoolSize=10, minPoolSize=1)
    return _mongo_client

# Cache thresholds to avoid recalculating every time
@lru_cache(maxsize=32)
def compute_thresholds_per_disease_cached(disease_cases_tuple, min_cases: float = 50.0) -> Dict[str, float]:
    """
    Cached version of threshold computation.
    disease_cases_tuple should be a tuple of (disease_name, tuple_of_cases)
    """
    thresholds = {}
    
    for disease, cases_tuple in disease_cases_tuple:
        try:
            # Convert tuple back to list for pandas Series
            cases_list = list(cases_tuple)
            vals = pd.Series(cases_list).fillna(0)
            q75 = float(vals.quantile(0.75))
        except Exception:
            q75 = 0.0
        threshold = max(q75, float(min_cases))
        if threshold <= 0:
            threshold = float(min_cases)
        thresholds[str(disease) if disease is not None else ""] = threshold
    
    return thresholds

def compute_payload() -> Dict[str, Any]:
    try:
        logger.info("Starting payload computation")
        
        client = get_mongo_client()
        # Try to use database from URI, fallback to 'water'
        try:
            db = client.get_default_database()
        except Exception:
            db = None
        if db is None:
            db_name = os.environ.get("MONGO_DB") or "water"
            db = client[db_name]

        coll_name = pick_collection(db)
        col = db[coll_name]

        # Limit query to reduce memory usage
        cursor = col.find({}, essential_fields).limit(10000)  # Limit to 10k records
        docs = list(cursor)
        
        logger.info(f"Retrieved {len(docs)} documents")

        if not docs:
            return {"areas": [], "redzones": [], "mapPath": None}

        # Use chunks to process large datasets
        chunk_size = 1000
        all_table_rows = []
        
        for i in range(0, len(docs), chunk_size):
            chunk_docs = docs[i:i + chunk_size]
            df = pd.DataFrame(chunk_docs)
            
            # Normalize columns and types
            for c in ["Cases", "Deaths", "Latitude", "Longitude", "year", "mon", "day"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            if "Cases" not in df.columns:
                df["Cases"] = 0
            else:
                df["Cases"] = df["Cases"].fillna(0)
            
            if "Deaths" in df.columns:
                df["Deaths"] = df["Deaths"].fillna(0)
            if "district" not in df.columns:
                df["district"] = ""
            if "Disease" not in df.columns:
                df["Disease"] = ""

            # Create cacheable tuple for thresholds - FIX: Convert lists to tuples
            disease_cases = {}
            for disease, group in df.groupby('Disease'):
                cases = pd.to_numeric(group['Cases'], errors='coerce').fillna(0).tolist()
                # Convert list to tuple to make it hashable
                disease_cases[str(disease) if disease is not None else ""] = tuple(cases)
            
            # Convert to tuple for caching - now all elements are hashable
            disease_cases_tuple = tuple(sorted(disease_cases.items()))
            thresholds = compute_thresholds_per_disease_cached(disease_cases_tuple, min_cases=50.0)
            
            df["threshold"] = df["Disease"].map(thresholds).fillna(50.0)

            # Risk percentage per row, capped [0, 100]
            df["outbreak_percent"] = (df["Cases"] / df["threshold"] * 100).clip(lower=0, upper=100).round(2)
            df["outbreak_percent"] = df["outbreak_percent"].fillna(0)
            df["band"] = df["outbreak_percent"].apply(band_from_percent)

            # Prepare table rows for this chunk
            chunk_table_rows = []
            df_sorted = df.sort_values(by=["outbreak_percent", "Cases"], ascending=[False, False])
            
            for _, row in df_sorted.iterrows():
                band = str(row.get("band", "green"))
                chunk_table_rows.append({
                    "state_ut": str(row.get("state_ut", "") or ""),
                    "district": str(row.get('district', '') or ''),
                    "disease": str(row.get("Disease", "") or ""),
                    "percentage": float(row.get("outbreak_percent", 0.0) or 0.0),
                    "date": row_date(row),
                    "band": band,
                    "lat": safe_float(row.get('Latitude')),
                    "lon": safe_float(row.get('Longitude')),
                    "cases": float(row.get('Cases', 0) or 0)
                })
            
            all_table_rows.extend(chunk_table_rows)
            
            # Clean up memory
            del df
            del chunk_table_rows
            gc.collect()

        # Sort all results
        all_table_rows.sort(key=lambda x: (x["percentage"], x["cases"]), reverse=True)

        # Build map with top 500 points to avoid memory issues
        map_url_path = None
        try:
            m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
            
            # Only plot top 500 points to avoid memory issues
            plot_data = [row for row in all_table_rows[:500] if row['lat'] is not None and row['lon'] is not None]
            
            for row in plot_data:
                band = row['band']
                color = 'red' if band == 'red' else ('yellow' if band == 'yellow' else 'green')
                popup_text = (
                    f"State/UT: {row['state_ut']}<br>"
                    f"District: {row['district']}<br>"
                    f"Disease: {row['disease']}<br>"
                    f"Date: {row['date']}<br>"
                    f"Cases: {row['cases']}<br>"
                    f"Risk Band: {band.capitalize()}<br>"
                    f"Outbreak %: {row['percentage']}%"
                )
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=popup_text
                ).add_to(m)

            # Since you're hosting Flask separately, save map to static folder instead
            # Remove the Node.js path logic
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            os.makedirs(static_dir, exist_ok=True)
            map_path = os.path.join(static_dir, 'india_hotspot_map.html')
            m.save(map_path)
            map_url_path = '/static/india_hotspot_map.html'
            
        except Exception as e:
            logger.error(f"Map generation error: {e}")
            map_url_path = None

        # Clean up
        gc.collect()
        
        logger.info(f"Payload computation completed with {len(all_table_rows)} records")
        return {"areas": all_table_rows, "redzones": all_table_rows, "mapPath": map_url_path}
        
    except Exception as e:
        logger.error(f"Error in compute_payload: {e}")
        return {"areas": [], "redzones": [], "mapPath": None, "error": str(e)}

# Essential fields and helper functions
essential_fields = {
    "state_ut": 1,
    "district": 1,
    "Disease": 1,
    "Cases": 1,
    "Deaths": 1,
    "Latitude": 1,
    "Longitude": 1,
    "year": 1,
    "mon": 1,
    "day": 1,
    "week_of_outbreak": 1,
}

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def build_date_string(year, mon, day):
    try:
        if pd.notnull(year) and pd.notnull(mon) and pd.notnull(day):
            return f"{int(year):04d}-{int(mon):02d}-{int(day):02d}"
        if pd.notnull(year) and pd.notnull(mon):
            return f"{int(year):04d}-{int(mon):02d}"
        if pd.notnull(year):
            return f"{int(year):04d}"
    except Exception:
        pass
    return "N/A"

def row_date(row: Dict[str, Any]) -> str:
    s = build_date_string(row.get('year'), row.get('mon'), row.get('day'))
    if s == 'N/A':
        w = row.get('week_of_outbreak')
        if isinstance(w, str) and w.strip():
            return w
    return s

def band_from_percent(p: float) -> str:
    # Green: <= 40, Yellow: > 40 and < 60, Red: >= 60
    if p >= 60:
        return 'red'
    if p > 40:
        return 'yellow'
    return 'green'

def band_emoji(band: str) -> str:
    return {'red': '🔴', 'yellow': '🟡', 'green': '🟢'}.get(band, '🟢')

def pick_collection(db) -> str:
    # Prefer explicit env override, else try 'Dieses_data' then fallback to 'Disease_Data'
    env_coll = os.environ.get('MONGO_COLL')
    if env_coll:
        return env_coll
    try:
        colls = set(db.list_collection_names())
    except Exception:
        colls = set()
    if 'Dieses_data' in colls:
        return 'Dieses_data'
    return 'Disease_Data'

# -------------------------
# Flask API wrapper with better error handling
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/": {"origins": "*"}})

# Add static file serving for maps
from flask import send_from_directory
@app.route('/static/<filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, filename)

# Add request timeout handling
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.timeout = 30

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True, "status": "healthy"})

@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        logger.info("Received prediction request")
        payload = compute_payload()
        logger.info("Sending response")
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"redzones": [], "areas": [], "mapPath": None, "error": str(e)}), 500

# Add graceful shutdown handling
def signal_handler(sig, frame):
    logger.info('Gracefully shutting down Flask server')
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT') or os.environ.get('PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)