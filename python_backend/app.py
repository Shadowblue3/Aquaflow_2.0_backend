import os
import json
from typing import Dict, Any, List

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import folium
from pymongo import MongoClient


# -------------------------
# Core data logic (ported from test.py)
# -------------------------

def get_mongo_client():
    """
    Build a MongoClient using MONGO_URI from environment if provided,
    otherwise fallback to the project default without trailing slash.
    """
    uri = os.environ.get("MONGO_URI") or "mongodb+srv://setudeyindia_db_user:G8z0myA0G7D9Hiwt@cluster0.pw3lt0s.mongodb.net/water"
    if uri.endswith('/'):
        uri = uri[:-1]
    return MongoClient(uri)


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


def compute_thresholds_per_disease(df: pd.DataFrame, min_cases: float = 50.0) -> Dict[str, float]:
    """
    Compute a dynamic threshold per disease using the 75th percentile of per-row Cases
    across all data for that disease, with a minimum floor.
    """
    thresholds: Dict[str, float] = {}
    for disease, group in df.groupby('Disease'):
        vals = pd.to_numeric(group['Cases'], errors='coerce').fillna(0)
        try:
            q75 = float(vals.quantile(0.75))
        except Exception:
            q75 = 0.0
        threshold = max(q75, float(min_cases))
        if threshold <= 0:
            threshold = float(min_cases)
        thresholds[str(disease) if disease is not None else ""] = threshold
    return thresholds


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


def compute_payload() -> Dict[str, Any]:
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

    cursor = col.find({}, essential_fields)
    docs = list(cursor)

    if not docs:
        return {"areas": [], "redzones": [], "mapPath": None}

    df = pd.DataFrame(docs)

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

    # Compute dynamic thresholds per disease using all rows
    thresholds = compute_thresholds_per_disease(df, min_cases=50.0)
    df["threshold"] = df["Disease"].map(thresholds).fillna(50.0)

    # Risk percentage per row, capped [0, 100]
    df["outbreak_percent"] = (df["Cases"] / df["threshold"] * 100).clip(lower=0, upper=100).round(2)
    df["outbreak_percent"] = df["outbreak_percent"].fillna(0)
    df["band"] = df["outbreak_percent"].apply(band_from_percent)

    # Prepare table rows for ALL documents
    table_rows: List[Dict[str, Any]] = []
    df_sorted = df.sort_values(by=["outbreak_percent", "Cases"], ascending=[False, False])
    for _, row in df_sorted.iterrows():
        band = str(row.get("band", "green"))
        emoji = band_emoji(band)
        table_rows.append({
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

    # Build map with ALL rows into Node's public/maps folder
    map_url_path = None
    try:
        m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)
        for _, row in df.iterrows():
            lat = safe_float(row.get('Latitude'))
            lon = safe_float(row.get('Longitude'))
            if lat is None or lon is None:
                continue
            band = str(row.get('band', 'green'))
            color = 'red' if band == 'red' else ('yellow' if band == 'yellow' else 'green')
            risk = float(row.get('outbreak_percent', 0) or 0)
            popup_text = (
                f"State/UT: {row.get('state_ut', '')}<br>"
                f"District: {row.get('district', '')}<br>"
                f"Disease: {row.get('Disease', '')}<br>"
                f"Date: {row_date(row)}<br>"
                f"Cases: {row.get('Cases', 0)}<br>"
                f"Risk Band: {band.capitalize()}<br>"
                f"Outbreak %: {risk}%"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(m)

        # Ensure map is saved under the Node app's public/maps directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
        maps_dir = os.path.join(base_dir, 'public', 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        map_path = os.path.join(maps_dir, 'india_hotspot_map.html')
        m.save(map_path)
        map_url_path = '/maps/india_hotspot_map.html'
    except Exception:
        map_url_path = None

    return {"areas": table_rows, "redzones": table_rows, "mapPath": map_url_path}


# -------------------------
# Flask API wrapper
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})


@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        payload = compute_payload()
        return jsonify(payload)
    except Exception as e:
        return jsonify({"redzones": [], "mapPath": None, "error": str(e)}), 500


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)
