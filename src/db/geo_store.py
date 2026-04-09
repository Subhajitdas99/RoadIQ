from src.db.database import fetch_geo_points


def add_geo_point(lat, lon, risk):
    # Phase 13 stores geo history through incidents in SQLite.
    # This function remains only as a compatibility shim.
    return {"lat": lat, "lon": lon, "risk": risk}


def get_geo_points(limit: int = 500):
    return fetch_geo_points(limit=limit)
