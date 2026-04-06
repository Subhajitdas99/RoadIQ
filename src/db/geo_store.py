# Simple in-memory geo storage (Phase-3.5 MVP)

geo_points = []

def add_geo_point(lat, lon, risk):
    geo_points.append({
        "lat": lat,
        "lon": lon,
        "risk": risk
    })

def get_geo_points():
    return geo_points
