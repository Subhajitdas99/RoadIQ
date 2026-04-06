from sklearn.cluster import DBSCAN
import numpy as np

def build_risk_zones(points):
    """
    points = [{lat, lon, risk}]
    """

    if len(points) < 2:
        return []

    coords = np.array([[p["lat"], p["lon"]] for p in points])

    # DBSCAN groups nearby geo points
    clustering = DBSCAN(eps=0.002, min_samples=2).fit(coords)

    labels = clustering.labels_

    zones = []

    for label in set(labels):
        if label == -1:
            continue

        cluster_points = [points[i] for i in range(len(points)) if labels[i] == label]

        avg_lat = sum(p["lat"] for p in cluster_points) / len(cluster_points)
        avg_lon = sum(p["lon"] for p in cluster_points) / len(cluster_points)
        avg_risk = sum(p["risk"] for p in cluster_points) / len(cluster_points)

        zones.append({
            "lat": avg_lat,
            "lon": avg_lon,
            "risk": round(avg_risk, 2),
            "count": len(cluster_points)
        })

    return zones
