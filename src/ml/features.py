import numpy as np

def build_geo_features(record):
    """
    Convert geo history record → ML feature vector
    """

    risk = record.get("risk", 0)
    density = record.get("density", 0)
    high = record.get("high", 0)
    medium = record.get("medium", 0)
    road_health = record.get("road_health", 0)

    return np.array([
        risk,
        density * 100,
        high,
        medium,
        100 - road_health
    ])
