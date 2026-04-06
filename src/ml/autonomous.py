from src.ml.features import build_geo_features
from src.ml.trainer import geo_trainer

# Global city memory
geo_history = []

def update_memory(lat, lon, risk, density, high, medium, road_health):

    record = {
        "lat": lat,
        "lon": lon,
        "risk": risk,
        "density": density,
        "high": high,
        "medium": medium,
        "road_health": road_health
    }

    geo_history.append(record)

    # Auto-train every few updates
    geo_trainer.train(geo_history)

    return record


def autonomous_predict(record):

    features = build_geo_features(record)

    pred = geo_trainer.predict(features)

    if pred is None:
        return {
            "predicted_risk": record["risk"],
            "trend": "learning"
        }

    if pred > record["risk"] + 5:
        trend = "increasing"
    elif pred < record["risk"] - 5:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "predicted_risk": pred,
        "trend": trend
    }
