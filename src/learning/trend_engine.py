# ======================================================
# 🧬 PHASE-11 SELF LEARNING TREND ENGINE
# ======================================================

from collections import defaultdict

# Simple in-memory learning store
trend_memory = defaultdict(list)


def update_trend(lat, lon, risk):
    """
    Store historical risk values per geo point.
    """
    key = f"{round(lat,2)}_{round(lon,2)}"
    trend_memory[key].append(risk)

    # limit memory to last 20 samples
    if len(trend_memory[key]) > 20:
        trend_memory[key] = trend_memory[key][-20:]


def get_trend_prediction(lat, lon):
    """
    Predict future risk using simple trend logic.
    """
    key = f"{round(lat,2)}_{round(lon,2)}"
    history = trend_memory.get(key, [])

    if len(history) < 3:
        return {
            "trend": "UNKNOWN",
            "predicted_risk": None,
            "growth_rate": 0
        }

    # calculate growth rate
    growth = history[-1] - history[0]
    avg_risk = sum(history) / len(history)

    if growth > 20:
        trend = "RAPIDLY_WORSENING"
    elif growth > 5:
        trend = "WORSENING"
    elif growth < -5:
        trend = "IMPROVING"
    else:
        trend = "STABLE"

    predicted_risk = min(100, round(avg_risk + growth * 0.5, 2))

    return {
        "trend": trend,
        "predicted_risk": predicted_risk,
        "growth_rate": round(growth, 2)
    }
