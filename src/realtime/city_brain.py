from src.realtime.event_bus import get_events

def compute_city_state():

    events = get_events(100)

    if not events:
        return {
            "status": "idle",
            "avg_risk": 0,
            "total_events": 0
        }

    total_detections = sum(e["num_detections"] for e in events)

    avg_risk = 0
    count = 0

    for e in events:
        for d in e["detections"]:
            avg_risk += d.get("severity", 0)
            count += 1

    avg_risk = round(avg_risk / max(count, 1), 2)

    if avg_risk > 40:
        city_status = "CRITICAL"
    elif avg_risk > 20:
        city_status = "WARNING"
    else:
        city_status = "NORMAL"

    return {
        "status": city_status,
        "avg_risk": avg_risk,
        "total_events": len(events),
        "detections_seen": total_detections
    }
