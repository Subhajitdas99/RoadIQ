# ======================================================
# 🧠 PHASE-10 AUTONOMOUS DECISION ENGINE
# ======================================================

def build_decision(risk_score, alert_level, maintenance_priority, detections):
    """
    Converts Geo-AI metrics into city actions.
    """

    # --------------------------------------------------
    # 🚦 Dispatch Score (0 - 100)
    # --------------------------------------------------
    dispatch_score = min(100, round(
        risk_score
        + (10 if alert_level == "WARNING" else 0)
        + (20 if alert_level == "CRITICAL" else 0)
        + (detections * 0.3),
        2
    ))

    # --------------------------------------------------
    # 🚧 Action Decision
    # --------------------------------------------------
    if dispatch_score > 85:
        action = "AUTO_DISPATCH_TEAM"
        decision_level = "AUTONOMOUS"
    elif dispatch_score > 60:
        action = "PRIORITY_QUEUE"
        decision_level = "AI_ASSIST"
    elif dispatch_score > 30:
        action = "SCHEDULE_REPAIR"
        decision_level = "SUGGESTED"
    else:
        action = "MONITOR_ONLY"
        decision_level = "PASSIVE"

    # --------------------------------------------------
    # 🛣️ Road Condition Class
    # --------------------------------------------------
    if risk_score > 75:
        road_state = "FAILED"
    elif risk_score > 50:
        road_state = "CRITICAL"
    elif risk_score > 30:
        road_state = "DEGRADED"
    else:
        road_state = "STABLE"

    return {
        "dispatch_score": dispatch_score,
        "action": action,
        "decision_level": decision_level,
        "road_state": road_state
    }
