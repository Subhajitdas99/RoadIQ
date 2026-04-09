from fastapi import FastAPI, UploadFile, File, Response, Form, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.inference.detector import run_inference
from src.inference.model_config import MODEL_OPTIONS
from src.db.database import (
    fetch_analytics_summary_filtered,
    fetch_damage_type_analytics,
    fetch_geo_hotspot_points,
    fetch_incident_by_id,
    fetch_incidents_filtered,
    fetch_incident_trends,
    fetch_top_risk_locations,
    init_db,
    save_incident,
    update_incident_status,
)
from src.db.geo_store import get_geo_points
from src.decision.decision_engine import build_decision
from src.geo.cluster_engine import build_risk_zones
from src.stream.live_manager import live_manager

import numpy as np
import cv2
import time

app = FastAPI(title="RoadIQ Geo-AI Smart City API")


VALID_INCIDENT_STATUSES = {
    "reported",
    "verified",
    "assigned",
    "in_progress",
    "resolved",
}


class IncidentStatusUpdate(BaseModel):
    status: str
    notes: str | None = None


def summarize_damage_types(detections: list[dict]) -> dict:
    summary: dict[str, int] = {}
    for det in detections:
        label = det.get("label", "unknown")
        summary[label] = summary.get(label, 0) + 1
    return summary


@app.on_event("startup")
def startup_event():
    init_db()


# ======================================================
# 🚧 MAIN DETECTION ENDPOINT
# ======================================================
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    model_key: str = Form("default"),
    response: Response = None,
):

    # ======================================================
    # 🛡️ FILE VALIDATION
    # ======================================================
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    contents = await file.read()

    if contents is None or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # ======================================================
    # 🖼️ DECODE IMAGE
    # ======================================================
    np_arr = np.frombuffer(contents, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_np is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ======================================================
    # 🤖 RUN AI INFERENCE
    # ======================================================
    start_time = time.time()
    inference_result = run_inference(image_np, model_key=model_key)
    detections = inference_result["detections"]
    inference_time = round(time.time() - start_time, 3)
    damage_type_summary = summarize_damage_types(detections)

    # ======================================================
    # 📊 PRIORITY STATS
    # ======================================================
    high = sum(1 for d in detections if d["priority"] == "High")
    medium = sum(1 for d in detections if d["priority"] == "Medium")
    low = sum(1 for d in detections if d["priority"] == "Low")

    # ======================================================
    # 🌍 GEO-AI CORE LOGIC
    # ======================================================
    severity_total = sum(d["severity"] for d in detections)
    risk_score = min(100, round(severity_total + (high * 20) + (medium * 10), 2))

    danger_zone = "TRUE" if high >= 3 or risk_score > 60 else "FALSE"

    if risk_score > 80:
        maintenance_priority = "IMMEDIATE_DISPATCH"
        maintenance_eta = "1_day"
    elif risk_score > 50:
        maintenance_priority = "URGENT_QUEUE"
        maintenance_eta = "2_days"
    elif risk_score > 25:
        maintenance_priority = "SCHEDULED"
        maintenance_eta = "1_week"
    else:
        maintenance_priority = "MONITOR"
        maintenance_eta = "monitor"

    # ======================================================
    # 🛰️ SMART METRICS
    # ======================================================
    road_health = max(0, round(100 - risk_score, 2))

    img_h, img_w = image_np.shape[:2]
    img_area = max(img_h * img_w, 1)

    density = round((len(detections) * 1000) / img_area, 4)

    if risk_score > 70:
        alert_level = "CRITICAL"
    elif risk_score > 40:
        alert_level = "WARNING"
    elif risk_score > 15:
        alert_level = "WATCH"
    else:
        alert_level = "NORMAL"

    decision = build_decision(
        risk_score=risk_score,
        alert_level=alert_level,
        maintenance_priority=maintenance_priority,
        detections=len(detections),
    )

    incident_id = save_incident(
        filename=file.filename,
        lat=lat,
        lon=lon,
        risk_score=risk_score,
        danger_zone=danger_zone,
        maintenance_eta=maintenance_eta,
        maintenance_priority=maintenance_priority,
        road_health=road_health,
        density=density,
        alert_level=alert_level,
        total_detections=len(detections),
        high_priority_count=high,
        medium_priority_count=medium,
        low_priority_count=low,
        inference_time=inference_time,
        detections=detections,
        decision=decision,
    )

    # ======================================================
    # 🔴 LIVE CITY BRAIN BROADCAST
    # ======================================================
    try:
        await live_manager.broadcast({
            "event": "NEW_ANALYSIS",
            "incident_id": incident_id,
            "lat": lat,
            "lon": lon,
            "risk_score": risk_score,
            "alert_level": alert_level,
            "maintenance_priority": maintenance_priority,
            "detections": len(detections)
        })
    except Exception:
        # Never crash API if websocket fails
        pass

    # ======================================================
    # 🧾 SMART MONITORING HEADERS
    # ======================================================
    if response is not None:
        response.headers["X-System"] = "RoadIQ Geo-AI"
        response.headers["X-Model-Key"] = inference_result["model_key"]
        response.headers["X-Model-Label"] = inference_result["model_label"]
        response.headers["X-Model-Version"] = inference_result["model_version"]
        response.headers["X-Model-Weights"] = inference_result["weights_path"]
        response.headers["X-Inference-Time"] = f"{inference_time}s"
        response.headers["X-Damage-Types"] = ",".join(sorted(damage_type_summary.keys()))

        response.headers["X-Total-Detections"] = str(len(detections))
        response.headers["X-High-Priority"] = str(high)
        response.headers["X-Medium-Priority"] = str(medium)
        response.headers["X-Low-Priority"] = str(low)

        response.headers["X-Road-Risk-Score"] = str(risk_score)
        response.headers["X-Danger-Zone"] = danger_zone
        response.headers["X-Maintenance-ETA"] = maintenance_eta
        response.headers["X-Maintenance-Priority"] = maintenance_priority
        response.headers["X-Road-Health"] = str(road_health)
        response.headers["X-Pothole-Density"] = str(density)
        response.headers["X-Alert-Level"] = alert_level

    # ======================================================
    # 📦 API RESPONSE
    # ======================================================
    return {
        "filename": file.filename,
        "incident_id": incident_id,
        "model_key": inference_result["model_key"],
        "model_label": inference_result["model_label"],
        "model_version": inference_result["model_version"],
        "location": {"lat": lat, "lon": lon},
        "num_detections": len(detections),
        "risk_score": risk_score,
        "danger_zone": danger_zone,
        "maintenance_eta": maintenance_eta,
        "maintenance_priority": maintenance_priority,
        "road_health": road_health,
        "density": density,
        "alert_level": alert_level,
        "damage_type_summary": damage_type_summary,
        "decision": decision,
        "detections": detections,
    }


@app.get("/models")
def available_models():
    return [
        {
            "key": model_key,
            "label": config["label"],
            "model_version": config["model_version"],
            "class_names": config["class_names"],
        }
        for model_key, config in MODEL_OPTIONS.items()
    ]


# ======================================================
# 🌍 GEO DATA STREAM
# ======================================================
@app.get("/geo-data")
def geo_data():
    return get_geo_points()


@app.get("/incidents")
def incident_history(
    limit: int = 100,
    status: str | None = None,
    alert_level: str | None = None,
):
    limit = max(1, min(limit, 500))
    return fetch_incidents_filtered(
        limit=limit,
        status=status,
        alert_level=alert_level,
    )


@app.get("/incidents/{incident_id}")
def incident_detail(incident_id: int):
    incident = fetch_incident_by_id(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@app.get("/analytics/summary")
def analytics_summary(days: int | None = None):
    return fetch_analytics_summary_filtered(days=_normalize_days(days))


def _normalize_days(days: int | None) -> int | None:
    if days is None or days <= 0:
        return None
    return min(days, 365)


@app.get("/analytics/trends")
def analytics_trends(limit: int = 30, days: int | None = None):
    limit = max(1, min(limit, 180))
    return fetch_incident_trends(limit=limit, days=_normalize_days(days))


@app.get("/analytics/damage-types")
def analytics_damage_types(days: int | None = None):
    return fetch_damage_type_analytics(days=_normalize_days(days))


@app.get("/analytics/top-risk-locations")
def analytics_top_risk_locations(limit: int = 5, days: int | None = None):
    limit = max(1, min(limit, 20))
    return fetch_top_risk_locations(limit=limit, days=_normalize_days(days))


@app.get("/analytics/hotspots")
def analytics_hotspots(days: int | None = None, limit: int = 500):
    limit = max(1, min(limit, 1000))
    points = fetch_geo_hotspot_points(limit=limit, days=_normalize_days(days))
    zones = build_risk_zones(
        [{"lat": point["lat"], "lon": point["lon"], "risk": point["risk_score"]} for point in points]
    )
    return {
        "points": points,
        "zones": zones,
    }


@app.patch("/incidents/{incident_id}/status")
def incident_status_update(incident_id: int, payload: IncidentStatusUpdate):
    normalized_status = payload.status.strip().lower()
    if normalized_status not in VALID_INCIDENT_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Use one of: {sorted(VALID_INCIDENT_STATUSES)}",
        )

    incident = update_incident_status(
        incident_id=incident_id,
        status=normalized_status,
        notes=payload.notes,
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


# ======================================================
# 🔴 GEO RISK ZONES
# ======================================================
@app.get("/geo-zones")
def geo_zones():
    points = get_geo_points()
    zones = build_risk_zones(points)
    return zones


# ======================================================
# 🔴 LIVE CITY BRAIN WEBSOCKET
# ======================================================
@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await live_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keepalive
    except WebSocketDisconnect:
        live_manager.disconnect(websocket)




