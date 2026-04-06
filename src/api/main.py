from fastapi import FastAPI, UploadFile, File, Response, Form, HTTPException
from fastapi import WebSocket, WebSocketDisconnect

from src.inference.detector import run_inference
from src.db.geo_store import add_geo_point, get_geo_points
from src.geo.cluster_engine import build_risk_zones
from src.stream.live_manager import live_manager

import numpy as np
import cv2
import time

app = FastAPI(title="RoadIQ Geo-AI Smart City API")


# ======================================================
# 🚧 MAIN DETECTION ENDPOINT
# ======================================================
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
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
    detections = run_inference(image_np)
    inference_time = round(time.time() - start_time, 3)

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

    # ======================================================
    # 🧠 SAVE GEO HISTORY
    # ======================================================
    add_geo_point(lat, lon, risk_score)

    # ======================================================
    # 🔴 LIVE CITY BRAIN BROADCAST
    # ======================================================
    try:
        await live_manager.broadcast({
            "event": "NEW_ANALYSIS",
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
        response.headers["X-Model-Version"] = "roadIQ-pothole-v1"
        response.headers["X-Inference-Time"] = f"{inference_time}s"

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
        "location": {"lat": lat, "lon": lon},
        "num_detections": len(detections),
        "risk_score": risk_score,
        "danger_zone": danger_zone,
        "maintenance_eta": maintenance_eta,
        "maintenance_priority": maintenance_priority,
        "road_health": road_health,
        "density": density,
        "alert_level": alert_level,
        "detections": detections,
    }


# ======================================================
# 🌍 GEO DATA STREAM
# ======================================================
@app.get("/geo-data")
def geo_data():
    return get_geo_points()


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




