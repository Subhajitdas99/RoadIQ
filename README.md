# RoadIQ

RoadIQ is a Geo-AI smart city platform for pothole detection, road-risk intelligence, and live infrastructure monitoring. It combines YOLOv8-based computer vision, FastAPI inference services, geo-spatial analytics, and a Streamlit command center to turn road images into actionable maintenance signals.

## What It Does

- detects potholes from uploaded road images using a YOLOv8 model
- calculates severity, repair priority, road health, and overall risk score
- stores latitude and longitude linked detections for geo analysis
- builds risk-zone intelligence from historical geo points
- streams live city events through a WebSocket-enabled backend
- visualizes detections, metrics, and map data in a command dashboard

## Core System Capabilities

### AI Detection Engine
- image inference with YOLOv8
- pothole-only filtering from model outputs
- bounding boxes, confidence scores, severity scoring, and priority labels

### Risk Intelligence
RoadIQ computes a road-risk score from detected pothole severity and priority counts, then derives:
- alert level
- maintenance priority
- maintenance ETA
- road health score
- pothole density

### Geo-AI Layer
- accepts latitude and longitude with every detection request
- stores geo-tagged risk history
- exposes geo-data and clustered risk-zone endpoints

### Realtime City Brain
- publishes live analysis events from the backend
- supports WebSocket clients for streaming updates
- keeps the command center synced with new detections

## Architecture

```text
User Upload + GPS
        |
        v
FastAPI API Layer
  - /detect
  - /geo-data
  - /geo-zones
  - /ws/live
        |
        v
AI + Decision Layer
  - YOLO inference
  - severity scoring
  - risk intelligence
  - autonomous decision logic
        |
        v
Geo / Stream Layer
  - geo history store
  - risk clustering
  - live event manager
        |
        v
Streamlit Command Center
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pillow
- Requests

## Project Structure

```text
RoadIQ/
|-- README.md
|-- requirements.txt
|-- src/
|   |-- api/
|   |-- dashboard/
|   |-- db/
|   |-- decision/
|   |-- geo/
|   |-- inference/
|   |-- learning/
|   |-- ml/
|   |-- neural/
|   |-- realtime/
|   |-- stream/
|-- tests/
|-- trainning/
```

## Main API Endpoints

### `POST /detect`
Accepts an image plus `lat` and `lon`, runs pothole detection, computes risk intelligence, stores geo history, and returns structured analysis.

Response includes values such as:
- `num_detections`
- `risk_score`
- `maintenance_priority`
- `maintenance_eta`
- `road_health`
- `alert_level`
- `detections`

Response headers also expose monitoring signals such as:
- `X-Road-Risk-Score`
- `X-Alert-Level`
- `X-Maintenance-ETA`
- `X-Pothole-Density`
- `X-Road-Health`

### `GET /geo-data`
Returns stored geo-tagged risk points.

### `GET /geo-zones`
Builds clustered road-risk zones from stored geo points.

### `WS /ws/live`
Streams live RoadIQ city events for realtime monitoring clients.

## Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI backend
```bash
uvicorn src.api.main:app --reload
```

### 3. Start the Streamlit dashboard
```bash
streamlit run src/dashboard/app.py
```

### 4. Open the app
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit dashboard: `http://localhost:8501`

## Current Workflow

1. Upload a road image in the dashboard
2. Enter latitude and longitude
3. Run detection through the FastAPI backend
4. Review bounding boxes, severity, and risk metrics
5. View geo intelligence on the live map
6. Stream new analysis events into the city command center

## Resume-Ready Summary

Built a full-stack Geo-AI smart city platform using YOLOv8, FastAPI, Streamlit, and WebSockets to detect potholes, compute severity-based risk scores, store geo-tagged road incidents, and deliver live infrastructure monitoring through a command center dashboard.

## Notes

- the trained YOLO weights are expected at `runs/detect/roadIQ_pothole/weights/best.pt`
- generated training artifacts are ignored in Git
- an untracked local sample image is intentionally not part of the repository
