import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd

# ================================
# ⚙️ CONFIG
# ================================
st.set_page_config(page_title="RoadIQ Command Center", layout="wide")

BACKEND_URL = "http://127.0.0.1:8000/detect"
GEO_URL = "http://127.0.0.1:8000/geo-data"
INCIDENTS_URL = "http://127.0.0.1:8000/incidents"
INCIDENT_DETAIL_URL = "http://127.0.0.1:8000/incidents/{incident_id}"
INCIDENT_STATUS_URL = "http://127.0.0.1:8000/incidents/{incident_id}/status"
ANALYTICS_SUMMARY_URL = "http://127.0.0.1:8000/analytics/summary"
ANALYTICS_TRENDS_URL = "http://127.0.0.1:8000/analytics/trends"
ANALYTICS_DAMAGE_TYPES_URL = "http://127.0.0.1:8000/analytics/damage-types"
ANALYTICS_TOP_LOCATIONS_URL = "http://127.0.0.1:8000/analytics/top-risk-locations"
ANALYTICS_HOTSPOTS_URL = "http://127.0.0.1:8000/analytics/hotspots"
MODELS_URL = "http://127.0.0.1:8000/models"

STATUS_OPTIONS = [
    "all",
    "reported",
    "verified",
    "assigned",
    "in_progress",
    "resolved",
]
ALERT_OPTIONS = ["all", "NORMAL", "WATCH", "WARNING", "CRITICAL"]
ANALYTICS_WINDOWS = {
    "Last 7 Days": 7,
    "Last 30 Days": 30,
    "All Time": None,
}
AUTO_REFRESH_SECONDS = 10
MODEL_LABELS = {
    "default": "Default Pothole Model",
    "three_class": "3-Class Experimental Model",
}

selected_window_label = st.selectbox(
    "Analytics Window",
    list(ANALYTICS_WINDOWS.keys()),
    index=1,
)
selected_days = ANALYTICS_WINDOWS[selected_window_label]
analytics_params = {}
if selected_days is not None:
    analytics_params["days"] = selected_days

st.title("🛰️ RoadIQ Phase-8.5 — Live City Command Center")

available_models = [{"key": key, "label": label} for key, label in MODEL_LABELS.items()]
try:
    models_res = requests.get(MODELS_URL, timeout=2)
    if models_res.status_code == 200:
        fetched_models = models_res.json()
        if fetched_models:
            available_models = fetched_models
except Exception:
    pass

model_options = {item["label"]: item["key"] for item in available_models}
selected_model_label = st.selectbox(
    "Detection Model",
    list(model_options.keys()),
    index=0,
)
selected_model_key = model_options[selected_model_label]
st.caption(f"Active detection model: {selected_model_label}")

# ================================
# 🌍 LIVE CITY STATUS PANEL
# ================================
col1, col2, col3, col4 = st.columns(4)

live_feed_counter = col1.empty()
risk_metric = col2.empty()
health_metric = col3.empty()
alert_metric = col4.empty()

# ================================
# 📡 LIVE GEO MAP
# ================================
st.subheader("🌍 Live City Risk Map")

map_placeholder = st.empty()


def render_map(points):

    if not points:
        map_placeholder.info("No geo data yet")
        return

    df = pd.DataFrame(points)

    if "lat" not in df or "lon" not in df:
        map_placeholder.warning("Invalid geo format")
        return

    df["latitude"] = df["lat"]
    df["longitude"] = df["lon"]

    map_placeholder.map(df[["latitude", "longitude"]])


def render_hotspot_map(points, zones):
    if not points and not zones:
        map_placeholder.info("No hotspot data yet")
        return

    frames = []

    if points:
        point_df = pd.DataFrame(points)
        if "lat" in point_df and "lon" in point_df:
            point_df["latitude"] = point_df["lat"]
            point_df["longitude"] = point_df["lon"]
            frames.append(point_df[["latitude", "longitude"]])

    if zones:
        zone_df = pd.DataFrame(zones)
        if "lat" in zone_df and "lon" in zone_df:
            zone_df["latitude"] = zone_df["lat"]
            zone_df["longitude"] = zone_df["lon"]
            frames.append(zone_df[["latitude", "longitude"]])
            st.caption("Hotspot Zones")
            if "risk" in zone_df:
                zone_df = zone_df.sort_values(by=["risk", "count"], ascending=[False, False])
            preferred_columns = [
                column
                for column in ["risk_band", "risk", "count", "lat", "lon"]
                if column in zone_df.columns
            ]
            st.dataframe(zone_df[preferred_columns], use_container_width=True)

    if frames:
        map_placeholder.map(pd.concat(frames, ignore_index=True))
    else:
        map_placeholder.warning("Invalid hotspot geo format")


# ================================
# 📤 IMAGE UPLOAD + GEO INPUT
# ================================
st.subheader("🚧 Live Detection Feed")

uploaded = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

lat = st.number_input("Latitude", value=0.0)
lon = st.number_input("Longitude", value=0.0)

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Run City Brain Detection"):

        with st.spinner("AI City Brain thinking..."):

            try:
                files = {
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
                }

                res = requests.post(
                    BACKEND_URL,
                    files=files,
                    data={"lat": lat, "lon": lon, "model_key": selected_model_key},
                    timeout=60
                )

            except Exception:
                st.error("❌ Cannot reach backend API")
                st.stop()

        if res.status_code == 200:

            data = res.json()
            detections = data.get("detections", [])

            # =============================
            # 🧠 COMMAND CENTER METRICS
            # =============================
            live_feed_counter.metric(
                "🚧 Live Detection Feed",
                data.get("num_detections", 0)
            )
            risk_metric.metric("⚠️ Risk Score", data.get("risk_score", 0))
            health_metric.metric("🛣️ Road Health", data.get("road_health", 0))
            alert_metric.metric("🚨 Alert Level", data.get("alert_level", "N/A"))

            damage_summary = data.get("damage_type_summary", {})
            model_label = data.get("model_label", selected_model_label)
            if damage_summary:
                st.caption(
                    f"Model: {model_label} | Damage Types: "
                    + ", ".join(
                        f"{label}={count}" for label, count in damage_summary.items()
                    )
                )

            # =============================
            # 🧩 DRAW SMART BOXES
            # =============================
            drawn = image_np.copy()

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                severity = det["severity"]
                priority = det["priority"]
                display_label = det.get("display_label", det.get("label", "Damage"))
                color_rgb = det.get("color_rgb", [255, 255, 255])
                color = tuple(int(channel) for channel in reversed(color_rgb))

                cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 3)

                text = f"{display_label} | {priority} | Sev:{severity}"
                cv2.putText(
                    drawn,
                    text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            st.success("✅ City Brain Detection Complete")
            st.image(drawn, use_container_width=True)

            if detections:
                legend_rows = []
                for det in detections:
                    legend_rows.append(
                        {
                            "damage_type": det.get("display_label", det.get("label")),
                            "priority": det.get("priority"),
                            "severity": det.get("severity"),
                            "confidence": det.get("confidence"),
                        }
                    )
                with st.expander("Damage Legend"):
                    st.dataframe(pd.DataFrame(legend_rows), use_container_width=True)

            with st.expander("📊 Raw Geo-AI Data"):
                st.json(data)

        else:
            st.error(f"❌ Backend Error {res.status_code}")

# ================================
# 🔄 LIVE CITY STREAM
# ================================
st.subheader("📡 Live Geo Intelligence Feed")

try:
    hotspot_params = {}
    if selected_days is not None:
        hotspot_params["days"] = selected_days
    hotspot_res = requests.get(ANALYTICS_HOTSPOTS_URL, params=hotspot_params, timeout=2)
    if hotspot_res.status_code == 200:
        hotspot_data = hotspot_res.json()
        render_hotspot_map(
            hotspot_data.get("points", []),
            hotspot_data.get("zones", []),
        )
    else:
        geo_res = requests.get(GEO_URL, timeout=2)
        if geo_res.status_code == 200:
            render_map(geo_res.json())
        else:
            st.warning("Geo endpoint returned error")
except Exception:
    st.warning("Geo stream offline")


st.subheader("📈 Analytics")

analytics_col1, analytics_col2 = st.columns(2)

try:
    summary_res = requests.get(ANALYTICS_SUMMARY_URL, params=analytics_params, timeout=2)
    trend_params = {"limit": 30, **analytics_params}
    trends_res = requests.get(ANALYTICS_TRENDS_URL, params=trend_params, timeout=2)
    damage_res = requests.get(ANALYTICS_DAMAGE_TYPES_URL, params=analytics_params, timeout=2)
    top_location_params = {"limit": 5, **analytics_params}
    top_locations_res = requests.get(
        ANALYTICS_TOP_LOCATIONS_URL,
        params=top_location_params,
        timeout=2,
    )

    if summary_res.status_code == 200:
        summary = summary_res.json()
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        sum_col1.metric("Total Incidents", summary.get("total_incidents", 0))
        sum_col2.metric("Avg Risk", summary.get("avg_risk_score", 0))
        sum_col3.metric("Avg Road Health", summary.get("avg_road_health", 0))
        sum_col4.metric("Total Detections", summary.get("total_detections", 0))

        alert_breakdown = summary.get("alert_breakdown", [])
        if alert_breakdown:
            analytics_col1.caption("Alert Level Distribution")
            analytics_col1.bar_chart(
                pd.DataFrame(alert_breakdown).set_index("alert_level")
            )

        status_breakdown = summary.get("status_breakdown", [])
        if status_breakdown:
            analytics_col2.caption("Workflow Status Distribution")
            analytics_col2.bar_chart(
                pd.DataFrame(status_breakdown).set_index("status")
            )

    if trends_res.status_code == 200:
        trends = trends_res.json()
        if trends:
            st.caption(f"Daily Incident Trend ({selected_window_label})")
            trend_df = pd.DataFrame(trends).set_index("day")
            st.line_chart(
                trend_df[["incident_count", "avg_risk_score", "avg_road_health"]]
            )

    lower_col1, lower_col2 = st.columns(2)
    if damage_res.status_code == 200:
        damage_rows = damage_res.json()
        if damage_rows:
            lower_col1.caption("Damage Type Analytics")
            lower_col1.dataframe(pd.DataFrame(damage_rows), use_container_width=True)

    if top_locations_res.status_code == 200:
        top_locations = top_locations_res.json()
        if top_locations:
            lower_col2.caption("Top Risk Locations")
            lower_col2.dataframe(pd.DataFrame(top_locations), use_container_width=True)

except Exception:
    st.warning("Analytics endpoints offline")


st.subheader("🗂 Incident History")

filter_col1, filter_col2, filter_col3 = st.columns(3)
selected_status = filter_col1.selectbox("Status", STATUS_OPTIONS, index=0)
selected_alert = filter_col2.selectbox("Alert Level", ALERT_OPTIONS, index=0)
incident_limit = filter_col3.slider("Rows", min_value=5, max_value=50, value=20, step=5)

incident_params = {"limit": incident_limit}
if selected_status != "all":
    incident_params["status"] = selected_status
if selected_alert != "all":
    incident_params["alert_level"] = selected_alert

try:
    incident_res = requests.get(INCIDENTS_URL, params=incident_params, timeout=2)
    if incident_res.status_code == 200:
        incidents = incident_res.json()
        if incidents:
            status_counts = {
                "reported": 0,
                "verified": 0,
                "assigned": 0,
                "in_progress": 0,
                "resolved": 0,
            }
            table_rows = []
            for incident in incidents:
                incident_status = incident.get("status", "reported")
                if incident_status in status_counts:
                    status_counts[incident_status] += 1
                table_rows.append({
                    "id": incident.get("id"),
                    "time": incident.get("created_at"),
                    "file": incident.get("filename"),
                    "lat": incident.get("lat"),
                    "lon": incident.get("lon"),
                    "detections": incident.get("total_detections"),
                    "risk_score": incident.get("risk_score"),
                    "alert_level": incident.get("alert_level"),
                    "status": incident.get("status"),
                    "maintenance_priority": incident.get("maintenance_priority"),
                })

            summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
            summary_col1.metric("Reported", status_counts["reported"])
            summary_col2.metric("Verified", status_counts["verified"])
            summary_col3.metric("Assigned", status_counts["assigned"])
            summary_col4.metric("In Progress", status_counts["in_progress"])
            summary_col5.metric("Resolved", status_counts["resolved"])

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            incident_ids = [incident["id"] for incident in incidents]
            selected_incident_id = st.selectbox(
                "Inspect Incident",
                incident_ids,
                format_func=lambda incident_id: f"Incident #{incident_id}",
            )

            detail_res = requests.get(
                INCIDENT_DETAIL_URL.format(incident_id=selected_incident_id),
                timeout=2,
            )
            if detail_res.status_code == 200:
                detail = detail_res.json()
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                detail_col1.metric("Selected Risk Score", detail.get("risk_score", 0))
                detail_col2.metric("Selected Status", detail.get("status", "reported"))
                detail_col3.metric("Alert Level", detail.get("alert_level", "NORMAL"))

                status_form_col1, status_form_col2 = st.columns([1, 2])
                new_status = status_form_col1.selectbox(
                    "Update Status",
                    STATUS_OPTIONS[1:],
                    index=max(
                        0,
                        STATUS_OPTIONS[1:].index(detail.get("status", "reported"))
                        if detail.get("status", "reported") in STATUS_OPTIONS[1:]
                        else 0,
                    ),
                    key=f"status_select_{selected_incident_id}",
                )
                new_notes = status_form_col2.text_input(
                    "Notes",
                    value=detail.get("notes", ""),
                    key=f"notes_input_{selected_incident_id}",
                )

                if st.button("Save Incident Update", key=f"save_incident_{selected_incident_id}"):
                    update_res = requests.patch(
                        INCIDENT_STATUS_URL.format(incident_id=selected_incident_id),
                        json={"status": new_status, "notes": new_notes},
                        timeout=4,
                    )
                    if update_res.status_code == 200:
                        st.success("Incident updated")
                        st.rerun()
                    else:
                        st.error("Failed to update incident status")

                with st.expander("Incident Detail"):
                    st.json(detail)
            else:
                st.warning("Unable to load selected incident detail")
        else:
            st.info("No incident history saved yet")
    else:
        st.warning("Incident history endpoint returned error")
except Exception:
    st.warning("Incident history offline")

# ================================
# 🔁 AUTO REFRESH LOOP (SAFE)
# ================================
st.caption(f"Live City Brain refreshing every {AUTO_REFRESH_SECONDS} seconds...")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

now = time.time()

if now - st.session_state.last_refresh > AUTO_REFRESH_SECONDS:
    st.session_state.last_refresh = now
    st.rerun()









