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

st.title("🛰️ RoadIQ Phase-8.5 — Live City Command Center")

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
                    data={"lat": lat, "lon": lon},
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

            # =============================
            # 🧩 DRAW SMART BOXES
            # =============================
            drawn = image_np.copy()

            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                severity = det["severity"]
                priority = det["priority"]

                if priority == "High":
                    color = (255, 0, 0)
                elif priority == "Medium":
                    color = (255, 255, 0)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 3)

                text = f"{priority} | Sev:{severity}"
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

            with st.expander("📊 Raw Geo-AI Data"):
                st.json(data)

        else:
            st.error(f"❌ Backend Error {res.status_code}")

# ================================
# 🔄 LIVE CITY STREAM
# ================================
st.subheader("📡 Live Geo Intelligence Feed")

try:
    geo_res = requests.get(GEO_URL, timeout=2)
    if geo_res.status_code == 200:
        render_map(geo_res.json())
    else:
        st.warning("Geo endpoint returned error")
except Exception:
    st.warning("Geo stream offline")

# ================================
# 🔁 AUTO REFRESH LOOP (SAFE)
# ================================
st.caption("🔄 Live City Brain refreshing every 2 seconds...")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

now = time.time()

if now - st.session_state.last_refresh > 2:
    st.session_state.last_refresh = now
    st.rerun()







