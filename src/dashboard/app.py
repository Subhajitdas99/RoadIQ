import streamlit as st
import requests

st.set_page_config(page_title="RoadIQ", layout="wide")
st.title("🚧 RoadIQ – Road Damage Detection Dashboard")

BACKEND_URL = "http://127.0.0.1:8000/detect"

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        res = requests.post(BACKEND_URL, files={"file": uploaded})

        if res.status_code == 200:
            st.success("✅ Detection completed!")
            st.json(res.json())
        else:
            st.error("❌ Backend error. Start FastAPI server first.")
