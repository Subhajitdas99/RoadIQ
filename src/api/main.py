from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from src.inference.detector import run_inference

app = FastAPI(title="RoadIQ API")

@app.get("/")
def root():
    return {"message": "RoadIQ backend running ✅"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    detections = run_inference(img)

    return {
        "filename": file.filename,
        "num_detections": len(detections),
        "detections": detections
    }
