from ultralytics import YOLO
import numpy as np

# MVP: pretrained YOLO model (replace later with pothole-trained weights)
model = YOLO("yolov8n.pt")

def run_inference(image_np: np.ndarray):
    """
    Input: image numpy array
    Output: list of detections (bbox + confidence + label)
    """
    results = model(image_np)
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

    return detections
