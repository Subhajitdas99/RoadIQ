from ultralytics import YOLO
import numpy as np

# ✅ Use your trained pothole model
model = YOLO("runs/detect/roadIQ_pothole/weights/best.pt")


def run_inference(image_np: np.ndarray):
    """
    Input: image numpy array (H, W, C)
    Output: list of pothole detections with severity & priority
    """

    results = model(image_np)
    detections = []

    # Image area for severity calculation
    img_h, img_w, _ = image_np.shape
    img_area = img_h * img_w

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:

            # --- Bounding box ---
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            # --- Confidence & class ---
            conf = float(box.conf[0].cpu())
            cls = int(box.cls[0].cpu())
            label = model.names.get(cls, str(cls))

            # ✅ Keep ONLY potholes
            if label != "pothole":
                continue

            # --- SEVERITY LOGIC ---
            bbox_area = abs((x2 - x1) * (y2 - y1))
            severity_score = round((bbox_area / img_area) * conf * 100, 2)

            if severity_score < 10:
                priority = "Low"
            elif severity_score < 30:
                priority = "Medium"
            else:
                priority = "High"

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ],
                "severity": severity_score,
                "priority": priority
            })

    return detections


