import cv2
import time
from src.inference.detector import run_inference
from src.realtime.event_bus import push_event

def start_camera_stream(camera_source=0):

    cap = cv2.VideoCapture(camera_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(frame)

        event = {
            "timestamp": time.time(),
            "num_detections": len(detections),
            "detections": detections
        }

        push_event(event)

        # simulate realtime
        time.sleep(1)
