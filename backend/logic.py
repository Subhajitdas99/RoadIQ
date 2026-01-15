import cv2
import numpy as np
import os
import uuid
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path safety: Ensures the model is found regardless of where you run the terminal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_images')

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Model (with fallback)
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Warning: Custom model '{MODEL_PATH}' not found.")
    print("Falling back to standard 'yolov8n.pt' for demonstration.")
    model = YOLO('yolov8n.pt') 
else:
    print(f"✅ Loading Custom Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

def calculate_severity(detections, img_area):
    """
    Calculates severity based on the total area of damage relative to the road.
    """
    total_damage_area = 0
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        total_damage_area += (width * height)

    # Severity Score: Percentage of image covered by damage (0.0 to 1.0)
    severity_score = total_damage_area / img_area
    
    # Prioritization Logic
    if severity_score > 0.10: # If >10% of the view is damaged
        return "Critical", severity_score, (0, 0, 255)   # Red
    elif severity_score > 0.02:
        return "High", severity_score, (0, 165, 255)     # Orange
    elif severity_score > 0:
        return "Medium", severity_score, (0, 255, 255)   # Yellow
    else:
        return "Safe", 0.0, (0, 255, 0)                  # Green

def process_frame(rgb_frame, filename, source_type="Image"):
    """
    Main logic function called by API.
    Returns: has_damage (bool), severity (float), priority (str), save_path (str)
    """
    try:
        # 1. Run Inference
        # We process the RGB frame directly
        results = model(rgb_frame, verbose=False)
        detections = results[0].boxes
        
        # 2. Analyze Detections
        h, w, _ = rgb_frame.shape
        img_area = w * h
        
        has_damage = len(detections) > 0
        priority = "Safe"
        severity = 0.0
        color = (0, 255, 0) 

        if has_damage:
            priority, severity, color = calculate_severity(detections, img_area)

        # 3. Visualization (Draw Boxes)
        # We work on a copy to avoid modifying the original array in place if reused
        annotated_frame = rgb_frame.copy()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Draw Rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"{priority} {int(box.conf[0]*100)}%"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. Save Image
        # Use UUID to ensure unique filenames if multiple people upload "image.jpg"
        unique_filename = f"{uuid.uuid4()}_{filename}"
        save_path = os.path.join(OUTPUT_DIR, unique_filename)
        
        # Convert RGB back to BGR for OpenCV saving
        bgr_save_img = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_save_img)
        
        return has_damage, round(severity, 4), priority, save_path

    except Exception as e:
        print(f"❌ Logic Error: {e}")
        # Return safe defaults so the API doesn't crash
        return False, 0.0, "Error", ""