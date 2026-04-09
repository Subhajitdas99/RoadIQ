from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.inference.model_config import (
    DATASET_CONFIG_PATH,
    DEFAULT_CLASS_NAMES,
    TRAINING_AMP,
    TRAINING_BATCH_SIZE,
    TRAINING_DEVICE,
    TRAINING_EPOCHS,
    TRAINING_IMAGE_SIZE,
    TRAINING_RUN_NAME,
    TRAINING_WORKERS,
)

model = YOLO("yolov8n.pt")
CLASS_NAMES = ", ".join(DEFAULT_CLASS_NAMES)

# Train this on a dataset with multiple classes such as:
# pothole, crack, waterlogging, road_patch, debris
# Keep the class order in dataset/data.yaml aligned with:
# pothole, crack, waterlogging, road_patch, debris
print(f"Training RoadIQ for classes: {CLASS_NAMES}")
model.train(
    data=str(DATASET_CONFIG_PATH),
    epochs=TRAINING_EPOCHS,
    imgsz=TRAINING_IMAGE_SIZE,
    batch=TRAINING_BATCH_SIZE,
    device=TRAINING_DEVICE,
    workers=TRAINING_WORKERS,
    amp=TRAINING_AMP,
    name=TRAINING_RUN_NAME,
)
