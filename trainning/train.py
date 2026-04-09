from pathlib import Path
import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RoadIQ YOLO model.")
    parser.add_argument(
        "--data",
        default=str(DATASET_CONFIG_PATH),
        help="Path to YOLO data.yaml",
    )
    parser.add_argument(
        "--name",
        default=TRAINING_RUN_NAME,
        help="Ultralytics run name under runs/detect/",
    )
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=TRAINING_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=TRAINING_BATCH_SIZE)
    parser.add_argument("--device", default=TRAINING_DEVICE)
    parser.add_argument("--workers", type=int, default=TRAINING_WORKERS)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=TRAINING_AMP,
        help="Enable or disable AMP",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = ", ".join(DEFAULT_CLASS_NAMES)

    print(f"Training RoadIQ for classes: {class_names}")
    print(f"Dataset config: {args.data}")
    print(f"Run name: {args.name}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        name=args.name,
    )


if __name__ == "__main__":
    main()
