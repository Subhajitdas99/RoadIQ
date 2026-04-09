from pathlib import Path

RUNS_DETECT_DIR = Path("runs/detect")
MODEL_WEIGHTS_PATH = Path("runs/detect/roadIQ_road_damage/weights/best.pt")
LEGACY_MODEL_WEIGHTS_PATH = Path("runs/detect/roadIQ_pothole/weights/best.pt")
DATASET_CONFIG_PATH = Path("dataset/data.yaml")

DEFAULT_CLASS_NAMES = [
    "pothole",
    "crack",
    "waterlogging",
    "road_patch",
    "debris",
]

TRAINING_RUN_NAME = "roadIQ_road_damage"
TRAINING_EPOCHS = 30
TRAINING_IMAGE_SIZE = 512
TRAINING_BATCH_SIZE = 2
TRAINING_DEVICE = "cpu"
TRAINING_WORKERS = 0
TRAINING_AMP = False


def resolve_best_model_weights() -> Path:
    candidates = []

    if RUNS_DETECT_DIR.exists():
        for run_dir in RUNS_DETECT_DIR.glob("roadIQ_road_damage*"):
            best_path = run_dir / "weights" / "best.pt"
            if best_path.exists():
                candidates.append(best_path)

    if MODEL_WEIGHTS_PATH.exists():
        candidates.append(MODEL_WEIGHTS_PATH)

    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    return LEGACY_MODEL_WEIGHTS_PATH
