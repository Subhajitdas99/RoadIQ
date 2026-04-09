from pathlib import Path

RUNS_DETECT_DIR = Path("runs/detect")
MODEL_WEIGHTS_PATH = Path("runs/detect/roadIQ_road_damage/weights/best.pt")
LEGACY_MODEL_WEIGHTS_PATH = Path("runs/detect/roadIQ_pothole/weights/best.pt")
THREE_CLASS_MODEL_WEIGHTS_PATH = Path("runs/detect/roadIQ_three_class/weights/best.pt")
DATASET_CONFIG_PATH = Path("dataset/data.yaml")
THREE_CLASS_DATASET_CONFIG_PATH = Path("dataset_three_class/data.yaml")

DEFAULT_CLASS_NAMES = [
    "pothole",
    "crack",
    "waterlogging",
    "road_patch",
    "debris",
]
THREE_CLASS_NAMES = [
    "pothole",
    "crack",
    "manhole",
]

TRAINING_RUN_NAME = "roadIQ_road_damage"
TRAINING_EPOCHS = 30
TRAINING_IMAGE_SIZE = 512
TRAINING_BATCH_SIZE = 2
TRAINING_DEVICE = "cpu"
TRAINING_WORKERS = 0
TRAINING_AMP = False


MODEL_OPTIONS = {
    "default": {
        "label": "Default Pothole Model",
        "run_prefix": "roadIQ_road_damage",
        "fallback_weights": MODEL_WEIGHTS_PATH,
        "legacy_fallback_weights": LEGACY_MODEL_WEIGHTS_PATH,
        "dataset_config": DATASET_CONFIG_PATH,
        "class_names": DEFAULT_CLASS_NAMES,
        "model_version": "roadIQ-road-damage-v2-ready",
    },
    "three_class": {
        "label": "3-Class Experimental Model",
        "run_prefix": "roadIQ_three_class",
        "fallback_weights": THREE_CLASS_MODEL_WEIGHTS_PATH,
        "legacy_fallback_weights": THREE_CLASS_MODEL_WEIGHTS_PATH,
        "dataset_config": THREE_CLASS_DATASET_CONFIG_PATH,
        "class_names": THREE_CLASS_NAMES,
        "model_version": "roadIQ-three-class-exp",
    },
}


def get_model_option(model_key: str) -> dict:
    normalized_key = (model_key or "default").strip().lower()
    return MODEL_OPTIONS.get(normalized_key, MODEL_OPTIONS["default"])


def resolve_best_model_weights(model_key: str = "default") -> Path:
    model_option = get_model_option(model_key)
    candidates = []

    if RUNS_DETECT_DIR.exists():
        for run_dir in RUNS_DETECT_DIR.glob(f"{model_option['run_prefix']}*"):
            best_path = run_dir / "weights" / "best.pt"
            if best_path.exists():
                candidates.append(best_path)

    fallback_weights = model_option["fallback_weights"]
    if fallback_weights.exists():
        candidates.append(fallback_weights)

    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    return model_option["legacy_fallback_weights"]
