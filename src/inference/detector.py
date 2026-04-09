from ultralytics import YOLO
import numpy as np

from src.inference.damage_config import (
    DEFAULT_DAMAGE_CONFIG,
    SUPPORTED_DAMAGE_TYPES,
)
from src.inference.model_config import get_model_option, resolve_best_model_weights

_loaded_models: dict[str, tuple[YOLO, str]] = {}


def load_model(model_key: str = "default") -> tuple[YOLO, str]:
    normalized_key = (model_key or "default").strip().lower()
    weights_path = str(resolve_best_model_weights(normalized_key))
    cached_model = _loaded_models.get(normalized_key)

    if cached_model and cached_model[1] == weights_path:
        return cached_model

    model = YOLO(weights_path)
    _loaded_models[normalized_key] = (model, weights_path)
    return model, weights_path


def run_inference(image_np: np.ndarray, model_key: str = "default"):
    """
    Input: image numpy array (H, W, C)
    Output: list of road-damage detections with severity and priority
    """
    model, weights_path = load_model(model_key)
    model_option = get_model_option(model_key)
    results = model(image_np)
    detections = []

    img_h, img_w, _ = image_np.shape
    img_area = max(img_h * img_w, 1)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            conf = float(box.conf[0].cpu())
            cls = int(box.cls[0].cpu())
            raw_label = model.names.get(cls, str(cls))
            normalized_label = str(raw_label).strip().lower().replace(" ", "_")

            if normalized_label not in SUPPORTED_DAMAGE_TYPES:
                continue

            damage_config = SUPPORTED_DAMAGE_TYPES.get(
                normalized_label,
                DEFAULT_DAMAGE_CONFIG,
            )

            bbox_area = abs((x2 - x1) * (y2 - y1))
            severity_score = round(
                (bbox_area / img_area)
                * conf
                * 100
                * damage_config["severity_weight"],
                2,
            )

            if severity_score < 10:
                priority = "Low"
            elif severity_score < 30:
                priority = "Medium"
            else:
                priority = "High"

            detections.append(
                {
                    "label": normalized_label,
                    "display_label": damage_config["display_name"],
                    "confidence": round(conf, 4),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    "severity": severity_score,
                    "priority": priority,
                    "severity_weight": damage_config["severity_weight"],
                    "color_rgb": list(damage_config["color_rgb"]),
                }
            )

    return {
        "detections": detections,
        "model_key": (model_key or "default").strip().lower(),
        "model_label": model_option["label"],
        "model_version": model_option["model_version"],
        "weights_path": weights_path,
    }
