from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from PIL import Image

from src.inference.model_config import DATASET_CONFIG_PATH, DEFAULT_CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_dataset_config() -> dict:
    config_path = PROJECT_ROOT / DATASET_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def resolve_split_dir(config: dict, key: str) -> Path:
    raw_path = config.get(key)
    if not raw_path:
        raise ValueError(f"Missing '{key}' in dataset config")
    return (PROJECT_ROOT / "dataset" / Path(raw_path).relative_to(".")).resolve()


def image_to_label_path(image_path: Path) -> Path:
    labels_dir = image_path.parent.parent / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def validate_image(image_path: Path) -> str | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return None
    except Exception:
        return "unreadable or unsupported image"


def validate_label_file(
    label_path: Path,
    max_class_id: int,
    class_counter: Counter,
) -> list[str]:
    errors: list[str] = []

    if not label_path.exists():
        return [f"missing label file: {label_path}"]

    lines = label_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [f"empty label file: {label_path}"]

    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no} invalid YOLO format")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no} non-numeric label values")
            continue

        if class_id < 0 or class_id > max_class_id:
            errors.append(
                f"{label_path}:{line_no} invalid class id {class_id} (expected 0-{max_class_id})"
            )
        else:
            class_counter[class_id] += 1

        for coord in coords:
            if coord < 0 or coord > 1:
                errors.append(
                    f"{label_path}:{line_no} bbox values must be normalized between 0 and 1"
                )
                break

    return errors


def validate_split(split_name: str, image_dir: Path, class_names: list[str]) -> dict:
    class_counter: Counter = Counter()
    errors: list[str] = []
    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    label_files_found = 0
    for image_path in images:
        image_error = validate_image(image_path)
        if image_error:
            errors.append(f"{image_path}: {image_error}")

        label_path = image_to_label_path(image_path)
        if label_path.exists():
            label_files_found += 1
        errors.extend(
            validate_label_file(
                label_path=label_path,
                max_class_id=len(class_names) - 1,
                class_counter=class_counter,
            )
        )

    return {
        "split": split_name,
        "image_count": len(images),
        "label_file_count": label_files_found,
        "class_counter": class_counter,
        "errors": errors,
    }


def print_report(results: list[dict], class_names: list[str]) -> None:
    print("RoadIQ Dataset Validation Report")
    print("=" * 40)
    print(f"Classes: {', '.join(class_names)}")
    print()

    total_counter: Counter = Counter()
    total_errors = 0

    for result in results:
        print(f"[{result['split']}]")
        print(f"Images: {result['image_count']}")
        print(f"Label files: {result['label_file_count']}")
        print("Class distribution:")
        for class_id, class_name in enumerate(class_names):
            count = result["class_counter"].get(class_id, 0)
            print(f"  {class_id} {class_name}: {count}")
            total_counter[class_id] += count

        error_count = len(result["errors"])
        total_errors += error_count
        print(f"Errors: {error_count}")
        if error_count:
            for error in result["errors"][:20]:
                print(f"  - {error}")
            if error_count > 20:
                print(f"  - ... {error_count - 20} more")
        print()

    print("[total]")
    for class_id, class_name in enumerate(class_names):
        print(f"  {class_name}: {total_counter.get(class_id, 0)}")
    print(f"Total errors: {total_errors}")


def main() -> int:
    config = load_dataset_config()
    class_names = config.get("names") or DEFAULT_CLASS_NAMES

    train_dir = resolve_split_dir(config, "train")
    val_dir = resolve_split_dir(config, "val")

    results = [
        validate_split("train", train_dir, class_names),
        validate_split("val", val_dir, class_names),
    ]
    print_report(results, class_names)

    has_errors = any(result["errors"] for result in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
