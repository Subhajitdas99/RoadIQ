from __future__ import annotations

from pathlib import Path
import random
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = PROJECT_ROOT / "new_dataset"
OUTPUT_DIR = PROJECT_ROOT / "dataset_three_class"
IMAGE_DIR = SOURCE_DIR / "images"
LABEL_DIR = SOURCE_DIR / "labels-YOLO"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CLASS_NAMES = ["pothole", "crack", "manhole"]


def ensure_output_dirs() -> None:
    for split in ("train", "valid"):
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def build_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def copy_pairs(pairs: list[tuple[Path, Path]], split: str) -> None:
    image_target_dir = OUTPUT_DIR / split / "images"
    label_target_dir = OUTPUT_DIR / split / "labels"

    for image_path, label_path in pairs:
        shutil.copy2(image_path, image_target_dir / image_path.name)
        shutil.copy2(label_path, label_target_dir / label_path.name)


def write_data_yaml() -> None:
    yaml_text = "\n".join(
        [
            "train: ./train/images",
            "val: ./valid/images",
            "",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  - {name}" for name in CLASS_NAMES],
            "",
        ]
    )
    (OUTPUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f"Missing source dataset: {SOURCE_DIR}")
        return 1

    pairs = build_pairs()
    if not pairs:
        print("No image/label pairs found in new_dataset")
        return 1

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    train_count = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:train_count]
    valid_pairs = pairs[train_count:]

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    ensure_output_dirs()
    copy_pairs(train_pairs, "train")
    copy_pairs(valid_pairs, "valid")
    write_data_yaml()

    print(f"Prepared 3-class dataset at: {OUTPUT_DIR}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Valid pairs: {len(valid_pairs)}")
    print("Classes: pothole, crack, manhole")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
