"""
match_labels.py
----------------

Bridges Step 1's output with externally-downloaded YOLO label files
(e.g. from a Roboflow export).

THE PROBLEM:
    Step 1 renames every collected image
        img001.jpg -> snake__img001__jpg.jpg
    but does not know about, move, or rename any .txt label file that
    came bundled with the original image.

    Step 2 expects label files in dataset/yolo_labels/ named to match
    the RENAMED image stem, not the original filename. Without this
    script, every snake image would silently be treated as a NEGATIVE
    (no snake) by Step 2, since it won't find a matching label file.

WHAT THIS SCRIPT DOES:
    Reads dataset/metadata.csv (written by Step 1). For every row where
    source_type == "snake", it uses the "original_source" column to find
    that image's original .txt label, and copies it into
    dataset/yolo_labels/ under the NEW filename's stem.

    Hard-negative images (rope/hose/cable/vine/other) are skipped on
    purpose -- they should have no label file, which Step 2 correctly
    interprets as "no snake in this image".

BEFORE RUNNING:
    1. Run step1_dataset_collection.py first.
    2. Make sure your downloaded .txt label files sit next to their
       original images in raw_data/snake/ (same filename stem), OR set
       EXTERNAL_LABELS_DIR below to wherever you kept them.

USAGE:
    python match_labels.py
"""

import csv
import shutil
from pathlib import Path
from typing import Optional

DATASET_DIR = Path("dataset")
METADATA_PATH = DATASET_DIR / "metadata.csv"
YOLO_LABELS_DIR = DATASET_DIR / "yolo_labels"

# If your downloaded .txt labels are NOT sitting next to the original
# images referenced in metadata.csv's "original_source" column, point
# this at the folder that contains them instead. Leave as None to look
# for a same-name .txt right next to each original image.
EXTERNAL_LABELS_DIR: Optional[Path] = None  # e.g. Path("raw_data/snake_labels")


def find_label_file(original_source: Path) -> Optional[Path]:
    """Locate the original YOLO label file for a given original image."""

    stem = original_source.stem

    if EXTERNAL_LABELS_DIR is not None:
        candidate = EXTERNAL_LABELS_DIR / f"{stem}.txt"
    else:
        candidate = original_source.with_suffix(".txt")

    return candidate if candidate.exists() else None


def main():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Run step1_dataset_collection.py first."
        )

    YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing = []

    with METADATA_PATH.open("r", newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            # Hard negatives intentionally get no label file.
            if row["source_type"] != "snake":
                continue

            new_stem = Path(row["filename"]).stem
            original_source = Path(row["original_source"])

            label_file = find_label_file(original_source)

            if label_file is None:
                missing.append(row["filename"])
                continue

            destination = YOLO_LABELS_DIR / f"{new_stem}.txt"
            shutil.copy(str(label_file), str(destination))
            matched += 1

    print(f"Matched and copied {matched} label files into {YOLO_LABELS_DIR}")

    if missing:

        print(
            f"\n⚠ {len(missing)} snake images had NO matching label file found:"
        )

        for name in missing[:20]:
            print(f"  - {name}")

        if len(missing) > 20:
            print(f"  ...and {len(missing) - 20} more")

        print(
            "\nThese will be treated as NEGATIVE (no snake) by Step 2 "
            "unless you locate/add their labels before running it."
        )


if __name__ == "__main__":
    main()