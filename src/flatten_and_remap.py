"""
flatten_and_remap.py
---------------------

Merges multiple pre-split, Roboflow-style datasets (each shaped like)

    dataset_root/train/images/*.jpg   dataset_root/train/labels/*.txt
    dataset_root/valid/images/*.jpg   dataset_root/valid/labels/*.txt
    dataset_root/test/images/*.jpg    dataset_root/test/labels/*.txt

into a single flat pool inside raw_data/snake/, ready for
step1_dataset_collection.py.

WHY THIS IS NEEDED:
    step1_dataset_collection.py does NOT scan subfolders -- it only
    looks at files directly inside raw_data/snake/. Dumping Roboflow's
    train/valid/test structure in there as-is means Step 1 will
    silently collect zero images (no error, just nothing found).

    This script also deliberately DISCARDS each dataset's own
    train/valid/test split. Those splits were made independently by
    each dataset's creator; mixing three unrelated splits doesn't give
    you a coherent split for your own project. Instead everything is
    flattened into one pool, and step3_dataset_split.py creates ONE
    consistent split across the combined total.

WHAT THIS SCRIPT DOES:
    1. Walks every dataset root you list in DATASET_ROOTS below.
    2. Finds every images/*.jpg + labels/*.txt pair, regardless of
       which train/valid/test subfolder it came from.
    3. Copies each image into raw_data/snake/ with a per-dataset
       prefix, guaranteeing no filename collisions between datasets.
    4. Remaps every class ID in every label file to 0 -- this project
       treats "snake" as one class regardless of species, and Step 2
       silently drops any box whose class ID isn't 0. Writes the
       remapped label next to its image, same stem, so
       match_labels.py can find it after Step 1 renames the image.

CONFIGURE:
    Edit DATASET_ROOTS below: (prefix, path_to_downloaded_dataset).
    Prefixes must be unique and filesystem-safe (letters/numbers/
    underscores).

USAGE:
    python flatten_and_remap.py
    (then) python step1_dataset_collection.py
    (then) python match_labels.py
"""

import shutil
from pathlib import Path

# ----------------------------------------------------------------------
# EDIT THIS: one entry per downloaded dataset
# ----------------------------------------------------------------------

DATASET_ROOTS = [
    ("ds1", Path("downloads/roboflow_dataset_1")),
    ("ds2", Path("downloads/roboflow_dataset_2")),
    ("ds3", Path("downloads/other_dataset_3")),
]

OUTPUT_DIR = Path("raw_data/snake")

SPLIT_SUBFOLDERS = ["train", "valid", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(dataset_root: Path):
    """
    Yield (image_path, label_path_or_None) for every image found under
    any split/images subfolder of dataset_root, or a flat images/
    folder directly under the root.
    """

    candidate_dirs = []

    for split in SPLIT_SUBFOLDERS:
        images_dir = dataset_root / split / "images"
        if images_dir.exists():
            candidate_dirs.append(images_dir)

    flat_images = dataset_root / "images"
    if flat_images.exists():
        candidate_dirs.append(flat_images)

    for images_dir in candidate_dirs:

        labels_dir = images_dir.parent / "labels"

        for image_path in images_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"

            yield image_path, (label_path if label_path.exists() else None)


def remap_label_to_single_class(label_path: Path) -> str:
    """Read a YOLO label file and remap every class ID in it to 0."""

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()

    remapped_lines = []

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            # Leave malformed lines as-is; Step 2 will flag them.
            remapped_lines.append(line)
            continue

        _, xc, yc, bw, bh = parts

        remapped_lines.append(f"0 {xc} {yc} {bw} {bh}")

    return "\n".join(remapped_lines) + ("\n" if remapped_lines else "")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0
    missing_labels = 0

    for prefix, dataset_root in DATASET_ROOTS:

        if not dataset_root.exists():
            print(f"⚠ Skipping missing dataset root: {dataset_root}")
            continue

        dataset_image_count = 0

        for image_path, label_path in find_pairs(dataset_root):

            new_stem = f"{prefix}__{image_path.stem}"

            new_image_path = OUTPUT_DIR / f"{new_stem}{image_path.suffix.lower()}"
            shutil.copy(str(image_path), str(new_image_path))

            dataset_image_count += 1
            total_images += 1

            if label_path is not None:

                remapped = remap_label_to_single_class(label_path)

                new_label_path = OUTPUT_DIR / f"{new_stem}.txt"
                new_label_path.write_text(remapped, encoding="utf-8")

                total_labels += 1

            else:
                missing_labels += 1

        print(f"{prefix}: {dataset_image_count} images copied from {dataset_root}")

    print(f"\nTotal images copied: {total_images}")
    print(f"Total labels remapped and copied: {total_labels}")

    if missing_labels:
        print(
            f"⚠ {missing_labels} images had no matching label file "
            "and were copied without one (Step 2 will treat these as "
            "negative/background unless that's a mistake in the source)."
        )

    print(
        "\nDone. raw_data/snake/ now contains a flat, single-class, "
        "collision-safe pool. Run step1_dataset_collection.py next."
    )


if __name__ == "__main__":
    main()


