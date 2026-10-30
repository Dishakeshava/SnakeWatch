"""
Step 1: Dataset Collection and Curation
---------------------------------------

Purpose:
    - Organize raw snake and hard-negative data
    - Extract frames from videos
    - Remove corrupt images
    - Remove near-duplicate images using perceptual hashing
    - Preserve source/video identity in metadata
    - Track negative subtype (rope, hose, cable, vine, other)
    - Report dataset composition and quality statistics

This script DOES NOT:
    - annotate images
    - train the detector
    - split train/val/test
    - perform augmentation

Expected raw structure:

raw_data/
    snake/
        snake_images...
        snake_video_01.mp4
        snake_video_02.mp4

    negative_rope/
        rope_images...
        rope_video_01.mp4

    negative_hose/
        ...

    negative_cable/
        ...

    negative_vine/
        ...

    negative_other/
        ...

Output:

dataset/
    images/
        *.jpg

    metadata.csv

    curation_report.json

    source_videos/
        ...

The metadata file preserves:
    filename
    source_id
    source_type
    negative_subtype
    frame_index
    original_source
    width
    height
    file_size_bytes
    quality information
"""

import csv
import json
import shutil
from pathlib import Path
from collections import Counter

import cv2
import imagehash

from PIL import Image
from tqdm import tqdm


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

RAW_DIR = Path("raw_data")
HARD_NEGATIVES_DIR = Path("hard_negatives")
DATASET_DIR = Path("dataset")

IMAGES_DIR = DATASET_DIR / "images"
METADATA_PATH = DATASET_DIR / "metadata.csv"
REPORT_PATH = DATASET_DIR / "curation_report.json"

HASH_THRESHOLD = 5

# Frame extraction:
# Save every Nth frame from a video.
DEFAULT_EVERY_N_FRAMES = 5

# Quality thresholds.
# These do NOT automatically remove an image.
# They are used to flag potentially problematic samples.
MIN_WIDTH = 320
MIN_HEIGHT = 240

# Very low Laplacian variance often indicates a blurry image.
BLUR_THRESHOLD = 40.0

# Very dark / bright image thresholds.
DARK_MEAN_THRESHOLD = 35.0
BRIGHT_MEAN_THRESHOLD = 220.0

RANDOM_SEED = 42

NEGATIVE_SUBTYPES = {
    "negative_rope": "rope",
    "negative_hose": "hose",
    "negative_cable": "cable",
    "negative_vine": "vine",
    "negative_other": "other",
}

# New hard negatives supplied separately at project root.
HARD_NEGATIVE_SUBTYPES = {
    "Vegetation": "vegetation",
    "Vines": "vine",
    "Ropes": "rope",
    "Shadows": "shadow",
    "Roots": "root",
    "Curled leaves": "curled_leaf",
    "Hoses": "hose",
    "Logs": "log",
    "Pipes": "pipe",
    "Branches": "branch",
    "Cables": "cable",
}


# ============================================================================
# 2. FOLDER SETUP
# ============================================================================

def setup_folders():
    """Create the required output folders."""

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 3. IMAGE QUALITY ANALYSIS
# ============================================================================

def analyze_image_quality(image):
    """
    Calculate simple quality indicators.

    These values are stored in metadata so that poor-quality samples
    can be reviewed later.

    The function does NOT automatically delete blurry/dark images.
    """

    if image is None:
        return {
            "width": 0,
            "height": 0,
            "blur_score": 0.0,
            "brightness": 0.0,
            "quality_flag": "unreadable",
        }

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variance of Laplacian is a simple blur/sharpness indicator.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = float(gray.mean())

    flags = []

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        flags.append("low_resolution")

    if blur_score < BLUR_THRESHOLD:
        flags.append("possible_blur")

    if brightness < DARK_MEAN_THRESHOLD:
        flags.append("very_dark")

    if brightness > BRIGHT_MEAN_THRESHOLD:
        flags.append("very_bright")

    if flags:
        quality_flag = "|".join(flags)
    else:
        quality_flag = "ok"

    return {
        "width": width,
        "height": height,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "quality_flag": quality_flag,
    }


# ============================================================================
# 4. SOURCE ID
# ============================================================================

def make_source_id(source_path: Path, source_type: str):
    """
    Create a stable source identifier.

    Example:

        raw_data/snake/snake_video_01.mp4
        -> snake__snake_video_01

        raw_data/negative_rope/rope_video_01.mp4
        -> negative_rope__rope_video_01
    """

    return f"{source_type}__{source_path.stem}"


# ============================================================================
# 5. COPY STATIC IMAGES
# ============================================================================

def collect_images_from_folder(
    source_folder: Path,
    source_type: str,
    negative_subtype: str,
    metadata_rows: list,
):
    """
    Copy images from a raw source folder into dataset/images.

    Every copied image receives a metadata record containing its source.
    """

    if not source_folder.exists():
        return

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_files = [
        f for f in source_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    for image_path in tqdm(
        image_files,
        desc=f"Collecting {source_type}"
    ):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"⚠ Could not read image: {image_path}")
            continue

        quality = analyze_image_quality(image)

        if source_type == "hard_negative":
            # Include subtype so identical filenames in different hard-negative
            # folders cannot collide.
            source_id = (
                f"hard_negative__{negative_subtype}__"
                f"{image_path.stem}"
            )
            output_name = (
                f"hard_negative__{negative_subtype}__"
                f"{image_path.stem}__"
                f"{image_path.suffix.lower().replace('.', '')}.jpg"
            )
        else:
            source_id = make_source_id(image_path, source_type)
            output_name = (
                f"{source_type}__"
                f"{image_path.stem}__"
                f"{image_path.suffix.lower().replace('.', '')}.jpg"
            )

        output_path = IMAGES_DIR / output_name

        # Prevent duplicate copies if the script is accidentally run twice.
        if output_path.exists():
            continue

        cv2.imwrite(str(output_path), image)

        metadata_rows.append({
            "filename": output_name,
            "source_id": source_id,
            "source_type": source_type,
            "negative_subtype": negative_subtype,
            "frame_index": "",
            "original_source": str(image_path),
            "width": quality["width"],
            "height": quality["height"],
            "blur_score": quality["blur_score"],
            "brightness": quality["brightness"],
            "quality_flag": quality["quality_flag"],
        })


# ============================================================================
# 6. VIDEO FRAME EXTRACTION
# ============================================================================

def extract_frames(
    video_path: Path,
    source_type: str,
    negative_subtype: str,
    metadata_rows: list,
    every_n_frames: int = DEFAULT_EVERY_N_FRAMES,
):
    """
    Extract every Nth frame from a video.

    IMPORTANT:
        The source video ID is stored for every frame.

    This allows Step 3 to split by source video rather than by individual
    frame, preventing train/validation/test leakage.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"⚠ Could not open video: {video_path}")
        return

    source_id = make_source_id(video_path, source_type)

    frame_index = 0
    saved_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % every_n_frames == 0:

            quality = analyze_image_quality(frame)

            output_name = (
                f"{source_id}__"
                f"frame_{frame_index:07d}.jpg"
            )

            output_path = IMAGES_DIR / output_name

            cv2.imwrite(str(output_path), frame)

            metadata_rows.append({
                "filename": output_name,
                "source_id": source_id,
                "source_type": source_type,
                "negative_subtype": negative_subtype,
                "frame_index": frame_index,
                "original_source": str(video_path),
                "width": quality["width"],
                "height": quality["height"],
                "blur_score": quality["blur_score"],
                "brightness": quality["brightness"],
                "quality_flag": quality["quality_flag"],
            })

            saved_index += 1

        frame_index += 1

    cap.release()

    print(
        f"Extracted {saved_index} frames from "
        f"{video_path.name}"
    )


# ============================================================================
# 7. COLLECT ALL RAW DATA
# ============================================================================

def collect_raw_dataset():
    """
    Collect images and frames from all configured source folders.
    """

    metadata_rows = []

    # ------------------------------------------------------------------------
    # Snake
    # ------------------------------------------------------------------------

    snake_dir = RAW_DIR / "snake"

    if snake_dir.exists():

        for path in snake_dir.iterdir():

            if path.is_file() and path.suffix.lower() in {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm",
            }:

                extract_frames(
                    path,
                    source_type="snake",
                    negative_subtype="",
                    metadata_rows=metadata_rows,
                )

        collect_images_from_folder(
            snake_dir,
            source_type="snake",
            negative_subtype="",
            metadata_rows=metadata_rows,
        )

    # ------------------------------------------------------------------------
    # Hard negatives
    # ------------------------------------------------------------------------

    for folder_name, subtype in NEGATIVE_SUBTYPES.items():

        source_dir = RAW_DIR / folder_name

        if not source_dir.exists():
            continue

        for path in source_dir.iterdir():

            if path.is_file() and path.suffix.lower() in {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".webm",
            }:

                extract_frames(
                    path,
                    source_type=folder_name,
                    negative_subtype=subtype,
                    metadata_rows=metadata_rows,
                )

        collect_images_from_folder(
            source_dir,
            source_type=folder_name,
            negative_subtype=subtype,
            metadata_rows=metadata_rows,
        )

    # ------------------------------------------------------------------------
    # NEW hard negatives at project root: hard_negatives/
    #
    # Expected:
    # hard_negatives/
    #   Branches/
    #   Cables/
    #   Curled leaves/
    #   Hoses/
    #   Pipes/
    # ------------------------------------------------------------------------
    if HARD_NEGATIVES_DIR.exists():
        for folder_name, subtype in HARD_NEGATIVE_SUBTYPES.items():
            source_dir = HARD_NEGATIVES_DIR / folder_name

            if not source_dir.exists():
                print(f"⚠ Hard-negative folder not found: {source_dir}")
                continue

            collect_images_from_folder(
                source_dir,
                source_type="hard_negative",
                negative_subtype=subtype,
                metadata_rows=metadata_rows,
            )
    else:
        print(f"⚠ Hard-negative directory not found: {HARD_NEGATIVES_DIR}")

    return metadata_rows


# ============================================================================
# 8. PERCEPTUAL-HASH DEDUPLICATION
# ============================================================================

def deduplicate_dataset(metadata_rows):
    """
    Remove near-duplicate images using perceptual hashing.

    When a duplicate is removed, its metadata entry is also removed.

    IMPORTANT:
        Deduplication is performed globally across the collected dataset,
        not separately for each class.
    """

    print("\n--- Deduplication ---")

    seen_hashes = {}
    kept_rows = []

    removed_count = 0

    image_paths = [
    IMAGES_DIR / row["filename"]
    for row in metadata_rows
    if (IMAGES_DIR / row["filename"]).exists()
    ]

    for image_path in tqdm(
        image_paths,
        desc="Deduplicating"
    ):

        try:
            with Image.open(image_path) as img:
                img_hash = imagehash.phash(img)

        except Exception as exc:

            print(
                f"⚠ Removing unreadable image "
                f"{image_path.name}: {exc}"
            )

            image_path.unlink(missing_ok=True)
            removed_count += 1
            continue

        duplicate = False

        for existing_hash in seen_hashes:

            if img_hash - existing_hash <= HASH_THRESHOLD:
                duplicate = True
                break

        if duplicate:

            image_path.unlink(missing_ok=True)
            removed_count += 1

        else:

            seen_hashes[img_hash] = image_path

            kept_rows.append(image_path.name)

    kept_rows_set = set(kept_rows)

    metadata_rows = [
        row
        for row in metadata_rows
        if row["filename"] in kept_rows_set
    ]

    print(
        f"Removed {removed_count} near-duplicate/corrupt images."
    )

    print(
        f"Remaining unique images: "
        f"{len(metadata_rows)}"
    )

    return metadata_rows


# ============================================================================
# 9. SAVE METADATA
# ============================================================================

def save_metadata(metadata_rows):
    """Write the dataset manifest."""

    fieldnames = [
        "filename",
        "source_id",
        "source_type",
        "negative_subtype",
        "frame_index",
        "original_source",
        "width",
        "height",
        "blur_score",
        "brightness",
        "quality_flag",
    ]

    with METADATA_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print(
        f"\nMetadata saved to: {METADATA_PATH}"
    )


# ============================================================================
# 10. DATASET STATISTICS
# ============================================================================

def generate_report(metadata_rows):
    """
    Generate dataset statistics.

    Hard-negative subtypes remain visible in the report even though they
    will later be treated as negative/background examples for detection.
    """

    total = len(metadata_rows)

    source_type_counts = Counter(
        row["source_type"]
        for row in metadata_rows
    )

    negative_counts = Counter(
        row["negative_subtype"]
        for row in metadata_rows
        if row["negative_subtype"]
    )

    source_ids = set(
        row["source_id"]
        for row in metadata_rows
    )

    quality_counts = Counter(
        row["quality_flag"]
        for row in metadata_rows
    )

    snake_count = source_type_counts.get("snake", 0)

    negative_count = sum(
        negative_counts.values()
    )

    report = {
        "total_images": total,
        "snake_images": snake_count,
        "hard_negative_images": negative_count,

        "snake_percentage": (
            round(snake_count / total * 100, 2)
            if total else 0
        ),

        "hard_negative_percentage": (
            round(negative_count / total * 100, 2)
            if total else 0
        ),

        "negative_subtypes": dict(
            negative_counts
        ),

        "source_type_counts": dict(
            source_type_counts
        ),

        "unique_source_ids": len(source_ids),

        "quality_flags": dict(
            quality_counts
        ),

        "hash_threshold": HASH_THRESHOLD,
        "frame_interval": DEFAULT_EVERY_N_FRAMES,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET CURATION REPORT")
    print("=" * 60)

    print(f"\nTotal images          : {total}")
    print(f"Snake images         : {snake_count}")
    print(f"Hard-negative images : {negative_count}")

    if total:
        print(
            f"Snake ratio          : "
            f"{snake_count / total * 100:.1f}%"
        )

        print(
            f"Hard-negative ratio  : "
            f"{negative_count / total * 100:.1f}%"
        )

    print("\n--- Hard-negative breakdown ---")

    for subtype in [
        "vegetation",
        "vine",
        "rope",
        "shadow",
        "root",
        "curled_leaf",
        "hose",
        "log",
        "pipe",
        "branch",
        "cable",
    ]:

        count = negative_counts.get(subtype, 0)

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{subtype:10s}: "
            f"{count:6d} "
            f"({percentage:.1f}%)"
        )

    print("\n--- Source information ---")

    print(
        f"Unique source IDs   : "
        f"{len(source_ids)}"
    )

    print("\n--- Quality flags ---")

    for flag, count in quality_counts.items():
        print(
            f"{flag:25s}: {count}"
        )

    print(
        f"\nReport saved to: {REPORT_PATH}"
    )


# ============================================================================
# 11. MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1 — ADD HARD NEGATIVES")
    print("=" * 60)

    setup_folders()

    # ------------------------------------------------------------------------
    # Collect ONLY the new hard-negative images.
    # Existing snake images are already present in dataset/images/.
    # ------------------------------------------------------------------------

    metadata = []

    if not HARD_NEGATIVES_DIR.exists():
        raise FileNotFoundError(
            f"Hard-negative directory not found: {HARD_NEGATIVES_DIR.resolve()}"
        )

    for folder_name, subtype in HARD_NEGATIVE_SUBTYPES.items():

        source_dir = HARD_NEGATIVES_DIR / folder_name

        if not source_dir.exists():
            print(f"⚠ Missing folder: {source_dir}")
            continue

        print(f"\nCollecting: {folder_name}")

        collect_images_from_folder(
            source_dir,
            source_type="hard_negative",
            negative_subtype=subtype,
            metadata_rows=metadata,
        )

    if not metadata:
        print("\n⚠ No hard-negative images were collected.")
        raise SystemExit(1)

    print(f"\nNew hard-negative metadata records: {len(metadata)}")

    # ------------------------------------------------------------------------
    # Deduplicate the NEW hard negatives only.
    # ------------------------------------------------------------------------

    metadata = deduplicate_dataset(metadata)

    # ------------------------------------------------------------------------
    # MERGE with the existing metadata.csv.
    # ------------------------------------------------------------------------

    existing_rows = []

    if METADATA_PATH.exists():

        with METADATA_PATH.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)
            existing_rows = list(reader)

    print(f"Existing metadata records: {len(existing_rows)}")

    existing_filenames = {
        row["filename"]
        for row in existing_rows
    }

    # Only add genuinely new records.
    new_rows = [
        row
        for row in metadata
        if row["filename"] not in existing_filenames
    ]

    combined_metadata = existing_rows + new_rows

    save_metadata(combined_metadata)

    # ------------------------------------------------------------------------
    # Generate report using the COMPLETE dataset.
    # ------------------------------------------------------------------------

    generate_report(combined_metadata)

    print("\n" + "=" * 60)
    print("HARD-NEGATIVE ADDITION COMPLETE")
    print("=" * 60)

    print(f"\nExisting records : {len(existing_rows)}")
    print(f"New records      : {len(new_rows)}")
    print(f"Total records     : {len(combined_metadata)}")

    print("\nHard-negative folders processed:")
    for folder_name, subtype in HARD_NEGATIVE_SUBTYPES.items():
        count = sum(
            1
            for row in new_rows
            if row["negative_subtype"] == subtype
        )
        print(f"  {folder_name:15s}: {count}")

    print("\nNext step:")
    print("  Step 2 — Annotation and validation")