"""
STEP 3 — Dataset Split + Training Augmentation Setup

Input:
    dataset/
        images/
        metadata.csv
        annotations.json

Output:
    dataset/
        splits/
            train.csv
            val.csv
            test.csv
            train_annotations.json
            val_annotations.json
            test_annotations.json
            split_report.json

Purpose:
    1. Split data at SOURCE level to prevent data leakage.
    2. Maintain reasonable positive/negative/hard-negative balance.
    3. Generate COCO annotations for each split.
    4. Define training-time augmentation.
    5. Do NOT create augmented images on disk.

Important:
    - Training augmentation should be applied on-the-fly.
    - Validation and test data must NOT be augmented.
    - RT-DETRv2-specific normalization should be handled by
      the actual training implementation.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import albumentations as A


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

IMAGES_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "metadata.csv"
COCO_FILE = DATASET_DIR / "annotations.json"

OUTPUT_DIR = DATASET_DIR / "splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Used only for reporting/checking.
# The exact final RT-DETRv2 input size can be configured
# later in the model/training pipeline.
TARGET_SIZE = 640


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    required_columns = {
        "filename",
        "source_id",
        "source_type",
        "negative_subtype",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"metadata.csv is missing columns: {sorted(missing)}"
        )

    return df


def load_coco():
    if not COCO_FILE.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {COCO_FILE}"
        )

    with open(COCO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_metadata(df, coco_data):
    """
    Add derived positive/negative labels based on
    the actual COCO annotations.

    Positive = image has at least one snake annotation.
    Negative = image has zero snake annotations.

    Metadata source_type and negative_subtype are retained
    only for source/category information.
    """

    df = df.copy()

    df["source_type"] = (
        df["source_type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    df["negative_subtype"] = (
        df["negative_subtype"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # --------------------------------------------------------
    # Determine positive/negative from ACTUAL annotations
    # --------------------------------------------------------

    annotation_counts = Counter(
        ann["image_id"]
        for ann in coco_data["annotations"]
    )

    image_id_by_filename = {
        str(image["file_name"]): image["id"]
        for image in coco_data["images"]
    }

    df["annotation_count"] = df["filename"].map(
        lambda filename: annotation_counts.get(
            image_id_by_filename.get(str(filename)),
            0
        )
    )

    # Positive = at least one actual snake annotation
    df["is_snake"] = df["annotation_count"] > 0

    # Negative = no annotations
    df["is_negative"] = df["annotation_count"] == 0

    return df


# ============================================================
# SOURCE-LEVEL STATISTICS
# ============================================================

def build_source_statistics(df):
    """
    Build statistics for every source.

    The source_id is the unit used for splitting.
    This prevents frames from the same video/source
    appearing in different splits.
    """

    source_stats = {}

    for source_id, group in df.groupby("source_id"):

        positive_count = int(group["is_snake"].sum())
        negative_count = int(group["is_negative"].sum())

        subtype_counts = Counter(
            group.loc[group["is_negative"], "negative_subtype"]
        )

        source_stats[source_id] = {
            "total": len(group),
            "positive": positive_count,
            "negative": negative_count,
            "subtypes": dict(subtype_counts),
        }

    return source_stats


# ============================================================
# SOURCE-LEVEL SPLITTING
# ============================================================

# def calculate_split_score(
#     current_counts,
#     group_counts,
#     target_counts,
# ):
#     """
#     Calculate how close a split would be to its target
#     after adding one source.

#     Lower score = better.
#     """

#     score = 0.0

#     for key in target_counts:

#         current = current_counts.get(key, 0)
#         addition = group_counts.get(key, 0)
#         target = target_counts[key]

#         new_value = current + addition

#         if target > 0:
#             score += abs(new_value - target) / target

#     return score


def split_sources(df):

    source_stats = build_source_statistics(df)
    source_ids = list(source_stats.keys())

    if len(source_ids) < 3:
        raise ValueError(
            "At least 3 unique source_id values are required "
            "to create train/validation/test splits."
        )

    rng = random.Random(RANDOM_SEED)

    # Group sources by their dominant stratum (positive, or a specific
    # hard-negative subtype) so each category's ratio is preserved
    # independently across splits.
    strata = defaultdict(list)

    for source_id, stats in source_stats.items():

        if stats["positive"] >= stats["negative"]:
            stratum_key = "positive"
        elif stats["subtypes"]:
            stratum_key = max(stats["subtypes"], key=stats["subtypes"].get)
        else:
            stratum_key = "negative"

        strata[stratum_key].append(source_id)

    split_assignments = {"train": [], "val": [], "test": []}

    for stratum_key, ids in strata.items():

        rng.shuffle(ids)

        total_in_stratum = sum(source_stats[sid]["total"] for sid in ids)

        train_cutoff = total_in_stratum * TRAIN_RATIO
        val_cutoff = train_cutoff + total_in_stratum * VAL_RATIO

        cumulative = 0

        for sid in ids:

            cumulative += source_stats[sid]["total"]

            if cumulative <= train_cutoff:
                split_assignments["train"].append(sid)
            elif cumulative <= val_cutoff:
                split_assignments["val"].append(sid)
            else:
                split_assignments["test"].append(sid)

    return split_assignments


# ============================================================
# BUILD SPLIT DATAFRAMES
# ============================================================

def build_split_dataframes(df, split_assignments):

    split_dfs = {}

    for split, source_ids in split_assignments.items():

        split_df = df[
            df["source_id"].isin(source_ids)
        ].copy()

        split_df = split_df.sort_values(
            ["source_id", "filename"]
        )

        split_df["split"] = split

        split_dfs[split] = split_df

    return split_dfs


# ============================================================
# CHECK SOURCE LEAKAGE
# ============================================================

def check_source_leakage(split_dfs):

    train_sources = set(
        split_dfs["train"]["source_id"]
    )

    val_sources = set(
        split_dfs["val"]["source_id"]
    )

    test_sources = set(
        split_dfs["test"]["source_id"]
    )

    train_val = train_sources & val_sources
    train_test = train_sources & test_sources
    val_test = val_sources & test_sources

    leakage = {
        "train_val": sorted(train_val),
        "train_test": sorted(train_test),
        "val_test": sorted(val_test),
    }

    if any(leakage.values()):
        raise RuntimeError(
            f"SOURCE LEAKAGE DETECTED: {leakage}"
        )

    return leakage


# ============================================================
# SPLIT STATISTICS
# ============================================================

def calculate_split_statistics(split_df):

    total = len(split_df)

    positive = int(split_df["is_snake"].sum())
    negative = int(split_df["is_negative"].sum())

    subtype_counts = Counter(
        split_df.loc[
            split_df["is_negative"],
            "negative_subtype",
        ]
    )

    return {
        "images": total,
        "sources": int(
            split_df["source_id"].nunique()
        ),
        "positive_snake": positive,
        "negative": negative,
        "positive_ratio": (
            positive / total if total else 0
        ),
        "negative_ratio": (
            negative / total if total else 0
        ),
        "negative_subtypes": dict(subtype_counts),
    }


# ============================================================
# SAVE CSV MANIFESTS
# ============================================================

def save_manifests(split_dfs):

    for split, split_df in split_dfs.items():

        columns = [
            "filename",
            "source_id",
            "source_type",
            "negative_subtype",
            "split",
        ]

        available_columns = [
            col
            for col in columns
            if col in split_df.columns
        ]

        output_file = OUTPUT_DIR / f"{split}.csv"

        split_df[available_columns].to_csv(
            output_file,
            index=False,
        )

        print(f"Saved: {output_file}")


# ============================================================
# COCO SPLIT CREATION
# ============================================================

def create_split_coco(
    coco_data,
    split_df,
    split_name,
):
    """
    Create a COCO JSON containing only images
    belonging to the selected split.

    IDs are remapped so each split has clean,
    independent image/annotation IDs.
    """

    filenames = set(
        split_df["filename"].astype(str)
    )

    selected_images = [
        img
        for img in coco_data["images"]
        if str(img["file_name"]) in filenames
    ]

    selected_image_ids = {
        img["id"]
        for img in selected_images
    }

    selected_annotations = [
        ann
        for ann in coco_data["annotations"]
        if ann["image_id"] in selected_image_ids
    ]

    # --------------------------------------------------------
    # Remap image IDs
    # --------------------------------------------------------

    image_id_map = {}

    new_images = []

    for new_id, image in enumerate(
        selected_images,
        start=1,
    ):

        old_id = image["id"]

        image_id_map[old_id] = new_id

        new_image = image.copy()
        new_image["id"] = new_id

        new_images.append(new_image)

    # --------------------------------------------------------
    # Remap annotation IDs
    # --------------------------------------------------------

    new_annotations = []

    for new_ann_id, annotation in enumerate(
        selected_annotations,
        start=1,
    ):

        new_annotation = annotation.copy()

        old_image_id = annotation["image_id"]

        new_annotation["id"] = new_ann_id
        new_annotation["image_id"] = image_id_map[
            old_image_id
        ]

        new_annotations.append(new_annotation)

    split_coco = {
        "info": {
            "description": (
                f"Snake Detection Dataset - {split_name}"
            )
        },
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }

    output_file = (
        OUTPUT_DIR /
        f"{split_name}_annotations.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            split_coco,
            f,
            indent=2,
        )

    print(f"Saved: {output_file}")

    return split_coco


# ============================================================
# VERIFY COCO SPLITS
# ============================================================

def verify_coco_split(
    split_coco,
    split_df,
    split_name,
):
    """
    Verify that every image in the CSV exists
    in the corresponding COCO split.
    """

    csv_files = set(
        split_df["filename"].astype(str)
    )

    coco_files = {
        str(img["file_name"])
        for img in split_coco["images"]
    }

    missing_from_coco = csv_files - coco_files
    extra_in_coco = coco_files - csv_files

    if missing_from_coco:
        raise RuntimeError(
            f"{split_name}: images missing from COCO: "
            f"{list(missing_from_coco)[:10]}"
        )

    if extra_in_coco:
        raise RuntimeError(
            f"{split_name}: unexpected COCO images: "
            f"{list(extra_in_coco)[:10]}"
        )


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

def get_train_transform():
    """
    Training-time augmentation.

    This transform should be applied ON-THE-FLY
    inside the training dataloader.

    Bounding boxes are handled in COCO format.
    """

    return A.Compose(
        [

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            A.HorizontalFlip(
                p=0.5
            ),

            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=114,
                p=0.35,
            ),

            # ------------------------------------------------
            # Lighting
            # ------------------------------------------------

            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.30,
            ),

            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.20,
            ),

            # ------------------------------------------------
            # Image quality variation
            # ------------------------------------------------

            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=(3, 5)
                    ),

                    A.GaussNoise(
                        std_range=(0.01, 0.05),
                        mean_range=(0.0, 0.0),
                    ),

                    A.ImageCompression(
                        quality_range=(60, 95)
                    ),
                ],
                p=0.20,
            ),

        ],

        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.30,
            clip=True,
        ),
    )


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

def get_eval_transform():
    """
    Validation/test transform.

    No random augmentation is applied here.

    Model-specific resize/normalization should be handled
    by the final RT-DETRv2 training/inference pipeline.
    """

    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.0,
            clip=True,
        ),
    )


# ============================================================
# OPTIONAL IMAGE RESIZE UTILITY
# ============================================================

def resize_with_padding(
    image,
    target_size=TARGET_SIZE,
):
    """
    Resize an image while preserving aspect ratio.

    This utility is provided for later integration.

    Do NOT use this function to permanently rewrite the
    dataset unless the selected RT-DETRv2 implementation
    specifically requires it.
    """

    h, w = image.shape[:2]

    scale = min(
        target_size / w,
        target_size / h,
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (
            target_size,
            target_size,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    x_offset = (
        target_size - new_w
    ) // 2

    y_offset = (
        target_size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return canvas


# ============================================================
# REPORT
# ============================================================

def create_report(
    df,
    split_dfs,
    leakage,
):

    report = {
        "configuration": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "random_seed": RANDOM_SEED,
            "target_size": TARGET_SIZE,
        },

        "dataset": {
            "total_images": len(df),
            "total_sources": int(
                df["source_id"].nunique()
            ),
            "total_snake_images": int(
                df["is_snake"].sum()
            ),
            "total_negative_images": int(
                df["is_negative"].sum()
            ),
        },

        "splits": {},

        "source_leakage": leakage,

        "notes": [
            "Splitting is performed at source level.",
            "Frames from the same source_id remain in one split.",
            "Training augmentation is intended to run on-the-fly.",
            "Validation and test data are not augmented.",
            "Hard-negative subtype is retained in metadata.",
            "Model-specific preprocessing should be handled by the final RT-DETRv2 implementation.",
        ],
    }

    for split, split_df in split_dfs.items():

        report["splits"][split] = (
            calculate_split_statistics(
                split_df
            )
        )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3 — DATASET SPLIT + AUGMENTATION SETUP")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\n[1] Loading metadata...")

    df = load_metadata()

    print(
        f"Loaded {len(df):,} metadata records."
    )

    print("\n[2] Loading COCO annotations...")

    coco_data = load_coco()

    print(
        f"Loaded {len(coco_data['images']):,} COCO images."
    )

    # --------------------------------------------------------
    # Prepare metadata
    # --------------------------------------------------------

    print("\n[3] Preparing metadata...")

    df = prepare_metadata(df, coco_data)

    # --------------------------------------------------------
    # Check filenames
    # --------------------------------------------------------

    print("\n[4] Checking image files...")

    missing_images = []

    for filename in df["filename"]:

        image_path = IMAGES_DIR / str(filename)

        if not image_path.exists():
            missing_images.append(str(filename))

    if missing_images:

        print(
            f"WARNING: {len(missing_images)} images "
            "listed in metadata are missing."
        )

        print(
            "Examples:",
            missing_images[:10],
        )

        df = df[
            ~df["filename"].isin(
                missing_images
            )
        ].copy()

    # --------------------------------------------------------
    # Source-level split
    # --------------------------------------------------------

    print("\n[5] Creating source-level split...")

    split_assignments = split_sources(df)

    print(
        f"Train sources: "
        f"{len(split_assignments['train'])}"
    )

    print(
        f"Validation sources: "
        f"{len(split_assignments['val'])}"
    )

    print(
        f"Test sources: "
        f"{len(split_assignments['test'])}"
    )

    # --------------------------------------------------------
    # Build split dataframes
    # --------------------------------------------------------

    split_dfs = build_split_dataframes(
        df,
        split_assignments,
    )

    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    print("\n[6] Checking source leakage...")

    leakage = check_source_leakage(
        split_dfs
    )

    print("✓ No source leakage detected.")

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print("\n[7] Split statistics")
    print("-" * 70)

    for split, split_df in split_dfs.items():

        stats = calculate_split_statistics(
            split_df
        )

        print(f"\n{split.upper()}")

        print(
            f"  Images:       {stats['images']:,}"
        )

        print(
            f"  Sources:      {stats['sources']:,}"
        )

        print(
            f"  Snake:        {stats['positive_snake']:,}"
        )

        print(
            f"  Negative:     {stats['negative']:,}"
        )

        print(
            f"  Snake ratio:  "
            f"{stats['positive_ratio']:.2%}"
        )

        print(
            f"  Hard negatives:"
        )

        for subtype, count in sorted(
            stats["negative_subtypes"].items()
        ):

            if subtype:
                print(
                    f"      {subtype}: {count:,}"
                )

    # --------------------------------------------------------
    # Save CSV manifests
    # --------------------------------------------------------

    print("\n[8] Saving split manifests...")

    save_manifests(
        split_dfs
    )

    # --------------------------------------------------------
    # Create COCO splits
    # --------------------------------------------------------

    print("\n[9] Creating split COCO files...")

    split_coco_data = {}

    for split, split_df in split_dfs.items():

        split_coco = create_split_coco(
            coco_data,
            split_df,
            split,
        )

        verify_coco_split(
            split_coco,
            split_df,
            split,
        )

        split_coco_data[split] = split_coco

    print(
        "✓ All COCO split files verified."
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    print("\n[10] Saving split report...")

    report = create_report(
        df,
        split_dfs,
        leakage,
    )

    report_file = (
        OUTPUT_DIR /
        "split_report.json"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"Saved: {report_file}"
    )

    # --------------------------------------------------------
    # Augmentation check
    # --------------------------------------------------------

    print("\n[11] Initializing augmentation pipeline...")

    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    print(
        "✓ Training augmentation pipeline created."
    )

    print(
        "✓ Validation/test pipeline contains no random augmentation."
    )

    # Prevent unused-variable warnings.
    _ = train_transform
    _ = eval_transform

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)

    print("\nGenerated:")

    print(
        f"  {OUTPUT_DIR / 'train.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'val.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'test.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'train_annotations.json'}"
    )

    print(
        f"  {OUTPUT_DIR / 'val_annotations.json'}"
    )

    print(
        f"  {OUTPUT_DIR / 'test_annotations.json'}"
    )

    print(
        f"  {OUTPUT_DIR / 'split_report.json'}"
    )

    print("\nImportant:")
    print(
        "  Augmented images were NOT written to disk."
    )

    print(
        "  Augmentation should be applied during model training."
    )

    print(
        "  RT-DETRv2-specific preprocessing will be configured later."
    )


if __name__ == "__main__":
    main()