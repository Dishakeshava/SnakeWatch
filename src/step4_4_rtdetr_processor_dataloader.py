"""
Step 4.4 — RT-DETRv2 ImageProcessor + collate_fn + DataLoader
------------------------------------------------------------------
Scope of this file:
    - Take the augmented (image, bboxes, category_ids) from 4.3
    - Convert COCO category_id=1 to model label=0
    - Rebuild annotations in the format expected by RTDetrImageProcessor
    - Let the pretrained processor handle resizing/rescaling
      (NOTE: RT-DETR checkpoints ship with do_normalize=false in their
      preprocessor_config.json — the processor rescales pixels to
      [0,1] via do_rescale, it does NOT apply ImageNet mean/std
      normalization by default, despite image_mean/image_std being
      present in the config)
    - Provide collate_fn for batching images with variable numbers of boxes
    - Assemble the complete 4.1 -> 4.4 pipeline

Still CPU-testable — no model forward/backward pass is performed here.

Expected class mapping:
    COCO category_id 1  ->  RT-DETRv2 model label 0
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import RTDetrImageProcessor

from step4_1_coco_dataset_class import SnakeCocoDataset
from step4_2_bbox_handling import coco_annotations_to_aligned_lists
from step3_dataset_split import get_train_transform, get_eval_transform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT = "PekingU/rtdetr_v2_r18vd"

# Your COCO files use category_id=1 for snake.
# RT-DETRv2 will be configured later as a one-class model:
# label 0 = snake.
COCO_TO_MODEL_LABEL = {
    1: 0
}


# ---------------------------------------------------------------------------
# Final Dataset
# ---------------------------------------------------------------------------

class SnakeDetectionDataset(SnakeCocoDataset):
    """
    Combines:

        4.1 -> COCO loading
        4.2 -> aligned bbox/category lists
        4.3 -> Albumentations
        4.4 -> RT-DETRv2 preprocessing

    Returns:

        {
            "pixel_values": Tensor[C, H, W],
            "labels": dict
        }

    The returned item is ready to be passed through a DataLoader
    and subsequently into RT-DETRv2.
    """

    def __init__(
        self,
        coco_json_path: Path,
        processor,
        transform,
        images_dir: Path = Path("dataset/images"),
    ):
        super().__init__(coco_json_path, images_dir)

        self.processor = processor
        self.transform = transform

    def __getitem__(self, index: int):

        # ---------------------------------------------------------------
        # 4.1 — Load raw image + COCO annotations
        # ---------------------------------------------------------------

        raw = super().__getitem__(index)

        # ---------------------------------------------------------------
        # 4.2 — Convert annotations to aligned lists
        # ---------------------------------------------------------------

        bboxes, category_ids = coco_annotations_to_aligned_lists(
            raw["annotations"]
        )

        assert len(bboxes) == len(category_ids), (
            "4.2 alignment failure: "
            "number of bboxes != number of category IDs"
        )

        # ---------------------------------------------------------------
        # 4.3 — Apply Albumentations
        # ---------------------------------------------------------------

        aug = self.transform(
            image=raw["image"],
            bboxes=bboxes,
            category_ids=category_ids,
        )

        assert len(aug["bboxes"]) == len(aug["category_ids"]), (
            "4.3 alignment failure: "
            "number of augmented bboxes != number of augmented labels"
        )

        # ---------------------------------------------------------------
        # 4.4 — Convert COCO category IDs to model labels
        # ---------------------------------------------------------------
        #
        # COCO:
        #       snake = category_id 1
        #
        # Model:
        #       snake = label 0
        #
        # This conversion happens ONLY at the model interface.
        # 4.2 continues to work with the original COCO category IDs.
        # ---------------------------------------------------------------

        model_category_ids = []

        for cat_id in aug["category_ids"]:

            if cat_id not in COCO_TO_MODEL_LABEL:
                raise ValueError(
                    f"Unexpected COCO category_id={cat_id}. "
                    f"Expected one of {list(COCO_TO_MODEL_LABEL.keys())}."
                )

            model_category_ids.append(
                COCO_TO_MODEL_LABEL[cat_id]
            )

        # ---------------------------------------------------------------
        # Rebuild annotations for RTDetrImageProcessor
        # ---------------------------------------------------------------

        annotations_for_processor = []

        for bbox, model_label in zip(
            aug["bboxes"],
            model_category_ids
        ):

            x, y, w, h = bbox

            # Defensive validation
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Invalid augmented bbox: {bbox}"
                )

            annotations_for_processor.append(
                {
                    "bbox": [float(x), float(y), float(w), float(h)],

                    # IMPORTANT:
                    # Model label is 0, not COCO category_id 1.
                    "category_id": int(model_label),

                    # Recompute area after augmentation.
                    "area": float(w * h),

                    "iscrowd": 0,
                }
            )

        target = {
            "image_id": raw["image_id"],
            "annotations": annotations_for_processor,
        }

        # ---------------------------------------------------------------
        # RT-DETRv2 processor
        # ---------------------------------------------------------------
        #
        # The pretrained processor handles resize (to 640x640) and
        # rescale (pixels / 255). It does NOT apply ImageNet mean/std
        # normalization for RT-DETR checkpoints (do_normalize=false in
        # preprocessor_config.json) — do not assume otherwise.
        #
        # We do NOT manually resize or rescale here.
        # ---------------------------------------------------------------

        encoding = self.processor(
            images=aug["image"],
            annotations=target,
            return_tensors="pt",
        )

        # Processor returns a batch dimension for a single image:
        #
        # [1, 3, 640, 640]
        #
        # Remove that dimension because DataLoader will create the
        # batch dimension later.

        pixel_values = encoding["pixel_values"].squeeze(0)

        # Labels are also returned with a single-image batch dimension.
        labels = encoding["labels"][0]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Creates a batch from multiple dataset items.

    pixel_values:
        Same tensor shape for every image, so they can be stacked.

    labels:
        Cannot be stacked because every image may contain a different
        number of bounding boxes.

        Therefore labels remain a list of per-image dictionaries.
    """

    pixel_values = torch.stack(
        [item["pixel_values"] for item in batch]
    )

    labels = [
        item["labels"]
        for item in batch
    ]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Find positive and negative samples
# ---------------------------------------------------------------------------

def find_positive_negative_indices(dataset):
    """
    Finds at least two positive and two negative images.

    This is used ONLY for the batch sanity check so that we don't rely
    on random shuffling to produce a mixed positive/negative batch.

    IMPORTANT: classification is based on the RAW COCO annotation
    (via dataset.image_ids + dataset.get_annotations from 4.1), NOT on
    dataset[i]'s post-augmentation box count. The train transform's
    min_visibility can legitimately drop a box during a harsh crop or
    rotation, so checking post-augmentation counts here would make
    this function's classification depend on random chance — and
    since dataset[i] is called AGAIN later (independently, with a
    fresh random draw) to build the actual test batch, an index
    correctly identified as "positive" here could come back with 0
    boxes there, failing the batch assertions for no real bug.
    Using the ground-truth annotation instead makes this deterministic.
    """

    positive_indices = []
    negative_indices = []

    for i, image_id in enumerate(dataset.image_ids):

        raw_annotations = dataset.get_annotations(image_id)

        if len(raw_annotations) > 0:
            if len(positive_indices) < 2:
                positive_indices.append(i)

        else:
            if len(negative_indices) < 2:
                negative_indices.append(i)

        if (
            len(positive_indices) >= 2
            and len(negative_indices) >= 2
        ):
            break

    if len(positive_indices) < 2:
        raise RuntimeError(
            "Could not find two positive images in the training dataset."
        )

    if len(negative_indices) < 2:
        raise RuntimeError(
            "Could not find two negative images in the training dataset."
        )

    return positive_indices, negative_indices


# ---------------------------------------------------------------------------
# CPU sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("STEP 4.4 — RT-DETRv2 PROCESSOR + DATALOADER SANITY CHECK")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load pretrained processor
    # ---------------------------------------------------------------

    print("\n[1] Loading RT-DETRv2 processor...")

    processor = RTDetrImageProcessor.from_pretrained(
        CHECKPOINT
    )

    print(f"✓ Processor loaded from: {CHECKPOINT}")

    # ---------------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------------

    print("\n[2] Creating training dataset...")

    train_dataset = SnakeDetectionDataset(
        Path("dataset/splits/train_annotations.json"),
        processor=processor,
        transform=get_train_transform(),
    )

    print(f"✓ Dataset size: {len(train_dataset)}")

    # ---------------------------------------------------------------
    # Single-item test
    # ---------------------------------------------------------------

    print("\n[3] Single-item test...")

    sample = train_dataset[0]

    print(
        f"  pixel_values shape: "
        f"{tuple(sample['pixel_values'].shape)}"
    )

    print(
        f"  labels keys: "
        f"{list(sample['labels'].keys())}"
    )

    print(
        f"  num boxes: "
        f"{len(sample['labels'].get('boxes', []))}"
    )

    # Check expected tensor shape.
    assert sample["pixel_values"].ndim == 3, (
        f"Expected [C,H,W] tensor, got {sample['pixel_values'].shape}"
    )

    assert sample["pixel_values"].shape[0] == 3, (
        f"Expected 3 channels, got {sample['pixel_values'].shape[0]}"
    )

    print(
        f"  ✓ Single-item tensor shape is valid: "
        f"{tuple(sample['pixel_values'].shape)}"
    )

    # ---------------------------------------------------------------
    # Find known positive and negative samples (from raw annotations,
    # not post-augmentation counts — see docstring above)
    # ---------------------------------------------------------------

    print("\n[4] Finding positive + negative samples...")

    positive_indices, negative_indices = (
        find_positive_negative_indices(train_dataset)
    )

    print(
        f"  Positive indices: {positive_indices}"
    )

    print(
        f"  Negative indices: {negative_indices}"
    )

    print("  ✓ Both positive and negative samples found.")

    # ---------------------------------------------------------------
    # Deliberately create a mixed batch
    # ---------------------------------------------------------------

    test_indices = (
        positive_indices[:2]
        + negative_indices[:2]
    )

    print(
        f"\n[5] Testing mixed DataLoader batch..."
    )

    print(
        f"  Test indices: {test_indices}"
    )

    test_subset = Subset(
        train_dataset,
        test_indices
    )

    loader = DataLoader(
        test_subset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))

    # ---------------------------------------------------------------
    # Batch checks
    # ---------------------------------------------------------------

    print("\n[6] Batch results...")

    print(
        f"  pixel_values batch shape: "
        f"{tuple(batch['pixel_values'].shape)}"
    )

    print(
        f"  labels type: "
        f"{type(batch['labels']).__name__}"
    )

    print(
        f"  number of label dictionaries: "
        f"{len(batch['labels'])}"
    )

    assert batch["pixel_values"].ndim == 4, (
        f"Expected [B,C,H,W] tensor, got {batch['pixel_values'].shape}"
    )

    assert batch["pixel_values"].shape[0] == 4, (
        f"Expected batch size 4, got {batch['pixel_values'].shape[0]}"
    )

    assert batch["pixel_values"].shape[1] == 3, (
        f"Expected 3 channels, got {batch['pixel_values'].shape[1]}"
    )

    print(
        f"  ✓ Batch tensor shape is valid: "
        f"{tuple(batch['pixel_values'].shape)}"
    )

    assert len(batch["labels"]) == 4, (
        "Expected 4 per-image label dictionaries."
    )

    # ---------------------------------------------------------------
    # Print boxes per image
    # ---------------------------------------------------------------
    #
    # NOTE: these counts still come from the actual augmented batch
    # (as they should — this is the real pipeline output), so a
    # positive image CAN legitimately show 0 boxes here if min_visibility
    # dropped it on this particular random draw. That's no longer a
    # test-harness bug (indices were chosen from ground truth), it's
    # a genuine, occasional, and informative augmentation outcome.
    # If it happens often, your min_visibility or augmentation
    # strength may be too aggressive — worth a quick look, not a crash.
    # ---------------------------------------------------------------

    box_counts = []

    for i, label in enumerate(batch["labels"]):

        num_boxes = len(
            label.get("boxes", [])
        )

        box_counts.append(num_boxes)

        print(
            f"    image {i}: "
            f"{num_boxes} boxes"
        )

    assert box_counts[2] == 0, (
        "Third test image was expected to be negative "
        "(ground-truth annotation had zero boxes)."
    )

    assert box_counts[3] == 0, (
        "Fourth test image was expected to be negative "
        "(ground-truth annotation had zero boxes)."
    )

    if box_counts[0] == 0 or box_counts[1] == 0:
        print(
            "\n⚠ A ground-truth-positive image came back with 0 boxes "
            "after augmentation. This is not a pipeline bug — it means "
            "min_visibility dropped the box on this random draw. If you "
            "see this often across repeated runs, consider loosening "
            "min_visibility or the crop/rotate intensity in step 3."
        )

    print("\n" + "=" * 70)
    print("✓ STEP 4.4 SANITY CHECK PASSED")
    print("=" * 70)

    print(
        "\nThe complete data pipeline is now validated:"
    )

    print(
        "  4.1 → COCO Dataset"
    )

    print(
        "  4.2 → Bounding-box handling"
    )

    print(
        "  4.3 → Albumentations"
    )

    print(
        "  4.4 → RT-DETRv2 processor + DataLoader"
    )

    print(
        "\nReady for the RT-DETRv2 training script."
    )