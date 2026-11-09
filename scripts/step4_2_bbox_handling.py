"""
Step 4.2 — Bounding-box handling: COCO annotations -> aligned lists
-----------------------------------------------------------------------
Scope of this file ONLY:
    - Take the raw `annotations` list that 4.1's Dataset returns
    - Produce two index-aligned lists: bboxes=[[x,y,w,h], ...] and
      category_ids=[1, 1, ...] — exactly what Albumentations'
      bbox_params expects
    - Handle the negative-image case (empty annotations -> empty lists)
      as a real, tested path, not an afterthought

Deliberately NOT in this file (comes in 4.3 / 4.4):
    - Actually running Albumentations transforms (4.3)
    - RT-DETR ImageProcessor + collate_fn (4.4)

Still CPU-testable, no GPU or model needed.
"""

from pathlib import Path
from step4_1_coco_dataset_class import SnakeCocoDataset


def coco_annotations_to_aligned_lists(annotations: list):
    """
    Converts a raw COCO annotation list into the two aligned lists
    Albumentations needs.

    Input (from 4.1, unchanged):
        [{"bbox": [x, y, w, h], "category_id": 1, "area": ..., ...}, ...]

    Output:
        bboxes:       [[x, y, w, h], ...]
        category_ids: [1, ...]

    An empty `annotations` list (negative/hard-negative image) correctly
    produces two empty lists — this is the expected, valid path, not
    an error.
    """
    bboxes = []
    category_ids = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]

        # Defensive check: a zero/negative-area box should never reach
        # here if step 2's validation ran correctly, but Albumentations
        # will raise a confusing internal error if one slips through —
        # better to catch it here with a clear message.
        if w <= 0 or h <= 0:
            raise ValueError(
                f"Zero/negative-area bbox reached 4.2 unfiltered: {ann}. "
                f"This should have been caught in step 2 validation."
            )

        bboxes.append([x, y, w, h])
        category_ids.append(ann["category_id"])

    return bboxes, category_ids


# ---------------------------------------------------------------------------
# CPU sanity check — confirms both branches produce correctly aligned,
# correctly-lengthed lists before 4.3 (Albumentations) is built on top
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = SnakeCocoDataset(Path("dataset/splits/train_annotations.json"))

    positive_sample, negative_sample = None, None
    for i in range(len(dataset)):
        item = dataset[i]
        if item["annotations"] and positive_sample is None:
            positive_sample = item
        if not item["annotations"] and negative_sample is None:
            negative_sample = item
        if positive_sample and negative_sample:
            break

    for label, sample in [("POSITIVE", positive_sample), ("NEGATIVE", negative_sample)]:
        print(f"--- {label} sample ---")
        bboxes, category_ids = coco_annotations_to_aligned_lists(sample["annotations"])

        print(f"  file_name:     {sample['file_name']}")
        print(f"  raw annotation count: {len(sample['annotations'])}")
        print(f"  bboxes:        {bboxes}")
        print(f"  category_ids:  {category_ids}")

        # The actual invariant that matters for Albumentations —
        # if this ever fails, something upstream is broken.
        assert len(bboxes) == len(category_ids), \
            "bboxes and category_ids length mismatch — do not proceed to 4.3"
        print(f"  ✓ lengths match ({len(bboxes)})\n")