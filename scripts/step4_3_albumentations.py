"""
Step 4.3 — Albumentations integration
-------------------------------------------
Scope of this file ONLY:
    - Feed the (image, bboxes, category_ids) from 4.2 into your actual
      get_train_transform() / get_eval_transform() from step 3
    - Confirm bboxes and category_ids are STILL aligned after the
      transform (min_visibility can drop boxes — this is the exact
      failure mode that breaks silently if untested)
    - Save a visual before/after so you can eyeball that boxes moved
      correctly with the image, not independently of it

Deliberately NOT in this file (comes in 4.4):
    - RT-DETR ImageProcessor resize/normalize
    - collate_fn / DataLoader batching

Still CPU-testable, no GPU or model needed.

IMPORTANT: update the import below to match your actual step 3
filename (yours appears to be step3_dataset_split.py based on your
folder structure — adjust if it's named differently).
"""

from pathlib import Path

import cv2
from step4_1_coco_dataset_class import SnakeCocoDataset
from step4_2_bbox_handling import coco_annotations_to_aligned_lists
from step3_dataset_split import get_train_transform, get_eval_transform

VIS_OUT_DIR = Path("dataset/augmentation_check")


def apply_transform(transform, image, bboxes, category_ids):
    """
    Runs one Albumentations transform and returns the result, with an
    explicit alignment check afterward. Works correctly for the empty
    (negative-image) case too — Albumentations handles empty bboxes
    natively, this just confirms it.
    """
    result = transform(image=image, bboxes=bboxes, category_ids=category_ids)

    # This is the actual thing that can go wrong: min_visibility in
    # your step-3 train transform can drop a box during a harsh crop/
    # rotate, and if the label isn't dropped in lockstep, everything
    # downstream (loss computation) silently misaligns.
    assert len(result["bboxes"]) == len(result["category_ids"]), (
        "Post-transform bboxes/category_ids length mismatch — "
        "this should never happen if bbox_params is configured correctly, "
        "investigate immediately rather than proceeding to 4.4"
    )

    return result


def draw_boxes(image, bboxes, out_path: Path, color=(0, 255, 0)):
    """COCO format [x, y, w, h] -> draw rectangle. Saves BGR for cv2."""
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    for x, y, w, h in bboxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# CPU sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    VIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SnakeCocoDataset(Path("dataset/splits/train_annotations.json"))
    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    positive_sample, negative_sample = None, None
    for i in range(len(dataset)):
        item = dataset[i]
        if item["annotations"] and positive_sample is None:
            positive_sample = item
        if not item["annotations"] and negative_sample is None:
            negative_sample = item
        if positive_sample and negative_sample:
            break

    for label, sample in [("positive", positive_sample), ("negative", negative_sample)]:
        print(f"--- {label.upper()} sample: {sample['file_name']} ---")
        bboxes, category_ids = coco_annotations_to_aligned_lists(sample["annotations"])

        # Save the un-augmented original for comparison
        draw_boxes(sample["image"], bboxes, VIS_OUT_DIR / f"{label}_original.jpg")

        # Run the TRAIN transform three times — since it's randomized,
        # running it once tells you almost nothing about whether it's
        # behaving consistently across the range of random parameters.
        for run in range(3):
            result = apply_transform(train_transform, sample["image"], bboxes, category_ids)
            print(f"  [train run {run}] bboxes before={len(bboxes)} after={len(result['bboxes'])}")
            draw_boxes(result["image"], result["bboxes"],
                       VIS_OUT_DIR / f"{label}_train_aug{run}.jpg")

        # Run the EVAL transform once — should be a no-op on the boxes
        # (no random augmentation), only clipping applies.
        eval_result = apply_transform(eval_transform, sample["image"], bboxes, category_ids)
        print(f"  [eval] bboxes before={len(bboxes)} after={len(eval_result['bboxes'])}")
        draw_boxes(eval_result["image"], eval_result["bboxes"],
                   VIS_OUT_DIR / f"{label}_eval.jpg")
        print()

    print(f"Saved visual checks to: {VIS_OUT_DIR}")
    print("Open these images and confirm boxes visually still sit on the snake "
          "(or correctly stay empty for the negative sample) after augmentation.")