"""
Step 4.1 — COCO Dataset class
---------------------------------
Scope of this file ONLY:
    - Load train_annotations.json / val_annotations.json (from step 3)
    - Index annotations by image_id for fast lookup
    - Return, per item: the loaded image (as a numpy array) + its raw
      COCO annotation list + its metadata (filename, width, height)

Deliberately NOT in this file (comes in 4.2 / 4.3 / 4.4):
    - Converting annotations into aligned bboxes/category_ids lists (4.2)
    - Albumentations augmentation (4.3)
    - RT-DETR ImageProcessor + collate_fn (4.4)

This is CPU-testable right now — no GPU or model needed. The sanity
check at the bottom loads one positive and one negative sample and
prints what came back, so you can confirm this piece works before
building anything on top of it.

Install deps:
    pip install opencv-python
"""

import json
from pathlib import Path
from collections import defaultdict

import cv2
from torch.utils.data import Dataset

IMAGES_DIR = Path("dataset/images")


class SnakeCocoDataset(Dataset):
    def __init__(self, coco_json_path: Path, images_dir: Path = IMAGES_DIR):
        coco = json.loads(Path(coco_json_path).read_text(encoding="utf-8"))

        self.images_dir = images_dir
        self.categories = coco["categories"]

        # image_id -> image metadata (filename, width, height)
        self.image_meta = {img["id"]: img for img in coco["images"]}

        # image_id -> list of its annotations (empty list if negative image)
        self.annotations_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)

        # Fixed order for indexing — list(dict) preserves insertion order
        # in Python 3.7+, but sorting explicitly makes this deterministic
        # regardless of how the JSON was written.
        self.image_ids = sorted(self.image_meta.keys())

    def __len__(self):
        return len(self.image_ids)

    def get_annotations(self, image_id: int):
        """Returns the raw COCO annotation list for one image.
        Empty list for negative (hard-negative) images — this is
        expected and correct, not an error case."""
        return self.annotations_by_image.get(image_id, [])

    def load_image(self, filename: str):
        """cv2 loads BGR by default — convert to RGB here so every
        downstream component (Albumentations, the model) gets a
        consistent color order, rather than each guessing separately."""
        img_path = self.images_dir / filename
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        meta = self.image_meta[image_id]

        image = self.load_image(meta["file_name"])
        annotations = self.get_annotations(image_id)

        # NOTE: annotations is returned RAW here — a list of COCO-format
        # dicts like {"bbox": [x,y,w,h], "category_id": 1, ...}.
        # Converting this into the two aligned lists Albumentations
        # needs (bboxes=[...], category_ids=[...]) is 4.2, not here.
        return {
            "image_id": image_id,
            "image": image,
            "annotations": annotations,
            "file_name": meta["file_name"],
            "width": meta["width"],
            "height": meta["height"],
        }


# ---------------------------------------------------------------------------
# CPU sanity check — run this before building 4.2 on top of it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset = SnakeCocoDataset(Path("dataset/splits/train_annotations.json"))
    print(f"Loaded {len(dataset)} images\n")

    # Find one positive (has annotations) and one negative (zero annotations)
    # sample specifically, rather than just checking index 0 — index 0
    # tells you nothing about whether the negative path works.
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
        if sample is None:
            print("  ⚠ None found in this split — check your data.")
            continue
        print(f"  file_name:   {sample['file_name']}")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  width/height (metadata): {sample['width']}x{sample['height']}")
        print(f"  num annotations: {len(sample['annotations'])}")
        if sample["annotations"]:
            print(f"  first bbox: {sample['annotations'][0]['bbox']}")
        print()