from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import shutil

import utils


def combine_coco_datasets(
    datasets: list[CocoKeyDataset], new_dir: str, transform=None
) -> CocoKeyDataset:
    if not datasets:
        return []

    # Ensure all datasets have the same categories
    base_categories = datasets[0].coco["categories"]
    for ds in datasets[1:]:
        if ds.coco["categories"] != base_categories:
            raise ValueError("All datasets must have the same categories to combine.")

    utils.make_directory_force_recursively(os.path.join(new_dir, "images"))

    combined_builder = CocoBuilder(categories=base_categories)
    for i, ds in enumerate(datasets):
        id_mapping = {}
        for img in ds.coco["images"]:
            # Copy image to new directory
            new_img_filename = f"ds{i}_{img['file_name']}"
            new_img_path = os.path.join(new_dir, "images", new_img_filename)
            src_path = os.path.join(ds.images_dir, img["file_name"])
            shutil.copy2(src_path, new_img_path)

            new_image_id = combined_builder.add_image(
                file_name=new_img_filename,
                width=img["width"],
                height=img["height"],
            )
            id_mapping[img["id"]] = new_image_id

        for ann in ds.coco["annotations"]:
            combined_builder.add_annotation(
                image_id=id_mapping[ann["image_id"]],
                bbox=ann["bbox"],
                category_id=ann["category_id"],
                iscrowd=ann.get("iscrowd", 0),
            )

    combined_builder.save(os.path.join(new_dir, "annotations.json"))
    return combined_builder.build(new_dir, transform)


class CocoBuilder:
    def __init__(self, categories=None):
        self.images = []
        self.annotations = []
        self.categories = categories or []
        self._image_id = 0
        self._annotation_id = 0

    def add_category(self, name, category_id=None):
        category_id = category_id if category_id is not None else len(self.categories)
        self.categories.append({"id": category_id, "name": name})
        return category_id

    def add_image(self, file_name, width, height) -> int:
        image_entry = {
            "id": self._image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        self.images.append(image_entry)
        self._image_id += 1
        return image_entry["id"]

    def add_annotation(self, image_id, bbox, category_id, iscrowd=0):
        """
        bbox = [x, y, width, height]
        """
        annotation_entry = {
            "id": self._annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": iscrowd,
        }
        self.annotations.append(annotation_entry)
        self._annotation_id += 1
        return annotation_entry["id"]

    def save(self, json_path):
        coco_data = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"Saved COCO JSON to {json_path}")

    def load(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        self.categories = data.get("categories", [])
        self._image_id = max([img["id"] for img in self.images], default=-1) + 1
        self._annotation_id = (
            max([ann["id"] for ann in self.annotations], default=-1) + 1
        )

    def build(self, output_dir: str, transform=None) -> CocoKeyDataset:
        img_dir = os.path.join(output_dir, "images")
        return CocoKeyDataset(
            images_dir=img_dir,
            coco_json={
                "images": self.images,
                "annotations": self.annotations,
                "categories": self.categories,
            },
            transform=transform,
        )


class CocoKeyDataset(Dataset):
    @staticmethod
    def from_json_file(images_dir: str, json_path: str, transform=None):
        try:
            with open(json_path, "r") as f:
                coco_json = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load COCO JSON from {json_path}: {e}")
        return CocoKeyDataset(images_dir, coco_json, transform)

    def __init__(self, images_dir, coco_json, transform=None):
        self.coco = coco_json

        # Build map from image_id -> image info
        self.images_info = {img["id"]: img for img in self.coco["images"]}

        # Flatten annotations: each annotation becomes one sample
        self.samples = []
        for ann in self.coco["annotations"]:
            image_info = self.images_info[ann["image_id"]]
            self.samples.append(
                {
                    "file_name": image_info["file_name"],
                    "bbox": ann["bbox"],  # [x, y, w, h] relative to crop
                    "category_id": ann["category_id"],
                }
            )

        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample["file_name"])
        image = Image.open(img_path).convert("RGB")  # or "L" for grayscale

        # Optionally crop based on bbox if needed (for per-frame COCO)
        # x, y, w, h = sample["bbox"]
        # image = image.crop((x, y, x+w, y+h))

        label = sample["category_id"]  # 0 or 1
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
