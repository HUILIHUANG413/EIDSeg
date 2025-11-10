# datasets.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

class EIDSegDataset(Dataset):
    """
    Base dataset: parses CVAT-style XML polygons and lists image files.

    Expected layout:
      data_dir/
        default/
          0001.jpg
          0002.png
          ...

    XML: <annotations><image name="0001.jpg"><polygon label="D_Building" points="x1,y1;..."/></image>...</annotations>
    """
    # --- EIDSeg global label spec (single source of truth) ---


    def __init__(self, annotation_xml: str, data_dir: str):
        self.data_dir = Path(data_dir)
        self.annotation_df = self._parse_xml(annotation_xml)

        default_dir = self.data_dir / "default"
        if not default_dir.exists():
            raise FileNotFoundError(f"Expected image folder: {default_dir}")

        self.image_filenames = sorted([
            f for f in os.listdir(default_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])
        if not self.image_filenames:
            raise RuntimeError(f"No images found in {default_dir}")

    def _parse_xml(self, annotation_xml: str) -> pd.DataFrame:
        import xml.etree.ElementTree as ET
        xml_path = Path(annotation_xml)
        if not xml_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {xml_path}")

        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        data: Dict[str, Dict[str, List]] = {}
        for image in root.findall('.//image'):
            img_name = f"default/{image.get('name')}"
            polygons = []
            for poly in image.findall('.//polygon'):
                pts = [
                    (float(p.split(',')[0]), float(p.split(',')[1]))
                    for p in poly.get('points').split(';') if p.strip()
                ]
                label = poly.get('label')
                polygons.append((pts, label))
            data[img_name] = {"Polygons": polygons}
        return pd.DataFrame(data).T


class UniversalSegmentationDataset(EIDSegDataset):
    """
    Task-ready dataset that:
      - Builds dense masks from polygons using CLASS_MAP.
      - Applies Albumentations transforms (resize, optional aug).
      - Encodes images via Hugging Face processor (OneFormer/Mask2Former/SegFormer/BEiT/EoMT).
    """

    CLASS_MAP = {
        "UD_Building": 0,
        "D_Building": 1,
        "Debris": 2,
        "UD_Road": 3,
        "D_Road": 4,
        "Undesignated": 5,   # Merged with Background
        "Background": 5
    }
    NUM_CLASSES = 6
    CLASS_NAMES = ["UD_Building", "D_Building", "Debris", "UD_Road", "D_Road", "void"]
    FOREGROUND_CLASSES: int = 5
    
    def __init__(
        self,
        annotation_xml: str,
        data_dir: str,
        image_processor,
        model_type: str = "mask2former",
        image_size: Tuple[int, int] = (1024, 1024),
        augment: bool = False
    ):
        super().__init__(annotation_xml, data_dir)
        self.image_processor = image_processor
        self.model_type = model_type
        self.image_size = image_size
        self.filenames = self.image_filenames

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Resize(height=image_size[0], width=image_size[1]),
            ], additional_targets={"mask": "mask"})
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1])
            ])

    def __len__(self) -> int:
        return len(self.filenames)

    def _draw_polygon(self, mask: np.ndarray, polygon, class_id: int) -> np.ndarray:
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], class_id)
        return mask

    def _create_segmentation_mask(self, img_name: str, image_wh: Tuple[int, int]) -> np.ndarray:
        w, h = image_wh
        mask = np.full((h, w), self.CLASS_MAP["Background"], dtype=np.uint8)
        key = "default/" + img_name
        try:
            polygon_points = self.annotation_df.loc[key]["Polygons"]
        except KeyError:
            return mask

        if polygon_points and not isinstance(polygon_points, float):
            for polygon, label in polygon_points:
                if label in self.CLASS_MAP:
                    mask = self._draw_polygon(mask, polygon, self.CLASS_MAP[label])
        return mask

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_path = self.data_dir / "default" / filename

        image = Image.open(image_path).convert("RGB")
        mask = self._create_segmentation_mask(filename, image.size)  # (W, H) from PIL

        sample = self.transform(image=np.array(image), mask=mask)
        image_np, mask_np = sample["image"], sample["mask"]
        image_pil = Image.fromarray(image_np)

        if self.model_type == "oneformer":
            encoded = self.image_processor(
                images=[image_pil], task_inputs=["semantic"], return_tensors="pt"
            )
            pixel_values = encoded["pixel_values"].squeeze(0)
            task_inputs = encoded["task_inputs"].squeeze(0)
        else:
            encoded = self.image_processor(images=image_pil, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)
            task_inputs = None

        target_h, target_w = pixel_values.shape[1], pixel_values.shape[2]
        mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()

        out = {"pixel_values": pixel_values, "pixel_mask": mask_tensor, "filenames": filename}
        if task_inputs is not None:
            out["task_inputs"] = task_inputs
        if "patch_start_indices" in encoded:
            out["patch_start_indices"] = encoded["patch_start_indices"].squeeze(0)
        return out


class DeepLabV3PlusDataset(EIDSegDataset):
    """
    For DeepLabV3+ training:
      - torchvision normalization (ImageNet)
      - albumentations for aug/resize
      - returns {"image": Tensor, "mask": LongTensor}
    """
    CLASS_MAP = {
        "UD_Building": 0, "D_Building": 1, "Debris": 2,
        "UD_Road": 3, "D_Road": 4, "Undesignated": 5, "Background": 5
    }
    NUM_CLASSES = 6
    CLASS_NAMES = ["UD_Building", "D_Building", "Debris", "UD_Road", "D_Road", "void"]

    def __init__(self, annotation_xml: str, data_dir: str,
                 image_size=(512, 512), augment: bool = False):
        super().__init__(annotation_xml, data_dir)
        self.image_size = tuple(image_size)
        self.augment = augment

        self.img_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
        ], additional_targets={"mask": "mask"}) if augment else None

        self.filenames = self.image_filenames

    def _draw_polygon(self, mask, polygon, class_id):
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], class_id)
        return mask

    def _create_segmentation_mask(self, img_name, image_wh):
        w, h = image_wh
        mask = np.full((h, w), self.CLASS_MAP["Background"], dtype=np.uint8)
        key = "default/" + img_name
        try:
            polygon_points = self.annotation_df.loc[key]["Polygons"]
        except KeyError:
            return mask
        if polygon_points and not isinstance(polygon_points, float):
            for polygon, label in polygon_points:
                if label in self.CLASS_MAP:
                    mask = self._draw_polygon(mask, polygon, self.CLASS_MAP[label])
        return mask

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = self.data_dir / "default" / filename
        image = Image.open(image_path).convert("RGB")
        mask = self._create_segmentation_mask(filename, image.size)

        if self.augment and self.aug_transform:
            aug = self.aug_transform(image=np.array(image), mask=mask)
            image = Image.fromarray(aug["image"])
            mask = aug["mask"]

        img_tensor = self.img_transform(image)
        # Ensure mask size matches image size expected by model
        mask_resized = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()

        return {"image": img_tensor, "mask": mask_tensor, "filename": filename}

