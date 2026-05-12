from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.dataset import AlbumentationsTransformVal, _resolve_image_path
from .utils import heatmap_cache_path


class AlbumentationsTransformStage2Train:
    def __init__(self, height: int = 256, width: int = 256):
        self.height = height
        self.width = width
        self.image_preprocess = A.Compose(
            [
                # 训练热力图是基于 val 预处理后的图像生成的，所以这里先把原图规整到同一画布。
                A.LongestMaxSize(max_size=max(height, width), p=1),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    value=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    position="top_left",
                    p=1,
                ),
            ]
        )
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ],
            additional_targets={"heatmap": "image"},
        )

    def __call__(self, img, heatmap):
        img = np.array(img)
        heatmap = np.array(heatmap)
        img = self.image_preprocess(image=img)["image"]

        expected_shape = (self.height, self.width)
        if heatmap.shape[:2] != expected_shape:
            raise ValueError(
                f"Cached heatmap shape {heatmap.shape[:2]} does not match expected {expected_shape}. "
                "Please regenerate the stage2 heatmap cache."
            )
        if img.shape[:2] != heatmap.shape[:2]:
            raise ValueError(
                f"Stage2 image/heatmap shape mismatch after preprocessing: image={img.shape[:2]}, "
                f"heatmap={heatmap.shape[:2]}"
            )

        augmented = self.transform(image=img, heatmap=heatmap)
        image_tensor = augmented["image"].float() / 255.0
        heatmap_tensor = augmented["heatmap"].float() / 255.0
        return torch.cat([image_tensor, heatmap_tensor], dim=0)


class _BaseStage2TxtDataset(Dataset):
    def __init__(self, txt_file_list, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for txt_file in txt_file_list:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    path = _resolve_image_path(line)
                    if not path:
                        continue
                    self.image_paths.append(path)
                    self.labels.append(int(self.get_label_from_path(path)))

        print(f"✅ Stage2 loaded {len(self.image_paths)} valid image paths")

    def __len__(self):
        return len(self.image_paths)

    def get_label_from_path(self, img_path):
        parts = Path(img_path).parts
        if len(parts) >= 3:
            label_dir = parts[-2]
            if label_dir.isdigit():
                return label_dir
        print(f"⚠️ Warning: unable to infer label from path: {img_path}")
        return 0


class Stage2TrainDatasetFromTxt(_BaseStage2TxtDataset):
    def __init__(self, txt_file_list, heatmap_root, transform=None):
        super().__init__(txt_file_list, transform=transform or AlbumentationsTransformStage2Train())
        self.heatmap_root = Path(heatmap_root)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        heatmap_path = heatmap_cache_path(self.heatmap_root, img_path, split="train")
        if not heatmap_path.exists():
            raise RuntimeError(f"Missing cached heatmap for training sample: {heatmap_path}")

        image = Image.open(img_path).convert("RGB")
        heatmap = Image.open(heatmap_path).convert("RGB")
        if self.transform:
            image = self.transform(image, heatmap)
        return image, self.labels[idx]


class Stage2EvalDatasetFromTxt(_BaseStage2TxtDataset):
    def __init__(self, txt_file_list, transform=None):
        super().__init__(txt_file_list, transform=transform or AlbumentationsTransformVal())

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
