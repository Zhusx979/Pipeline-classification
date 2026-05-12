from pathlib import Path
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from common.path_utils import PROJECT_ROOT, resolve_project_path


def _resolve_image_path(raw_path: str) -> Optional[str]:
    raw_path = raw_path.strip().strip('"').strip("'")
    if not raw_path:
        return None

    normalized = raw_path.replace("\\", "/")
    candidates = [normalized]
    for legacy_name in ("dataset_0309", "dataset_0303", "dataset_0126"):
        if legacy_name in normalized:
            candidates.append(normalized.replace(legacy_name, "dataset"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        candidate_path = Path(candidate)
        path_options = [candidate_path]
        if not candidate_path.is_absolute():
            resolved_project_path = resolve_project_path(candidate_path)
            if resolved_project_path is not None:
                path_options.append(resolved_project_path)
            path_options.append(Path.cwd() / candidate_path)

        for path in path_options:
            if path.exists() and path.is_file():
                return str(path.resolve())

    return None


class AlbumentationsTransformTrain:
    def __init__(self, height=256, width=256):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=height, p=1),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                value=0,
                border_mode=cv2.BORDER_CONSTANT,
                position="top_left",
                p=1,
            ),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"].float() / 255.0


class AlbumentationsTransformVal:
    def __init__(self, height=256, width=256):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=height, p=1),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                value=0,
                border_mode=cv2.BORDER_CONSTANT,
                position="top_left",
                p=1,
            ),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented["image"].float() / 255.0


class ImageDatasetFromTxt(Dataset):
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

        print(f"✅ 成功加载 {len(self.image_paths)} 个有效图像路径")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"无法加载图像: {img_path} | 错误: {str(e)}")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

    def get_label_from_path(self, img_path):
        parts = Path(img_path).parts
        if len(parts) >= 3:
            label_dir = parts[-2]
            if label_dir.isdigit():
                return label_dir
        print(f"⚠️ 警告：无法从路径提取标签: {img_path}")
        return 0
