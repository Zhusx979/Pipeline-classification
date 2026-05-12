import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent


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
            path_options.append(PROJECT_ROOT / candidate_path)
            path_options.append(Path.cwd() / candidate_path)

        for path in path_options:
            if path.exists() and path.is_file():
                return str(path.resolve())

    return None
# 自定义增强函数
class AlbumentationsTransformTrain:
    def __init__(self, height=256, width=256):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=height, p = 1),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5), # 随机调整亮度和对比度
            A.HorizontalFlip(p=0.5),              # 随机水平翻转
            A.VerticalFlip(p=0.5),                # 随机垂直翻转
            # A.RandomResizedCrop(height=height, width=width, p=0.5),  # 随机裁剪并调整到224x224
            # A.Affine(shear=random.uniform(-30, 30), p=0.5),     # 随机剪切
            A.PadIfNeeded(min_height=height, min_width=width, value=0, border_mode =cv2.BORDER_CONSTANT, position="top_left", p = 1),
            ToTensorV2(),                         # 转换为Tensor
        ])

    def __call__(self, img):
        img = np.array(img)  # PIL Image 转为 NumPy 数组
        augmented = self.transform(image=img)
        image_tensor = augmented['image'].float() / 255.0  # 转为 float 并归一化到 0-1
        return image_tensor

# 自定义增强函数
class AlbumentationsTransformVal:
    def __init__(self, height=256, width=256):
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=height, p = 1),  # 统一调整图片尺寸
            A.PadIfNeeded(min_height=height, min_width=width, value=0, border_mode =cv2.BORDER_CONSTANT, position="top_left", p = 1),
            ToTensorV2(),                         # 转换为Tensor
        ])

    def __call__(self, img):
        img = np.array(img)  # PIL Image 转为 NumPy 数组
        augmented = self.transform(image=img)
        image_tensor = augmented['image'].float() / 255.0  # 转为 float 并归一化到 0-1
        return image_tensor
    
# 自定义Dataset类，读取txt文件中的图片路径，并加载图片
class ImageDatasetFromTxt(Dataset):
    # def __init__(self, txt_file_list, transform=None):
    #     """
    #     Args:
    #         txt_file (string): 包含图片路径的txt文件。
    #         transform (callable, optional): Optional transform to be applied
    #             on a sample.
    #     """
    #     self.image_paths = []
    #     for txt_file in txt_file_list:
    #         with open(txt_file, 'r') as f:
    #              self.image_paths.extend(f.readlines())
    #         self.image_paths = [x.strip() for x in self.image_paths]  # 移除换行符
    #     self.transform = transform
    def __init__(self, txt_file_list, transform=None):
        self.image_paths = []
        self.labels = []  # 新增：预存标签
        
        for txt_file in txt_file_list:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    path = _resolve_image_path(line)
                    if not path:  # 跳过空行或无法解析的路径
                        continue

                    self.image_paths.append(path)
                    
                    # 提前提取并存储标签
                    label = self.get_label_from_path(path)
                    self.labels.append(int(label))
        
        print(f"✅ 成功加载 {len(self.image_paths)} 个有效图像路径")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]
    #     image = Image.open(img_path).convert('RGB')  # 打开图片并转换为RGB模式
    #     label = self.get_label_from_path(img_path)   # 根据文件路径提取标签（自定义函数）

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, int(label)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 添加详细错误信息
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 提供具体错误信息
            raise RuntimeError(f"无法加载图像: {img_path} | 错误: {str(e)}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx] 

    # def get_label_from_path(self, img_path):
    #     """
    #     根据文件路径提取标签（根据实际需求进行修改）。
    #     假设文件夹名称表示类别，例如 '/path/to/dataset/class1/image1.jpg'。
    #     """
    #     return os.path.basename(os.path.dirname(img_path))
    def get_label_from_path(self, img_path):
        """改进的标签提取方法"""
        parts = Path(img_path).parts
        
        # 根据您的实际路径结构调整
        if len(parts) >= 3:
            # 取倒数第二级目录名（如 '0'）
            label_dir = parts[-2]
            
            # 确保目录名是数字
            if label_dir.isdigit():
                return label_dir
        
        # 默认返回0（根据您的类别调整）
        print(f"⚠️ 警告：无法从路径提取标签: {img_path}")
        return 0
