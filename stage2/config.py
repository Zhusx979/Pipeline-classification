from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional
import os

from common.path_utils import PROJECT_ROOT

DATASET_ROOT = PROJECT_ROOT / "dataset"
LOCAL_DINO_PATH = PROJECT_ROOT / "dino_model"
DEFAULT_DINO_PATH = os.environ.get(
    "DINO_PATH",
    str(LOCAL_DINO_PATH) if LOCAL_DINO_PATH.exists() else "facebook/dinov2-base",
)


@dataclass
class Stage2DatasetConfig:
    train_txt: str = str(DATASET_ROOT / "train.txt")
    val_txt: str = str(DATASET_ROOT / "valid.txt")
    test_txt: str = str(DATASET_ROOT / "test.txt")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    drop_last: bool = True
    shuffle_train: bool = True
    heatmap_cache_root: str = str(PROJECT_ROOT / "stage2_heatmaps")


@dataclass
class Stage2ExperimentConfig:
    preset_name: str
    name: str
    backbone: str
    num_classes: int = 4
    epochs: int = 1
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    use_multi_gpu: bool = False
    dino_path: str = DEFAULT_DINO_PATH
    dino_frozen: bool = True
    backbone_pretrained: bool = True
    backbone_lr: float = 1e-4
    dino_lr: float = 0.0
    projection_lr: float = 1e-3
    classifier_lr: float = 1e-3
    adapter_reconstruction_weight: float = 0.1
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    save_root: str = "./stage2_result"
    stage1_preset_name: str = "resnet50_dino_pointwise"
    stage1_checkpoint: str = ""
    dataset: Stage2DatasetConfig = field(default_factory=Stage2DatasetConfig)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _stage2_preset(name: str, backbone: str) -> Dict[str, object]:
    return {
        "name": name,
        "backbone": backbone,
        "backbone_pretrained": True,
        "dino_frozen": True,
        "backbone_lr": 1e-4,
        "dino_lr": 0.0,
        "projection_lr": 1e-3,
        "classifier_lr": 1e-3,
        "save_root": "./stage2_result",
    }


MODEL_PRESETS: Dict[str, Dict[str, object]] = {
    "resnet18_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 ResNet18 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "resnet18",
    ),
    "resnet34_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 ResNet34 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "resnet34",
    ),
    "resnet50_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 ResNet50 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "resnet50",
    ),
    "resnet101_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 ResNet101 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "resnet101",
    ),
    "densenet121_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 DenseNet121 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "densenet121",
    ),
    "inception_v3_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 InceptionV3 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "inception_v3",
    ),
    "mobilenetv3_small_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 MobileNetV3-Small + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "mobilenetv3_small_100",
    ),
    "efficientnet_b0_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 EfficientNet-B0 + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "efficientnet_b0",
    ),
    "coat_tiny_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 CoaT Tiny + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "coat_tiny.in1k",
    ),
    "coat_small_heatmap_dino_pointwise": _stage2_preset(
        "Stage2 CoaT Small + GradCAM Heatmap + DINOv2 Pointwise Fusion",
        "coat_small.in1k",
    ),
}


def build_stage2_experiment_config(
    preset_name: str,
    *,
    stage1_preset_name: str = "resnet50_dino_pointwise",
    stage1_checkpoint: Optional[str] = None,
    train_txt: Optional[str] = None,
    val_txt: Optional[str] = None,
    test_txt: Optional[str] = None,
    image_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    epochs: Optional[int] = None,
    num_classes: Optional[int] = None,
    use_multi_gpu: bool = False,
    extra: Optional[Dict[str, object]] = None,
) -> Stage2ExperimentConfig:
    if preset_name not in MODEL_PRESETS:
        raise KeyError(f"Unknown stage2 preset: {preset_name}")

    preset = MODEL_PRESETS[preset_name]
    dataset = Stage2DatasetConfig(
        train_txt=train_txt or str(DATASET_ROOT / "train.txt"),
        val_txt=val_txt or str(DATASET_ROOT / "valid.txt"),
        test_txt=test_txt or str(DATASET_ROOT / "test.txt"),
        image_size=image_size or 224,
        batch_size=batch_size or 32,
        num_workers=num_workers or 0,
    )

    cfg = Stage2ExperimentConfig(
        preset_name=preset_name,
        name=str(preset["name"]),
        backbone=str(preset["backbone"]),
        num_classes=num_classes or 4,
        epochs=epochs or 50,
        image_size=image_size or 224,
        batch_size=batch_size or 32,
        num_workers=num_workers or 0,
        use_multi_gpu=use_multi_gpu,
        dino_path=DEFAULT_DINO_PATH,
        dino_frozen=bool(preset["dino_frozen"]),
        backbone_pretrained=bool(preset["backbone_pretrained"]),
        backbone_lr=float(preset["backbone_lr"]),
        dino_lr=float(preset["dino_lr"]),
        projection_lr=float(preset["projection_lr"]),
        classifier_lr=float(preset["classifier_lr"]),
        adapter_reconstruction_weight=0.1,
        weight_decay=1e-4,
        eta_min=1e-6,
        save_root=str(preset["save_root"]),
        stage1_preset_name=stage1_preset_name,
        stage1_checkpoint=stage1_checkpoint or "",
        dataset=dataset,
        extra=extra or {},
    )
    return cfg
