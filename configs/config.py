from dataclasses import asdict, dataclass, field
import os
from typing import Dict, Optional

from common.path_utils import PROJECT_ROOT

DATASET_ROOT = PROJECT_ROOT / "dataset"
LOCAL_DINO_PATH = PROJECT_ROOT / "dino_model"
DEFAULT_DINO_PATH = os.environ.get(
    "DINO_PATH",
    str(LOCAL_DINO_PATH) if LOCAL_DINO_PATH.exists() else "facebook/dinov2-base",
)


@dataclass
class DatasetConfig:
    train_txt: str = str(DATASET_ROOT / "train.txt")
    val_txt: str = str(DATASET_ROOT / "valid.txt")
    test_txt: str = str(DATASET_ROOT / "test.txt")
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    drop_last: bool = True
    shuffle_train: bool = True


@dataclass
class ExperimentConfig:
    preset_name: str
    name: str
    backbone: str
    fusion: str
    num_classes: int = 4
    epochs: int = 1
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    use_multi_gpu: bool = False
    dino_path: str = DEFAULT_DINO_PATH
    dino_frozen: bool = False
    transformer_name: str = "none"
    transformer_source: str = "none"
    transformer_path: str = ""
    transformer_frozen: bool = False
    backbone_pretrained: bool = True
    projection_dim: int = 512
    classifier_hidden_dim: int = 256
    backbone_lr: float = 1e-4
    dino_lr: float = 1e-5
    projection_lr: float = 1e-3
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    save_root: str = "./result"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _preset(
    *,
    name: str,
    backbone: str,
    fusion: str,
    projection_dim: int,
    backbone_pretrained: bool = True,
    backbone_lr: float = 1e-4,
    dino_lr: float = 0.0,
    projection_lr: float = 1e-3,
    classifier_lr: float = 1e-3,
    save_root: str = "./result",
    dino_frozen: bool = True,
    transformer_name: str = "none",
    transformer_source: str = "none",
    transformer_frozen: bool = False,
) -> Dict[str, object]:
    return {
        "name": name,
        "backbone": backbone,
        "fusion": fusion,
        "backbone_pretrained": backbone_pretrained,
        "dino_frozen": dino_frozen,
        "projection_dim": projection_dim,
        "backbone_lr": backbone_lr,
        "dino_lr": dino_lr,
        "projection_lr": projection_lr,
        "classifier_lr": classifier_lr,
        "save_root": save_root,
        "transformer_name": transformer_name,
        "transformer_source": transformer_source,
        "transformer_frozen": transformer_frozen,
    }


MODEL_PRESETS: Dict[str, Dict[str, object]] = {
    "resnet18": _preset(name="ResNet18", backbone="resnet18", fusion="none", projection_dim=512),
    "resnet34": _preset(name="ResNet34", backbone="resnet34", fusion="none", projection_dim=512),
    "resnet50": _preset(name="ResNet50", backbone="resnet50", fusion="none", projection_dim=2048),
    "resnet101": _preset(name="ResNet101", backbone="resnet101", fusion="none", projection_dim=2048),
    "densenet121": _preset(name="DenseNet121", backbone="densenet121", fusion="none", projection_dim=1024),
    "inception_v3": _preset(name="Inception v3", backbone="inception_v3", fusion="none", projection_dim=2048),
    "mobilenetv3_small": _preset(
        name="MobileNetV3 Small",
        backbone="mobilenetv3_small_100",
        fusion="none",
        projection_dim=1024,
    ),
    "efficientnet_b0": _preset(
        name="EfficientNet-B0",
        backbone="efficientnet_b0",
        fusion="none",
        projection_dim=1280,
    ),
    "coat_tiny": _preset(name="CoaT Tiny", backbone="coat_tiny.in1k", fusion="none", projection_dim=320),
    "coat_small": _preset(name="CoaT Small", backbone="coat_small.in1k", fusion="none", projection_dim=320),
    "hrnet_w18_scratch": _preset(
        name="HRNet-W18 (Scratch)",
        backbone="hrnet_w18",
        fusion="none",
        projection_dim=2048,
        backbone_pretrained=False,
    ),
    "rexnet_100_scratch": _preset(
        name="ReXNet-1.0 (Scratch)",
        backbone="rexnet_100",
        fusion="none",
        projection_dim=1280,
        backbone_pretrained=False,
    ),
    "regnety_002_scratch": _preset(
        name="RegNetY-002 (Scratch)",
        backbone="regnety_002",
        fusion="none",
        projection_dim=368,
        backbone_pretrained=False,
    ),
    "swin_tiny_cls_scratch": _preset(
        name="Swin-Tiny (Scratch)",
        backbone="swin_tiny_patch4_window7_224.ms_in1k",
        fusion="none",
        projection_dim=768,
        backbone_pretrained=False,
        transformer_name="swin_tiny_patch4_window7_224.ms_in1k",
        transformer_source="timm",
    ),
    "xcit_nano_cls_scratch": _preset(
        name="XCiT-Nano (Scratch)",
        backbone="xcit_nano_12_p16_224.fb_in1k",
        fusion="none",
        projection_dim=128,
        backbone_pretrained=False,
        transformer_name="xcit_nano_12_p16_224.fb_in1k",
        transformer_source="timm",
    ),
    "twins_svt_small_cls_scratch": _preset(
        name="Twins-SVT-Small (Scratch)",
        backbone="twins_svt_small.in1k",
        fusion="none",
        projection_dim=512,
        backbone_pretrained=False,
        transformer_name="twins_svt_small.in1k",
        transformer_source="timm",
    ),
    "mambaout_femto_cls_scratch": _preset(
        name="MambaOut-Femto (Scratch)",
        backbone="mambaout_femto.in1k",
        fusion="none",
        projection_dim=160,
        backbone_pretrained=False,
    ),
    "mambaout_kobe_cls_scratch": _preset(
        name="MambaOut-Kobe (Scratch)",
        backbone="mambaout_kobe.in1k",
        fusion="none",
        projection_dim=192,
        backbone_pretrained=False,
    ),
    "mambaout_tiny_cls_scratch": _preset(
        name="MambaOut-Tiny (Scratch)",
        backbone="mambaout_tiny.in1k",
        fusion="none",
        projection_dim=384,
        backbone_pretrained=False,
    ),
    "dinov2": _preset(
        name="DINOv2",
        backbone="dinov2",
        fusion="none",
        projection_dim=768,
        backbone_lr=0.0,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "deit_tiny_cls": _preset(
        name="DeiT-Tiny",
        backbone="deit_tiny_patch16_224",
        fusion="none",
        projection_dim=192,
        transformer_name="deit_tiny_patch16_224",
        transformer_source="timm",
    ),
    "vit_tiny_cls": _preset(
        name="ViT-Tiny",
        backbone="vit_tiny_patch16_224",
        fusion="none",
        projection_dim=192,
        transformer_name="vit_tiny_patch16_224",
        transformer_source="timm",
    ),
    "resnet50_dino_pointwise": _preset(
        name="ResNet50 + DINOv2 Pointwise Multiplication",
        backbone="resnet50",
        fusion="mul",
        projection_dim=2048,
        dino_lr=0.0,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "resnet18_dino_pointwise": _preset(
        name="ResNet18 + DINOv2 Pointwise Multiplication",
        backbone="resnet18",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "resnet34_dino_pointwise": _preset(
        name="ResNet34 + DINOv2 Pointwise Multiplication",
        backbone="resnet34",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "resnet101_dino_pointwise": _preset(
        name="ResNet101 + DINOv2 Pointwise Multiplication",
        backbone="resnet101",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "densenet121_dino_pointwise": _preset(
        name="DenseNet121 + DINOv2 Pointwise Multiplication",
        backbone="densenet121",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "inception_v3_dino_pointwise": _preset(
        name="Inception v3 + DINOv2 Pointwise Multiplication",
        backbone="inception_v3",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "mobilenetv3_small_dino_pointwise": _preset(
        name="MobileNetV3 Small + DINOv2 Pointwise Multiplication",
        backbone="mobilenetv3_small_100",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "efficientnet_b0_dino_pointwise": _preset(
        name="EfficientNet-B0 + DINOv2 Pointwise Multiplication",
        backbone="efficientnet_b0",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "coat_tiny_dino_pointwise": _preset(
        name="CoaT Tiny + DINOv2 Pointwise Multiplication",
        backbone="coat_tiny.in1k",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "coat_small_dino_pointwise": _preset(
        name="CoaT Small + DINOv2 Pointwise Multiplication",
        backbone="coat_small.in1k",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "inception_v3_dino_add": _preset(
        name="Inception v3 + DINOv2 Projection Add",
        backbone="inception_v3",
        fusion="add",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "cait_s24_dino_pointwise": _preset(
        name="CaiT-S24 + DINOv2 Pointwise Multiplication",
        backbone="cait_s24_224",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="facebook/dinov2-base",
        transformer_source="hf",
    ),
    "resnet50_coat_tiny_pointwise": _preset(
        name="ResNet50 + CoaT Tiny Pointwise Multiplication",
        backbone="resnet50",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="coat_tiny.in1k",
        transformer_source="timm",
    ),
    "resnet50_coat_small_pointwise": _preset(
        name="ResNet50 + CoaT Small Pointwise Multiplication",
        backbone="resnet50",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="coat_small.in1k",
        transformer_source="timm",
    ),
    "resnet50_deit_tiny_pointwise": _preset(
        name="ResNet50 + DeiT-Tiny Pointwise Multiplication",
        backbone="resnet50",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="deit_tiny_patch16_224",
        transformer_source="timm",
    ),
    "resnet50_vit_tiny_pointwise": _preset(
        name="ResNet50 + ViT-Tiny Pointwise Multiplication",
        backbone="resnet50",
        fusion="mul",
        projection_dim=512,
        dino_lr=1e-5,
        dino_frozen=False,
        transformer_name="vit_tiny_patch16_224",
        transformer_source="timm",
    ),
}


def build_experiment_config(
    preset_name: str,
    *,
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
) -> ExperimentConfig:
    if preset_name not in MODEL_PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    preset = MODEL_PRESETS[preset_name]
    dataset = DatasetConfig(
        train_txt=train_txt or str(DATASET_ROOT / "train.txt"),
        val_txt=val_txt or str(DATASET_ROOT / "valid.txt"),
        test_txt=test_txt or str(DATASET_ROOT / "test.txt"),
        image_size=image_size or 224,
        batch_size=batch_size or 32,
        num_workers=num_workers or 0,
    )

    transformer_name = str(preset.get("transformer_name", "none"))
    transformer_source = str(preset.get("transformer_source", "none"))
    transformer_path = DEFAULT_DINO_PATH if transformer_name == "facebook/dinov2-base" else ""
    transformer_frozen = bool(preset.get("transformer_frozen", preset["dino_frozen"]))

    cfg = ExperimentConfig(
        preset_name=preset_name,
        name=str(preset["name"]),
        backbone=str(preset["backbone"]),
        fusion=str(preset["fusion"]),
        num_classes=num_classes or 4,
        epochs=epochs or 50,
        image_size=image_size or 224,
        batch_size=batch_size or 32,
        num_workers=num_workers or 0,
        use_multi_gpu=use_multi_gpu,
        dino_path=DEFAULT_DINO_PATH,
        dino_frozen=bool(preset["dino_frozen"]),
        transformer_name=transformer_name,
        transformer_source=transformer_source,
        transformer_path=transformer_path,
        transformer_frozen=transformer_frozen,
        backbone_pretrained=bool(preset.get("backbone_pretrained", True)),
        projection_dim=int(preset["projection_dim"]),
        classifier_hidden_dim=256,
        backbone_lr=float(preset["backbone_lr"]),
        dino_lr=float(preset["dino_lr"]),
        projection_lr=float(preset["projection_lr"]),
        classifier_lr=float(preset["classifier_lr"]),
        weight_decay=1e-4,
        eta_min=1e-6,
        save_root=str(preset["save_root"]),
        dataset=dataset,
        extra=extra or {},
    )
    return cfg
