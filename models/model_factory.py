from .fusion_models import (
    AddFusionModel,
    BackboneClassifier,
    CaiTDinoPointwiseFusion,
    DenseNetDinoPointwiseFusion,
    DinoClassifier,
    FusionSpec,
    InceptionDinoFusion,
    PointwiseFusionModel,
    ResNetDinoPointwiseFusion,
    ResNetDinoPointwiseMulFusion,
    TransformerClassifier,
)


BACKBONE_CLASSIFIER_PRESETS = {
    "resnet18": ("resnet18", True),
    "resnet34": ("resnet34", True),
    "resnet50": ("resnet50", True),
    "resnet101": ("resnet101", True),
    "densenet121": ("densenet121", True),
    "inception_v3": ("inception_v3", True),
    "mobilenetv3_small": ("mobilenetv3_small_100", True),
    "efficientnet_b0": ("efficientnet_b0", True),
    "hrnet_w18_scratch": ("hrnet_w18", False),
    "rexnet_100_scratch": ("rexnet_100", False),
    "regnety_002_scratch": ("regnety_002", False),
    "coat_tiny": ("coat_tiny.in1k", True),
    "coat_small": ("coat_small.in1k", True),
    "swin_tiny_cls_scratch": ("swin_tiny_patch4_window7_224.ms_in1k", False),
    "xcit_nano_cls_scratch": ("xcit_nano_12_p16_224.fb_in1k", False),
    "twins_svt_small_cls_scratch": ("twins_svt_small.in1k", False),
    "mambaout_femto_cls_scratch": ("mambaout_femto.in1k", False),
    "mambaout_kobe_cls_scratch": ("mambaout_kobe.in1k", False),
    "mambaout_tiny_cls_scratch": ("mambaout_tiny.in1k", False),
}

TIMM_TRANSFORMER_PRESETS = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "beit_base": "beit_base_patch16_224",
    "cait_s24": "cait_s24_224",
}

HF_TRANSFORMER_PRESETS = {
    "dinov2": None,
}


def _build_fusion_spec(backbone_name: str, transformer_key: str, num_classes: int, dino_path: str, fusion_dim: int = 512):
    if transformer_key in HF_TRANSFORMER_PRESETS:
        transformer_name = dino_path if transformer_key == "dinov2" else transformer_key
        transformer_path = dino_path if transformer_key == "dinov2" else None
        return FusionSpec(
            backbone_name=backbone_name,
            transformer_name=transformer_name,
            transformer_path=transformer_path,
            transformer_source="hf",
            num_classes=num_classes,
            fusion_dim=fusion_dim,
        )

    if transformer_key in TIMM_TRANSFORMER_PRESETS:
        return FusionSpec(
            backbone_name=backbone_name,
            transformer_name=TIMM_TRANSFORMER_PRESETS[transformer_key],
            transformer_source="timm",
            num_classes=num_classes,
            fusion_dim=fusion_dim,
        )

    raise KeyError(f"Unknown transformer key: {transformer_key}")


def build_model(preset_name: str, num_classes: int, dino_path: str):
    if preset_name in BACKBONE_CLASSIFIER_PRESETS:
        backbone_name, backbone_pretrained = BACKBONE_CLASSIFIER_PRESETS[preset_name]
        return BackboneClassifier(
            backbone_name,
            num_classes=num_classes,
            backbone_pretrained=backbone_pretrained,
        )

    if preset_name == "dinov2":
        return DinoClassifier(num_classes=num_classes, dino_path=dino_path)
    if preset_name == "deit_tiny_cls":
        return TransformerClassifier("deit_tiny_patch16_224", num_classes=num_classes, transformer_source="timm")
    if preset_name == "deit_small_cls":
        return TransformerClassifier("deit_small_patch16_224", num_classes=num_classes, transformer_source="timm")
    if preset_name == "vit_tiny_cls":
        return TransformerClassifier("vit_tiny_patch16_224", num_classes=num_classes, transformer_source="timm")
    if preset_name == "vit_small_cls":
        return TransformerClassifier("vit_small_patch16_224", num_classes=num_classes, transformer_source="timm")
    if preset_name == "beit_base_cls":
        return TransformerClassifier("beit_base_patch16_224", num_classes=num_classes, transformer_source="timm")
    if preset_name == "cait_s24_cls":
        return TransformerClassifier("cait_s24_224", num_classes=num_classes, transformer_source="timm")

    if preset_name == "resnet50_dino_pointwise":
        return ResNetDinoPointwiseFusion(num_classes=num_classes, dino_path=dino_path)
    if preset_name == "resnet101_dino_pointwise":
        return ResNetDinoPointwiseMulFusion(num_classes=num_classes, dino_path=dino_path)
    if preset_name == "resnet101_dino_pointwise_unfrozen":
        return ResNetDinoPointwiseMulFusion(num_classes=num_classes, dino_path=dino_path)
    if preset_name == "densenet121_dino_pointwise":
        return DenseNetDinoPointwiseFusion(num_classes=num_classes, dino_path=dino_path, densenet_version="densenet121")
    if preset_name == "inception_v3_dino_add":
        return InceptionDinoFusion(num_classes=num_classes, dino_path=dino_path)
    if preset_name == "cait_s24_dino_pointwise":
        return CaiTDinoPointwiseFusion(num_classes=num_classes, dino_path=dino_path, cait_model_name="cait_s24_224")

    dynamic_mul_presets = {
        "resnet18_dino_pointwise": ("resnet18", "dinov2"),
        "resnet34_dino_pointwise": ("resnet34", "dinov2"),
        "mobilenetv3_small_dino_pointwise": ("mobilenetv3_small_100", "dinov2"),
        "efficientnet_b0_dino_pointwise": ("efficientnet_b0", "dinov2"),
        "coat_tiny_dino_pointwise": ("coat_tiny.in1k", "dinov2"),
        "coat_small_dino_pointwise": ("coat_small.in1k", "dinov2"),
        "inception_v3_dino_pointwise": ("inception_v3", "dinov2"),
        "resnet50_coat_tiny_pointwise": ("resnet50", "coat_tiny"),
        "resnet50_coat_small_pointwise": ("resnet50", "coat_small"),
        "resnet50_deit_tiny_pointwise": ("resnet50", "deit_tiny"),
        "resnet50_vit_tiny_pointwise": ("resnet50", "vit_tiny"),
    }
    if preset_name in dynamic_mul_presets:
        backbone_name, transformer_key = dynamic_mul_presets[preset_name]
        return PointwiseFusionModel(_build_fusion_spec(backbone_name, transformer_key, num_classes, dino_path))

    dynamic_add_presets = {
        "inception_v3_dino_add": ("inception_v3", "dinov2"),
    }
    if preset_name in dynamic_add_presets:
        backbone_name, transformer_key = dynamic_add_presets[preset_name]
        return AddFusionModel(_build_fusion_spec(backbone_name, transformer_key, num_classes, dino_path))

    raise KeyError(f"Unknown model preset: {preset_name}")
