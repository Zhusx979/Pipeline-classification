from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import timm
from transformers import AutoModel

from .backbones import create_backbone


@dataclass
class FusionSpec:
    backbone_name: str
    transformer_name: str
    num_classes: int
    transformer_path: str | None = None
    transformer_source: str = "hf"
    fusion_dim: int = 512
    transformer_frozen: bool = False
    backbone_pretrained: bool = True
    transformer_pretrained: bool = True
    classifier_hidden_dim: int = 256


def _freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class TransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        transformer_name: str,
        *,
        transformer_source: str = "hf",
        transformer_path: str | None = None,
        pretrained: bool = True,
        frozen: bool = False,
    ):
        super().__init__()
        self.transformer_name = transformer_name
        self.transformer_source = transformer_source

        if transformer_source == "hf":
            model_id = transformer_path or transformer_name
            self.encoder = AutoModel.from_pretrained(model_id)
            self.num_features = getattr(self.encoder.config, "hidden_size", None) or 768
        elif transformer_source == "timm":
            self.encoder = timm.create_model(transformer_name, pretrained=pretrained, num_classes=0)
            self.num_features = getattr(self.encoder, "num_features", None)
            if self.num_features is None:
                raise ValueError(f"Transformer {transformer_name} does not expose num_features.")
        else:
            raise ValueError(f"Unsupported transformer source: {transformer_source}")

        if frozen:
            _freeze_module(self.encoder)

    def forward(self, x):
        if self.transformer_source == "hf":
            outputs = self.encoder(x)
            return outputs.last_hidden_state[:, 0, :]
        return self.encoder(x)


class BackboneClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, backbone_pretrained: bool = True):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained=backbone_pretrained)
        self.backbone_dim = getattr(self.backbone, "num_features", None)
        if self.backbone_dim is None:
            raise ValueError(f"Backbone {backbone_name} does not expose num_features.")
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class DinoClassifier(nn.Module):
    def __init__(self, num_classes: int, dino_path: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(dino_path)
        self.backbone_dim = getattr(self.backbone.config, "hidden_size", None) or 768
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        transformer_name: str,
        num_classes: int,
        *,
        transformer_source: str = "hf",
        transformer_path: str | None = None,
        transformer_pretrained: bool = True,
        transformer_frozen: bool = False,
    ):
        super().__init__()
        self.backbone = TransformerEncoderWrapper(
            transformer_name,
            transformer_source=transformer_source,
            transformer_path=transformer_path,
            pretrained=transformer_pretrained,
            frozen=transformer_frozen,
        )
        self.transformer_name = transformer_name
        self.transformer_source = transformer_source
        self.backbone_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class PointwiseFusionModel(nn.Module):
    def __init__(self, spec: FusionSpec, backbone_dim: int | None = None):
        super().__init__()
        self.backbone = create_backbone(spec.backbone_name, pretrained=spec.backbone_pretrained)
        self.backbone_dim = backbone_dim or getattr(self.backbone, "num_features", None)
        if self.backbone_dim is None:
            raise ValueError(f"Backbone {spec.backbone_name} does not expose num_features.")
        self.transformer = TransformerEncoderWrapper(
            spec.transformer_name,
            transformer_source=spec.transformer_source,
            transformer_path=spec.transformer_path,
            pretrained=spec.transformer_pretrained,
            frozen=spec.transformer_frozen,
        )
        self.transformer_name = spec.transformer_name
        self.transformer_source = spec.transformer_source
        self.transformer_dim = self.transformer.num_features

        self.backbone_projection = nn.Sequential(
            nn.Linear(self.backbone_dim, spec.fusion_dim),
            nn.BatchNorm1d(spec.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.transformer_projection = nn.Sequential(
            nn.Linear(self.transformer_dim, spec.fusion_dim),
            nn.BatchNorm1d(spec.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(spec.fusion_dim, spec.classifier_hidden_dim),
            nn.BatchNorm1d(spec.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(spec.classifier_hidden_dim, spec.num_classes),
        )

    def forward(self, x):
        backbone_feat = self.backbone(x)
        transformer_feat = self.transformer(x)
        backbone_proj = self.backbone_projection(backbone_feat)
        transformer_proj = self.transformer_projection(transformer_feat)
        fused_feat = backbone_proj * transformer_proj
        return self.classifier(fused_feat)


class AddFusionModel(nn.Module):
    def __init__(self, spec: FusionSpec, backbone_dim: int | None = None):
        super().__init__()
        self.backbone = create_backbone(spec.backbone_name, pretrained=spec.backbone_pretrained)
        self.backbone_dim = backbone_dim or getattr(self.backbone, "num_features", None)
        if self.backbone_dim is None:
            raise ValueError(f"Backbone {spec.backbone_name} does not expose num_features.")
        self.transformer = TransformerEncoderWrapper(
            spec.transformer_name,
            transformer_source=spec.transformer_source,
            transformer_path=spec.transformer_path,
            pretrained=spec.transformer_pretrained,
            frozen=spec.transformer_frozen,
        )
        self.transformer_name = spec.transformer_name
        self.transformer_source = spec.transformer_source
        self.transformer_dim = self.transformer.num_features

        self.backbone_projection = nn.Sequential(
            nn.Linear(self.backbone_dim, spec.fusion_dim),
            nn.BatchNorm1d(spec.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.transformer_projection = nn.Sequential(
            nn.Linear(self.transformer_dim, spec.fusion_dim),
            nn.BatchNorm1d(spec.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(spec.fusion_dim, spec.classifier_hidden_dim),
            nn.BatchNorm1d(spec.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(spec.classifier_hidden_dim, spec.num_classes),
        )

    def forward(self, x):
        backbone_feat = self.backbone(x)
        transformer_feat = self.transformer(x)
        backbone_proj = self.backbone_projection(backbone_feat)
        transformer_proj = self.transformer_projection(transformer_feat)
        fused_feat = backbone_proj + transformer_proj
        return self.classifier(fused_feat)


class ResNetDinoPointwiseMulFusion(nn.Module):
    def __init__(self, num_classes, dino_path):
        super().__init__()
        self.resnet = timm.create_model("resnet101", pretrained=True, num_classes=0)
        self.resnet_dim = 2048
        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino_dim = 768
        self.resnet_projection = nn.Sequential(
            nn.Linear(self.resnet_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.dino_projection = nn.Sequential(
            nn.Linear(self.dino_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        resnet_feat = self.resnet(x)
        dino_out = self.dino(x)
        dino_feat = dino_out.last_hidden_state[:, 0, :]
        resnet_proj = self.resnet_projection(resnet_feat)
        dino_proj = self.dino_projection(dino_feat)
        fusion_feat = resnet_proj * dino_proj
        return self.classifier(fusion_feat)


class ResNetDinoPointwiseFusion(nn.Module):
    def __init__(self, num_classes, dino_path):
        super().__init__()
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.resnet_dim = 2048
        self.dino = AutoModel.from_pretrained(dino_path)
        _freeze_module(self.dino)
        self.dino_dim = 768
        self.dino_project = nn.Sequential(
            nn.Linear(self.dino_dim, self.resnet_dim),
            nn.LayerNorm(self.resnet_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(self.resnet_dim, num_classes)

    def forward(self, x):
        resnet_feat = self.resnet(x)
        with torch.no_grad():
            dino_out = self.dino(x)
            dino_feat = dino_out.last_hidden_state[:, 0, :]
        dino_feat_proj = self.dino_project(dino_feat)
        fusion_feat = resnet_feat * dino_feat_proj
        return self.classifier(fusion_feat)


class DenseNetDinoPointwiseFusion(nn.Module):
    def __init__(self, num_classes, dino_path, densenet_version="densenet121"):
        super().__init__()
        self.densenet = timm.create_model(densenet_version, pretrained=True, num_classes=0)
        self.densenet_dim = self.densenet.num_features
        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino_dim = 768
        self.densenet_projection = nn.Sequential(
            nn.Linear(self.densenet_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.dino_projection = nn.Sequential(
            nn.Linear(self.dino_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        densenet_feat = self.densenet(x)
        dino_out = self.dino(x)
        dino_feat = dino_out.last_hidden_state[:, 0, :]
        densenet_proj = self.densenet_projection(densenet_feat)
        dino_proj = self.dino_projection(dino_feat)
        fusion_feat = densenet_proj * dino_proj
        return self.classifier(fusion_feat)


class InceptionDinoFusion(nn.Module):
    def __init__(self, num_classes, dino_path):
        super().__init__()
        self.inception = timm.create_model("inception_v3", pretrained=True, num_classes=0)
        self.inception_dim = 2048
        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino_dim = 768
        self.inception_projection = nn.Sequential(
            nn.Linear(self.inception_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.dino_projection = nn.Sequential(
            nn.Linear(self.dino_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        inception_feat = self.inception(x)
        dino_out = self.dino(x)
        dino_feat = dino_out.last_hidden_state[:, 0, :]
        proj_inception = self.inception_projection(inception_feat)
        proj_dino = self.dino_projection(dino_feat)
        fused_feat = proj_inception + proj_dino
        return self.classifier(fused_feat)


class CaiTDinoPointwiseFusion(nn.Module):
    def __init__(self, num_classes, dino_path, cait_model_name="cait_s24_224"):
        super().__init__()
        self.cait = timm.create_model(cait_model_name, pretrained=True, num_classes=0)
        if "s24" in cait_model_name or "s36" in cait_model_name:
            self.cait_dim = 384
        else:
            self.cait_dim = 768
        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino_dim = 768
        self.fusion_dim = 512
        self.cait_projection = nn.Sequential(
            nn.Linear(self.cait_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.dino_projection = nn.Sequential(
            nn.Linear(self.dino_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.norm = nn.LayerNorm(self.fusion_dim)

    def forward(self, x):
        cait_feat = self.cait(x)
        dino_out = self.dino(x)
        dino_feat = dino_out.last_hidden_state[:, 0, :]
        cait_proj = self.cait_projection(cait_feat)
        dino_proj = self.dino_projection(dino_feat)
        fusion_feat = cait_proj * dino_proj
        fusion_feat = self.norm(fusion_feat)
        return self.classifier(fusion_feat)
