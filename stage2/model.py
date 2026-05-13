from __future__ import annotations

import torch
import torch.nn as nn
import timm
from transformers import AutoModel


def _freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class Stage2BackboneDinoHeatmapFusion(nn.Module):
    def __init__(self, num_classes: int, dino_path: str, backbone_name: str = "resnet50"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=3)
        self.backbone_dim = getattr(self.backbone, "num_features", None)
        if self.backbone_dim is None:
            raise ValueError(f"Stage2 backbone {backbone_name} does not expose num_features.")

        self.dino = AutoModel.from_pretrained(dino_path)
        self.dino_dim = getattr(self.dino.config, "hidden_size", None) or 768
        _freeze_module(self.dino)

        self.dino_project = nn.Sequential(
            nn.Linear(self.dino_dim, self.backbone_dim),
            nn.LayerNorm(self.backbone_dim),
            nn.ReLU(inplace=True),
        )
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.heatmap_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, self.backbone_dim),
            nn.LayerNorm(self.backbone_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Dropout(p=0.3),
            nn.Linear(self.backbone_dim, self.backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.backbone_dim // 2, num_classes),
        )

    def _split_input(self, image, heatmap=None):
        if heatmap is None:
            if image.shape[1] == 4:
                return image[:, :3, :, :], image[:, 3:, :, :]
            if image.shape[1] == 3:
                zero_heatmap = torch.zeros(
                    image.shape[0],
                    1,
                    image.shape[2],
                    image.shape[3],
                    device=image.device,
                    dtype=image.dtype,
                )
                return image, zero_heatmap
            raise ValueError(f"Stage2 model expects 3-channel image (+ heatmap) or packed 4 channels, got {image.shape[1]}.")
        return image, heatmap

    def forward(self, image, heatmap=None, guidance_scale: float = 1.0):
        raw_input, heatmap_input = self._split_input(image, heatmap)
        if heatmap_input.dim() == 3:
            heatmap_input = heatmap_input.unsqueeze(1)
        heatmap_input = heatmap_input.float()
        if heatmap_input.shape[1] != 1:
            heatmap_input = heatmap_input[:, :1, :, :]

        attention_map = self.heatmap_encoder(heatmap_input)
        guided_input = raw_input * (1.0 + guidance_scale * attention_map)
        backbone_feat = self.backbone(guided_input)
        with torch.no_grad():
            dino_out = self.dino(raw_input)
            dino_feat = dino_out.last_hidden_state[:, 0, :]
        dino_proj = self.dino_project(dino_feat)
        heatmap_gate = self.heatmap_pool(attention_map)
        fused_feat = backbone_feat * dino_proj
        fused_feat = fused_feat * (1.0 + heatmap_gate)
        return self.classifier(fused_feat)
