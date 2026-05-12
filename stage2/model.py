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
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=6)
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
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

        self.eval_input_adapter = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        self._init_eval_input_adapter()

    def _init_eval_input_adapter(self):
        with torch.no_grad():
            self.eval_input_adapter.weight.zero_()
            for channel in range(3):
                self.eval_input_adapter.weight[channel, channel, 0, 0] = 1.0
                self.eval_input_adapter.weight[channel + 3, channel, 0, 0] = 1.0

    def _split_input(self, x):
        if x.shape[1] == 6:
            return x, x[:, :3, :, :]
        if x.shape[1] == 3:
            return self.eval_input_adapter(x), x
        raise ValueError(f"Stage2 model expects 3 or 6 input channels, got {x.shape[1]}.")

    def forward(self, x):
        heatmap_input, raw_input = self._split_input(x)
        backbone_feat = self.backbone(heatmap_input)
        with torch.no_grad():
            dino_out = self.dino(raw_input)
            dino_feat = dino_out.last_hidden_state[:, 0, :]
        dino_proj = self.dino_project(dino_feat)
        fused_feat = backbone_feat * dino_proj
        return self.classifier(fused_feat)
