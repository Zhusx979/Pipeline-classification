from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any

import torch

from common.path_utils import resolve_project_path


def stable_hash(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def heatmap_cache_path(heatmap_root: str | Path, image_path: str, split: str = "train") -> Path:
    root = resolve_project_path(heatmap_root)
    digest = stable_hash(str(Path(image_path).resolve()))
    return root / split / f"{digest}.png"


def build_stage2_heatmap_root(
    heatmap_cache_root: str | Path,
    checkpoint_path: str | Path,
    cache_version: str,
) -> Path:
    root = resolve_project_path(heatmap_cache_root)
    checkpoint_stem = Path(checkpoint_path).stem
    return root / f"{checkpoint_stem}_{cache_version}"


def load_state_dict_compat(model: torch.nn.Module, checkpoint_path: str | Path) -> torch.nn.Module:
    resolved_checkpoint_path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def resolve_stage1_target_layer(model: torch.nn.Module):
    if hasattr(model, "resnet") and hasattr(model.resnet, "layer4"):
        return model.resnet.layer4[-1]
    if hasattr(model, "densenet") and hasattr(model.densenet, "features"):
        features = model.densenet.features
        if hasattr(features, "norm5"):
            return features.norm5
        return list(features.children())[-1]
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1]
        if hasattr(backbone, "features"):
            features = backbone.features
            if hasattr(features, "norm5"):
                return features.norm5
            children = list(features.children())
            if children:
                return children[-1]
        if hasattr(backbone, "conv_head"):
            return backbone.conv_head
        if hasattr(backbone, "blocks"):
            try:
                return backbone.blocks[-1]
            except Exception:
                pass
        for attr in ("norm", "head_drop"):
            if hasattr(backbone, attr):
                return getattr(backbone, attr)
        blocks = getattr(backbone, "blocks", None) or getattr(backbone, "serial_blocks", None)
        if blocks is not None:
            try:
                last_block = blocks[-1]
                if isinstance(last_block, torch.nn.Sequential) and len(last_block) > 0:
                    return last_block[-1]
                return last_block
            except Exception:
                pass
    if hasattr(model, "cait"):
        backbone = model.cait
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            return backbone.blocks[-1]
    raise ValueError("Unsupported stage1 backbone for stage2 Grad-CAM target-layer resolution.")
