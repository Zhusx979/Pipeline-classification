from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common.plotting import blend_heatmap_overlay


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_indices = torch.argmax(output, dim=1)
        else:
            class_indices = torch.full(
                (output.size(0),),
                int(class_idx),
                device=output.device,
                dtype=torch.long,
            )

        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(output.size(0), device=output.device), class_indices] = 1.0
        output.backward(gradient=one_hot)

        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")
        if grads.dim() == 3:
            grads = grads.transpose(1, 2).unsqueeze(-1)
        if acts.dim() == 3:
            acts = acts.transpose(1, 2).unsqueeze(-1)
        if grads.dim() != 4 or acts.dim() != 4:
            raise RuntimeError(
                f"Grad-CAM expects 4D or token-like 3D activations/gradients, got grads={tuple(grads.shape)}, acts={tuple(acts.shape)}."
            )

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)
        cam = F.relu(cam)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam[0].detach().cpu().numpy(), int(class_indices[0].item())


def save_pure_heatmap(heatmap: np.ndarray, save_path: str | Path, output_size: tuple[int, int] | None = None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    if output_size is not None and heatmap.shape[:2] != output_size:
        heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LINEAR)

    heatmap_rgb = (plt.get_cmap("magma")(heatmap)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(heatmap_rgb).save(save_path)


def save_grayscale_heatmap(heatmap: np.ndarray, save_path: str | Path, output_size: tuple[int, int] | None = None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    if output_size is not None and heatmap.shape[:2] != output_size:
        heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LINEAR)

    heatmap_gray = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heatmap_gray, mode="L").save(save_path)


def save_overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | Path,
    output_size: tuple[int, int] | None = None,
    alpha: float = 0.48,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    if output_size is not None and heatmap.shape[:2] != output_size:
        heatmap = cv2.resize(heatmap, output_size, interpolation=cv2.INTER_LINEAR)
    if output_size is not None and image_rgb.shape[:2] != output_size:
        image_rgb = cv2.resize(image_rgb, output_size, interpolation=cv2.INTER_AREA)

    overlay = blend_heatmap_overlay(image_rgb, heatmap, alpha=alpha)
    Image.fromarray(overlay).save(save_path)
