from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common.path_utils import ensure_file, resolve_project_path
from common.plotting import blend_heatmap_overlay
from models.model_factory import build_model
from stage2.gradcam import GradCAM
from stage2.utils import load_state_dict_compat, resolve_stage1_target_layer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for a stage1 model.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="Stage1 checkpoint path.")
    parser.add_argument("--preset", default="resnet50_dino_pointwise")
    parser.add_argument("--output", default="gradcam_overlay.png")
    parser.add_argument("--dino-path", default=None, help="Local model path or Hugging Face model id.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_input_tensor(image_path, image_size, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor


def main():
    args = parse_args()
    if args.dino_path:
        os.environ["DINO_PATH"] = args.dino_path

    image_path = ensure_file(args.image, description="input image")
    checkpoint_path = ensure_file(args.checkpoint, description="model checkpoint")
    output_path = resolve_project_path(args.output)

    device = torch.device(args.device)
    model = build_model(args.preset, args.num_classes, os.environ.get("DINO_PATH", "facebook/dinov2-base")).to(device)
    load_state_dict_compat(model, checkpoint_path)
    target_layer = resolve_stage1_target_layer(model)
    cam_tool = GradCAM(model, target_layer)

    input_tensor = build_input_tensor(image_path, args.image_size, device)
    heatmap, predicted_class = cam_tool(input_tensor)
    cam_tool.remove_hooks()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image with OpenCV: {image_path}")
    image_rgb = cv2.cvtColor(cv2.resize(image_bgr, (args.image_size, args.image_size)), cv2.COLOR_BGR2RGB)
    overlay = blend_heatmap_overlay(image_rgb, heatmap, alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(overlay)).save(output_path)
    print(f"Saved Grad-CAM overlay to {output_path} | predicted_class={predicted_class}")


if __name__ == "__main__":
    main()
