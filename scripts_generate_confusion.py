from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.path_utils import resolve_project_path
from common.plotting import apply_publication_style
from configs.config import build_experiment_config
from metrics.classification_metrics import compute_classification_metrics
from models.model_factory import build_model
import timm
from transformers import AutoModel


ROOT = resolve_project_path(".")
RESULT_ROOT = ROOT / "result_20260511"
DATASET_ROOT = ROOT / "dataset"
OUTPUT_DIR = ROOT / "paper" / "fig"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Crack", "Lack\nfusion", "Incomp.\npenetr.", "Porosity\nslag incl."]

MODEL_SPECS = [
    ("ResNet18", "ResNet18", "resnet18"),
    ("ResNet34", "ResNet34", "resnet34"),
    ("ResNet50", "ResNet50", "resnet50"),
    ("ResNet101", "ResNet101", "resnet101"),
    ("ResNet18 + DINOv2", "EfficientNet-B0_+_DINOv2_Pointwise_Multiplication", "efficientnet_b0_dino_pointwise"),
    ("ResNet34 + DINOv2", "Inception_v3_+_DINOv2_Pointwise_Multiplication", "inception_v3_dino_pointwise" ),
    ("ResNet50 + DINOv2", "ResNet50_+_DINOv2_Pointwise_Multiplication", "resnet50_dino_pointwise"),
    ("ResNet101 + DINOv2", "ResNet101_+_DINOv2_Pointwise_Multiplication", "resnet101_dino_pointwise"),
]


_ORIG_TIMM_CREATE_MODEL = timm.create_model
_ORIG_AUTO_FROM_PRETRAINED = AutoModel.from_pretrained


def _offline_create_model(*args, **kwargs):
    kwargs["pretrained"] = False
    return _ORIG_TIMM_CREATE_MODEL(*args, **kwargs)


def _offline_auto_from_pretrained(*args, **kwargs):
    kwargs["local_files_only"] = True
    return _ORIG_AUTO_FROM_PRETRAINED(*args, **kwargs)


timm.create_model = _offline_create_model
AutoModel.from_pretrained = _offline_auto_from_pretrained


def load_test_loader():
    dataset = RadiographTestDataset(DATASET_ROOT / "test.txt", image_size=224)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


def resolve_image_path(raw_path: str) -> Path | None:
    raw_path = raw_path.strip().strip('"').strip("'")
    if not raw_path:
        return None
    normalized = raw_path.replace("\\", "/")
    candidates = [normalized]
    for legacy_name in ("dataset_0309", "dataset_0303", "dataset_0126"):
        if legacy_name in normalized:
            candidates.append(normalized.replace(legacy_name, "dataset"))

    for candidate in candidates:
        candidate_path = Path(candidate)
        options = [candidate_path]
        if not candidate_path.is_absolute():
            resolved = resolve_project_path(candidate_path)
            if resolved is not None:
                options.append(resolved)
            options.append(Path.cwd() / candidate_path)
        for path in options:
            if path.exists() and path.is_file():
                return path.resolve()
    return None


class RadiographTestDataset(Dataset):
    def __init__(self, txt_file: Path, image_size: int = 224):
        self.image_size = image_size
        self.image_paths = []
        self.labels = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                path = resolve_image_path(line)
                if path is None:
                    continue
                self.image_paths.append(path)
                self.labels.append(int(self.get_label_from_path(path)))
        print(f"Loaded {len(self.image_paths)} test images.")

    def __len__(self):
        return len(self.image_paths)

    def get_label_from_path(self, img_path):
        parts = Path(img_path).parts
        if len(parts) >= 3:
            label_dir = parts[-2]
            if label_dir.isdigit():
                return label_dir
        return 0

    def _transform(self, img: Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        scale = self.image_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        canvas.paste(resized, (0, 0))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self._transform(image), self.labels[idx]


@torch.no_grad()
def evaluate_confusion(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds)
    metrics = compute_classification_metrics(y_true, y_pred, labels=list(range(4)))
    return np.asarray(metrics["confusion_matrix"]), metrics


def load_model(model_dir: Path, preset_name: str):
    cfg = build_experiment_config(preset_name, epochs=50, num_classes=4)
    model = build_model(preset_name, num_classes=4, dino_path=cfg.dino_path)
    checkpoint = torch.load(model_dir / "best_val_model.pth", map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def draw_heatmap(ax, cm, title, vmax, show_ylabel, show_xlabel):
    sns.heatmap(
        cm,
        ax=ax,
        cmap="BuGn",
        vmin=0,
        vmax=vmax,
        cbar=False,
        square=True,
        linewidths=1.4,
        linecolor="#F8FAFC",
        annot=True,
        fmt="d",
        annot_kws={"fontsize": 18, "fontweight": "bold", "color": "#0F172A"},
    )
    ax.set_title(title, fontsize=18, fontweight="bold", pad=10)
    ax.set_xticklabels(CLASS_NAMES, rotation=0, fontsize=16, fontweight="bold")
    ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted" if show_xlabel else "", fontsize=14)
    ax.set_ylabel("True" if show_ylabel else "", fontsize=14)
    ax.tick_params(length=0)
    ax.set_facecolor("#F8FBFD")


def main():
    apply_publication_style()
    loader = load_test_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for display_name, folder, preset_name in MODEL_SPECS:
        model = load_model(RESULT_ROOT / folder, preset_name).to(device)
        cm, metrics = evaluate_confusion(model, loader, device)
        results[display_name] = (cm, metrics)
        print(display_name, metrics["accuracy"], metrics["f1_macro"])

    vmax = max(cm.max() for cm, _ in results.values())
    fig, axes = plt.subplots(2, 4, figsize=(24, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for idx, (display_name, _, _) in enumerate(MODEL_SPECS):
        row = 0 if idx < 4 else 1
        col = idx % 4
        cm, metrics = results[display_name]
        title = display_name
        draw_heatmap(ax=axes[row, col], cm=cm, title=title, vmax=vmax, show_ylabel=(col == 0), show_xlabel=(row == 1))
    fig.savefig(OUTPUT_DIR / "confusion_matrix_resnet_dino_2x4.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "confusion_matrix_resnet_dino_2x4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
