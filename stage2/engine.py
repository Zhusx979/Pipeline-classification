from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import AlbumentationsTransformVal, ImageDatasetFromTxt
from metrics.classification_metrics import compute_classification_metrics, flatten_metric_dict
from common.model_profile import format_model_profile, model_profile_to_dict, profile_model
from common.path_utils import build_timestamped_output_dir, ensure_file
from common.plotting import save_training_curve_grid
from models.model_factory import build_model
from .config import Stage2ExperimentConfig
from .dataset import AlbumentationsTransformStage2Eval, AlbumentationsTransformStage2Train, Stage2EvalDatasetFromTxt, Stage2TrainDatasetFromTxt
from .gradcam import GradCAM, save_grayscale_heatmap
from .model import Stage2BackboneDinoHeatmapFusion
from .utils import build_stage2_heatmap_root, load_state_dict_compat, heatmap_cache_path, resolve_stage1_target_layer


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _infer_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model_for_training(model: nn.Module, cfg: Stage2ExperimentConfig, device: torch.device) -> nn.Module:
    model = model.to(device)
    if cfg.use_multi_gpu and device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Multi-GPU enabled: using DataParallel on {gpu_count} GPUs.")
            model = nn.DataParallel(model)
        else:
            print("Multi-GPU requested, but fewer than 2 CUDA devices were found. Falling back to single GPU.")
    return model


def _make_loaders(cfg: Stage2ExperimentConfig):
    train_txt = ensure_file(cfg.dataset.train_txt, description="train txt file")
    val_txt = ensure_file(cfg.dataset.val_txt, description="validation txt file")
    test_txt = ensure_file(cfg.dataset.test_txt, description="test txt file")
    checkpoint_path = ensure_file(cfg.stage1_checkpoint, description="stage1 checkpoint")
    heatmap_root = build_stage2_heatmap_root(
        cfg.dataset.heatmap_cache_root,
        checkpoint_path,
        cfg.dataset.heatmap_cache_version,
    )
    train_dataset = Stage2TrainDatasetFromTxt(
        [train_txt],
        heatmap_root,
        AlbumentationsTransformStage2Train(cfg.image_size, cfg.image_size),
    )
    val_dataset = Stage2EvalDatasetFromTxt(
        [val_txt],
        heatmap_root,
        split="valid",
        transform=AlbumentationsTransformStage2Eval(cfg.image_size, cfg.image_size),
    )
    test_dataset = Stage2EvalDatasetFromTxt(
        [test_txt],
        heatmap_root,
        split="test",
        transform=AlbumentationsTransformStage2Eval(cfg.image_size, cfg.image_size),
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.dataset.shuffle_train,
        num_workers=cfg.num_workers,
        drop_last=cfg.dataset.drop_last,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, test_loader, heatmap_root


def _build_optimizer(model: nn.Module, cfg: Stage2ExperimentConfig):
    model = _unwrap_model(model)
    params = []
    if hasattr(model, "backbone"):
        params.append({"params": model.backbone.parameters(), "lr": cfg.backbone_lr})
    elif hasattr(model, "resnet"):
        params.append({"params": model.resnet.parameters(), "lr": cfg.backbone_lr})
    if hasattr(model, "dino"):
        params.append({"params": model.dino.parameters(), "lr": cfg.dino_lr})
    if hasattr(model, "dino_project"):
        params.append({"params": model.dino_project.parameters(), "lr": cfg.projection_lr})
    if hasattr(model, "heatmap_encoder"):
        params.append({"params": model.heatmap_encoder.parameters(), "lr": cfg.projection_lr})
    if hasattr(model, "heatmap_pool"):
        params.append({"params": model.heatmap_pool.parameters(), "lr": cfg.projection_lr})
    if hasattr(model, "classifier"):
        params.append({"params": model.classifier.parameters(), "lr": cfg.classifier_lr})

    if not params:
        params = [{"params": model.parameters(), "lr": cfg.backbone_lr}]

    return optim.AdamW(params, weight_decay=cfg.weight_decay)


def _collect_probs(logits: torch.Tensor):
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _format_metric_value(value):
    if value is None:
        return "None"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _print_confusion_table(cm: List[List[int]], labels: Optional[List[int]], title: str):
    labels = labels or list(range(len(cm)))
    cell_width = 8
    header = " " * 10 + " ".join(f"{str(label):>{cell_width}}" for label in labels)
    print(title)
    print(header)
    for i, row in enumerate(cm):
        row_label = f"True {labels[i]}"
        cells = " ".join(f"{int(v):>{cell_width}d}" for v in row)
        print(f"{row_label:<10} {cells}")


def _print_class_metrics(metrics: Dict[str, object]):
    labels = metrics.get("labels") or []
    per_keys = [
        ("precision", "per_class_precision"),
        ("recall", "per_class_recall"),
        ("f1", "per_class_f1"),
        ("spec", "per_class_specificity"),
        ("npv", "per_class_npv"),
    ]
    rows = []
    for idx, label in enumerate(labels):
        row = [f"{label}"]
        for _, key in per_keys:
            values = metrics.get(key, [])
            row.append(_format_metric_value(values[idx] if idx < len(values) else None))
        rows.append(row)

    if not rows:
        return

    headers = ["class", "precision", "recall", "f1", "spec", "npv"]
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("Per-class metrics:")
    print(" | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(headers))))


def _model_forward(model, images, heatmaps, cfg: Stage2ExperimentConfig):
    return model(images, heatmaps, guidance_scale=cfg.heatmap_guidance_scale)


def _evaluate(model, loader, criterion, device, cfg: Stage2ExperimentConfig):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, heatmaps, labels in loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)
            outputs = _model_forward(model, images, heatmaps, cfg)
            loss = criterion(outputs, labels)
            probs = _collect_probs(outputs)
            preds = np.argmax(probs, axis=1)

            total_loss += loss.item()
            total_correct += (preds == labels.detach().cpu().numpy()).sum()
            total_samples += labels.size(0)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.append(probs)

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_classification_metrics(all_labels, all_preds, y_prob=y_prob)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["acc_percent"] = 100.0 * total_correct / max(total_samples, 1)
    metrics["samples"] = total_samples
    return metrics


def _write_csv_header(csv_path: Path, columns: List[str]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def _append_csv_row(csv_path: Path, row: List[object]):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _save_summary_artifacts(save_path: Path, history_df: pd.DataFrame, cfg: Stage2ExperimentConfig, model_profile):
    best_val_idx = history_df["val_acc"].astype(float).idxmax()
    best_test_idx = history_df["test_acc"].astype(float).idxmax()
    best_val_row = history_df.iloc[best_val_idx]
    best_test_row = history_df.iloc[best_test_idx]

    summary_report = f"""
Model: {cfg.name}
Backbone: {cfg.backbone}
DINO Path: {cfg.dino_path}
Stage1 Preset: {cfg.stage1_preset_name}
Stage1 Checkpoint: {cfg.stage1_checkpoint}
Heatmap Cache Version: {cfg.dataset.heatmap_cache_version}
Heatmap Guidance Scale: {cfg.heatmap_guidance_scale}
Train Heatmap Uses GT Label: {cfg.heatmap_train_use_gt}
Eval Heatmap Uses GT Label: {cfg.heatmap_eval_use_gt}
Image Size: {cfg.image_size}
Classes: {cfg.num_classes}
Epochs: {cfg.epochs}

Model Profile:
{format_model_profile(model_profile)}

Best Val:
  Epoch: {best_val_row['epoch']}
  Acc: {float(best_val_row['val_acc']):.4f}%
  F1: {float(best_val_row['val_f1_macro']):.6f}
  Balanced Acc: {float(best_val_row['val_balanced_accuracy']):.6f}
  MCC: {float(best_val_row['val_mcc']):.6f}

Best Test:
  Epoch: {best_test_row['epoch']}
  Acc: {float(best_test_row['test_acc']):.4f}%
  F1: {float(best_test_row['test_f1_macro']):.6f}
  Balanced Acc: {float(best_test_row['test_balanced_accuracy']):.6f}
  MCC: {float(best_test_row['test_mcc']):.6f}
""".strip()
    (save_path / "training_summary.txt").write_text(summary_report, encoding="utf-8")

    save_training_curve_grid(history_df, save_path / "training_curves")


def _build_stage1_model(cfg: Stage2ExperimentConfig, device):
    if not cfg.stage1_checkpoint:
        raise ValueError("stage1_checkpoint must be set before running stage2.")
    checkpoint_path = ensure_file(cfg.stage1_checkpoint, description="stage1 checkpoint")
    model = build_model(cfg.stage1_preset_name, cfg.num_classes, cfg.dino_path).to(device)
    load_state_dict_compat(model, checkpoint_path)
    model.eval()
    return model


def _prepare_heatmaps(cfg: Stage2ExperimentConfig, device):
    if not cfg.stage1_checkpoint:
        raise ValueError("Please set STAGE1_CHECKPOINT in run_stage2.py before running stage2.")

    checkpoint_path = ensure_file(cfg.stage1_checkpoint, description="stage1 checkpoint")
    heatmap_root = build_stage2_heatmap_root(
        cfg.dataset.heatmap_cache_root,
        checkpoint_path,
        cfg.dataset.heatmap_cache_version,
    )
    heatmap_root.mkdir(parents=True, exist_ok=True)
    stage1_model = _build_stage1_model(cfg, device)
    target_layer = resolve_stage1_target_layer(stage1_model)
    cam_tool = GradCAM(stage1_model, target_layer)

    split_specs = [
        ("train", ensure_file(cfg.dataset.train_txt, description="train txt file"), cfg.heatmap_train_use_gt),
        ("valid", ensure_file(cfg.dataset.val_txt, description="validation txt file"), cfg.heatmap_eval_use_gt),
        ("test", ensure_file(cfg.dataset.test_txt, description="test txt file"), cfg.heatmap_eval_use_gt),
    ]

    for split_name, txt_path, use_gt_label in split_specs:
        (heatmap_root / split_name).mkdir(parents=True, exist_ok=True)
        split_dataset = ImageDatasetFromTxt([txt_path], AlbumentationsTransformVal(cfg.image_size, cfg.image_size))
        for idx in tqdm(range(len(split_dataset)), desc=f"Generating stage2 heatmaps [{split_name}]"):
            image, label = split_dataset[idx]
            img_path = split_dataset.image_paths[idx]
            save_path = heatmap_cache_path(heatmap_root, img_path, split=split_name)
            if save_path.exists():
                try:
                    cached_heatmap = Image.open(save_path)
                    if cached_heatmap.size == (cfg.image_size, cfg.image_size):
                        continue
                except Exception:
                    pass

            input_tensor = image.unsqueeze(0).to(device)
            input_tensor.requires_grad_(True)
            target_class = int(label) if use_gt_label else None
            heatmap, _ = cam_tool(input_tensor, class_idx=target_class)
            save_grayscale_heatmap(heatmap, save_path, output_size=(cfg.image_size, cfg.image_size))

    cam_tool.remove_hooks()
    return heatmap_root


def train_stage2_experiment(cfg: Stage2ExperimentConfig):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = _infer_device()
    _prepare_heatmaps(cfg, device)
    train_loader, val_loader, test_loader, _ = _make_loaders(cfg)

    base_model = Stage2BackboneDinoHeatmapFusion(
        num_classes=cfg.num_classes,
        dino_path=cfg.dino_path,
        backbone_name=cfg.backbone,
    )
    model_profile = profile_model(base_model.to(device), device, (1, 3, cfg.image_size, cfg.image_size))
    model = _prepare_model_for_training(base_model, cfg, device)
    print("\nModel Profile:")
    print(format_model_profile(model_profile))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    save_root = build_timestamped_output_dir(cfg.save_root, cfg.name, f"{pd.Timestamp.now():%Y%m%d}")
    save_root.mkdir(parents=True, exist_ok=True)
    (save_root / "config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    metric_columns = [
        "epoch",
        "train_loss",
        "train_acc",
        "model_total_params",
        "model_trainable_params",
        "model_frozen_params",
        "model_flops",
        "model_inference_ms",
        "model_throughput_images_per_sec",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "val_precision_weighted",
        "val_recall_weighted",
        "val_f1_weighted",
        "val_balanced_accuracy",
        "val_mcc",
        "val_cohen_kappa",
        "val_specificity_macro",
        "val_npv_macro",
        "val_fpr_macro",
        "val_fnr_macro",
        "val_fdr_macro",
        "val_for_macro",
        "val_log_loss",
        "val_roc_auc_ovr_macro",
        "val_roc_auc_ovr_weighted",
        "val_pr_auc_macro",
        "val_pr_auc_weighted",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "test_balanced_accuracy",
        "test_mcc",
        "test_cohen_kappa",
        "test_specificity_macro",
        "test_npv_macro",
        "test_fpr_macro",
        "test_fnr_macro",
        "test_fdr_macro",
        "test_for_macro",
        "test_log_loss",
        "test_roc_auc_ovr_macro",
        "test_roc_auc_ovr_weighted",
        "test_pr_auc_macro",
        "test_pr_auc_weighted",
    ]
    csv_path = save_root / "training_metrics.csv"
    _write_csv_header(csv_path, metric_columns)

    best_val_acc = float("-inf")
    best_epoch = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, heatmaps, labels in tqdm(train_loader, desc=f"Stage2 Epoch {epoch + 1}/{cfg.epochs}"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)
            outputs = _model_forward(model, images, heatmaps, cfg)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)
        val_metrics = _evaluate(model, val_loader, criterion, device, cfg)
        test_metrics = _evaluate(model, test_loader, criterion, device, cfg)
        is_best = val_metrics["acc_percent"] > best_val_acc

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "model_total_params": model_profile.total_params,
            "model_trainable_params": model_profile.trainable_params,
            "model_frozen_params": model_profile.frozen_params,
            "model_flops": model_profile.flops or "",
            "model_inference_ms": model_profile.avg_inference_ms,
            "model_throughput_images_per_sec": model_profile.throughput_images_per_sec,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc_percent"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc_percent"],
        }
        row.update(flatten_metric_dict("val", val_metrics, exclude={"confusion_matrix", "samples"}))
        row.update(flatten_metric_dict("test", test_metrics, exclude={"confusion_matrix", "samples"}))

        history.append(row)
        _append_csv_row(csv_path, [row.get(col, "") for col in metric_columns])

        print("\n" + "=" * 88)
        print(f"Stage2 Epoch {epoch + 1} {'(BEST)' if is_best else ''}")
        print("-" * 88)
        print(f"Train: loss={_format_metric_value(train_loss)} | acc={_format_metric_value(train_acc)}")
        print(f"Val  : loss={_format_metric_value(val_metrics['loss'])} | acc={_format_metric_value(val_metrics['acc_percent'])}")
        print(f"Test : loss={_format_metric_value(test_metrics['loss'])} | acc={_format_metric_value(test_metrics['acc_percent'])}")
        print("-" * 88)
        _print_confusion_table(val_metrics.get("confusion_matrix", []), val_metrics.get("labels"), "Val Confusion Matrix:")
        _print_class_metrics(val_metrics)
        print("-" * 88)
        _print_confusion_table(test_metrics.get("confusion_matrix", []), test_metrics.get("labels"), "Test Confusion Matrix:")
        _print_class_metrics(test_metrics)
        print("=" * 88)

        if is_best:
            best_val_acc = val_metrics["acc_percent"]
            best_epoch = epoch + 1
            torch.save(_unwrap_model(model).state_dict(), save_root / "best_val_model.pth")

    if best_epoch > 0:
        best_row = next(item for item in history if item["epoch"] == best_epoch)
        print(f"Best stage2 validation model saved at epoch {best_epoch} with val_acc={best_row['val_acc']:.4f}%")

    history_df = pd.DataFrame(history)
    _save_summary_artifacts(save_root, history_df, cfg, model_profile)
    (save_root / "model_profile.json").write_text(json.dumps(model_profile_to_dict(model_profile), ensure_ascii=False, indent=2), encoding="utf-8")
    return save_root
