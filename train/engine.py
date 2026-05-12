from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import ExperimentConfig
from data.dataset import AlbumentationsTransformTrain, AlbumentationsTransformVal, ImageDatasetFromTxt
from metrics.classification_metrics import compute_classification_metrics, flatten_metric_dict
from common.model_profile import format_model_profile, model_profile_to_dict, profile_model
from common.path_utils import build_timestamped_output_dir, ensure_file
from common.plotting import save_training_curve_grid
from models.model_factory import build_model


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _make_loaders(cfg: ExperimentConfig):
    train_txt = ensure_file(cfg.dataset.train_txt, description="train txt file")
    val_txt = ensure_file(cfg.dataset.val_txt, description="validation txt file")
    test_txt = ensure_file(cfg.dataset.test_txt, description="test txt file")

    train_dataset = ImageDatasetFromTxt([train_txt], AlbumentationsTransformTrain(cfg.image_size, cfg.image_size))
    val_dataset = ImageDatasetFromTxt([val_txt], AlbumentationsTransformVal(cfg.image_size, cfg.image_size))
    test_dataset = ImageDatasetFromTxt([test_txt], AlbumentationsTransformVal(cfg.image_size, cfg.image_size))

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
    return train_loader, val_loader, test_loader


def _build_optimizer(model: nn.Module, cfg: ExperimentConfig):
    model = _unwrap_model(model)
    params = []
    if hasattr(model, "backbone"):
        params.append({"params": model.backbone.parameters(), "lr": cfg.backbone_lr})
    elif hasattr(model, "resnet"):
        params.append({"params": model.resnet.parameters(), "lr": cfg.backbone_lr})
    elif hasattr(model, "densenet"):
        params.append({"params": model.densenet.parameters(), "lr": cfg.backbone_lr})
    elif hasattr(model, "inception"):
        params.append({"params": model.inception.parameters(), "lr": cfg.backbone_lr})
    elif hasattr(model, "cait"):
        params.append({"params": model.cait.parameters(), "lr": cfg.backbone_lr})

    if hasattr(model, "transformer"):
        params.append({"params": model.transformer.parameters(), "lr": cfg.dino_lr})
    elif hasattr(model, "dino"):
        params.append({"params": model.dino.parameters(), "lr": cfg.dino_lr})

    projection_modules = [
        "backbone_projection",
        "transformer_projection",
        "dino_projection",
        "resnet_projection",
        "densenet_projection",
        "inception_projection",
        "cait_projection",
        "dino_project",
    ]
    for module_name in projection_modules:
        if hasattr(model, module_name):
            params.append({"params": getattr(model, module_name).parameters(), "lr": cfg.projection_lr})

    if hasattr(model, "classifier"):
        params.append({"params": model.classifier.parameters(), "lr": cfg.classifier_lr})

    if not params:
        params = [{"params": model.parameters(), "lr": cfg.backbone_lr}]

    return optim.AdamW(params, weight_decay=cfg.weight_decay)


def _infer_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model_for_training(model: nn.Module, cfg: ExperimentConfig, device: torch.device) -> nn.Module:
    model = model.to(device)
    if cfg.use_multi_gpu and device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Multi-GPU enabled: using DataParallel on {gpu_count} GPUs.")
            model = nn.DataParallel(model)
        else:
            print("Multi-GPU requested, but fewer than 2 CUDA devices were found. Falling back to single GPU.")
    return model


def _collect_probs(logits: torch.Tensor):
    return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _extract_transformer_features(module: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = module(images)
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported transformer output type: {type(outputs)!r}")


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
    widths = [max(len(str(label)), 6) for label in labels]
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

    extra_keys = ["per_class_tp", "per_class_fp", "per_class_fn", "per_class_tn", "per_class_fpr", "per_class_fnr"]
    if any(metrics.get(k) is not None for k in extra_keys):
        print("Confusion-derived metrics:")
        print(
            " | ".join(
                [
                    "class",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "fpr",
                    "fnr",
                ]
            )
        )
        for idx, label in enumerate(labels):
            print(
                " | ".join(
                    [
                        str(label),
                        _format_metric_value((metrics.get("per_class_tp") or [None])[idx] if idx < len(metrics.get("per_class_tp") or []) else None),
                        _format_metric_value((metrics.get("per_class_fp") or [None])[idx] if idx < len(metrics.get("per_class_fp") or []) else None),
                        _format_metric_value((metrics.get("per_class_fn") or [None])[idx] if idx < len(metrics.get("per_class_fn") or []) else None),
                        _format_metric_value((metrics.get("per_class_tn") or [None])[idx] if idx < len(metrics.get("per_class_tn") or []) else None),
                        _format_metric_value((metrics.get("per_class_fpr") or [None])[idx] if idx < len(metrics.get("per_class_fpr") or []) else None),
                        _format_metric_value((metrics.get("per_class_fnr") or [None])[idx] if idx < len(metrics.get("per_class_fnr") or []) else None),
                    ]
                )
            )


def _print_epoch_report(epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, object], test_metrics: Dict[str, object], is_best: bool):
    def line(title: str, metrics: Dict[str, object], keys: List[str]):
        body = " | ".join(f"{k}={_format_metric_value(metrics.get(k))}" for k in keys)
        print(f"{title}: {body}")

    print("\n" + "=" * 88)
    print(f"Epoch {epoch} {'(BEST)' if is_best else ''}")
    print("-" * 88)
    line("Train", train_metrics, ["loss", "acc_percent"])
    line("Val  ", val_metrics, ["loss", "acc_percent", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa"])
    line("Test ", test_metrics, ["loss", "acc_percent", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa"])
    print("-" * 88)
    _print_confusion_table(val_metrics.get("confusion_matrix", []), val_metrics.get("labels"), "Val Confusion Matrix:")
    _print_class_metrics(val_metrics)
    print("-" * 88)
    _print_confusion_table(test_metrics.get("confusion_matrix", []), test_metrics.get("labels"), "Test Confusion Matrix:")
    _print_class_metrics(test_metrics)
    print("=" * 88)


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
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


def _save_summary_artifacts(save_path: Path, history_df: pd.DataFrame, cfg: ExperimentConfig, model_profile):
    best_val_idx = history_df["val_acc"].astype(float).idxmax()
    best_test_idx = history_df["test_acc"].astype(float).idxmax()
    best_val_row = history_df.iloc[best_val_idx]
    best_test_row = history_df.iloc[best_test_idx]

    summary_report = f"""
Model: {cfg.name}
Backbone: {cfg.backbone}
DINO Path: {cfg.dino_path}
Frozen DINO: {cfg.dino_frozen}
Transformer: {cfg.transformer_name}
Transformer Source: {cfg.transformer_source}
Frozen Transformer: {cfg.transformer_frozen}
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


def _save_feature_artifacts(model: nn.Module, loader, save_path: Path):
    model = _unwrap_model(model)
    backbone_prefix = None
    backbone_module = None
    backbone_project = None

    if hasattr(model, "resnet"):
        backbone_prefix = "resnet"
        backbone_module = model.resnet
        backbone_project = getattr(model, "resnet_projection", None)
    elif hasattr(model, "densenet"):
        backbone_prefix = "densenet"
        backbone_module = model.densenet
        backbone_project = getattr(model, "densenet_projection", None)
    elif hasattr(model, "inception"):
        backbone_prefix = "inception"
        backbone_module = model.inception
        backbone_project = getattr(model, "inception_projection", None)
    elif hasattr(model, "cait"):
        backbone_prefix = "cait"
        backbone_module = model.cait
        backbone_project = getattr(model, "cait_projection", None)
    elif hasattr(model, "backbone"):
        backbone_prefix = "backbone"
        backbone_module = model.backbone
        backbone_project = getattr(model, "backbone_projection", None)

    transformer_module = getattr(model, "transformer", None)
    if transformer_module is None:
        transformer_module = getattr(model, "dino", None)

    if backbone_prefix is None or backbone_module is None or transformer_module is None:
        return

    transformer_project = getattr(model, "transformer_projection", None)
    if transformer_project is None:
        transformer_project = getattr(model, "dino_projection", None)
    if transformer_project is None:
        transformer_project = getattr(model, "dino_project", None)

    transformer_prefix = getattr(getattr(transformer_module, "encoder", transformer_module), "__class__", type(transformer_module)).__name__.lower()
    if cfg_transformer_name := getattr(model, "transformer_name", None):
        transformer_prefix = str(cfg_transformer_name).replace("/", "_").replace("-", "_")

    backbone_features = []
    transformer_features = []
    backbone_projected = []
    transformer_projected = []
    fused_features = []
    feature_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting features"):
            images = images.to(next(model.parameters()).device)
            labels = labels.to(images.device)

            backbone_feat = backbone_module(images)
            transformer_feat = _extract_transformer_features(transformer_module, images)

            backbone_features.append(backbone_feat.cpu().numpy())
            transformer_features.append(transformer_feat.cpu().numpy())
            feature_labels.append(labels.cpu().numpy())

            if backbone_project is not None and transformer_project is not None:
                backbone_proj = backbone_project(backbone_feat)
                transformer_proj = transformer_project(transformer_feat)
                backbone_projected.append(backbone_proj.cpu().numpy())
                transformer_projected.append(transformer_proj.cpu().numpy())
                if hasattr(model, "classifier") and backbone_proj.shape == transformer_proj.shape:
                    if hasattr(model, "norm"):
                        fused = model.norm(backbone_proj * transformer_proj)
                    else:
                        fused = backbone_proj * transformer_proj
                else:
                    fused = backbone_proj * transformer_proj
                fused_features.append(fused.cpu().numpy())

    np.save(save_path / f"{backbone_prefix}_features.npy", np.concatenate(backbone_features, axis=0))
    np.save(save_path / f"{transformer_prefix}_features.npy", np.concatenate(transformer_features, axis=0))
    if backbone_projected:
        np.save(save_path / f"{backbone_prefix}_projected_features.npy", np.concatenate(backbone_projected, axis=0))
    if transformer_projected:
        np.save(save_path / f"{transformer_prefix}_projected_features.npy", np.concatenate(transformer_projected, axis=0))
    if fused_features:
        np.save(save_path / "fused_features.npy", np.concatenate(fused_features, axis=0))
    np.save(save_path / "feature_labels.npy", np.concatenate(feature_labels, axis=0))


def train_experiment(cfg: ExperimentConfig):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = _infer_device()
    train_loader, val_loader, test_loader = _make_loaders(cfg)
    base_model = build_model(cfg.preset_name, cfg.num_classes, cfg.dino_path)
    model_profile = profile_model(base_model.to(device), device, (1, 3, cfg.image_size, cfg.image_size))
    model = _prepare_model_for_training(base_model, cfg, device)
    print("\nModel Profile:")
    print(format_model_profile(model_profile))
    criterion = nn.CrossEntropyLoss()
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

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        train_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        test_metrics = _evaluate(model, test_loader, criterion, device)
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

        _print_epoch_report(
            epoch=epoch + 1,
            train_metrics={"loss": train_loss, "acc_percent": train_acc},
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            is_best=is_best,
        )

        if is_best:
            best_val_acc = val_metrics["acc_percent"]
            best_epoch = epoch + 1
            torch.save(_unwrap_model(model).state_dict(), save_root / "best_val_model.pth")

    if best_epoch > 0:
        best_row = next(item for item in history if item["epoch"] == best_epoch)
        print(f"Best validation model saved at epoch {best_epoch} with val_acc={best_row['val_acc']:.4f}%")

    history_df = pd.DataFrame(history)
    _save_summary_artifacts(save_root, history_df, cfg, model_profile)
    (save_root / "model_profile.json").write_text(json.dumps(model_profile_to_dict(model_profile), ensure_ascii=False, indent=2), encoding="utf-8")
    if cfg.extra.get("save_features"):
        _save_feature_artifacts(model, val_loader, save_root)
    return save_root
