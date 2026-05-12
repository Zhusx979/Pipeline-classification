from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _specificity_from_cm(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificities.append(float(tn / denom) if denom else 0.0)
    return np.asarray(specificities, dtype=float)


def _npv_from_cm(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    npvs = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fn
        npvs.append(float(tn / denom) if denom else 0.0)
    return np.asarray(npvs, dtype=float)


def _confusion_rate_metrics(cm: np.ndarray):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn

    def safe_div(numer, denom):
        return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)

    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    fdr = safe_div(fp, tp + fp)
    for_rate = safe_div(fn, fn + tn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "fdr": fdr,
        "for": for_rate,
    }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = per_class
    specificity = _specificity_from_cm(cm)
    npv = _npv_from_cm(cm)
    cm_rates = _confusion_rate_metrics(cm)

    metrics = {
        "labels": list(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "error_rate": float(1.0 - accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "specificity_macro": float(np.mean(specificity)) if len(specificity) else 0.0,
        "specificity_weighted": float(np.average(specificity, weights=per_class_support)) if per_class_support.sum() else 0.0,
        "npv_macro": float(np.mean(npv)) if len(npv) else 0.0,
        "npv_weighted": float(np.average(npv, weights=per_class_support)) if per_class_support.sum() else 0.0,
        "fpr_macro": float(np.mean(cm_rates["fpr"])) if len(cm_rates["fpr"]) else 0.0,
        "fnr_macro": float(np.mean(cm_rates["fnr"])) if len(cm_rates["fnr"]) else 0.0,
        "fdr_macro": float(np.mean(cm_rates["fdr"])) if len(cm_rates["fdr"]) else 0.0,
        "for_macro": float(np.mean(cm_rates["for"])) if len(cm_rates["for"]) else 0.0,
        "confusion_matrix": cm.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_support": per_class_support.tolist(),
        "per_class_specificity": specificity.tolist(),
        "per_class_npv": npv.tolist(),
        "per_class_tp": cm_rates["tp"].tolist(),
        "per_class_fp": cm_rates["fp"].tolist(),
        "per_class_fn": cm_rates["fn"].tolist(),
        "per_class_tn": cm_rates["tn"].tolist(),
        "per_class_fpr": cm_rates["fpr"].tolist(),
        "per_class_fnr": cm_rates["fnr"].tolist(),
        "per_class_fdr": cm_rates["fdr"].tolist(),
        "per_class_for": cm_rates["for"].tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        if len(labels) > 1:
            y_true_bin = label_binarize(y_true, classes=labels)
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
                )
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr")
                )
                metrics["pr_auc_macro"] = float(
                    average_precision_score(y_true_bin, y_prob, average="macro")
                )
                metrics["pr_auc_weighted"] = float(
                    average_precision_score(y_true_bin, y_prob, average="weighted")
                )
            except Exception:
                metrics["roc_auc_ovr_macro"] = None
                metrics["roc_auc_ovr_weighted"] = None
                metrics["pr_auc_macro"] = None
                metrics["pr_auc_weighted"] = None

    return metrics


def flatten_metric_dict(prefix: str, metrics: Dict[str, object], exclude: Optional[Iterable[str]] = None) -> Dict[str, object]:
    exclude = set(exclude or [])
    flat = {}
    for key, value in metrics.items():
        if key in exclude:
            continue
        if isinstance(value, (list, tuple, dict)):
            continue
        flat[f"{prefix}_{key}"] = value
    return flat
