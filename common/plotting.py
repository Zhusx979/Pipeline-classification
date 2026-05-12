from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PALETTE = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#457B9D",
    "#8D99AE",
    "#9D4EDD",
]
OUR_COLOR = "#E76F51"
GRID_COLOR = "#D9DEE7"
TRAIN_COLOR = "#457B9D"
VAL_COLOR = "#F4A261"
TEST_COLOR = "#E76F51"
CURVE_COLOR_MAP = {
    "Train": TRAIN_COLOR,
    "Val": VAL_COLOR,
    "Test": TEST_COLOR,
}


def apply_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "axes.edgecolor": "#B9C0CC",
            "axes.labelcolor": "#1F2937",
            "axes.titlecolor": "#111827",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _finalize_axis(ax, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None) -> None:
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.margins(x=0.02, y=0.08)
    ax.set_axisbelow(True)


def _mark_best(ax, x: Sequence[float], y: Sequence[float], color: str) -> None:
    if len(x) == 0:
        return
    idx = int(np.nanargmax(np.asarray(y, dtype=float)))
    ax.scatter([x[idx]], [y[idx]], s=42, color=color, edgecolor="white", linewidth=0.8, zorder=5)


def save_training_curve_grid(history_df, save_path: str | Path, dpi: int = 300) -> None:
    apply_publication_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    epochs = history_df["epoch"].to_numpy()

    panels = [
        (axes[0, 0], [("train_loss", "Train"), ("val_loss", "Val"), ("test_loss", "Test")], "Loss", "Epoch"),
        (axes[0, 1], [("train_acc", "Train"), ("val_acc", "Val"), ("test_acc", "Test")], "Accuracy", "Epoch"),
        (axes[1, 0], [("val_precision_macro", "Precision"), ("val_recall_macro", "Recall"), ("val_f1_macro", "F1"), ("val_balanced_accuracy", "Balanced Acc")], "Validation Metrics", "Epoch"),
        (axes[1, 1], [("test_precision_macro", "Precision"), ("test_recall_macro", "Recall"), ("test_f1_macro", "F1"), ("test_balanced_accuracy", "Balanced Acc")], "Test Metrics", "Epoch"),
    ]

    for ax, series_list, title, xlabel in panels:
        for i, (col, label) in enumerate(series_list):
            if col not in history_df.columns:
                continue
            color = CURVE_COLOR_MAP.get(label, PALETTE[i % len(PALETTE)])
            values = history_df[col].to_numpy(dtype=float)
            ax.plot(epochs, values, label=label, color=color, marker="o")
            _mark_best(ax, epochs, values, color)
        _finalize_axis(ax, xlabel=xlabel, ylabel=title, title=title)
        ax.legend(loc="best")

    fig.savefig(save_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def save_metric_comparison(
    dfs: dict[str, object],
    metric: str,
    save_path: str | Path,
    *,
    title: str | None = None,
    xlabel: str = "Epoch",
    ylabel: str | None = None,
    highlight: str | None = None,
) -> None:
    apply_publication_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for i, (label, df) in enumerate(dfs.items()):
        if metric not in df.columns or "epoch" not in df.columns:
            continue
        color = OUR_COLOR if label == highlight else PALETTE[i % len(PALETTE)]
        line_style = "-" if label == highlight else "--"
        ax.plot(
            df["epoch"],
            df[metric],
            label=label,
            color=color,
            linestyle=line_style,
            marker="o",
            markersize=4.5,
        )
    _finalize_axis(ax, xlabel=xlabel, ylabel=ylabel or metric, title=title or metric)
    ax.legend(loc="best")
    fig.savefig(save_path.with_suffix(".png"), dpi=300)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def blend_heatmap_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    if image_rgb.dtype != np.float32:
        image = image_rgb.astype(np.float32) / 255.0
    else:
        image = np.clip(image_rgb, 0.0, 1.0)
    heat_rgb = plt.get_cmap("magma")(heatmap)[..., :3]
    composite = np.clip((1.0 - alpha) * image + alpha * heat_rgb, 0.0, 1.0)
    return (composite * 255).astype(np.uint8)
