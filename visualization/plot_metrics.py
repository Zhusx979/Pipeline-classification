from pathlib import Path

import pandas as pd

from common.plotting import save_metric_comparison


def plot_training_metrics(csv_path: str, save_dir: str):
    df = pd.read_csv(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "train_loss", "val_loss", "test_loss",
        "train_acc", "val_acc", "test_acc",
        "val_precision_macro", "val_recall_macro", "val_f1_macro", "val_balanced_accuracy",
        "test_precision_macro", "test_recall_macro", "test_f1_macro", "test_balanced_accuracy",
    ]

    for metric in metrics:
        if metric not in df.columns:
            continue
        save_metric_comparison({"run": df}, metric, save_dir / metric, title=metric)
