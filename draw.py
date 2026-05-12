from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.path_utils import resolve_project_path, sanitize_filename
from common.plotting import save_metric_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compare metrics from multiple training CSV files.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV_PATH"),
        help="Add one run as: --run LABEL path/to/training_metrics.csv",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["train_loss", "train_acc", "val_acc", "val_loss", "test_acc", "test_loss"],
    )
    parser.add_argument("--output-dir", default="image")
    parser.add_argument("--highlight", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.run:
        raise ValueError("At least one --run LABEL CSV_PATH pair is required.")

    dfs = {}
    for label, csv_path in args.run:
        resolved_csv = resolve_project_path(csv_path)
        if resolved_csv is None or not resolved_csv.is_file():
            raise FileNotFoundError(f"CSV file not found for run '{label}': {resolved_csv}")
        dfs[label] = pd.read_csv(resolved_csv)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        save_path = output_dir / sanitize_filename(metric)
        save_metric_comparison(
            dfs,
            metric,
            save_path,
            title=f"{metric} Comparison",
            highlight=args.highlight,
        )
        print(f"Saved: {save_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
