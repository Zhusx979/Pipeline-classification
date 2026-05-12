from __future__ import annotations

import argparse
import os

def parse_args():
    from stage2.config import MODEL_PRESETS

    parser = argparse.ArgumentParser(description="Stage2 training entry point.")
    parser.add_argument("--preset", default="resnet50_heatmap_dino_pointwise", choices=sorted(MODEL_PRESETS.keys()))
    parser.add_argument("--stage1-preset", default="resnet50_dino_pointwise")
    parser.add_argument("--stage1-checkpoint", default=None, help="Required stage1 checkpoint path.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--multi-gpu", action="store_true", help="Use DataParallel when multiple CUDA devices are available.")
    parser.add_argument("--train-txt", default=None)
    parser.add_argument("--val-txt", default=None)
    parser.add_argument("--test-txt", default=None)
    parser.add_argument("--dino-path", default=None, help="Local model path or Hugging Face model id.")
    return parser.parse_args()


def main():
    args = parse_args()

    from configs.config import DEFAULT_DINO_PATH
    from stage2.config import build_stage2_experiment_config
    from stage2.engine import train_stage2_experiment

    if args.dino_path:
        os.environ["DINO_PATH"] = args.dino_path

    cfg = build_stage2_experiment_config(
        preset_name=args.preset,
        stage1_preset_name=args.stage1_preset,
        stage1_checkpoint=args.stage1_checkpoint,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        test_txt=args.test_txt,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        num_classes=args.num_classes,
        use_multi_gpu=args.multi_gpu,
    )
    print(f"Using DINO path: {cfg.dino_path or DEFAULT_DINO_PATH}")
    train_stage2_experiment(cfg)


if __name__ == "__main__":
    main()
