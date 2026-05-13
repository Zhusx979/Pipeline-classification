# 电科管道分类

该项目已整理为适合在 Linux 服务器上“解压后直接使用”的结构。默认训练入口为 `run.py`，第二阶段热力图融合训练入口为 `run_stage2.py`。

## 建议保留的目录

- `common/`
- `configs/`
- `data/`
- `dataset/`
- `metrics/`
- `models/`
- `stage2/`
- `train/`
- `visualization/`
- `run.py`
- `run_stage2.py`
- `draw.py`
- `grad.py`
- `requirements.txt`

## 建议不要打包上传的内容

- `__pycache__/`
- `result_*`、`stage2_result_*`
- `stage2_heatmaps/`
- `paper/*.aux`、`paper/*.log`、`paper/*.out`、`paper/*.synctex.gz`

这些文件不影响训练启动，只会增加压缩包体积。

## Linux 环境准备

推荐使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果服务器需要指定 PyTorch CUDA 版本，请先按服务器 CUDA 环境安装匹配的 `torch` 和 `torchvision`，再执行：

```bash
pip install -r requirements.txt --no-deps
```

## 数据组织要求

默认数据目录为项目根目录下的 `dataset/`：

```text
dataset/
  train/
  valid/
  test/
  train.txt
  valid.txt
  test.txt
```

文本清单 `train.txt`、`valid.txt`、`test.txt` 可以写相对路径。项目也兼容历史写法 `dataset_0309/...`，会自动映射到当前 `dataset/...`。

## DINO 模型说明

代码会优先使用以下两种方式之一加载 DINO：

1. 项目根目录下的本地 `dino_model/`
2. 环境变量或命令行指定的 `DINO_PATH`
3. Hugging Face 模型名 `facebook/dinov2-base`

如果服务器不能联网，建议提前把 DINO 模型目录放到项目根目录并命名为 `dino_model/`，或者在运行时显式指定：

```bash
export DINO_PATH=/path/to/dino_model
```

## Stage1 训练

```bash
python run.py \
  --preset resnet50_dino_pointwise \
  --epochs 50 \
  --batch-size 32 \
  --image-size 224 \
  --num-workers 4
```

如需在检测到多张 CUDA 显卡时启用多 GPU 训练，可在命令末尾追加 `--multi-gpu`。

常用参数：

- `--train-txt`
- `--val-txt`
- `--test-txt`
- `--dino-path`
- `--save-features`
- `--multi-gpu`

`--multi-gpu` 使用 PyTorch `DataParallel`。如果当前环境只有 1 张或 0 张 CUDA 显卡，会自动回退到单卡或 CPU。

## 消融实验推荐命令

以下命令都使用“最小参数版本”模型，方便快速完成论文中的 backbone / transformer 消融。

### 1. 证明 ResNet 优于其他 CNN

CNN 单分支基线：

```bash
python run.py --preset resnet18 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet34 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet50 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset densenet121 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset inception_v3 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset mobilenetv3_small --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset efficientnet_b0 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
```

固定 Transformer 为 DINOv2，比较不同 CNN：

```bash
python run.py --preset resnet18_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet34_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet50_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet101_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset densenet121_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset inception_v3_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset mobilenetv3_small_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset efficientnet_b0_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
```

### 2. 证明 DINOv2 优于其他 Transformer

Transformer 单分支基线：

```bash
python run.py --preset dinov2 --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset deit_tiny_cls --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset vit_tiny_cls --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset coat_tiny --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset coat_small --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
```

固定 CNN 为 ResNet18，比较不同 Transformer：

```bash
python run.py --preset resnet18_dino_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet18_deit_tiny_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet18_vit_tiny_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet18_coat_tiny_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
python run.py --preset resnet18_coat_small_pointwise --epochs 50 --batch-size 32 --image-size 224 --num-workers 4
```

如果服务器不能联网，请额外追加：

```bash
--dino-path /path/to/dino_model
```

## Stage2 训练

Stage2 依赖一个已经训练好的 Stage1 checkpoint。

```bash
python run_stage2.py \
  --preset resnet50_heatmap_dino_pointwise \
  --stage1-preset resnet50_dino_pointwise \
  --stage1-checkpoint result_YYYYMMDD/your_stage1_run/best_val_model.pth \
  --epochs 50 \
  --batch-size 32 \
  --image-size 224 \
  --num-workers 4 \
  --multi-gpu
```

首次运行时会在 `stage2_heatmaps/` 下自动生成训练集热力图缓存。

### Stage2 全部 CNN 消融命令

下面给出与 Stage1 CNN 消融一一对应的 Stage2 多 GPU 命令示例。请先完成对应的 Stage1 训练，再把 `--stage1-checkpoint` 替换成实际生成的 `best_val_model.pth` 路径。

```bash
python run_stage2.py --preset resnet18_heatmap_dino_pointwise --stage1-preset resnet18_dino_pointwise --stage1-checkpoint result_YYYYMMDD/ResNet18_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset resnet34_heatmap_dino_pointwise --stage1-preset resnet34_dino_pointwise --stage1-checkpoint result_YYYYMMDD/ResNet34_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset resnet50_heatmap_dino_pointwise --stage1-preset resnet50_dino_pointwise --stage1-checkpoint result_YYYYMMDD/ResNet50_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset resnet101_heatmap_dino_pointwise --stage1-preset resnet101_dino_pointwise --stage1-checkpoint result_YYYYMMDD/ResNet101_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset densenet121_heatmap_dino_pointwise --stage1-preset densenet121_dino_pointwise --stage1-checkpoint result_YYYYMMDD/DenseNet121_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset inception_v3_heatmap_dino_pointwise --stage1-preset inception_v3_dino_pointwise --stage1-checkpoint result_YYYYMMDD/Inception_v3_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset mobilenetv3_small_heatmap_dino_pointwise --stage1-preset mobilenetv3_small_dino_pointwise --stage1-checkpoint result_YYYYMMDD/MobileNetV3_Small_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
python run_stage2.py --preset efficientnet_b0_heatmap_dino_pointwise --stage1-preset efficientnet_b0_dino_pointwise --stage1-checkpoint result_YYYYMMDD/EfficientNet_B0_+_DINOv2_Pointwise_Multiplication/best_val_model.pth --epochs 50 --batch-size 64 --image-size 224 --num-workers 8 --multi-gpu
```

## 训练结果输出

Stage1 默认输出到：

```text
result_YYYYMMDD/<experiment_name>/
```

Stage2 默认输出到：

```text
stage2_result_YYYYMMDD/<experiment_name>/
```

每次输出目录中包含：

- `config.json`
- `training_metrics.csv`
- `training_summary.txt`
- `training_curves.png`
- `training_curves.pdf`
- `best_val_model.pth`

## 其他工具

对比多个训练曲线：

```bash
python draw.py \
  --run stage1_a result_YYYYMMDD/exp_a/training_metrics.csv \
  --run stage1_b result_YYYYMMDD/exp_b/training_metrics.csv \
  --highlight stage1_b
```

生成单张图的 Grad-CAM：

```bash
python grad.py \
  --image dataset/test/0/example.png \
  --checkpoint result_YYYYMMDD/your_stage1_run/best_val_model.pth \
  --preset resnet50_dino_pointwise \
  --output gradcam_overlay.png
```

## 迁移前最后建议

上传到服务器前，建议先在本地压缩时排除：

```text
__pycache__/
result_*/
stage2_result_*/
stage2_heatmaps/
```

这样可以减少上传时间，也避免服务器端误用旧实验结果。
