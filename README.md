# YOLOv8 仪表状态识别训练工程

本项目用于构建 YOLOv8 目标检测训练流水线，面向 `偏低`、`正常`、`偏高` 三类仪表状态识别。数据集主要通过少量前景素材、背景图和随机增强自动合成，并输出 YOLO 格式训练数据。

## 项目结构

```text
yolo_v8_train/
├── row/                    # 原始前景素材
├── transparent/            # 抠图后的透明前景素材
├── backgrounds/            # 背景图
├── test/                   # 测试图片
├── data.yaml               # YOLO 数据集配置
├── make_transparent.py     # 使用 rembg 批量抠图
├── video_to_bg.py          # 从视频抽帧生成背景图
├── generate_data.py        # 合成训练图片并生成 YOLO 标签
├── check_data.py           # 抽查标签框是否正确
├── train.py                # YOLOv8 训练入口
└── predict.py              # 推理测试入口
```

`datasets/`、`runs/`、`check_results/` 和模型权重文件默认不提交到 Git，因为它们属于可再生成的数据、训练结果或较大的二进制产物。

## 环境依赖

建议使用 Python 虚拟环境安装依赖：

```bash
pip install ultralytics opencv-python pillow numpy albumentations rembg
```

如果使用 GPU 训练，还需要安装与你的 CUDA 环境匹配的 PyTorch。

## 数据准备流程

1. 将原始前景素材放入 `row/偏低`、`row/正常`、`row/偏高`。
2. 修改 `make_transparent.py` 中的 `INPUT_DIR`、`OUTPUT_DIR`、`FILE_PREFIX`，按类别运行抠图。
3. 将背景图放入 `backgrounds/`，或运行 `video_to_bg.py` 从 `bg_video.mp4` 抽帧生成背景图。
4. 修改 `generate_data.py` 中的 `INPUT_DIR`、`FILE_PREFIX`、`CLASS_ID`，按类别生成合成训练数据。
5. 运行 `check_data.py` 抽查生成标签框是否正确。

类别 ID 对应关系：

```text
0: low     # 偏低
1: normal  # 正常
2: high    # 偏高
```

## 训练

确认 `data.yaml` 中的 `path` 指向当前机器上的 `datasets` 目录，然后运行：

```bash
python train.py
```

训练结果会输出到 `runs/train/meter_model`。

## 推理

训练完成后，根据 `predict.py` 中的模型路径和测试图片路径进行推理：

```bash
python predict.py
```
