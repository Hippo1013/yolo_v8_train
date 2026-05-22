# YOLOv8 仪表盘状态识别训练工程

```text
yolo_v8_train/
├── backgrounds/                    # 640x640 背景图，供数据生成脚本随机选取
│   ├── bg1.jpg
│   ├── ...
│   └── bg10.jpg
├── raw/                            # 原始仪表盘前景图，每类 5 张
│   ├── 偏低/
│   ├── 正常/
│   └── 偏高/
├── transparent/                    # 使用 rembg 抠图后的透明前景图，每类 5 张
│   ├── 偏低/
│   ├── 正常/
│   └── 偏高/
├── dataset/                        # 训练集生成位置
├── bg_video.mp4                    # 用于抽取背景源的视频
├── data.yaml                       # YOLO 数据集配置
├── generate_data.py                # 合成训练数据并生成 YOLO 标签
├── make_transparent.py             # 将 raw/ 中的仪表盘图片抠成透明前景
├── README.md                       # 项目说明
├── realsense_live_test.py          # Windows 本机 RealSense 实时测试脚本
├── requirement_yolov8_train.txt    # yolo_v8_train 环境依赖冻结结果
├── resize_backgrounds.py           # 将背景源裁剪缩放为 640x640
├── train.py                        # YOLOv8 训练入口
└── video_to_bg.py                  # 从 bg_video.mp4 抽帧生成背景源
```

本项目用于在 Windows 设备上训练 YOLOv8 仪表盘状态识别模型。训练完成后，将模型权重复制到机器狗主机，在机器狗主机上完成实时推理。

识别类别为三类：

```text
0: low     # 偏低
1: normal  # 正常
2: high    # 偏高
```

## 1. 准备训练环境

```bash
conda create -n yolo_v8_train python=3.10
conda activate yolo_v8_train
pip install -r requirement_yolov8_train.txt
```

`requirement_yolov8_train.txt` 是当前 `yolo_v8_train` conda 环境的冻结结果。该环境用于运行数据处理、数据生成和 YOLOv8 训练脚本。

本步骤主要使用 `conda` 管理 Python 环境，并通过 `pip` 安装 `ultralytics`、`torch`、`opencv-python`、`Pillow`、`albumentations`、`rembg` 等库；构建逻辑是先固定环境，再保证数据处理和训练脚本在同一依赖版本下运行。

## 2. 准备原数据

原数据放在 `raw/` 目录下：

```text
raw/
├── 偏低/
├── 正常/
└── 偏高/
```

主办方原始提供的数据为每类 2 张仪表盘图片。为了增加前景素材数量，本项目使用 Google NanoBanana 生图对每类仪表盘进行了 AI P 图，将每类扩展到 5 张图片。

每个状态目录下放置 5 张仪表盘图片，文件名分别为：

```text
偏低1.png, 偏低2.png, 偏低3.png, 偏低4.png, 偏低5.png
正常1.png, 正常2.png, 正常3.png, 正常4.png, 正常5.png
偏高1.png, 偏高2.png, 偏高3.png, 偏高4.png, 偏高5.png
```

这 15 张图片是后续合成训练数据的前景源图，其中每类前 2 张来自主办方原数据，其余图片来自 Google NanoBanana AI P 图结果。

本步骤不依赖 Python 库，核心逻辑是按类别整理输入数据，并让文件名与后续脚本中的类别名、编号范围保持一致。

## 3. 提取透明前景

```bash
python make_transparent.py
```

`make_transparent.py` 会读取 `raw/偏低`、`raw/正常`、`raw/偏高` 中的原始仪表盘图片，使用 `rembg` 将背景透明化，只保留仪表盘主体画面。

输出结果保存在：

```text
transparent/
├── 偏低/
├── 正常/
└── 偏高/
```

`transparent/` 中的图片会作为合成训练数据时的前景素材。

本步骤主要使用 `rembg` 和 `Pillow`：`Pillow` 负责读取和保存图片，`rembg` 负责分割主体并生成带 alpha 通道的透明前景。

## 4. 从视频提取背景源

```bash
python video_to_bg.py
```

`video_to_bg.py` 会读取根目录下的 `bg_video.mp4`，每隔 `SAVE_EVERY_N_FRAMES` 帧抽取一张图片，作为背景源。

抽帧结果会保存到：

```text
backgrounds_raw/
```

`backgrounds_raw/` 是中间目录。如果直接使用项目中已有的 `backgrounds/`，可以跳过抽帧和背景缩放步骤。

本步骤主要使用 `opencv-python`：通过 `cv2.VideoCapture` 读取视频流，再按固定帧间隔用 `cv2.imwrite` 保存背景源图。

## 5. 统一背景尺寸

```bash
python resize_backgrounds.py
```

`resize_backgrounds.py` 会读取 `backgrounds_raw/` 中的背景源图，先居中裁剪成正方形，再缩放为 YOLO 训练使用的 `640x640`。

输出结果保存到：

```text
backgrounds/
```

`backgrounds/` 是最终背景库，`generate_data.py` 会直接从这里随机选择背景。项目中已包含一批处理好的背景图；如果不重新抽帧，可以直接使用现有 `backgrounds/`。

本步骤主要使用 `Pillow`：先对背景源图做居中正方形裁剪，再统一缩放到 `640x640`，保证后续合成数据尺寸稳定。

## 6. 生成 YOLO 训练集

```bash
python generate_data.py
```

`generate_data.py` 会读取：

```text
transparent/   # 透明仪表盘前景
backgrounds/   # 640x640 背景图
```

然后将前景和背景随机组合，生成 YOLO 格式训练集：

```text
dataset/
└── train/
    ├── images/
    └── labels/
```

该脚本默认每张透明前景生成 `350` 张合成图片。当前每类 5 张前景、共 3 类，因此全量生成约 `5 * 3 * 350 = 5250` 张训练图片。

数据增强包括：

- 随机旋转、缩放和拉伸；
- 小仪表盘占比增强，用于提高远距离泛化能力；
- 目标越小，前景分辨率越低；
- 高斯模糊、运动模糊、噪声、亮度/对比度变化；
- 小目标颜色交界软化、对比度和饱和度轻微降低。

`dataset/` 是训练集输出目录，实际训练图片和标签会在运行 `generate_data.py` 后生成。

本步骤主要使用 `Pillow`、`numpy`、`opencv-python` 和 `albumentations`：先用 `Pillow` 合成透明前景和背景，再用 `albumentations` 做图像增强，最后用 OpenCV 保存图片并同步写出 YOLO 标签。

## 7. 训练 YOLOv8 模型

训练 YOLOv8n：

```bash
python train.py --model yolov8n.pt --name v1_yolov8n
```

训练 YOLOv8s：

```bash
python train.py --model yolov8s.pt --name v1_yolov8s
```

`train.py` 默认读取 `data.yaml`，其中数据集根目录为：

```yaml
path: dataset
train: train/images
val: train/images
```

当前训练参数默认值为：

```text
epochs=150
patience=30
imgsz=640
batch=16
workers=4
device=0
degrees=0.0
mosaic=0.5
close_mosaic=10
```

训练结果默认保存到：

```text
runs/train/<name>/
```

最终权重通常使用：

```text
runs/train/<name>/weights/best.pt
```

本步骤主要使用 `ultralytics` 和 `torch`：`ultralytics.YOLO` 加载预训练权重并读取 `data.yaml`，底层由 PyTorch 在 GPU 或 CPU 上执行训练。

## 8. 将权重复制到机器狗主机

训练完成后，将 `best.pt` 复制到机器狗主机的推理工程中。例如：

```bash
scp runs/train/v1_yolov8s/weights/best.pt dog@<机器狗主机IP>:/home/dog/dog_ws/dashboard_recognition/models/
```

本项目负责 Windows 侧训练流水线。机器狗主机上的实时推理脚本和运行环境应在机器狗主机对应工程中维护。

本步骤通常使用 `scp` 或其他文件传输工具，不涉及 Python 代码；核心逻辑是把训练阶段产出的 `best.pt` 作为推理阶段的模型输入。

## 9. Windows 本机 RealSense 测试

如果 Windows 设备连接了 Intel RealSense D435i，也可以运行：

```bash
python realsense_live_test.py
```

`realsense_live_test.py` 会读取脚本中的 `MODEL_PATH`，打开 RealSense 彩色流和深度流，使用训练好的 YOLO 权重进行实时识别，并在窗口中显示标注框、中文类别和深度距离。

如果训练任务名称或输出目录不同，需要先修改脚本顶部的：

```python
MODEL_PATH = Path("runs/detect/runs2/train/v1.1_yolov8s/weights/best.pt")
```

本步骤主要使用 `pyrealsense2`、`opencv-python`、`ultralytics`、`torch`、`numpy` 和 `Pillow`：RealSense 负责采集彩色图和深度图，YOLO 负责检测，OpenCV 与 Pillow 负责实时显示和中文标注。

## 10. Git 提交范围

仓库只提交训练流水线必要内容：

```text
backgrounds/
raw/
transparent/
dataset/.gitkeep
bg_video.mp4
data.yaml
generate_data.py
make_transparent.py
README.md
realsense_live_test.py
resize_backgrounds.py
video_to_bg.py
train.py
requirement_yolov8_train.txt
```

以下内容属于可再生成数据、训练结果或本地临时文件，不提交到 Git：

```text
backgrounds_raw/
dataset/train/
datasets/
datasets_new/
runs/
*.pt
*.onnx
*.engine
```
