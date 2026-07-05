# YOLOv8 v2 仪表盘与字母联合检测训练工程

本目录用于生成并训练一个同时检测 3 种仪表盘状态和 4 个字母目标的 YOLOv8 检测模型。

类别定义：

```text
0: dashboard_low
1: dashboard_normal
2: dashboard_high
3: letter_A
4: letter_B
5: letter_C
6: letter_D
```

## 数据生成策略

`generate_data.py` 会生成三类样本：

```text
dashboard-only: 一张图只有 1 个仪表盘
letter-only:    一张图只有 1 个字母
pair:           一张图同时有 1 个仪表盘和 1 个字母
```

默认数据量：

```text
train:
- 仪表盘单目标：每类 500 张，共 1500 张
- 字母单目标：每类 800 张，共 3200 张
- 双目标组合：12 个组合，每组合 300 张，共 3600 张
- 合计 8300 张图

val:
- 仪表盘单目标：每类 150 张，共 450 张
- 字母单目标：每类 220 张，共 880 张
- 双目标组合：12 个组合，每组合 80 张，共 960 张
- 合计 2290 张图
```

仪表盘目标的旋转、缩放、拉伸、小目标降质等参数与 `yolo_v8_train/generate_data.py` 保持一致。字母目标不做旋转，不做水平/垂直翻转；只保留尺度、轻微透视、光照、噪声、模糊和压缩等增强。双目标图生成时会检查两个 YOLO 标注框，默认要求两个框之间至少保留 `8px` 间隔，避免相交或贴边。

## 生成数据

请使用现有 `yolo_v8_train` conda 环境：

```bash
conda activate yolo_v8_train
cd yolo_v8_train_v2
python generate_data.py
```

生成结果：

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

## 生成预览图

```bash
python preview_dataset.py --split train --count 10
```

默认输出到：

```text
previews_train/
```

## 校验数据集

```bash
python validate_dataset.py
```

该脚本会统计 `train/val` 的图片数、标注文件数、各类别目标数，并检查双目标样本中的两个标注框是否相交。

## 训练模型

先训练 `yolov8n.pt`：

```bash
python train.py --model yolov8n.pt --name v2_yolov8n
```

再训练 `yolov8s.pt`：

```bash
python train.py --model yolov8s.pt --name v2_yolov8s
```

默认训练参数：

```text
epochs=180
patience=35
imgsz=640
batch=16
workers=4
device=0
mosaic=0.3
close_mosaic=15
degrees=0.0
fliplr=0.0
flipud=0.0
mixup=0.0
copy_paste=0.0
```

训练结果默认保存到：

```text
runs/train/<name>/weights/best.pt
```
