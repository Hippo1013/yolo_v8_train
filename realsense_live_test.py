import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ================= 配置区域 =================
MODEL_PATH = Path("runs/detect/runs2/train/v1.1_yolov8s/weights/best.pt")

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30

CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.7
IMGSZ = 640

SHOW_DEPTH_DISTANCE = True
WINDOW_NAME = "RealSense YOLOv8 Live Test"

CLASS_NAMES_CN = {
    0: "偏低",
    1: "正常",
    2: "偏高",
}

# OpenCV BGR 颜色
CLASS_COLORS = {
    0: (255, 80, 80),    # 偏低：蓝
    1: (80, 200, 80),    # 正常：绿
    2: (80, 80, 255),    # 偏高：红
}

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]
# ===========================================


def load_chinese_font(size=22):
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def get_device():
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("未检测到 CUDA，将使用 CPU 推理。")
    return "cpu"


def draw_label_with_pil(frame_bgr, text, box, color_bgr, font):
    x1, y1, x2, y2 = box
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    label_x = max(0, x1)
    label_y = max(0, y1 - text_h - 8)
    if label_y == 0:
        label_y = min(y2 + 4, frame_bgr.shape[0] - text_h - 4)

    draw.rectangle(
        [label_x, label_y, label_x + text_w + 8, label_y + text_h + 6],
        fill=color_rgb,
    )
    draw.text((label_x + 4, label_y + 2), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def get_center_distance(depth_frame, box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    if cx < 0 or cy < 0 or cx >= COLOR_WIDTH or cy >= COLOR_HEIGHT:
        return None

    # 取中心附近小窗口的有效深度中位数，比单点深度稳定。
    distances = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            px = cx + dx
            py = cy + dy
            if 0 <= px < COLOR_WIDTH and 0 <= py < COLOR_HEIGHT:
                dist = depth_frame.get_distance(px, py)
                if dist > 0:
                    distances.append(dist)

    if not distances:
        return None
    return float(np.median(distances))


def main():
    if not MODEL_PATH.exists():
        print(f"找不到模型文件: {MODEL_PATH}")
        print("请先确认训练结果路径是否存在。")
        return

    device = get_device()

    print(f"正在加载模型: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    font = load_chinese_font(size=22)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    if SHOW_DEPTH_DISTANCE:
        config.enable_stream(rs.stream.depth, COLOR_WIDTH, COLOR_HEIGHT, rs.format.z16, FPS)

    align = rs.align(rs.stream.color) if SHOW_DEPTH_DISTANCE else None

    print("正在启动 Intel RealSense D435i...")
    pipeline.start(config)

    print("开始实时识别。按 q 或 Esc 退出。")
    last_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if SHOW_DEPTH_DISTANCE:
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
            else:
                depth_frame = None

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data()).copy()

            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    class_name = CLASS_NAMES_CN.get(cls_id, str(model.names.get(cls_id, cls_id)))
                    color = CLASS_COLORS.get(cls_id, (0, 255, 255))

                    label = f"{class_name} {conf:.2f}"
                    if SHOW_DEPTH_DISTANCE and depth_frame:
                        dist = get_center_distance(depth_frame, (x1, y1, x2, y2))
                        if dist is not None:
                            label += f" {dist:.2f}m"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    frame = draw_label_with_pil(frame, label, (x1, y1, x2, y2), color, font)

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 3, color, -1)

            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
