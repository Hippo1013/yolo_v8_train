import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 仪表状态识别训练入口")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型权重，如 yolov8n.pt 或 yolov8s.pt。")
    parser.add_argument("--data", default="data.yaml", help="YOLO 数据集配置文件。")
    parser.add_argument("--project", default="runs/train", help="训练结果输出根目录。")
    parser.add_argument("--name", default="dashboard_yolov8n", help="本次训练任务名称。")
    parser.add_argument("--epochs", type=int, default=150, help="最大训练轮次。")
    parser.add_argument("--patience", type=int, default=30, help="早停等待轮次。")
    parser.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸。")
    parser.add_argument("--batch", type=int, default=16, help="批次大小，显存不足时可改为 8。")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数。")
    parser.add_argument("--device", default="0", help="训练设备，GPU 用 0，CPU 用 cpu。")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖同名训练任务目录。")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"正在加载预训练模型: {args.model}")
    model = YOLO(args.model)

    print("开始训练。训练过程中按 Ctrl+C 可以提前结束并保存当前结果。")
    # 训练集已在 generate_data.py 中做过大量合成增强，这里只保留较轻的 YOLO 内置增强。
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        degrees=0.0,
        mosaic=0.5,
        close_mosaic=10,
    )

    print("训练完成，开始执行最终验证。")
    model.val()


if __name__ == "__main__":
    main()
