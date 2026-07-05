import argparse
from pathlib import Path

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 v2 detector for dashboard states and letters.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, for example yolov8n.pt or yolov8s.pt.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.yaml")
    parser.add_argument("--project", type=Path, default=SCRIPT_DIR / "runs" / "train")
    parser.add_argument("--name", default=None, help="Run name. Default is derived from the base model.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a previous last.pt checkpoint.")
    return parser.parse_args()


def default_run_name(model_name):
    stem = Path(model_name).stem
    return f"dashboard_letter_{stem}_v2"


def main():
    args = parse_args()
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        print(f"resuming training from: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
        print("resume training finished; validating best available weights")
        model.val()
        return

    data_path = args.data.resolve()
    project_path = args.project.resolve()
    run_name = args.name or default_run_name(args.model)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    print(f"loading base model: {args.model}")
    print(f"dataset yaml: {data_path}")
    print(f"run name: {run_name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=str(project_path),
        name=run_name,
        exist_ok=args.exist_ok,
        degrees=0.0,
        mosaic=0.3,
        close_mosaic=15,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
    )

    print("training finished; validating best available weights")
    model.val(data=str(data_path), imgsz=args.imgsz, device=args.device, batch=args.batch)


if __name__ == "__main__":
    main()
