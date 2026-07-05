import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = {
    0: "dashboard_low",
    1: "dashboard_normal",
    2: "dashboard_high",
    3: "letter_A",
    4: "letter_B",
    5: "letter_C",
    6: "letter_D",
}
COLORS = {
    0: (230, 57, 70),
    1: (42, 157, 143),
    2: (244, 162, 97),
    3: (38, 70, 83),
    4: (69, 123, 157),
    5: (131, 56, 236),
    6: (255, 183, 3),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO labels onto generated dataset samples.")
    parser.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "previews_train")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def yolo_to_xyxy(values, width, height):
    xc, yc, w, h = values
    box_w = w * width
    box_h = h * height
    x1 = xc * width - box_w / 2.0
    y1 = yc * height - box_h / 2.0
    x2 = xc * width + box_w / 2.0
    y2 = yc * height + box_h / 2.0
    return x1, y1, x2, y2


def draw_wide_rectangle(draw, box, color, width=3):
    x1, y1, x2, y2 = box
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)


def label_text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def draw_preview(image_path, label_path, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        box = yolo_to_xyxy(values, width, height)
        color = COLORS.get(class_id, (255, 255, 255))
        text = f"{class_id} {CLASS_NAMES.get(class_id, 'unknown')}"
        draw_wide_rectangle(draw, box, color, width=3)

        text_w, text_h = label_text_size(draw, text, font)
        x1, y1, _, _ = box
        text_x = max(0, int(x1))
        text_y = max(0, int(y1) - text_h - 4)
        draw.rectangle((text_x, text_y, text_x + text_w + 6, text_y + text_h + 4), fill=color)
        draw.text((text_x + 3, text_y + 2), text, fill=(255, 255, 255), font=font)

    image.save(output_path, quality=95)


def main():
    args = parse_args()
    random.seed(args.seed)
    image_dir = args.dataset_dir / args.split / "images"
    label_dir = args.dataset_dir / args.split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(path for path in image_dir.glob("*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    sample_count = min(args.count, len(image_files))
    selected = random.sample(image_files, sample_count)

    for index, image_path in enumerate(selected):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        output_path = args.output_dir / f"preview_{index:02d}_{image_path.stem}.jpg"
        draw_preview(image_path, label_path, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
