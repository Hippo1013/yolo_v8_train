import argparse
from collections import Counter
from pathlib import Path


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


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated YOLO labels for v2 dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=SCRIPT_DIR / "dataset")
    return parser.parse_args()


def yolo_to_xyxy(values):
    x, y, w, h = values
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def boxes_intersect(box_a, box_b):
    return not (
        box_a[2] <= box_b[0]
        or box_b[2] <= box_a[0]
        or box_a[3] <= box_b[1]
        or box_b[3] <= box_a[1]
    )


def read_label(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} invalid YOLO row: {line}")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        rows.append((class_id, values))
    return rows


def validate_split(dataset_dir, split):
    label_dir = dataset_dir / split / "labels"
    image_dir = dataset_dir / split / "images"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    counts = Counter()
    intersection_violations = []
    malformed_counts = []
    missing_images = []

    for label_path in sorted(label_dir.glob("*.txt")):
        rows = read_label(label_path)
        if len(rows) not in (1, 2):
            malformed_counts.append(str(label_path))
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            missing_images.append(str(image_path))

        for class_id, _ in rows:
            counts[class_id] += 1

        if len(rows) == 2:
            first_box = yolo_to_xyxy(rows[0][1])
            second_box = yolo_to_xyxy(rows[1][1])
            if boxes_intersect(first_box, second_box):
                intersection_violations.append(str(label_path))

    image_count = len(list(image_dir.glob("*.jpg")))
    label_count = len(list(label_dir.glob("*.txt")))
    return {
        "split": split,
        "image_count": image_count,
        "label_file_count": label_count,
        "class_counts": counts,
        "intersection_violations": intersection_violations,
        "malformed_counts": malformed_counts,
        "missing_images": missing_images,
    }


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    has_errors = False
    for split in ("train", "val"):
        result = validate_split(dataset_dir, split)
        print(f"{split}: images={result['image_count']} labels={result['label_file_count']}")
        for class_id in sorted(CLASS_NAMES):
            print(f"  {class_id}: {CLASS_NAMES[class_id]} = {result['class_counts'][class_id]}")

        for key in ("intersection_violations", "malformed_counts", "missing_images"):
            values = result[key]
            print(f"  {key}: {len(values)}")
            if values:
                has_errors = True
                for value in values[:10]:
                    print(f"    {value}")

    if has_errors:
        raise SystemExit(1)
    print("validation passed")


if __name__ == "__main__":
    main()
