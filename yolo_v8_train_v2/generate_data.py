import argparse
import random
import shutil
from io import BytesIO
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "dashboard_low",
    1: "dashboard_normal",
    2: "dashboard_high",
    3: "letter_A",
    4: "letter_B",
    5: "letter_C",
    6: "letter_D",
}

DASHBOARD_CLASSES = [
    {"class_id": 0, "name": "dashboard_low", "asset_dir": "偏低"},
    {"class_id": 1, "name": "dashboard_normal", "asset_dir": "正常"},
    {"class_id": 2, "name": "dashboard_high", "asset_dir": "偏高"},
]

LETTER_CLASSES = [
    {"class_id": 3, "name": "letter_A", "source": "A.png"},
    {"class_id": 4, "name": "letter_B", "source": "B.png"},
    {"class_id": 5, "name": "letter_C", "source": "C.png"},
    {"class_id": 6, "name": "letter_D", "source": "D.png"},
]

# Dashboard geometry constants must stay aligned with yolo_v8_train/generate_data.py.
SCALE_MIN = 0.12
SCALE_MAX = 0.58
SMALL_SCALE_MAX = 0.24
MEDIUM_SCALE_MAX = 0.40
SMALL_SCALE_PROB = 0.45
MEDIUM_SCALE_PROB = 0.40
ROTATE_ANGLE = 180
STRETCH_INTENSITY = 0.2
MIN_OBJECT_SIZE = 64
MIN_RESOLUTION_FACTOR = 0.35
MAX_RESOLUTION_FACTOR = 1.0
COLOR_EDGE_BLUR_MAX = 0.9
COLOR_EDGE_BLUR_PROB = 0.85
CONTRAST_REDUCTION_MAX = 0.12
SATURATION_REDUCTION_MAX = 0.12

transform_pipeline = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.ISONoise(p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_visibility=0.0,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate v2 synthetic YOLO data for dashboard states and letters.")
    parser.add_argument("--background-dir", type=Path, default=SCRIPT_DIR / "backgrounds")
    parser.add_argument("--dashboard-dir", type=Path, default=SCRIPT_DIR / "assets" / "dashboard_transparent")
    parser.add_argument("--letter-dir", type=Path, default=SCRIPT_DIR / "assets" / "letter_sources")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "dataset")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-dashboard-only-per-class", type=int, default=500)
    parser.add_argument("--train-letter-only-per-class", type=int, default=800)
    parser.add_argument("--train-pair-per-combo", type=int, default=300)
    parser.add_argument("--val-dashboard-only-per-class", type=int, default=150)
    parser.add_argument("--val-letter-only-per-class", type=int, default=220)
    parser.add_argument("--val-pair-per-combo", type=int, default=80)
    parser.add_argument("--letter-scale-min", type=float, default=0.16)
    parser.add_argument("--letter-scale-max", type=float, default=0.60)
    parser.add_argument("--letter-perspective", type=float, default=0.06)
    parser.add_argument("--letter-threshold", type=int, default=180)
    parser.add_argument("--letter-box-padding-ratio", type=float, default=0.035)
    parser.add_argument("--pair-gap", type=float, default=8.0, help="Minimum pixel gap between label boxes in pair images.")
    parser.add_argument("--placement-attempts", type=int, default=120)
    parser.add_argument("--generation-attempt-factor", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def load_backgrounds(background_dir):
    if not background_dir.exists():
        raise FileNotFoundError(f"Background directory not found: {background_dir}")
    files = sorted(path for path in background_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No background images found in: {background_dir}")
    return files


def load_dashboard_sources(dashboard_dir):
    sources = {}
    for class_config in DASHBOARD_CLASSES:
        class_dir = dashboard_dir / class_config["asset_dir"]
        if not class_dir.exists():
            raise FileNotFoundError(f"Dashboard source directory not found: {class_dir}")
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"No dashboard images found in: {class_dir}")
        sources[class_config["class_id"]] = [Image.open(path).convert("RGBA") for path in files]
    return sources


def load_letter_sources(letter_dir, threshold):
    sources = {}
    for class_config in LETTER_CLASSES:
        path = letter_dir / class_config["source"]
        if not path.exists():
            raise FileNotFoundError(f"Letter source image not found: {path}")
        image = Image.open(path).convert("RGBA")
        sources[class_config["class_id"]] = {
            "paper": image,
            "letter_mask": extract_letter_mask(image, threshold),
        }
    return sources


def prepare_output_dirs(output_dir, keep_existing):
    for split in ("train", "val"):
        for subdir in ("images", "labels"):
            target = output_dir / split / subdir
            if target.exists() and not keep_existing:
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]


def valid_box(box, min_size=2.0):
    return box[2] - box[0] >= min_size and box[3] - box[1] >= min_size


def boxes_intersect(box_a, box_b, gap=0.0):
    return not (
        box_a[2] + gap <= box_b[0]
        or box_b[2] + gap <= box_a[0]
        or box_a[3] + gap <= box_b[1]
        or box_b[3] + gap <= box_a[1]
    )


def blur_rgb_keep_alpha(obj_img, radius):
    alpha = obj_img.getchannel("A")
    rgb = obj_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    rgb.putalpha(alpha)
    return rgb


def sample_dashboard_scale_ratio():
    """Same segmented scale distribution as the original dashboard generator."""
    r = random.random()
    if r < SMALL_SCALE_PROB:
        return random.uniform(SCALE_MIN, SMALL_SCALE_MAX)
    if r < SMALL_SCALE_PROB + MEDIUM_SCALE_PROB:
        return random.uniform(SMALL_SCALE_MAX, MEDIUM_SCALE_MAX)
    return random.uniform(MEDIUM_SCALE_MAX, SCALE_MAX)


def degrade_object_resolution(obj_img, scale_ratio):
    """Same small-object degradation logic as the original dashboard generator."""
    target_size = obj_img.size

    if SCALE_MAX <= SCALE_MIN:
        resolution_factor = MAX_RESOLUTION_FACTOR
    else:
        normalized = (scale_ratio - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)
        normalized = max(0.0, min(1.0, normalized))
        resolution_factor = MIN_RESOLUTION_FACTOR + normalized * (MAX_RESOLUTION_FACTOR - MIN_RESOLUTION_FACTOR)

    low_w = max(8, int(obj_img.width * resolution_factor))
    low_h = max(8, int(obj_img.height * resolution_factor))

    if low_w < obj_img.width or low_h < obj_img.height:
        obj_img = obj_img.resize((low_w, low_h), Image.Resampling.BILINEAR)
        obj_img = obj_img.resize(target_size, Image.Resampling.BILINEAR)

    blur_strength = (1.0 - resolution_factor) * 0.8
    if blur_strength > 0.08:
        obj_img = blur_rgb_keep_alpha(obj_img, blur_strength)

    edge_blur = (1.0 - resolution_factor) * COLOR_EDGE_BLUR_MAX
    if edge_blur > 0.08 and random.random() < COLOR_EDGE_BLUR_PROB:
        obj_img = blur_rgb_keep_alpha(obj_img, edge_blur)

    degradation_strength = 1.0 - resolution_factor
    if degradation_strength > 0.05:
        alpha = obj_img.getchannel("A")
        rgb = obj_img.convert("RGB")
        contrast = 1.0 - degradation_strength * CONTRAST_REDUCTION_MAX
        saturation = 1.0 - degradation_strength * SATURATION_REDUCTION_MAX
        rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
        rgb = ImageEnhance.Color(rgb).enhance(saturation)
        rgb.putalpha(alpha)
        obj_img = rgb

    if resolution_factor < 0.85:
        rgb = Image.new("RGB", obj_img.size, (0, 0, 0))
        rgb.paste(obj_img, mask=obj_img.getchannel("A"))
        buffer = BytesIO()
        quality = int(45 + resolution_factor * 45)
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGBA")
        compressed.putalpha(obj_img.getchannel("A"))
        obj_img = compressed

    return obj_img


def make_dashboard_object(class_id, dashboard_sources, image_size):
    obj_orig = random.choice(dashboard_sources[class_id])
    angle = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)
    obj_rot = obj_orig.rotate(angle, expand=True, resample=Image.BICUBIC)

    crop_box = obj_rot.getbbox()
    if crop_box:
        obj_rot = obj_rot.crop(crop_box)

    base_scale = sample_dashboard_scale_ratio()
    stretch_w = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)
    stretch_h = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)

    obj_aspect = obj_rot.width / obj_rot.height
    target_w_base = image_size * base_scale
    target_h_base = target_w_base / obj_aspect

    final_w = int(target_w_base * stretch_w)
    final_h = int(target_h_base * stretch_h)

    if final_w > image_size:
        final_w = int(image_size * 0.9)
    if final_h > image_size:
        final_h = int(image_size * 0.9)
    if final_w < MIN_OBJECT_SIZE or final_h < MIN_OBJECT_SIZE:
        return None

    obj_final = obj_rot.resize((final_w, final_h), Image.Resampling.LANCZOS)
    obj_final = degrade_object_resolution(obj_final, base_scale)
    return {"class_id": class_id, "image": obj_final, "letter_mask": None}


def extract_letter_mask(source_img, threshold):
    rgb = source_img.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    alpha = source_img.getchannel("A")

    gray_np = np.array(gray)
    alpha_np = np.array(alpha)
    mask_np = ((gray_np < threshold) & (alpha_np > 0)).astype(np.uint8) * 255
    return Image.fromarray(mask_np, mode="L")


def crop_pair_to_alpha(obj_img, letter_mask):
    bbox = obj_img.getchannel("A").getbbox()
    if not bbox:
        return None, None
    return obj_img.crop(bbox), letter_mask.crop(bbox)


def perspective_coefficients(start_points, end_points):
    matrix = []
    vector = []
    for (x1, y1), (x2, y2) in zip(start_points, end_points):
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
        vector.extend([x2, y2])
    coeffs = np.linalg.lstsq(np.array(matrix, dtype=np.float64), np.array(vector, dtype=np.float64), rcond=None)[0]
    return coeffs.tolist()


def apply_perspective_pair(obj_img, letter_mask, max_shift_ratio):
    if max_shift_ratio <= 0:
        return obj_img, letter_mask

    w, h = obj_img.size
    if w < 4 or h < 4:
        return obj_img, letter_mask

    max_dx = max(1, int(w * max_shift_ratio))
    max_dy = max(1, int(h * max_shift_ratio))

    src = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    dst = [
        (random.randint(0, max_dx), random.randint(0, max_dy)),
        (w - 1 - random.randint(0, max_dx), random.randint(0, max_dy)),
        (w - 1 - random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)),
        (random.randint(0, max_dx), h - 1 - random.randint(0, max_dy)),
    ]

    coeffs = perspective_coefficients(dst, src)
    obj_img = obj_img.transform(
        (w, h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    letter_mask = letter_mask.transform(
        (w, h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
        fillcolor=0,
    )
    return obj_img, letter_mask


def make_letter_object(class_id, letter_sources, image_size, args):
    source = letter_sources[class_id]
    source_img = source["paper"]
    source_letter_mask = source["letter_mask"]
    source_aspect = source_img.width / source_img.height

    scale_ratio = random.uniform(args.letter_scale_min, args.letter_scale_max)
    target_w = int(image_size * scale_ratio)
    target_h = int(target_w / source_aspect)

    max_h = int(image_size * 0.78)
    if target_h > max_h:
        target_h = max_h
        target_w = int(target_h * source_aspect)

    stretch_w = random.uniform(0.94, 1.06)
    stretch_h = random.uniform(0.94, 1.06)
    target_w = max(32, int(target_w * stretch_w))
    target_h = max(32, int(target_h * stretch_h))

    obj = source_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    letter_mask = source_letter_mask.resize((target_w, target_h), Image.Resampling.LANCZOS)
    obj, letter_mask = apply_perspective_pair(obj, letter_mask, args.letter_perspective)

    # Deliberately no letter rotation. Real scenes should not contain upside-down letters.
    obj, letter_mask = crop_pair_to_alpha(obj, letter_mask)
    if obj is None:
        return None

    if obj.width > image_size * 0.95 or obj.height > image_size * 0.95:
        ratio = min((image_size * 0.95) / obj.width, (image_size * 0.95) / obj.height)
        resized_size = (max(1, int(obj.width * ratio)), max(1, int(obj.height * ratio)))
        obj = obj.resize(resized_size, Image.Resampling.LANCZOS)
        letter_mask = letter_mask.resize(resized_size, Image.Resampling.LANCZOS)

    obj = degrade_object_resolution(obj, scale_ratio)
    return {"class_id": class_id, "image": obj, "letter_mask": letter_mask}


def square_letter_box(letter_mask, paste_x, paste_y, image_size, padding_ratio):
    mask_box = letter_mask.point(lambda value: 255 if value > 24 else 0).getbbox()
    if not mask_box:
        return None

    x1, y1, x2, y2 = mask_box
    width = x2 - x1
    height = y2 - y1
    side = max(width, height)
    padding = int(round(side * padding_ratio))
    side = side + padding * 2

    cx = (x1 + x2) / 2.0 + paste_x
    cy = (y1 + y2) / 2.0 + paste_y
    bx1 = cx - side / 2.0
    by1 = cy - side / 2.0
    bx2 = cx + side / 2.0
    by2 = cy + side / 2.0

    if bx1 < 0:
        bx2 -= bx1
        bx1 = 0
    if by1 < 0:
        by2 -= by1
        by1 = 0
    if bx2 > image_size:
        shift = bx2 - image_size
        bx1 -= shift
        bx2 = image_size
    if by2 > image_size:
        shift = by2 - image_size
        by1 -= shift
        by2 = image_size

    return clip_box([bx1, by1, bx2, by2], image_size, image_size)


def candidate_label_box(obj, paste_x, paste_y, image_size, args):
    if obj["letter_mask"] is None:
        return [paste_x, paste_y, paste_x + obj["image"].width, paste_y + obj["image"].height]
    return square_letter_box(obj["letter_mask"], paste_x, paste_y, image_size, args.letter_box_padding_ratio)


def place_objects(objects, bg_img, args):
    placed = []

    for obj in objects:
        image = obj["image"]
        if image.width >= args.image_size or image.height >= args.image_size:
            return None

        max_x = args.image_size - image.width
        max_y = args.image_size - image.height
        success = False

        for _ in range(args.placement_attempts):
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            label_box = candidate_label_box(obj, paste_x, paste_y, args.image_size, args)
            if label_box is None or not valid_box(label_box, min_size=20.0):
                continue
            if any(boxes_intersect(label_box, item["bbox"], gap=args.pair_gap) for item in placed):
                continue

            placed.append(
                {
                    "class_id": obj["class_id"],
                    "bbox": label_box,
                    "image": image,
                    "paste_xy": (paste_x, paste_y),
                }
            )
            success = True
            break

        if not success:
            return None

    comp_img = bg_img.copy()
    # In pair samples, draw letter paper first and dashboard last so the dashboard
    # is never hidden by the white letter background.
    for item in sorted(placed, key=lambda value: 1 if value["class_id"] <= 2 else 0):
        comp_img.paste(item["image"], item["paste_xy"], mask=item["image"].getchannel("A"))

    labels = [{"class_id": item["class_id"], "bbox": item["bbox"]} for item in placed]
    return comp_img, labels


def compose_and_save(objects, background_files, split, save_name, args):
    bg_path = random.choice(background_files)
    bg_img = Image.open(bg_path).convert("RGB")
    bg_img = ImageOps.fit(bg_img, (args.image_size, args.image_size), method=Image.Resampling.BICUBIC)

    objects = list(objects)
    random.shuffle(objects)
    placed_result = place_objects(objects, bg_img, args)
    if placed_result is None:
        return False

    comp_img, placed = placed_result
    if len(placed) > 1 and boxes_intersect(placed[0]["bbox"], placed[1]["bbox"], gap=0.0):
        return False

    img_np = np.array(comp_img)
    class_labels = [item["class_id"] for item in placed]
    bboxes = [item["bbox"] for item in placed]
    augmented = transform_pipeline(image=img_np, bboxes=bboxes, class_labels=class_labels)

    final_img = augmented["image"]
    final_bboxes = [clip_box(box, final_img.shape[1], final_img.shape[0]) for box in augmented["bboxes"]]
    final_labels = list(augmented["class_labels"])

    if len(final_bboxes) != len(placed):
        return False
    if len(final_bboxes) > 1 and boxes_intersect(final_bboxes[0], final_bboxes[1], gap=0.0):
        return False
    if any(not valid_box(box, min_size=20.0) for box in final_bboxes):
        return False

    image_path = args.output_dir / split / "images" / f"{save_name}.jpg"
    label_path = args.output_dir / split / "labels" / f"{save_name}.txt"

    Image.fromarray(final_img).save(image_path, format="JPEG", quality=92)
    lines = []
    for class_id, box in zip(final_labels, final_bboxes):
        yolo_box = xyxy_to_yolo(box, final_img.shape[1], final_img.shape[0])
        lines.append(f"{class_id} {' '.join(f'{value:.6f}' for value in yolo_box)}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def generate_dashboard_only(split, count_per_class, dashboard_sources, background_files, args):
    total = 0
    for class_config in DASHBOARD_CLASSES:
        class_id = class_config["class_id"]
        generated = 0
        attempts = 0
        max_attempts = count_per_class * args.generation_attempt_factor
        print(f"[{split}] dashboard-only {class_config['name']}: target={count_per_class}")
        while generated < count_per_class and attempts < max_attempts:
            attempts += 1
            obj = make_dashboard_object(class_id, dashboard_sources, args.image_size)
            if obj is None:
                continue
            save_name = f"{split}_dashboard_only_{class_config['name']}_{generated:05d}"
            if compose_and_save([obj], background_files, split, save_name, args):
                generated += 1
                total += 1
        if generated < count_per_class:
            print(f"[warn] dashboard-only {class_config['name']} generated {generated}/{count_per_class}")
    return total


def generate_letter_only(split, count_per_class, letter_sources, background_files, args):
    total = 0
    for class_config in LETTER_CLASSES:
        class_id = class_config["class_id"]
        generated = 0
        attempts = 0
        max_attempts = count_per_class * args.generation_attempt_factor
        print(f"[{split}] letter-only {class_config['name']}: target={count_per_class}")
        while generated < count_per_class and attempts < max_attempts:
            attempts += 1
            obj = make_letter_object(class_id, letter_sources, args.image_size, args)
            if obj is None:
                continue
            save_name = f"{split}_letter_only_{class_config['name']}_{generated:05d}"
            if compose_and_save([obj], background_files, split, save_name, args):
                generated += 1
                total += 1
        if generated < count_per_class:
            print(f"[warn] letter-only {class_config['name']} generated {generated}/{count_per_class}")
    return total


def generate_pairs(split, count_per_combo, dashboard_sources, letter_sources, background_files, args):
    total = 0
    if count_per_combo <= 0:
        return total

    for dashboard_config in DASHBOARD_CLASSES:
        for letter_config in LETTER_CLASSES:
            generated = 0
            attempts = 0
            max_attempts = count_per_combo * args.generation_attempt_factor
            print(f"[{split}] pair {dashboard_config['name']} + {letter_config['name']}: target={count_per_combo}")
            while generated < count_per_combo and attempts < max_attempts:
                attempts += 1
                dashboard_obj = make_dashboard_object(dashboard_config["class_id"], dashboard_sources, args.image_size)
                letter_obj = make_letter_object(letter_config["class_id"], letter_sources, args.image_size, args)
                if dashboard_obj is None or letter_obj is None:
                    continue
                save_name = (
                    f"{split}_pair_{dashboard_config['name']}_{letter_config['name']}_{generated:05d}"
                )
                if compose_and_save([dashboard_obj, letter_obj], background_files, split, save_name, args):
                    generated += 1
                    total += 1
            if generated < count_per_combo:
                print(
                    f"[warn] pair {dashboard_config['name']} + {letter_config['name']} "
                    f"generated {generated}/{count_per_combo}"
                )
    return total


def count_labels(output_dir, split):
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    label_dir = output_dir / split / "labels"
    for path in label_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts[int(line.split()[0])] += 1
    return counts


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.background_dir = args.background_dir.resolve()
    args.dashboard_dir = args.dashboard_dir.resolve()
    args.letter_dir = args.letter_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    prepare_output_dirs(args.output_dir, args.keep_existing)
    background_files = load_backgrounds(args.background_dir)
    dashboard_sources = load_dashboard_sources(args.dashboard_dir)
    letter_sources = load_letter_sources(args.letter_dir, args.letter_threshold)

    print(f"background_dir: {args.background_dir}")
    print(f"dashboard_dir: {args.dashboard_dir}")
    print(f"letter_dir: {args.letter_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"image_size: {args.image_size}")
    print(f"pair_gap: {args.pair_gap}")

    train_total = 0
    train_total += generate_dashboard_only(
        "train", args.train_dashboard_only_per_class, dashboard_sources, background_files, args
    )
    train_total += generate_letter_only("train", args.train_letter_only_per_class, letter_sources, background_files, args)
    train_total += generate_pairs(
        "train", args.train_pair_per_combo, dashboard_sources, letter_sources, background_files, args
    )

    val_total = 0
    val_total += generate_dashboard_only("val", args.val_dashboard_only_per_class, dashboard_sources, background_files, args)
    val_total += generate_letter_only("val", args.val_letter_only_per_class, letter_sources, background_files, args)
    val_total += generate_pairs("val", args.val_pair_per_combo, dashboard_sources, letter_sources, background_files, args)

    print("-" * 60)
    print(f"generated train images: {train_total}")
    print(f"generated val images: {val_total}")
    print(f"generated total images: {train_total + val_total}")
    for split in ("train", "val"):
        print(f"{split} label counts:")
        for class_id, count in count_labels(args.output_dir, split).items():
            print(f"  {class_id}: {CLASS_NAMES[class_id]} = {count}")


if __name__ == "__main__":
    main()
