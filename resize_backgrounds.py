from pathlib import Path

from PIL import Image, ImageOps


INPUT_DIR = Path("backgrounds_raw")
OUTPUT_DIR = Path("backgrounds")
TARGET_SIZE = 640
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_crop_square(image):
    # 先裁成正方形，再缩放到训练尺寸，避免背景被强行拉伸变形。
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        print("Run video_to_bg.py first, or put source background images in backgrounds_raw/.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print(f"No background images found in: {INPUT_DIR}")
        return

    success_count = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = center_crop_square(image)
                image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

                output_path = OUTPUT_DIR / f"{image_path.stem}.jpg"
                image.save(output_path, quality=95, optimize=True)
                success_count += 1
                print(f"Saved {output_path}")
        except Exception as exc:
            print(f"Failed to process {image_path}: {exc}")

    print(f"Done. Generated {success_count} resized backgrounds in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
