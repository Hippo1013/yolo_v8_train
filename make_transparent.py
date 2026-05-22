from pathlib import Path

from PIL import Image
from rembg import remove


RAW_DIR = Path("raw")
OUTPUT_DIR = Path("transparent")
CLASSES = ["偏低", "正常", "偏高"]
START_NUM = 1
END_NUM = 5


def process_class(class_name):
    input_dir = RAW_DIR / class_name
    output_dir = OUTPUT_DIR / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始处理 {class_name}: {input_dir} -> {output_dir}")
    for idx in range(START_NUM, END_NUM + 1):
        filename = f"{class_name}{idx}.png"
        input_path = input_dir / filename
        output_path = output_dir / filename

        if not input_path.exists():
            print(f"[跳过] 找不到 {input_path}")
            continue

        try:
            print(f"正在抠图: {filename} ...", end="", flush=True)
            image = Image.open(input_path)
            transparent = remove(image)
            transparent.save(output_path)
            print("完成")
        except Exception as exc:
            print(f"失败: {exc}")


def main():
    if not RAW_DIR.exists():
        print(f"找不到原数据目录: {RAW_DIR}")
        return

    for class_name in CLASSES:
        process_class(class_name)

    print(f"处理结束，透明前景已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
