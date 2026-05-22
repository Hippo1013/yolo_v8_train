import os
import random
from io import BytesIO
import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageEnhance, ImageFilter

# ================= 配置区域 =================

# 1. 输入设置
BG_DIR = "backgrounds"            # 640x640 背景库，先运行 resize_backgrounds.py 生成

# 2. 类别设置
CLASSES = [
    {"class_id": 0, "name": "偏低", "input_dir": "transparent/偏低", "file_prefix": "偏低", "start_num": 1, "end_num": 5},
    {"class_id": 1, "name": "正常", "input_dir": "transparent/正常", "file_prefix": "正常", "start_num": 1, "end_num": 5},
    {"class_id": 2, "name": "偏高", "input_dir": "transparent/偏高", "file_prefix": "偏高", "start_num": 1, "end_num": 5},
]

# 3. 输出设置
OUTPUT_IMG_DIR = "dataset/train/images"
OUTPUT_TXT_DIR = "dataset/train/labels"
GENERATE_PER_IMAGE = 350          # 每张素材生成 350 张
MAX_TOTAL_IMAGES = None           # 试生成总数；全量生成时改为 None

# 4. 变换参数
SCALE_MIN = 0.12                  # 仪表盘占背景宽度的最小比例
SCALE_MAX = 0.58                  # 仪表盘占背景宽度的最大比例
SMALL_SCALE_MAX = 0.24            # 小仪表盘占比上限
MEDIUM_SCALE_MAX = 0.40           # 中等仪表盘占比上限
SMALL_SCALE_PROB = 0.45           # 小仪表盘采样占比
MEDIUM_SCALE_PROB = 0.40          # 中等仪表盘采样占比
ROTATE_ANGLE = 180                # 全角度旋转
STRETCH_INTENSITY = 0.2           # 拉伸强度
MIN_OBJECT_SIZE = 64              # 过小样本跳过，避免目标不可辨
MIN_RESOLUTION_FACTOR = 0.35      # 小目标前景降采样比例
MAX_RESOLUTION_FACTOR = 1.0       # 大目标前景保留比例
COLOR_EDGE_BLUR_MAX = 0.9         # 小目标颜色交界最大软化半径
COLOR_EDGE_BLUR_PROB = 0.85       # 颜色交界软化概率
CONTRAST_REDUCTION_MAX = 0.12     # 小目标最大对比度降低幅度
SATURATION_REDUCTION_MAX = 0.12   # 小目标最大饱和度降低幅度

# ================= 增强管道 =================
transform_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.MotionBlur(blur_limit=7, p=0.2),
    A.ISONoise(p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# ================= 工具函数 =================
def xyxy_to_yolo(box, img_w, img_h):
    x_min, y_min, x_max, y_max = box
    dw, dh = 1. / img_w, 1. / img_h
    w_px = x_max - x_min
    h_px = y_max - y_min
    x_c = x_min + w_px / 2.0
    y_c = y_min + h_px / 2.0
    return (x_c * dw, y_c * dh, w_px * dw, h_px * dh)

def blur_rgb_keep_alpha(obj_img, radius):
    alpha = obj_img.getchannel("A")
    rgb = obj_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    rgb.putalpha(alpha)
    return rgb

def sample_scale_ratio():
    """分段连续采样，让小仪表盘占比更高，同时保留中、大目标。"""
    r = random.random()
    if r < SMALL_SCALE_PROB:
        return random.uniform(SCALE_MIN, SMALL_SCALE_MAX)
    if r < SMALL_SCALE_PROB + MEDIUM_SCALE_PROB:
        return random.uniform(SMALL_SCALE_MAX, MEDIUM_SCALE_MAX)
    return random.uniform(MEDIUM_SCALE_MAX, SCALE_MAX)

def degrade_object_resolution(obj_img, scale_ratio):
    """按仪表盘占比连续降低前景分辨率：目标越小，细节越糊。"""
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

def generate_one_sample(obj_orig, bg_files, class_id, img_idx, sample_idx):
    # --- [Step 1: 前景处理 (全角度旋转 + 裁剪)] ---
    angle = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)
    obj_rot = obj_orig.rotate(angle, expand=True, resample=Image.BICUBIC)

    crop_box = obj_rot.getbbox()
    if crop_box:
        obj_rot = obj_rot.crop(crop_box)

    # --- [Step 2: 准备背景] ---
    bg_path = random.choice(bg_files)
    bg_img = Image.open(bg_path).convert("RGB")
    bg_w, bg_h = bg_img.size

    # --- [Step 3: 计算缩放 + 拉伸变形] ---
    base_scale = sample_scale_ratio()
    stretch_w = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)
    stretch_h = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)

    obj_aspect = obj_rot.width / obj_rot.height
    target_w_base = bg_w * base_scale
    target_h_base = target_w_base / obj_aspect

    final_w = int(target_w_base * stretch_w)
    final_h = int(target_h_base * stretch_h)

    if final_w > bg_w:
        final_w = int(bg_w * 0.9)
    if final_h > bg_h:
        final_h = int(bg_h * 0.9)
    if final_w < MIN_OBJECT_SIZE or final_h < MIN_OBJECT_SIZE:
        return False

    obj_final = obj_rot.resize((final_w, final_h), Image.Resampling.LANCZOS)
    obj_final = degrade_object_resolution(obj_final, base_scale)

    # --- [Step 4: 合成] ---
    max_x = bg_w - final_w
    max_y = bg_h - final_h
    paste_x = random.randint(0, max(0, max_x))
    paste_y = random.randint(0, max(0, max_y))

    comp_img = bg_img.copy()
    comp_img.paste(obj_final, (paste_x, paste_y), mask=obj_final)

    bbox_orig = [paste_x, paste_y, paste_x + final_w, paste_y + final_h]

    # --- [Step 5: 增强 & 保存] ---
    # albumentations 同时更新图像和 bbox，保证增强后的标签仍然对齐目标。
    img_np = np.array(comp_img)
    augmented = transform_pipeline(image=img_np, bboxes=[bbox_orig], class_labels=[class_id])

    final_img = augmented['image']
    final_bboxes = augmented['bboxes']

    if not final_bboxes:
        return False

    save_name = f"aug_{class_id}_src{img_idx}_{sample_idx:04d}"
    cv2.imwrite(f"{OUTPUT_IMG_DIR}/{save_name}.jpg", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    yolo_box = xyxy_to_yolo(final_bboxes[0], final_img.shape[1], final_img.shape[0])
    with open(f"{OUTPUT_TXT_DIR}/{save_name}.txt", "w") as f:
        f.write(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_box])}\n")

    return True

def generate_class(class_config, bg_files, total_success):
    class_id = class_config["class_id"]
    input_dir = class_config["input_dir"]
    file_prefix = class_config["file_prefix"]
    start_num = class_config["start_num"]
    end_num = class_config["end_num"]

    print("-" * 30)
    print(f"开始生成类别: {class_config['name']} (ID={class_id})")
    print(f"   输入目录: {os.path.abspath(input_dir)}")
    print(f"   查找文件: {file_prefix}{start_num}.png -> {file_prefix}{end_num}.png")

    if not os.path.exists(input_dir):
        print("[跳过] 输入文件夹不存在")
        return total_success

    class_success = 0

    for img_idx in range(start_num, end_num + 1):
        filename = f"{file_prefix}{img_idx}.png"
        obj_path = os.path.join(input_dir, filename)

        if not os.path.exists(obj_path):
            print(f"[跳过] 没找到文件: {obj_path}")
            if os.path.exists(obj_path.replace('.png', '.jpg')):
                print("   提示: 发现同名 .jpg 文件，请修改脚本里的后缀逻辑或重命名文件。")
            continue

        try:
            obj_orig = Image.open(obj_path).convert("RGBA")
            print(f"正在处理: {filename} ...")

            generated_for_image = 0
            attempts = 0
            max_attempts = GENERATE_PER_IMAGE * 5

            while generated_for_image < GENERATE_PER_IMAGE and attempts < max_attempts:
                if MAX_TOTAL_IMAGES is not None and total_success >= MAX_TOTAL_IMAGES:
                    break

                attempts += 1
                try:
                    if generate_one_sample(obj_orig, bg_files, class_id, img_idx, generated_for_image):
                        generated_for_image += 1
                        class_success += 1
                        total_success += 1
                except Exception:
                    pass

            if generated_for_image < GENERATE_PER_IMAGE:
                print(f"   警告: {filename} 只生成 {generated_for_image}/{GENERATE_PER_IMAGE} 张")

        except Exception as e:
            print(f"[错误] 处理素材 {filename} 时崩溃: {e}")

        if MAX_TOTAL_IMAGES is not None and total_success >= MAX_TOTAL_IMAGES:
            break

    print(f"类别 {class_config['name']} 生成完成: {class_success} 张")
    return total_success

def main():
    # 0. 检查背景库
    if not os.path.exists(BG_DIR):
        print(f"[错误] 找不到背景文件夹: {os.path.abspath(BG_DIR)}")
        return

    bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not bg_files:
        print(f"[错误] 背景文件夹里是空的: {BG_DIR}")
        return

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    print("开始生成三类数据...")
    print(f"   背景目录: {os.path.abspath(BG_DIR)}")
    print(f"   输出目录: {os.path.abspath(os.path.dirname(OUTPUT_IMG_DIR))}")
    if MAX_TOTAL_IMAGES is not None:
        print(f"   试生成上限: {MAX_TOTAL_IMAGES} 张")

    # 依次生成三类数据，类别 ID 与 data.yaml 中的 names 顺序保持一致。
    total_success = 0
    for class_config in CLASSES:
        if MAX_TOTAL_IMAGES is not None and total_success >= MAX_TOTAL_IMAGES:
            break
        total_success = generate_class(class_config, bg_files, total_success)

    print("-" * 30)
    print(f"任务结束。共生成 {total_success} 张图片。")
    if total_success == 0:
        print("警告: 一张图都没生成！请仔细检查上面的 [跳过] 提示。")
        print("   可能是文件夹路径写错了，或者文件名不匹配 (例如是 .jpg 却写成了 .png)")

if __name__ == "__main__":
    main()
