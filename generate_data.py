import os
import random
import cv2
import numpy as np
import albumentations as A
from PIL import Image

# ================= 配置区域 (每次换类别记得改这里!) =================

# 1. 输入设置
INPUT_DIR = "transparent/偏高"    # 素材文件夹 (请确认文件夹名字对不对)
FILE_PREFIX = "偏高"              # 文件前缀 (比如 "偏高1.png")
START_NUM = 1                     # 1
END_NUM = 5                       # 5
BG_DIR = "backgrounds"            # 背景库

# 2. 类别设置
CLASS_ID = 2                      # 偏低=0, 正常=1, 偏高=2

# 3. 输出设置
OUTPUT_IMG_DIR = "datasets/train/images"
OUTPUT_TXT_DIR = "datasets/train/labels"
GENERATE_PER_IMAGE = 350          # 每张素材生成 350 张

# 4. 变换参数
SCALE_MIN = 0.3                   # 缩放最小值
SCALE_MAX = 0.8                   # 缩放最大值
ROTATE_ANGLE = 180                # 全角度旋转
STRETCH_INTENSITY = 0.2           # 拉伸强度

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

def main():
    # 0. 检查背景库
    if not os.path.exists(BG_DIR):
        print(f"❌ [错误] 找不到背景文件夹: {os.path.abspath(BG_DIR)}")
        return
    
    bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not bg_files:
        print(f"❌ [错误] 背景文件夹里是空的: {BG_DIR}")
        return

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
    
    print(f"🚀 开始生成 (Debug模式)...")
    print(f"   输入目录: {os.path.abspath(INPUT_DIR)}")
    print(f"   查找文件: {FILE_PREFIX}{START_NUM}.png -> {FILE_PREFIX}{END_NUM}.png")
    
    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ [致命错误] 输入文件夹不存在！")
        print(f"   请检查你的项目里有没有这个文件夹: {INPUT_DIR}")
        return

    total_success = 0

    for img_idx in range(START_NUM, END_NUM + 1):
        filename = f"{FILE_PREFIX}{img_idx}.png"
        obj_path = os.path.join(INPUT_DIR, filename)

        # 🔍 调试信息：告诉你它在找哪个文件
        if not os.path.exists(obj_path):
            print(f"⚠️ [跳过] 没找到文件: {obj_path}")
            # 尝试找一下可能存在的其他后缀，帮用户排查
            if os.path.exists(obj_path.replace('.png', '.jpg')):
                 print(f"   💡 提示: 发现同名 .jpg 文件，请修改脚本里的后缀逻辑或重命名文件。")
            continue

        try:
            # 加载素材
            obj_orig = Image.open(obj_path).convert("RGBA")
            print(f"📸 正在处理: {filename} ...")

            for i in range(GENERATE_PER_IMAGE):
                try:
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
                    base_scale = random.uniform(SCALE_MIN, SCALE_MAX)
                    stretch_w = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)
                    stretch_h = random.uniform(1.0 - STRETCH_INTENSITY, 1.0 + STRETCH_INTENSITY)

                    obj_aspect = obj_rot.width / obj_rot.height
                    target_w_base = bg_w * base_scale
                    target_h_base = target_w_base / obj_aspect
                    
                    final_w = int(target_w_base * stretch_w)
                    final_h = int(target_h_base * stretch_h)
                    
                    if final_w > bg_w: final_w = int(bg_w * 0.9)
                    if final_h > bg_h: final_h = int(bg_h * 0.9)
                    
                    obj_final = obj_rot.resize((final_w, final_h), Image.Resampling.LANCZOS)

                    # --- [Step 4: 合成] ---
                    max_x = bg_w - final_w
                    max_y = bg_h - final_h
                    paste_x = random.randint(0, max(0, max_x))
                    paste_y = random.randint(0, max(0, max_y))

                    comp_img = bg_img.copy()
                    comp_img.paste(obj_final, (paste_x, paste_y), mask=obj_final)

                    bbox_orig = [paste_x, paste_y, paste_x + final_w, paste_y + final_h]

                    # --- [Step 5: 增强 & 保存] ---
                    img_np = np.array(comp_img)
                    augmented = transform_pipeline(image=img_np, bboxes=[bbox_orig], class_labels=[CLASS_ID])
                    
                    final_img = augmented['image']
                    final_bboxes = augmented['bboxes']

                    if not final_bboxes: continue

                    save_name = f"aug_{CLASS_ID}_src{img_idx}_{i:04d}"
                    cv2.imwrite(f"{OUTPUT_IMG_DIR}/{save_name}.jpg", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

                    yolo_box = xyxy_to_yolo(final_bboxes[0], final_img.shape[1], final_img.shape[0])
                    with open(f"{OUTPUT_TXT_DIR}/{save_name}.txt", "w") as f:
                        f.write(f"{CLASS_ID} {' '.join([f'{x:.6f}' for x in yolo_box])}\n")
                    
                    total_success += 1

                except Exception:
                    pass

        except Exception as e:
            print(f"❌ 处理素材 {filename} 时崩溃: {e}")

    print("-" * 30)
    print(f"✅ 任务结束。共生成 {total_success} 张图片。")
    if total_success == 0:
        print("⚠️ 警告: 一张图都没生成！请仔细检查上面的 [跳过] 提示。")
        print("   可能是文件夹路径写错了，或者文件名不匹配 (例如是 .jpg 却写成了 .png)")

if __name__ == "__main__":
    main()