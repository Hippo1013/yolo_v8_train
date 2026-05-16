import cv2
import os
import glob
import random

# ================= 配置区域 =================
IMG_DIR = "datasets/train/images"
TXT_DIR = "datasets/train/labels"
OUTPUT_DIR = "check_results"  # 结果保存在这就
CHECK_COUNT = 50                     # 随机抽查多少张？
# ===========================================

def main():
    # 1. 准备工作
    if not os.path.exists(IMG_DIR):
        print("❌ 找不到图片文件夹")
        return
    
    # 清空并重建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        # 清除旧图
        for f in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, f))

    # 获取所有 jpg 图片
    img_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
    if not img_paths:
        print("❌ 没有找到图片，请检查路径")
        return

    # 随机打乱，抽查 20 张
    random.shuffle(img_paths)
    sample_paths = img_paths[:CHECK_COUNT]

    print(f"🚀 开始检查 {len(sample_paths)} 张图片...")

    for img_path in sample_paths:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w, _ = img.shape
        basename = os.path.basename(img_path)
        
        # 找对应的 txt
        txt_name = basename.replace(".jpg", ".txt")
        txt_path = os.path.join(TXT_DIR, txt_name)
        
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f.readlines():
                    # 解析 YOLO 格式: id x_c y_c w h
                    data = line.strip().split()
                    class_id = int(data[0])
                    x_c, y_c, bw, bh = map(float, data[1:])
                    
                    # 坐标转换: 归一化 -> 像素
                    w_px = int(bw * w)
                    h_px = int(bh * h)
                    x_px = int(x_c * w - w_px / 2)
                    y_px = int(y_c * h - h_px / 2)
                    
                    # 画框 (绿色, 线宽2)
                    cv2.rectangle(img, (x_px, y_px), (x_px + w_px, y_px + h_px), (0, 255, 0), 2)
                    
                    # 写个 ID 在头顶
                    cv2.putText(img, f"ID: {class_id}", (x_px, y_px - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存结果
        save_path = os.path.join(OUTPUT_DIR, f"check_{basename}")
        cv2.imwrite(save_path, img)
        print(f"✅ 已生成: check_{basename}")

    print(f"\n🎉 检查完毕！快去打开文件夹 '{OUTPUT_DIR}' 看看框准不准！")

if __name__ == "__main__":
    main()