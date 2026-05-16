from ultralytics import YOLO
import cv2
import os

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH = "runs/detect/runs/train/meter_model/weights/best.pt"

# 2. 测试图片路径
IMAGE_PATH = "test/test2.jpg" # 我看你报错里是这个路径，帮你改好了
# ===========================================

def main():
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        print("   请检查 runs/train/ 文件夹下的路径是否正确")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到测试图片: {IMAGE_PATH}")
        print("   请确认图片路径是否正确")
        return

    print(f"🚀 正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    print(f"📸 正在预测图片: {IMAGE_PATH} ...")
    
    # 开始推理
    results = model.predict(source=IMAGE_PATH, conf=0.5)

    # 解析结果
    for result in results:
        print("-" * 30)
        print(f"检测到 {len(result.boxes)} 个目标:")
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            print(f"   🎯 类别: {class_name} (ID: {class_id})")
            print(f"      置信度: {confidence:.2f}")
            
        # 获取画好框的图片
        im_array = result.plot()
        
        # === 核心修改区 ===
        # 以前是用 cv2.imshow 弹窗，现在改成 cv2.imwrite 保存文件
        # 这样就能避开 headless 版本的报错
        save_name = "predict_result.jpg"
        cv2.imwrite(save_name, im_array)
        
        print("-" * 30)
        print(f"✅ 预测成功！结果图片已保存为: {save_name}")
        print("👉 请去文件夹里双击打开这张图，看看框准不准！")

if __name__ == "__main__":
    main()