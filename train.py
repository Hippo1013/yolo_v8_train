from ultralytics import YOLO

def main():
    # 1. 加载预训练模型
    # yolov8s.pt 是小号版 (Small)，速度和精度平衡，适合机器狗这种边缘设备
    # 第一次运行会自动从网上下载这个文件，不用担心
    print("🚀 正在加载 YOLOv8 Small 模型...")
    model = YOLO('yolov8s.pt') 

    # 2. 开始训练 (让 RTX 4050 火力全开)
    print("🔥 开始训练！(训练过程中按 Ctrl+C 可以提前结束并保存)")
    
    results = model.train(
        data='data.yaml',   # 指定刚才写的配置文件
        epochs=150,         # 训练 150 轮 (通常 50-100 轮就能收敛得很好)
        imgsz=640,          # 图片大小 (标准 640x640)
        device=0,           # device=0 代表使用第一块显卡 (你的 RTX 4050)
        batch=16,           # 批次大小 (4050 显存 6G，设 16 比较稳，如果报错 OOM 就改成 8)
        workers=4,          # 数据加载线程数
        project='runs/train', # 结果保存在 runs/train 目录下
        name='meter_model',   # 这一次训练任务的名字
        exist_ok=True,        # 如果文件夹已存在，允许覆盖(或追加)
        
        # 增强设置 (因为我们已经手动增强过数据了，这里可以稍微关小一点自带的增强)
        degrees=0.0,        # 关闭自带旋转 (我们自己转过了)
        mosaic=1.0,         # 开启马赛克增强 (对小目标很有用)
    )
    
    print("✅ 训练完成！")
    
    # 3. 自动验证一下效果
    print("📊 正在进行最终验证...")
    metrics = model.val()

if __name__ == '__main__':
    # Windows 下多线程训练必须加这行保护，否则会报错
    main()