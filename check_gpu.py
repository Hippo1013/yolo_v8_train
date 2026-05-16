import torch

print("-" * 30)
try:
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"发现 {gpu_count} 块显卡:")
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
        print("✅ 恭喜！你的 RTX 4050 已经准备好狂飙了！")
    else:
        print("❌ 警告：PyTorch 没找到显卡，训练会很慢！")
        print("可能是 torch 版本装错了，装成了 CPU 版。")
except Exception as e:
    print(f"❌ 环境有问题: {e}")
print("-" * 30)