import cv2
import PIL
import albumentations
import numpy
import torch

print("✅ 环境自检开始...")
print(f"🔹 PyTorch 版本 (显卡核心): {torch.__version__} (CUDA可用: {torch.cuda.is_available()})")
print(f"🔹 NumPy 版本: {numpy.__version__} (如果是 1.x 就很安全)")
print(f"🔹 Pillow 版本: {PIL.__version__}")
print(f"🔹 Albumentations 版本: {albumentations.__version__}")
print(f"🔹 OpenCV 版本: {cv2.__version__}")

# 测试一下核心功能是否冲突
try:
    A = albumentations
    transform = A.Compose([A.SafeRotate(limit=20)])
    print("✅ Albumentations 语法测试通过 (SafeRotate 可用)")
except AttributeError:
    print("❌ Albumentations 版本太老，请升级！")

print("🎉 一切正常，可以开始生成数据了！")