import os
from rembg import remove
from PIL import Image

# ================= 配置区域 =================
# 1. 图片所在的文件夹 (你的原始图片放在哪？)
INPUT_DIR = "row/偏高" 

# 2. 输出文件夹 (处理好的图片放哪？)
OUTPUT_DIR = "transparent/偏高"

# 3. 文件名前缀
FILE_PREFIX = "偏高"

# 4. 遍历范围 (从1到5)
START_NUM = 1
END_NUM = 5
# ===========================================

def main():
    # 自动创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已自动创建输出目录: {OUTPUT_DIR}")

    print(f"🚀 开始 CPU 批量抠图 (范围: {START_NUM}-{END_NUM})...\n")

    for i in range(START_NUM, END_NUM + 1):
        # 拼接文件名: 偏低1.png, 偏低2.png ...
        filename = f"{FILE_PREFIX}{i}.png"
        
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # 检查文件是否存在
        if not os.path.exists(input_path):
            print(f"⚠️ 跳过: 找不到 {input_path}")
            continue

        try:
            print(f"⏳ 正在处理: {filename} ...", end="", flush=True)
            
            # 打开图片
            inp = Image.open(input_path)
            
            # 抠图 (CPU模式)
            output = remove(inp)
            
            # 保存
            output.save(output_path)
            print(" ✅ 完成")
            
        except Exception as e:
            print(f"\n❌ 失败: {e}")

    print("\n🎉 全部搞定！快去看看 transparent 文件夹吧。")

if __name__ == "__main__":
    main()