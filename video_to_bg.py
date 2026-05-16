import cv2
import os

# === 配置 ===
VIDEO_PATH = "bg_video.mp4"      # 你的视频文件
OUTPUT_DIR = "backgrounds"       # 输出文件夹
SAVE_EVERY_N_FRAMES = 10         # 每隔几帧保存一张 (避免图片太重复)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 找不到视频 {VIDEO_PATH}，请录一段环境视频放过来！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    save_count = 0
    
    print("🚀 开始从视频提取背景图...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频结束
            
        # 每隔 N 帧保存一张
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            save_name = os.path.join(OUTPUT_DIR, f"bg_frame_{save_count}.jpg")
            cv2.imwrite(save_name, frame)
            save_count += 1
            print(f"   📸 已保存: {save_name}", end="\r")
            
        frame_count += 1
        
    cap.release()
    print(f"\n\n✅ 搞定！从视频里提取了 {save_count} 张背景图到 {OUTPUT_DIR} 文件夹。")
    print("👉 现在去运行 generate_data.py 吧！")

if __name__ == "__main__":
    main()