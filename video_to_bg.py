import cv2

from pathlib import Path


VIDEO_PATH = Path("bg_video.mp4")
OUTPUT_DIR = Path("backgrounds_raw")
SAVE_EVERY_N_FRAMES = 10

def main():
    if not VIDEO_PATH.exists():
        print(f"找不到视频文件: {VIDEO_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_count = 0
    save_count = 0
    
    print(f"开始从 {VIDEO_PATH} 抽帧到 {OUTPUT_DIR} ...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频结束
            
        # 每隔 N 帧保存一张
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            save_name = OUTPUT_DIR / f"bg_frame_{save_count}.jpg"
            cv2.imwrite(str(save_name), frame)
            save_count += 1
            print(f"已保存: {save_name}", end="\r")
            
        frame_count += 1
        
    cap.release()
    print(f"\n\n完成。从视频里提取了 {save_count} 张背景图到 {OUTPUT_DIR}。")
    print("下一步运行 resize_backgrounds.py，将背景统一裁剪缩放为 640x640。")

if __name__ == "__main__":
    main()
