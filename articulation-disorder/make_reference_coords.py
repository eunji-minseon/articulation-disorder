import os
from video.extract_mouth_landmarks import extract_mouth_landmarks

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

for i in range(1, 10):
    video_path = os.path.join(RAW_DIR, f"normal{i}.mp4")
    coords_path = os.path.join(PROCESSED_DIR, f"normal{i}_coords.txt")

    if os.path.exists(video_path):
        print(f"🎥 Extracting from {video_path} → {coords_path}")
        extract_mouth_landmarks(video_path, coords_path)
    else:
        print(f"⚠️ 영상 파일 없음: {video_path}")
