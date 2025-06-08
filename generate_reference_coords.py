# 사용자 영상 → 좌표 추출
from articulation_disorder.video.extract_mouth_landmarks import extract_mouth_landmarks

user_video_path = "data/raw/user_video.mp4"
output_path = "data/processed/user_coords.txt"

print(f"🎞️ {user_video_path} → {output_path} 변환 중...")
extract_mouth_landmarks(user_video_path, output_path)
