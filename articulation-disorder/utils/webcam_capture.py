import cv2
import os

def capture_video(output_path="data/raw/webcam_capture.mp4", duration=5, fps=20):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(duration * fps)
    print(f"🔴 {duration}초 동안 영상 촬영 중...")

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 저장 완료: {output_path}")
