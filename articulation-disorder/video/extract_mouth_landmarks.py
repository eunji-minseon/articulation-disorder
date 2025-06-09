import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 입술 관련 landmark index 
LIPS_IDX = sorted(set([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,     # 윗입술
    146, 91, 181, 84, 17, 314, 405, 321, 375             # 아랫입술
]))

def extract_mouth_landmarks(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    coords_all = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success_count = 0

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            mouth_coords = [
                (lm.x, lm.y)
                for i, lm in enumerate(landmarks.landmark)
                if i in LIPS_IDX
            ]
            coords_all.append(mouth_coords)
            success_count += 1

    cap.release()
    face_mesh.close()

    # 결과 저장
    with open(output_txt_path, "w") as f:
        for frame_coords in coords_all:
            f.write(str(frame_coords) + "\n")

    print(f"📽️ {video_path}: 총 {total_frames}프레임 중 {success_count}개에서 얼굴 인식 성공") 