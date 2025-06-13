import cv2
import mediapipe as mp
import numpy as np

# 입술 landmark index
LIPS_IDX = sorted(set([
    0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185,
    267, 269, 270, 291, 314, 321, 375, 405, 409  # 상하 입술 포함
]))

# 프레임 명도 조정 (LAB + CLAHE)
def enhance_frame_quality(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# 좌표 정규화 (0~1 범위)
def normalize_coordinates(coords, frame_width=None, frame_height=None):
    coords_array = np.array(coords)
    min_xy = np.min(coords_array, axis=0)
    max_xy = np.max(coords_array, axis=0)
    box_size = max_xy - min_xy
    box_size[box_size == 0] = 1e-6  # division by zero 방지
    normalized_coords = (coords_array - min_xy) / box_size
    return normalized_coords.tolist()

# 입모양 좌표 추출 함수
def extract_mouth_landmarks(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"❌ 영상 프레임 없음: {video_path}")
        return 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    coords_all = []
    success_count = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 프레임 {i} 읽기 실패")
            continue

        # 프레임 품질 향상
        enhanced_frame = enhance_frame_quality(frame)
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 추출
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            mouth_coords = []
            for idx in LIPS_IDX:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x = lm.x * frame_width
                    y = lm.y * frame_height
                    mouth_coords.append((x, y))

            if len(mouth_coords) == len(LIPS_IDX):
                norm_coords = normalize_coordinates(mouth_coords)
                coords_all.append(norm_coords)
                success_count += 1

        elif coords_all:
            coords_all.append(coords_all[-1])  # 마지막 성공 프레임 복사

    cap.release()
    face_mesh.close()

    if success_count == 0:
        print("❌ 얼굴 인식된 프레임 없음 → 좌표 추출 실패")
        return 0

    # 좌표 저장
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for coords in coords_all:
            f.write(str([(round(x, 6), round(y, 6)) for x, y in coords]) + "\n")

    print(f"✅ 좌표 추출 완료: {success_count}/{total_frames} 프레임")
    return (success_count / total_frames) * 100
