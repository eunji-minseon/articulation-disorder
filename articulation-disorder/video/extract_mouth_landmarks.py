import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화 (개선된 설정)
mp_face_mesh = mp.solutions.face_mesh

# 입술 관련 landmark index 
LIPS_IDX = sorted(set([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,     # 윗입술
    146, 91, 181, 84, 17, 314, 405, 321, 375             # 아랫입술
]))

class CoordinateSmoothing:
    """좌표 스무딩을 위한 클래스"""
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_coords = None
    
    def smooth(self, coords):
        if self.prev_coords is None:
            self.prev_coords = coords
            return coords
        
        # 지수 이동 평균으로 스무딩
        smoothed = []
        for i, (x, y) in enumerate(coords):
            prev_x, prev_y = self.prev_coords[i]
            smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
            smooth_y = self.alpha * y + (1 - self.alpha) * prev_y
            smoothed.append((smooth_x, smooth_y))
        
        self.prev_coords = smoothed
        return smoothed

def enhance_frame_quality(frame):
    """프레임 품질 향상"""
    # 명도 조정
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def normalize_coordinates(coords, frame_width, frame_height):
    """좌표 정규화 (0-1 범위로)"""
    normalized = []
    for x, y in coords:
        norm_x = x / frame_width if frame_width > 0 else x
        norm_y = y / frame_height if frame_height > 0 else y
        normalized.append((norm_x, norm_y))
    return normalized

def extract_mouth_landmarks(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    # 개선된 FaceMesh 설정
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # 더 정확한 랜드마크
        min_detection_confidence=0.7,  # 검출 신뢰도 증가
        min_tracking_confidence=0.5    # 추적 신뢰도
    )

    # 스무딩 객체 초기화
    smoother = CoordinateSmoothing(alpha=0.7)
    
    coords_all = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success_count = 0
    
    # 프레임 크기 정보
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"🎬 영상 분석 시작: {video_path}")
    print(f"📏 프레임 크기: {frame_width}x{frame_height}")
    print(f"🎞️ 총 프레임 수: {total_frames}")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 품질 향상
        enhanced_frame = enhance_frame_quality(frame)
        frame_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # 입술 좌표 추출 (픽셀 좌표로 변환)
            mouth_coords = []
            for i in LIPS_IDX:
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    x = lm.x * frame_width
                    y = lm.y * frame_height
                    mouth_coords.append((x, y))
            
            if len(mouth_coords) == len(LIPS_IDX):
                # 좌표 스무딩 적용
                smoothed_coords = smoother.smooth(mouth_coords)
                
                # 정규화된 좌표 저장 (0-1 범위)
                normalized_coords = normalize_coordinates(smoothed_coords, frame_width, frame_height)
                coords_all.append(normalized_coords)
                success_count += 1
        else:
            # 얼굴 인식 실패 시 이전 프레임 좌표 사용 (있다면)
            if coords_all:
                coords_all.append(coords_all[-1])  # 마지막 성공한 좌표 재사용
            
        # 진행률 표시
        if (frame_idx + 1) % 100 == 0:
            progress = (frame_idx + 1) / total_frames * 100
            print(f"⏳ 진행률: {progress:.1f}% ({frame_idx + 1}/{total_frames})")

    cap.release()
    face_mesh.close()

    # 결과 저장
    with open(output_txt_path, "w", encoding='utf-8') as f:
        for frame_coords in coords_all:
            # 더 정확한 좌표 저장 (소수점 6자리)
            formatted_coords = [(round(x, 6), round(y, 6)) for x, y in frame_coords]
            f.write(str(formatted_coords) + "\n")

    success_rate = (success_count / total_frames) * 100 if total_frames > 0 else 0
    print(f"✅ 완료: {output_txt_path}")
    print(f"📊 성공률: {success_count}/{total_frames} ({success_rate:.1f}%)")
    print(f"💾 저장된 프레임 수: {len(coords_all)}")

# 사용 예시
if __name__ == "__main__":
    video_path = "your_video.mp4"
    output_path = "mouth_landmarks.txt"
    extract_mouth_landmarks(video_path, output_path)