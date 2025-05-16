import os
import streamlit as st
from utils.webcam_capture import capture_video
from video.extract_mouth_landmarks import extract_mouth_landmarks
from analysis.compare_shapes import calculate_similarity
import ast

# 폴더 경로 설정
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

st.title("🗣️ 조음장애 분석 시스템")
st.markdown("두 개의 영상을 업로드하거나 직접 촬영하여 조음 정확도를 분석할 수 있습니다.")

# 사용자 입력 방식 선택
method = st.radio("영상 입력 방식 선택", ["업로드", "촬영"])

user_video_path = None
ref_video_path = os.path.join(RAW_DIR, "normal_video.mp4")  # 기준 영상은 고정된 파일 사용

# 영상 입력 처리
if method == "업로드":
    user_file = st.file_uploader("📥 사용자 영상 업로드 (mp4)", type=["mp4"], key="user")
    if user_file:
        user_video_path = os.path.join(RAW_DIR, "user_video.mp4")
        with open(user_video_path, "wb") as f:
            f.write(user_file.read())
        st.video(user_video_path)

elif method == "촬영":
    if st.button("📸 사용자 영상 촬영"):
        user_video_path = os.path.join(RAW_DIR, "user_video.mp4")
        capture_video(user_video_path)
        st.video(user_video_path)

# 좌표 파일 경로
user_coords_path = os.path.join(PROCESSED_DIR, "user_coords.txt")
ref_coords_path = os.path.join(PROCESSED_DIR, "normal_coords.txt")

# 유사도 분석 실행
if user_video_path and st.button("🚀 분석 시작"):
    st.info("입모양 좌표 추출 중...")
    extract_mouth_landmarks(user_video_path, user_coords_path)
    extract_mouth_landmarks(ref_video_path, ref_coords_path)  # 기준 영상도 항상 처리

    def load_coords(path):
        coords = []
        with open(path, "r") as f:
            for line in f:
                try:
                    coords.append(ast.literal_eval(line.strip()))
                except:
                    continue
        return coords

    user_coords = load_coords(user_coords_path)
    ref_coords = load_coords(ref_coords_path)

    if not user_coords or not ref_coords:
        st.error("🚨 좌표 정보가 비어있습니다. 다시 시도해주세요.")
        st.stop()

    similarity = calculate_similarity(user_coords[0], ref_coords[0])
    st.markdown(f"### ✅ 정상 발음 유사도: `{similarity}%`")

    if similarity >= 85:
        st.success("발음이 매우 정확합니다! 😄")
    elif similarity >= 60:
        st.warning("조금 더 연습이 필요합니다. 🙂")
    else:
        st.error("입모양이 많이 다릅니다. 다시 확인해보세요. 😢")
