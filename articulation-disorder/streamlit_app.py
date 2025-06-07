import os
import time
import numpy as np
import streamlit as st
import ast
from video.extract_mouth_landmarks import extract_mouth_landmarks

# 경로 설정
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 문장 → 기준 좌표 파일 prefix
sentence_to_file = {
    "강아지가 짖고 있어요": "normal1",
    "토끼가 풀을 먹어요": "normal2",
    "코끼리는 코가 길어요": "normal3",
    "창문을 열어 주세요": "normal4",
    "고양이가 야옹해요": "normal5",
    "무지개가 떴어요": "normal6",
    "자동차가 달려요": "normal7",
    "별이 반짝반짝 빛나요": "normal8",
    "나는 사과를 좋아해요": "normal9",
    "오늘은 기분이 좋아요": "normal10"
}

# 문장 → 분석 음소
sentence_analysis = {
    "강아지가 짖고 있어요": ["ㄱ", "ㅇ", "ㅈ", "ㅆ"],
    "토끼가 풀을 먹어요": ["ㅌ", "ㄲ", "ㅍ", "ㄹ"],
    "코끼리는 코가 길어요": ["ㅋ", "ㄲ", "ㄹ"],
    "창문을 열어 주세요": ["ㅊ", "ㅁ", "ㅈ", "ㅅ"],
    "고양이가 야옹해요": ["ㄱ", "ㅇ", "ㅑ", "ㅎ"],
    "무지개가 떴어요": ["ㅁ", "ㅈ", "ㄱ", "ㄸ"],
    "자동차가 달려요": ["ㅈ", "ㄷ", "ㅊ", "ㄹ"],
    "별이 반짝반짝 빛나요": ["ㅂ", "ㅈ", "ㄴ", "ㅉ"],
    "나는 사과를 좋아해요": ["ㄴ", "ㅅ", "ㄱ", "ㅘ", "ㅈ"],
    "오늘은 기분이 좋아요": ["ㅇ", "ㄴ", "ㅈ", "ㅗ"]
}

st.title("\U0001F5E3️ 조음장애 진단 시스템")

if 'selected_sentence' not in st.session_state:
    st.session_state.selected_sentence = list(sentence_to_file.keys())[0]

selected_sentence = st.selectbox(
    "진단할 문장을 선택하세요:",
    list(sentence_to_file.keys()),
    index=list(sentence_to_file.keys()).index(st.session_state.selected_sentence)
)

st.session_state.selected_sentence = selected_sentence
phonemes = sentence_analysis[selected_sentence]
st.markdown(f"### \U0001F3AF 분석할 음소: `{', '.join(phonemes)}`")

file_prefix = sentence_to_file[selected_sentence]
ref_coords_path = os.path.join(PROCESSED_DIR, f"{file_prefix}_coords.txt")
user_video_path = os.path.join(RAW_DIR, "user_video.mp4")
user_coords_path = os.path.join(PROCESSED_DIR, "user_coords.txt")

user_file = st.file_uploader("\U0001F4C5 사용자 영상 업로드 (mp4)", type=["mp4"])

if user_file:
    with open(user_video_path, "wb") as f:
        f.write(user_file.read())
    st.video(user_video_path)

    if st.button("\U0001F680 분석 시작"):
        if not os.path.exists(ref_coords_path):
            st.error(f"❌ 기준 좌표 파일이 없습니다: {ref_coords_path}")
            st.stop()

        st.info("📍 사용자 영상 → 입모양 좌표 추출 중...")
        extract_mouth_landmarks(user_video_path, user_coords_path)

        timeout = 5
        elapsed = 0
        while (not os.path.exists(user_coords_path) or os.path.getsize(user_coords_path) == 0) and elapsed < timeout:
            time.sleep(0.2)
            elapsed += 0.2

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
            st.error("🚨 좌표 데이터가 비어있습니다.")
            st.stop()

        # ✅ 좌표 차이 통계 확인 (프레임 0 기준)
        try:
            u0 = np.array(user_coords[0])
            r0 = np.array(ref_coords[0])
            cut_len = min(len(u0), len(r0))
            u0 = u0[:cut_len].flatten()
            r0 = r0[:cut_len].flatten()

            diff_vector = np.abs(u0 - r0)
            mean_diff = np.mean(diff_vector)
            max_diff = np.max(diff_vector)
            min_diff = np.min(diff_vector)

            st.markdown("### 📊 첫 프레임 좌표 차이 분석")
            st.text(f"차이 평균: {mean_diff:.6f}")
            st.text(f"차이 최대: {max_diff:.6f}")
            st.text(f"차이 최소: {min_diff:.6f}")
        except Exception as e:
            st.warning(f"좌표 차이 분석 실패: {e}")

        # 프레임별 유사도 계산
        similarities = []
        min_len = min(len(user_coords), len(ref_coords))
        warned = False

        for i in range(min_len):
            c1 = user_coords[i]
            c2 = ref_coords[i]

            if len(c1) != len(c2):
                if not warned:
                    st.warning(f"⚠️ 좌표 개수 다름 (예시 프레임): 사용자 {len(c1)} vs 기준 {len(c2)}")
                    warned = True
                cut_len = min(len(c1), len(c2))
                c1 = c1[:cut_len]
                c2 = c2[:cut_len]

            try:
                c1_np = np.array(c1)
                c2_np = np.array(c2)
                distances = np.linalg.norm(c1_np - c2_np, axis=1)
                avg_dist = np.mean(distances)
                similarity_score = max(0.0, 100 - avg_dist * 300)  # 감도 조정 가능
                similarities.append(similarity_score)
            except Exception as e:
                st.warning(f"좌표 차이 분석 실패 (프레임 {i}): {e}")
                continue

        similarity = round(sum(similarities) / len(similarities), 1) if similarities else 0.0

        st.markdown(f"### ✅ 유사도: `{similarity}%`")
        if similarity >= 85:
            st.success("발음이 매우 정확합니다! 😄")
        elif similarity >= 60:
            st.warning("조금 더 연습이 필요해요. 🙂")
        else:
            st.error("입모양이 많이 다르네요. 연습이 필요해요. 🤭")
