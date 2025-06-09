# ✅ 완전히 정리한 streamlit_app.py (안정형 구조)

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import time
import numpy as np
import streamlit as st
import ast
from datetime import datetime
import pandas as pd
from video.extract_mouth_landmarks import extract_mouth_landmarks

# 경로 설정
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SCORE_LOG_PATH = "data/user_scores.csv"
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

selected_sentence = st.selectbox("진단할 문장을 선택하세요:", list(sentence_to_file.keys()))
phonemes = sentence_analysis[selected_sentence]
st.markdown(f"### \U0001F3AF 분석할 음소: `{', '.join(phonemes)}`")

file_prefix = sentence_to_file[selected_sentence]
ref_coords_path = os.path.join(PROCESSED_DIR, f"{file_prefix}_coords.txt")
user_video_path = os.path.join(RAW_DIR, "user_video.mp4")

@st.cache_data
def load_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f:
            try:
                coords.append(ast.literal_eval(line.strip()))
            except:
                continue
    return coords

ref_coords = load_coords(ref_coords_path)

user_file = st.file_uploader("\U0001F4C5 사용자 영상 업로드 (mp4, mov)", type=["mp4", "mpeg4", "mov"])

if user_file:
    with open(user_video_path, "wb") as f:
        f.write(user_file.read())
    st.video(user_video_path)

    if st.button("\U0001F680 분석 시작"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_coords_path = os.path.join(PROCESSED_DIR, f"user_coords_{timestamp}.txt")

        st.info("📍 사용자 영상 → 입모양 좌표 추출 중...")
        extract_mouth_landmarks(user_video_path, user_coords_path)
        user_coords = load_coords(user_coords_path)

        if not user_coords or not ref_coords:
            st.error("🚨 좌표 데이터가 비어있습니다.")
            st.stop()

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
                similarity_score = round(100 * np.exp(-6 * avg_dist), 1)
                similarities.append(similarity_score)
            except:
                continue

        similarity = round(sum(similarities) / len(similarities), 1) if similarities else 0.0

        st.markdown(f"### ✅ 유사도: `{similarity}%`")
        if similarity >= 85:
            st.success("발음이 매우 정확합니다! 😄")
        elif similarity >= 60:
            st.warning("조금 더 연습이 필요해요. 🙂")
        else:
            st.error("입모양이 많이 다르네요. 연습이 필요해요. 🤭")

        # 점수 저장
        result_row = pd.DataFrame([{
            "timestamp": timestamp,
            "sentence": selected_sentence,
            "similarity": similarity
        }])

        if os.path.exists(SCORE_LOG_PATH):
            score_df = pd.read_csv(SCORE_LOG_PATH)
            score_df = pd.concat([score_df, result_row], ignore_index=True)
        else:
            score_df = result_row

        score_df.to_csv(SCORE_LOG_PATH, index=False)

# 기록 출력
if os.path.exists(SCORE_LOG_PATH):
    score_df = pd.read_csv(SCORE_LOG_PATH)
else:
    score_df = pd.DataFrame()

st.markdown("---")
st.markdown("### 🗂️ 이전 분석 기록")
st.dataframe(score_df.sort_values("timestamp", ascending=False).reset_index(drop=True))

score_df.to_csv(SCORE_LOG_PATH, index=False)
print(f"✅ 점수 저장됨: {SCORE_LOG_PATH}")
