import os
from gtts import gTTS
import base64
import whisper
from difflib import SequenceMatcher
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "articulation-disorder"))

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import time
import numpy as np
import streamlit as st
import ast
from datetime import datetime
import pandas as pd
from video.extract_mouth_landmarks import extract_mouth_landmarks

def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='ko')
    tts.save(filename)
    return filename

def get_stt_text(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path, language='ko')
    return result["text"]

def compare_texts(ref_text, stt_text):
    ratio = SequenceMatcher(None, ref_text, stt_text).ratio()
    return round(ratio * 100, 1)

def normalize_coordinates(coords):
    #좌표 정규화
    if not coords:
        return coords
    
    coords_array = np.array(coords)
    
    # 중심점 계산 (얼굴 위치 보정)
    center = np.mean(coords_array, axis=0)
    centered_coords = coords_array - center
    
    # 스케일 정규화 (얼굴 크기 보정)
    scale = np.std(centered_coords)
    if scale > 0:
        normalized_coords = centered_coords / scale
    else:
        normalized_coords = centered_coords
    
    return normalized_coords.tolist()

def calculate_improved_similarity(user_coords, ref_coords):
    #거리 기반 계산 유사도
    similarities = []
    min_len = min(len(user_coords), len(ref_coords))

    for i in range(min_len):
        c1 = user_coords[i]
        c2 = ref_coords[i]

        if len(c1) != len(c2):
            cut_len = min(len(c1), len(c2))
            c1 = c1[:cut_len]
            c2 = c2[:cut_len]

        try:
            c1_np = np.array(c1)
            c2_np = np.array(c2)

            if np.allclose(c1_np, c2_np, atol=1e-6):
                print(f"✅ Frame {i}: 좌표 완전 일치 (유사도 100%)")
            else:
                print(f"❌ Frame {i}: 좌표 다름")
                print("차이:", np.abs(c1_np - c2_np).max())

            # 좌표가 거의 완전히 같으면 유사도 100%
            if np.allclose(c1_np, c2_np, atol=1e-6):
                similarity_score = 100.0
            else:
                distances = np.linalg.norm(c1_np - c2_np, axis=1)
                avg_dist = np.mean(distances)
                similarity_score = round(100 * np.exp(-2 * avg_dist), 1)

            similarities.append(similarity_score)

        except Exception as e:
            print(f"Error at frame {i}: {e}")
            continue

    return round(sum(similarities) / len(similarities), 1) if similarities else 0.0

if 'user_id' not in st.session_state:
    st.session_state.user_id = ""

st.sidebar.markdown("## 🔐 사용자 로그인")
nickname_or_email = st.sidebar.text_input("닉네임 또는 이메일을 입력하세요.", value=st.session_state.user_id)

if nickname_or_email:
    st.session_state.user_id = nickname_or_email
    st.sidebar.success(f"✅ {nickname_or_email} 님으로 로그인됨")

user_id = st.session_state.user_id
if not user_id:
    st.error("로그인이 필요합니다. 왼쪽 사이드바에서 닉네임 또는 이메일을 입력해주세요.")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SCORE_LOG_PATH = os.path.join(BASE_DIR, "data", "user_scores.csv")

os.makedirs("data", exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

sentence_to_file = {
    "그대로 멈춰라": "normal1",
    "꼭두각시 인형 피노키오": "normal2",
    "코끼리 아저씨는 코가 손이래": "normal3",
    "또 만나요 뽀뽀뽀": "normal4",
    "반짝반짝 작은 별": "normal5",
    "산토끼 토끼야": "normal6",
    "시계는 아침부터 똑딱똑딱": "normal7",
    "쌩쌩 불어도 괜찮아요": "normal8",
    "예쁜 아기 곰": "normal9",
}

sentence_analysis = {
    "그대로 멈춰라": ["ㅊ", "ㅝ"],
    "꼭두각시 인형 피노키오": ["ㄲ", "ㅍ", "ㅋ"],
    "코끼리 아저씨는 코가 손이래": ["ㅋ", "ㄲ", "ㅆ"],
    "또 만나요 뽀뽀뽀": ["ㄸ", "ㅛ", "ㅃ"],
    "반짝반짝 작은 별": ["ㅉ", "ㅕ"],
    "산토끼 토끼야": ["ㅌ", "ㄲ", "ㅑ"],
    "시계는 아침부터 똑딱똑딱": ["ㄸ", "ㅊ", "ㅖ"],
    "쌩쌩 불어도 괜찮아요": ["ㅆ", "ㅊ", "ㅙ"],
    "예쁜 아기 곰": ["ㅖ", "ㅃ"],
}

st.title("\U0001F5E3️ 조음장애 진단 시스템")

selected_sentence = st.selectbox("진단할 문장을 선택하세요.:", list(sentence_to_file.keys()))
phonemes = sentence_analysis[selected_sentence]
st.markdown(f"### ✅ 유의할 음소: `{', '.join(phonemes)}`")
st.info("👉 해당 음소들을 집중해서 발음하면 더 좋은 결과를 얻을 수 있어요!")


file_prefix = sentence_to_file[selected_sentence]
ref_coords_path = os.path.join(PROCESSED_DIR, f"{file_prefix}_coords.txt")
user_video_path = os.path.join(RAW_DIR, "user_video.mp4")
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

user_file = st.file_uploader("📹 사용자 영상 업로드 (mp4, mov)", type=["mp4", "mpeg4", "mov"])

if user_file:
    with open(user_video_path, "wb") as f:
        f.write(user_file.read())
    st.video(user_video_path)

    if st.button("🚀 분석 시작"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_coords_path = os.path.join(PROCESSED_DIR, f"user_coords_{timestamp}.txt")

        with st.spinner(" 사용자 영상 → 입모양 좌표 추출 중입니다..."):
            extract_mouth_landmarks(user_video_path, user_coords_path)

        user_coords = load_coords(user_coords_path)

        if not user_coords or not ref_coords:
            st.error("🚨 좌표 데이터가 비어있습니다.")
            st.stop()
        # 좌표 불러온 후 정규화
        user_coords = load_coords(user_coords_path)
        ref_coords = load_coords(ref_coords_path)
        print("ref shape:", np.array(ref_coords).shape)
        print("user shape:", np.array(user_coords).shape)

        similarity = calculate_improved_similarity(user_coords, ref_coords)

        # 👄 입모양 유사도
        st.markdown(f"#### ✓ 조음 정확도: `{similarity}%`")

        # 🧠 STT 기반 발화 유사도
        with st.spinner("🎙️ 사용자의 실제 발화 내용을 인식 중입니다..."):
            try:
                stt_result = get_stt_text(user_video_path)
                text_similarity = compare_texts(selected_sentence, stt_result)
                st.markdown(f"#### ✓ 발화 정확도: `{text_similarity}%`")

                st.markdown(f"#### ✓ 인식된 음성 결과: `{stt_result}`")

                with st.expander("📊 진단 결과", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("조음 정확도", f"{similarity}%")

                    with col2:
                        st.metric("발화 정확도", f"{text_similarity}%")

                    st.markdown("#### 💬 종합 피드백")
                    if similarity >= 70 and text_similarity >= 80:
                        st.success("발음과 내용 모두 아주 정확합니다! 😎")
                    elif similarity >= 50 and text_similarity >= 60:
                        st.warning("전반적으로 괜찮지만, 조음이나 발화 중 일부가 부족해요.")
                    else:
                        st.error("입모양과 발화 모두 연습이 필요해요. 다시 시도해보세요.")
            
            except Exception as e:
                st.error(f"🚨 STT 분석 중 오류 발생: {e}")

        # 문장만 읽기
        sentence_text = f"{selected_sentence}"
        tts_path = text_to_speech(sentence_text, "sentence_only.mp3")

        with open(tts_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True,
            )

        if similarity is not None and timestamp is not None:
            st.markdown(f"📌 최근 점수: {similarity}% ({timestamp})")    

        result_row = pd.DataFrame([{
            "user_id": str(user_id),
            "timestamp": timestamp,
            "sentence": selected_sentence,
            "similarity": similarity
        }])

        if os.path.exists(SCORE_LOG_PATH) and os.path.getsize(SCORE_LOG_PATH) > 0:
            score_df = pd.read_csv(SCORE_LOG_PATH)
            score_df = pd.concat([score_df, result_row], ignore_index=True)
        else:
            score_df = result_row

        score_df.to_csv(SCORE_LOG_PATH, index=False)
        st.success("📈 분석 결과 저장 완료!")

if os.path.exists(SCORE_LOG_PATH) and os.path.getsize(SCORE_LOG_PATH) > 0:
    score_df = pd.read_csv(SCORE_LOG_PATH)
else:
    score_df = pd.DataFrame(columns=["user_id", "timestamp", "sentence", "similarity"])

st.markdown("---")
st.markdown("### 🗂️ 내 분석 기록")

user_history = score_df[score_df["user_id"] == user_id] if "user_id" in score_df.columns else pd.DataFrame()
try:
    st.dataframe(user_history.sort_values("timestamp", ascending=False).reset_index(drop=True))
except KeyError:
    st.warning("❌ 'timestamp' 열이 없어서 정렬할 수 없습니다.")
    st.dataframe(user_history)

if st.button("🗑️ 기존 기록 완전 삭제"):
    try:
        os.remove(SCORE_LOG_PATH)
        st.success("✅ 기존 기록 삭제 완료! 앱을 다시 실행해주세요.")
    except:
        st.warning("❌ 삭제할 파일이 없거나 이미 삭제되었습니다.")