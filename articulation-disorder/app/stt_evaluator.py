# app/stt_evaluator.py
from jiwer import wer, cer

def calculate_stt_score(user_text, correct_text):

    user_text = user_text.lower().strip()
    correct_text = correct_text.lower().strip()

    try:
        word_error = wer(correct_text, user_text)  # 0.0 ~ 1.0
        accuracy = round((1 - word_error) * 100, 2)
        return accuracy
    except Exception as e:
        print("STT 점수 계산 중 오류:", e)
        return 0.0

import whisper

def run_whisper_stt(video_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        print("Whisper STT 처리 중 오류:", e)
        return ""
