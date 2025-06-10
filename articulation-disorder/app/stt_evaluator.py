# app/stt_evaluator.py
import whisper
from jiwer import wer

def calculate_stt_score(user_text, correct_text):
    user_text = user_text.lower().strip()
    correct_text = correct_text.lower().strip()

    try:
        word_error = wer(correct_text, user_text)
        accuracy = round((1 - word_error) * 100, 2)
        return accuracy
    except Exception as e:
        print("STT 점수 계산 오류:", e)
        return 0.0

def run_whisper_stt(video_path):
    model = whisper.load_model("base")  # 필요 시 "small", "medium"도 가능
    result = model.transcribe(video_path)
    return result["text"]
