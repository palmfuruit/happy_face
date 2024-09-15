import cv2
from fer import FER

# FERの感情分析モデルを初期化
emotion_detector = FER()

def analyze_emotions(aligned_face_cropped):
    # 感情分析を実行
    emotions = emotion_detector.detect_emotions(aligned_face_cropped)
    if emotions:
        emotion_scores = emotions[0]['emotions']  # 最初の顔の感情スコアを取得
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        top_score = int(emotion_scores[top_emotion] * 100)  # 0-100に変換
        return top_emotion, top_score
    return None, None