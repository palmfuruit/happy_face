import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForImageClassification
import torch.nn.functional as F


from transformers import AutoModel, AutoTokenizer

# https://huggingface.co/motheecreator/vit-Facial-Expression-Recognition
model_name = "motheecreator/vit-Facial-Expression-Recognition"
model = AutoModelForImageClassification.from_pretrained(model_name)


# クラスラベルの定義（FERライブラリのクラスラベルに合わせる）
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_input(face_image):
    # 224x224にリサイズ
    face_image = cv2.resize(face_image, (224, 224))
    face_image = face_image.astype('float32') / 255.0  # 0-1にスケーリング
    
    # グレースケール画像なら、3チャンネルに複製
    if face_image.shape[-1] == 1:
        face_image = np.repeat(face_image, 3, axis=-1)  # 3チャンネルに複製
    
    # 正規化: 平均0.5、標準偏差0.5に基づいて正規化
    face_image = (face_image - 0.5) / 0.5
    
    # PyTorch用に [batch_size, channels, height, width] へ変換
    face_image = np.transpose(face_image, (2, 0, 1))  # チャンネルを先頭に移動
    face_image = np.expand_dims(face_image, axis=0)  # バッチ次元を追加
    
    # NumPyからPyTorchテンソルへ変換
    face_image = torch.tensor(face_image).to(torch.float32)
    
    return face_image


def analyze_emotions(aligned_face_cropped):
    # 画像を前処理
    processed_face = preprocess_input(aligned_face_cropped)
    
    # デバイスの設定（CPUまたはGPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_face = processed_face.to(device)
    model.to(device)
    
    # モデルで推論を実行
    with torch.no_grad():  # 勾配を計算しないようにする
        predictions = model(processed_face)
    
    # 感情スコアを取得し、softmaxで確率に変換
    emotion_scores = predictions.logits[0]  # logitsを使用
    emotion_probs = F.softmax(emotion_scores, dim=0)  # softmaxを適用して合計が1になるようにする

    # 全ての感情スコア（確率）を表示
    for i, prob in enumerate(emotion_probs):
        emotion = emotion_labels[i]
        prob_percentage = prob.item() * 100  # 確率を0-100%に変換
        print(f"{emotion}: {prob_percentage:.2f}%")
    
    # 最も高い確率の感情を取得
    top_emotion_index = torch.argmax(emotion_probs).item()
    top_emotion = emotion_labels[top_emotion_index]
    top_prob = round(emotion_probs[top_emotion_index].item() * 100)
    
    return top_emotion, top_prob