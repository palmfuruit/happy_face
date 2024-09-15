# 必要なモジュールのインポート
import torch
from flask import Flask, request, render_template, redirect 
import io
from PIL import Image
import base64
import numpy as np
import cv2
from mtcnn import MTCNN  # 顔認識
from fer import FER  # 感情分析



# Flask のインスタンスを作成
app = Flask(__name__)
app.secret_key = 'secret_key'

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# アスペクト比を固定して、幅が指定した値になるようリサイズする。
def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))


# MTCNNの顔検出モデルを初期化（閾値を調整）
detector = MTCNN(steps_threshold=[0.5, 0.6, 0.7])  # ここで閾値を調整

# MTCNNの顔検出モデルを初期化
detector = MTCNN()

# FERの感情分析モデルを初期化
emotion_detector = FER()


# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')

            # リサイズ
            if image.width > 512:
                image = scale_to_width(image, 512)

            # PIL画像をOpenCV用のnumpy配列に変換
            image_cv = np.array(image)

            # 顔検出を実行
            faces = detector.detect_faces(image_cv)

        # 検出された顔に枠を描画し、感情分析を実行
        for face in faces:
            # 顔の位置を取得
            x, y, width, height = face['box']

            # 30%上下左右に拡大
            x_offset = int(width * 0.3)
            y_offset = int(height * 0.3)

            # 枠を拡大（画像の範囲を超えないように調整）
            x = max(0, x - x_offset)
            y = max(0, y - y_offset)
            width = min(image_cv.shape[1] - x, width + 2 * x_offset)
            height = min(image_cv.shape[0] - y, height + 2 * y_offset)

            # 顔のランドマークを取得（目、鼻、口など）
            keypoints = face['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']

            # 目の位置を使って画像を正面向きに補正
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(delta_y, delta_x))

            # 回転行列を計算し、顔を補正
            center_of_eyes = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center_of_eyes, angle, 1)
            aligned_face = cv2.warpAffine(image_cv, rotation_matrix, (image_cv.shape[1], image_cv.shape[0]))

            # 顔の部分を切り出し（拡大後の枠）
            aligned_face_cropped = aligned_face[y:y + height, x:x + width]

            # 感情分析を実行
            emotions = emotion_detector.detect_emotions(aligned_face_cropped)

            if emotions:
                emotion_scores = emotions[0]['emotions']  # 最初の顔の感情スコアを取得
                print(f'顔の感情スコア: {emotion_scores}')  # それぞれの感情スコアを表示

                # 最も高い感情を取得
                top_emotion = max(emotion_scores, key=emotion_scores.get)
                top_score = int(emotion_scores[top_emotion] * 100)  # 0-100に変換

                # 拡大した枠に描画
                cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # 最も確率の高い感情名とスコアを枠の上に表示
                cv2.putText(image_cv, f'{top_emotion} ({top_score})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            else:
                # 拡大した枠に描画
                cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(image_cv, 'No emotion detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # OpenCV画像をエンコードしてBase64に変換
        _, buffer = cv2.imencode('.png', image_cv)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_data = f"data:image/png;base64,{image_base64}"

        return render_template('result.html', num_faces=len(faces), image=base64_data)
    
    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
