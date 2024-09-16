import io
import base64
from flask import Flask, request, render_template, redirect
from PIL import Image
import numpy as np
import cv2

from .face_detection import detect_faces, align_face
from .emotion_analysis_nolib import analyze_emotions

# Flask のインスタンスを作成
app = Flask(__name__)
app.secret_key = 'secret_key'

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# FERライブラリに対応した感情ごとの枠と文字の色を定義
emotion_colors = {
    'angry': {'box_color': (0, 0, 255), 'text_color': (255, 255, 255)},         # 赤 (文字: 白)
    'disgust': {'box_color': (0, 128, 0), 'text_color': (255, 255, 255)},       # 深緑 (文字: 白)
    'fear': {'box_color': (128, 0, 128), 'text_color': (255, 255, 255)},        # 紫 (文字: 白)
    'happy': {'box_color': (0, 255, 255), 'text_color': (0, 0, 0)},             # 黄色 (文字: 黒)
    'sad': {'box_color': (255, 0, 0), 'text_color': (255, 255, 255)},           # 青 (文字: 白)
    'surprise': {'box_color': (0, 165, 255), 'text_color': (0, 0, 0)},          # オレンジ (文字: 黒)
    'neutral': {'box_color': (128, 128, 128), 'text_color': (255, 255, 255)}    # グレー (文字: 白)
}

#　拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# アスペクト比を固定して、幅が指定した値になるようリサイズする。
def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods=['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        if 'filename' not in request.files:
            return redirect(request.url)

        file = request.files['filename']
        if file and allowed_file(file.filename):
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')

            if image.width > 512:
                image = scale_to_width(image, 512)

            image_cv_rgb = np.array(image)
            image_cv_bgr = cv2.cvtColor(image_cv_rgb, cv2.COLOR_RGB2BGR)

            # 顔検出を実行
            faces = detect_faces(image_cv_rgb)

###

            # 感情ごとの枠と文字の色を定義（BGR形式）
            emotion_colors = {
                'angry': {'box_color': (0, 0, 255), 'text_color': (255, 255, 255)},         # 赤 (文字: 白)
                'disgust': {'box_color': (0, 128, 0), 'text_color': (255, 255, 255)},       # 深緑 (文字: 白)
                'fear': {'box_color': (128, 0, 128), 'text_color': (255, 255, 255)},        # 紫 (文字: 白)
                'happy': {'box_color': (0, 255, 255), 'text_color': (0, 0, 0)},             # 黄色 (文字: 黒)
                'sad': {'box_color': (255, 0, 0), 'text_color': (255, 255, 255)},           # 青 (文字: 白)
                'surprise': {'box_color': (0, 165, 255), 'text_color': (0, 0, 0)},          # オレンジ (文字: 黒)
                'neutral': {'box_color': (128, 128, 128), 'text_color': (255, 255, 255)}    # グレー (文字: 白)
            }

            # すべての顔の感情スコアを調べ、感情ごとに最大スコアを取得する
            max_scores = {}

            # まず、最高スコアを探すフェーズ
            for face in faces:
                aligned_face_cropped, x, y, width, height = align_face(image_cv_rgb, face)
                
                # 感情分析を実行
                top_emotion, top_score = analyze_emotions(aligned_face_cropped)

                if top_emotion:
                    top_emotion_lower = top_emotion.lower()
                    # その感情の最大スコアを保持
                    if top_emotion_lower not in max_scores or max_scores[top_emotion_lower] < top_score:
                        max_scores[top_emotion_lower] = top_score

            # 次に、顔を表示するフェーズ
            for face in faces:
                aligned_face_cropped, x, y, width, height = align_face(image_cv_rgb, face)

                # 感情分析を実行
                top_emotion, top_score = analyze_emotions(aligned_face_cropped)

                # デフォルトの色設定（感情が認識できない場合）
                box_color = (0, 255, 0)  # 緑の枠
                text_color = (0, 180, 0)  # 落ち着いた緑の文字
                box_thickness = 2  # デフォルトの枠の太さ
                font_thickness = 1  # デフォルトの文字の太さ

                # 感情に基づいて色と枠の太さを設定
                if top_emotion and top_emotion.lower() in emotion_colors:
                    box_color = emotion_colors[top_emotion.lower()]['box_color']
                    text_color = emotion_colors[top_emotion.lower()]['text_color']
                    
                    # 同じ感情の中で一番スコアが高ければ枠と文字を太くする
                    if top_emotion.lower() in max_scores and max_scores[top_emotion.lower()] == top_score:
                        box_thickness = 4  # 一番高いスコアの場合、枠を太くする
                        font_thickness = 2  # 一番高いスコアの場合、文字も太くする

                # 表示するテキストの位置を枠の内側に設定
                text_position_y = y + 20  # 枠の内側の上部に設定
                font_scale = 1.5  # 文字を少し大きく

                # フォントを細くするためにHERSHEY_PLAINを使用
                font = cv2.FONT_HERSHEY_PLAIN

                # 文字のサイズを計算
                (text_width, text_height), baseline = cv2.getTextSize(f'{top_emotion} ({top_score})', 
                                                                    font, 
                                                                    font_scale, font_thickness)
                
                # 文字の背景を塗りつぶす（枠の色で）
                cv2.rectangle(image_cv_bgr, (x, y), (x + text_width + 10, y + text_height + baseline), box_color, -1)

                # スコアに基づいて感情ラベルを表示
                if top_emotion and top_score >= 50:
                    cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), box_color, box_thickness)  # 感情ごとの枠色と太さ
                    cv2.putText(image_cv_bgr, f'{top_emotion} ({top_score})', (x + 5, y + text_height), 
                                font, font_scale, text_color, font_thickness)  # 感情ごとの文字色と太さ
                else:
                    cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), box_color, box_thickness)  # デフォルトの枠色と太さ
                    cv2.putText(image_cv_bgr, 'No emotion detected', (x + 5, y + text_height), 
                                font, font_scale, text_color, font_thickness)  # デフォルトの文字色と太さ



####
            _, buffer = cv2.imencode('.png', image_cv_bgr)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_data = f"data:image/png;base64,{image_base64}"

            return render_template('result.html', num_faces=len(faces), image=base64_data)

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
