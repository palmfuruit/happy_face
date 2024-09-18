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


# アップロードされた画像を処理し、OpenCVの形式に変換する
def process_image(file):
    image = Image.open(file).convert('RGB')
    if image.width > 512:
        image = scale_to_width(image, 512)
    image_cv_rgb = np.array(image)
    image_cv_bgr = cv2.cvtColor(image_cv_rgb, cv2.COLOR_RGB2BGR)
    return image_cv_rgb, image_cv_bgr

# 感情スコアを解析し、最大スコアを保持する
def get_max_emotion_scores(faces, image_cv_bgr):
    max_scores = {}
    for face in faces:
        aligned_face_cropped, _, _, _, _ = align_face(image_cv_bgr, face)
        top_emotion, top_score = analyze_emotions(aligned_face_cropped)
        if top_emotion:
            top_emotion_lower = top_emotion.lower()
            if top_emotion_lower not in max_scores or max_scores[top_emotion_lower] < top_score:
                max_scores[top_emotion_lower] = top_score
    return max_scores

# 顔に感情の枠とラベルを描画する
def draw_emotion_boxes(faces, image_cv_bgr, max_scores):
    for face in faces:
        aligned_face_cropped, x, y, width, height = align_face(image_cv_bgr, face)
        top_emotion, top_score = analyze_emotions(aligned_face_cropped)
        box_color, text_color, box_thickness, font_thickness = get_emotion_box_colors(top_emotion, top_score, max_scores)

        # Draw the bounding box around the face first
        cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), box_color, box_thickness)

        # Font settings
        font_scale = 1.5
        font = cv2.FONT_HERSHEY_PLAIN

        # Calculate position and size for emotion text
        emotion_text = f'{top_emotion}'
        (text_width, text_height), baseline = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
        emotion_y_position = y + text_height + 5
        
        # Draw background for emotion text
        cv2.rectangle(image_cv_bgr, (x, y), (x + text_width + 10, y + text_height + baseline), box_color, -1)
        # Draw emotion text on top of the background
        cv2.putText(image_cv_bgr, emotion_text, (x + 5, emotion_y_position), font, font_scale, text_color, font_thickness)

        # Calculate position and size for score text
        score_text = f'Score: {round(top_score * 100)}'
        score_font_scale = 1.0
        score_font_thickness = 1
        (score_text_width, score_text_height), _ = cv2.getTextSize(score_text, font, score_font_scale, score_font_thickness)
        score_y_position = y + height - 5
        
        # Draw background for score text
        cv2.rectangle(image_cv_bgr, (x, score_y_position - score_text_height), 
                      (x + score_text_width + 10, score_y_position + baseline), box_color, -1)
        # Draw score text on top of the background
        cv2.putText(image_cv_bgr, score_text, (x + 5, score_y_position), font, score_font_scale, text_color, score_font_thickness)


# 感情ごとの枠と文字の色を取得する
def get_emotion_box_colors(top_emotion, top_score, max_scores):
    box_color = (0, 255, 0)
    text_color = (0, 180, 0)
    box_thickness = 2
    font_thickness = 1
    if top_emotion and top_emotion.lower() in emotion_colors:
        box_color = emotion_colors[top_emotion.lower()]['box_color']
        text_color = emotion_colors[top_emotion.lower()]['text_color']
        if top_emotion.lower() in max_scores and max_scores[top_emotion.lower()] == top_score:
            box_thickness = 4
            font_thickness = 2
    return box_color, text_color, box_thickness, font_thickness



### Route
@app.route('/', methods=['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        if 'filename' not in request.files:
            return redirect(request.url)
        file = request.files['filename']
        if file and allowed_file(file.filename):
            image_cv_rgb, image_cv_bgr = process_image(file)
            faces = detect_faces(image_cv_rgb)
            max_scores = get_max_emotion_scores(faces, image_cv_bgr)
            draw_emotion_boxes(faces, image_cv_bgr, max_scores)
            _, buffer = cv2.imencode('.png', image_cv_bgr)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_data = f"data:image/png;base64,{image_base64}"
            return render_template('result.html', num_faces=len(faces), image=base64_data)
    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
