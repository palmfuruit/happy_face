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


            for face in faces:
                aligned_face_cropped, x, y, width, height = align_face(image_cv_rgb, face)
                print(f"Face detected at x={x}, y={y}, width={width}, height={height}")

                # 感情分析を実行
                top_emotion, top_score = analyze_emotions(aligned_face_cropped)

                # 表示するテキストの位置を枠の内側に設定
                text_position_y = y + 20  # 枠の内側の上部に設定

                # 落ち着いた緑色
                calm_green = (0, 180, 0)

                # スコアに基づいて感情ラベルを表示
                if top_emotion and top_score >= 50:
                    cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)  # 明るい緑の枠
                    cv2.putText(image_cv_bgr, f'{top_emotion} ({top_score})', (x + 5, text_position_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, calm_green, 2)  # 太字、落ち着いた緑色のテキスト
                else:
                    cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)  # 明るい緑の枠
                    cv2.putText(image_cv_bgr, 'No emotion detected', (x + 5, text_position_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, calm_green, 2)  # 太字、落ち着いた緑色のテキスト
                    


            _, buffer = cv2.imencode('.png', image_cv_bgr)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_data = f"data:image/png;base64,{image_base64}"

            return render_template('result.html', num_faces=len(faces), image=base64_data)

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
