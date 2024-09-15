import io
import base64
from flask import Flask, request, render_template, redirect
from PIL import Image
import numpy as np
import cv2

from face_detection import detect_faces, align_face
from emotion_analysis import analyze_emotions

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

            image_cv = np.array(image)

            # 顔検出を実行
            faces = detect_faces(image_cv)

            for face in faces:
                aligned_face_cropped, x, y, width, height = align_face(image_cv, face)

                # 感情分析を実行
                top_emotion, top_score = analyze_emotions(aligned_face_cropped)

                if top_emotion:
                    cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(image_cv, f'{top_emotion} ({top_score})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    cv2.rectangle(image_cv, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(image_cv, 'No emotion detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.png', image_cv)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_data = f"data:image/png;base64,{image_base64}"

            return render_template('result.html', num_faces=len(faces), image=base64_data)

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
