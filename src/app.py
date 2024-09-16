import io
import base64
from flask import Flask, request, render_template, redirect
from PIL import Image
import numpy as np
import cv2

from .face_detection import detect_faces, align_face
from .emotion_analysis import analyze_emotions

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
            # return redirect(request.url)
            return render_template('index.html')

        file = request.files['filename']
        if file and allowed_file(file.filename):
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')

            if image.width > 512:
                image = scale_to_width(image, 512)

                
            # PIL 画像を OpenCV 形式に変換
            image_cv_rgb = np.array(image)
            image_cv_bgr = cv2.cvtColor(image_cv_rgb, cv2.COLOR_RGB2BGR)

            # 顔検出を実行
            faces = detect_faces(image_cv_rgb)

            # for face in faces:
            #     aligned_face_cropped, x, y, width, height = align_face(image_cv_rgb, face)

            #     # 感情分析を実行
            #     top_emotion, top_score = analyze_emotions(aligned_face_cropped)

            #     if top_emotion:
            #         cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
            #         cv2.putText(image_cv_bgr, f'{top_emotion} ({top_score})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            #     else:
            #         cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
            #         cv2.putText(image_cv_bgr, 'No emotion detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


            # 検出された顔に枠を描画
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(image_cv_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # image のメモリを解放
            image.close()

           # 画像をエンコードして base64 に変換
            _, buffer = cv2.imencode('.png', image_cv_bgr)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_data = f"data:image/png;base64,{image_base64}"

            # # 画像データをバッファに書き込む
            # image.save(buf, 'png')
            # # バイナリデータを base64 でエンコードして utf-8 でデコード
            # base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            # # HTML 側の src の記述に合わせるために付帯情報付与する
            # base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # image のメモリを解放
            image.close()

            # return render_template('result.html', num_faces=len(faces), image=base64_data)
            return render_template('result.html', image=base64_data, num_faces=len(faces))

    elif request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
