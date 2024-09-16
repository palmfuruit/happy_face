import numpy as np
import cv2
from mtcnn import MTCNN

# MTCNNの顔検出モデルを初期化（閾値を調整）
detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7])  # ここで閾値を調整

def detect_faces(image_cv):
    # 顔検出を実行
    faces = detector.detect_faces(image_cv)
    return faces

def align_face(image_cv, face):
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

    return aligned_face_cropped, x, y, width, height
