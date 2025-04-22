import os
import cv2
import mediapipe as mp
import numpy as np
import math
import gradio as gr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calc_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def analyze_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(rgb)
        if not results.multi_face_landmarks:
            raise RuntimeError("未检测到人脸")

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])

    # 嘴角角度
    lm_left, lm_right, lm_mid = pts[61], pts[291], pts[13]
    left_angle = calc_angle(lm_mid, lm_left)
    right_angle = calc_angle(lm_mid, lm_right)
    mouth_tilt = right_angle - left_angle

    # EAR
    left_indices  = [33, 160, 158, 133, 153, 144]
    right_indices = [362, 385, 387, 263, 373, 380]
    left_ear = eye_aspect_ratio(pts[left_indices])
    right_ear = eye_aspect_ratio(pts[right_indices])

    return {
        'mouth_tilt': round(mouth_tilt, 2),
        'left_ear': round(left_ear, 3),
        'right_ear': round(right_ear, 3)
    }, results

def visualize_landmarks(image_path, results):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    
    annotated = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    return annotated

def process_image(img_path):
    try:
        features, results = analyze_face(img_path)
        img = visualize_landmarks(img_path, results)

        msg = f"""嘴角歪斜角度差：{features['mouth_tilt']} 度  
左眼 EAR：{features['left_ear']}  
右眼 EAR：{features['right_ear']}  
"""

        if abs(features['mouth_tilt']) > 200:
            msg += "⚠️ 可能存在嘴角歪斜\n"
        if abs(features['left_ear'] - features['right_ear']) > 0.1:
            msg += "⚠️ 可能存在眼部不对称\n"

        return img, msg
    except Exception as e:
        return None, f"❌ 出错：{str(e)}"

# 创建 Gradio 界面
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", label="上传面部图像"),
    outputs=[
        gr.Image(label="面部关键点可视化"),
        gr.Textbox(label="分析结果")
    ],
    title="面部歪斜与EAR检测",
    description="上传一张正面人脸图像，自动分析嘴角歪斜角度与眼部开合程度（EAR）"
)

if __name__ == "__main__":
    demo.launch()
