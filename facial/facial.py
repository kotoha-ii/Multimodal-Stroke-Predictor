import os
import cv2
import mediapipe as mp
import numpy as np
import math
import gradio as gr


# 只显示 ERROR 级别以上日志，屏蔽 INFO/WARN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calc_angle(p1, p2):
    """计算两点连线与水平线的夹角（度）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def eye_aspect_ratio(eye_landmarks):
    """计算 Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def analyze_face(image_path):
    """
    提取面部特征：
      - 嘴角歪斜角度差(mouth_tilt)
      - 左右眼 EAR(left_ear, right_ear)
    返回字典：{'mouth_tilt': float, 'left_ear': float, 'right_ear': float}
    """
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

    # 嘴角：61 左嘴角，291 右嘴角，13 嘴中点
    lm_left, lm_right, lm_mid = pts[61], pts[291], pts[13]
    left_angle  = calc_angle(lm_mid, lm_left)
    right_angle = calc_angle(lm_mid, lm_right)
    mouth_tilt  = right_angle - left_angle

    # 眼睛 EAR
    left_indices  = [33, 160, 158, 133, 153, 144]
    right_indices = [362,385,387,263,373,380]
    left_ear  = eye_aspect_ratio(pts[left_indices])
    right_ear = eye_aspect_ratio(pts[right_indices])

    return {
        'mouth_tilt': round(mouth_tilt, 2),
        'left_ear'  : round(left_ear, 3),
        'right_ear' : round(right_ear, 3)
    }

def visualize_face_landmarks(image_path, save_path=None, display=True):
    """
    在图像上绘制面部关键点网格并显示/保存。
    :param image_path: 输入图像路径
    :param save_path: (可选) 保存结果的路径
    :param display: 是否弹窗显示
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(rgb)
        if not results.multi_face_landmarks:
            raise RuntimeError("未检测到人脸")

        annotated = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style()
            )

    if save_path:
        cv2.imwrite(save_path, annotated)
    if display:
        cv2.imshow("Face Landmarks", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return annotated

if __name__ == "__main__":
    path = "facial_data/"
    pic = "2.jpg" 
    img_path = path + pic       # 图片路径
    # 提取特征
    feats = analyze_face(img_path)
    print(f"嘴角歪斜角度差：{feats['mouth_tilt']}")
    print(f"左眼 EAR:{feats['left_ear']} \n右眼 EAR:{feats['right_ear']}")
    
    if abs(feats['mouth_tilt']) > 200:
        print("⚠️ 可能存在嘴角歪斜")
    if abs(feats['left_ear'] - feats['right_ear']) > 0.1:
        print("⚠️ 可能存在眼部不对称")

    # 可视化关键点并保存
    out = visualize_face_landmarks(img_path, save_path=path + "landmark/" + pic)
    print("可视化结果已保存")
