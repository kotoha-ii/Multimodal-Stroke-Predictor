# body/main.py
import cv2
import os
import time
import numpy as np
from body.config import Config
from body.nih_utils import ArmEvaluator
from PIL import ImageFont, ImageDraw, Image

# 添加控制变量
is_running = False

def draw_chinese_text(image, text, position, color, font):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def start_pose_assessment(video_input=0):
    """启动摄像头评估，支持 Gradio 按钮触发"""
    global is_running
    is_running = True

    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except IOError:
        print("未找到 simhei.ttf，使用默认字体")
        font = ImageFont.load_default()

    config = Config()
    evaluator = ArmEvaluator(config)
    last_terminal_update = 0

    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
    )
    
    if isinstance(video_input, dict) and "name" in video_input:
        video_input = video_input["name"]
    elif video_input==None:
        return "无法打开视频源"
        
    cap = cv2.VideoCapture(int(video_input) if str(video_input).isdigit() else video_input)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_input}")
        return "无法打开视频源"

    while cap.isOpened() and is_running:
        success, frame = cap.read()
        if not success:
            break
        current_time = time.time()
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
            evaluator.dynamic_calibration(landmarks)
            corrected_landmarks = evaluator.get_corrected_landmarks(landmarks)
            if evaluator.check_posture(corrected_landmarks):
                angles = evaluator.get_arm_angles(landmarks)
                evaluator.update_scores(angles, corrected_landmarks, current_time)
                frame = draw_chinese_text(frame, config.GUIDELINE_TEXT, (30, 30),
                                          tuple(config.COLOR_CORRECT), font)
                for i, (side, angle) in enumerate(angles.items()):
                    text = f"{side}臂: {angle:.1f}° ({evaluator.scores[side]}分)"
                    color = config.COLOR_CORRECT if evaluator.scores[side] < 2 else config.COLOR_WARNING
                    frame = draw_chinese_text(frame, text, (30, 70+i*40), tuple(color), font)
            else:
                frame = draw_chinese_text(frame, "请保持直立姿势!", (30, 30),
                                          tuple(config.COLOR_WARNING), font)
            if current_time - last_terminal_update > 0.5:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"校准状态: {evaluator.get_calibration_status()}")
                print(f"左臂: {angles.get('left', 0):.1f}° | 右臂: {angles.get('right', 0):.1f}°")
                last_terminal_update = current_time

        cv2.imshow('NIH Upper Limb Assessment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n最终评估结果:")
    print(f"左臂: {evaluator.scores['left']}分 | 右臂: {evaluator.scores['right']}分")

    return f"""评估结束: 左臂 {evaluator.scores['left']}分 | 右臂 {evaluator.scores['right']}分
        Scale Definition
        0 No drift; limb holds 90 (or 45) degrees for full 10 seconds.
        1 Drift; limb holds 90 (or 45) degrees, but drifts down before full 10 seconds; does not hit bed or other support.
        2 Some effort against gravity; limb cannot get to or maintain (if cued) 90 (or 45) degrees, drifts down to bed, but has some effort against gravity.
        3 No effort against gravity; limb falls.
        """

def stop_pose_assessment():
    """终止摄像头评估"""
    global is_running
    is_running = False
    return "上肢评估已终止"
