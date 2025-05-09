import argparse
import os
import cv2
import time
import numpy as np
from config import Config
from nih_utils import ArmEvaluator
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image  # 新增导入

def draw_chinese_text(image, text, position, color, font):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description='NIH上肢评估系统')
    parser.add_argument('-i', '--input', required=True, help='输入视频路径或摄像头ID')
    args = parser.parse_args()

    # 初始化中文字体
    try:
        font = ImageFont.truetype("simhei.ttf", 20)  # 或指定完整路径如"C:/Windows/Fonts/simhei.ttf"
    except IOError:
        print("警告：未找到中文字体，将使用默认字体")
        font = ImageFont.load_default()

    config = Config()
    evaluator = ArmEvaluator(config)
    last_terminal_update = 0
    pose = mp.solutions.pose.Pose(
        model_complexity=config.MODEL_COMPLEXITY,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
    )

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print(f"无法打开视频源: {args.input}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        current_time = time.time()
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark])
            # 新增校准调用
            evaluator.dynamic_calibration(landmarks)
            corrected_landmarks = evaluator.get_corrected_landmarks(landmarks)
            if evaluator.check_posture(corrected_landmarks):
                angles = evaluator.get_arm_angles(landmarks)
                evaluator.update_scores(angles, corrected_landmarks, time.time())
                
                # 渲染指导文本
                frame = draw_chinese_text(frame, config.GUIDELINE_TEXT, (30, 30),
                                         tuple(config.COLOR_CORRECT), font)
                
                # 渲染角度和分数
                for i, (side, angle) in enumerate(angles.items()):
                    text = f"{side}臂: {angle:.1f}° ({evaluator.scores[side]}分)"
                    color = config.COLOR_CORRECT if evaluator.scores[side]<2 else config.COLOR_WARNING
                    frame = draw_chinese_text(frame, text, (30, 70+i*40), tuple(color), font)
            else:
                frame = draw_chinese_text(frame, "请保持直立姿势!", (30, 30),
                                         tuple(config.COLOR_WARNING), font)
            #####每0.5s打印信息#####每0.5s打印信息#####每0.5s打印信息
            if current_time - last_terminal_update > 0.5:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"校准状态: {evaluator.get_calibration_status()}")
                print(f"左臂角度: {angles.get('left', 0):.1f}° 右臂角度: {angles.get('right', 0):.1f}°")
                last_terminal_update = current_time
            #####每0.5s打印信息#####每0.5s打印信息#####每0.5s打印信息
        cv2.imshow('NIH Upper Limb Assessment', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n最终评估结果:")
    print(f"左臂: {evaluator.scores['left']}分 | 右臂: {evaluator.scores['right']}分")

if __name__ == "__main__":
    main()