# class Config:
#     # MediaPipe参数
#     MODEL_COMPLEXITY = 1
#     MIN_DETECTION_CONFIDENCE = 0.7
    
#     # 评估参数
#     TARGET_ANGLE = 90      # 目标角度
#     ANGLE_TOLERANCE = 8     # 角度容差
#     REQUIRED_TIME = 10      # 需要保持的秒数
    
#     # 姿势验证
#     VERTICAL_THRESHOLD = 0.85  # 肩-髋垂直度阈值
    
#     # 可视化参数
#     GUIDELINE_TEXT = "请直立并平举双臂至90度"
#     COLOR_CORRECT = (0, 255, 0)
#     COLOR_WARNING = (0, 0, 255)
class Config:
    MODEL_COMPLEXITY = 2  # 提高模型精度
    MIN_DETECTION_CONFIDENCE = 0.8
    TARGET_ANGLE = 80     # 实际临床允许10度偏差
    ANGLE_TOLERANCE = 12  # 扩大瞬时容差
    REQUIRED_TIME = 8     # 临床标准8秒
    VERTICAL_THRESHOLD = 5  # 身体倾斜容差(度)
    GUIDELINE_TEXT = "请直立并平举双臂至90度"
    COLOR_CORRECT = (0, 255, 0)
    COLOR_WARNING = (0, 0, 255)
    # 新增校准参数
    CALIBRATION_FRAMES = 30       # 校准帧数
    GOLDEN_RATIO = 0.618          # 肩宽/躯干高理想比例
    CALIBRATION_EMA_ALPHA = 0.9   # 平滑系数
    
    # 新增可视化参数
    FONT_PATH = "simhei.ttf"      # 字体文件路径
    FONT_SIZE = 20
    DEBUG_DISPLAY = True          # 是否显示调试信息

        # 新增支撑物检测参数
    SUPPORT_THRESHOLD = 0.85    # Y坐标阈值(屏幕底部为1.0)
    FALL_VELOCITY_THRESH = 30   # 坠落速度阈值(度/秒)
    MIN_MOVEMENT_ANGLE = 5      # 视为有效运动的最小角度变化