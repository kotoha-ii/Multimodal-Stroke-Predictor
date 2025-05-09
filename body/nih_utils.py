# nih_utils.py
from collections import deque
import numpy as np
class BodyProportions:
    def __init__(self, config):
        self.config = config
        self.reference_scale = None
        self.torso_vectors = deque(maxlen=config.CALIBRATION_FRAMES)
        
    def update_proportions(self, landmarks):
        """动态身体比例计算"""
        shoulder_width = np.linalg.norm(landmarks[11] - landmarks[12])
        torso_height = np.mean([landmarks[23][1]-landmarks[11][1], 
                               landmarks[24][1]-landmarks[12][1]])
        
        current_ratio = shoulder_width / (torso_height + 1e-6)
        ideal_ratio = self.config.GOLDEN_RATIO
        
        if self.reference_scale is None:
            self.reference_scale = current_ratio / ideal_ratio
        else:
            self.reference_scale = (self.config.CALIBRATION_EMA_ALPHA * self.reference_scale +
                                   (1 - self.config.CALIBRATION_EMA_ALPHA) * (current_ratio / ideal_ratio))
        
        # 记录躯干向量
        shoulder_mid = (landmarks[11] + landmarks[12]) / 2
        hip_mid = (landmarks[23] + landmarks[24]) / 2
        self.torso_vectors.append(hip_mid - shoulder_mid)
        
        return self.reference_scale


class ArmEvaluator:
    def __init__(self, config):
        self.config = config
        self.reset()
        self.proportions = BodyProportions(config)
        self.calibration_matrix = np.eye(3)
        # 新增支撑物检测参数
        self.support_y_threshold = 0.9  # 手腕y坐标超过此值视为碰到支撑物（需根据实际坐标系调整）
        self.drop_speed_threshold = 30  # 度/秒，超过此速度视为自由下落
        
    def reset(self):
        self.scores = {'left':4, 'right':4}
        self.angle_history = {
            'left': deque(maxlen=90),  # 3秒数据缓存(假设30fps)
            'right': deque(maxlen=90)
        }
        self.status = {
            'left': {
                'timing_start': None,
                'max_angle': 0,
                'min_angle': 180,
                'last_angles': deque(maxlen=5),  # 用于计算下落速度
                'support_contact': False
            },
            'right': {
                'timing_start': None,
                'max_angle': 0,
                'min_angle': 180,
                'last_angles': deque(maxlen=5),
                'support_contact': False
            }
        }
    ######################################校准
    def dynamic_calibration(self, landmarks):
        """实时校准主方法"""
        scale = self.proportions.update_proportions(landmarks)
        torso_vector = np.mean(self.proportions.torso_vectors, axis=0)
        
        theta = np.arctan2(torso_vector[0], torso_vector[1])
        
        # 构建3x3仿射变换矩阵
        self.calibration_matrix = np.array([
            [scale*np.cos(theta), -scale*np.sin(theta), 0],
            [scale*np.sin(theta),  scale*np.cos(theta), 0],
            [0,                   0,                    1]
        ])
        
    def get_corrected_landmarks(self, landmarks):
        """获取校正后坐标"""
        homogenous = np.hstack((landmarks, np.ones((len(landmarks),1))))
        return (homogenous @ self.calibration_matrix.T)[:,:2]
    
    def get_calibration_status(self):
        """返回校准状态信息"""
        if len(self.proportions.torso_vectors) < self.config.CALIBRATION_FRAMES:
            return f"校准中 ({len(self.proportions.torso_vectors)}/{self.config.CALIBRATION_FRAMES})"
        
        angle_std = np.std([np.arctan2(v[0],v[1]) for v in self.proportions.torso_vectors])
        return f"已校准 | 稳定度: {np.degrees(angle_std):.1f}°"
    ######################################校准
    def update_scores(self, current_angles, landmarks, timestamp):
        """修改后的更新逻辑"""
        for side in ['left', 'right']:
            angle = current_angles[side]
            wrist_y = landmarks[15 if side == 'left' else 16][1]
            
            # 更新状态参数
            self.angle_history[side].append(angle)
            self.status[side]['last_angles'].append(angle)
            self.status[side]['max_angle'] = max(self.status[side]['max_angle'], angle)
            self.status[side]['min_angle'] = min(self.status[side]['min_angle'], angle)
            self.status[side]['support_contact'] = wrist_y > self.support_y_threshold
            
            # 检测初始姿势建立
            if angle >= self.config.TARGET_ANGLE - self.config.ANGLE_TOLERANCE:
                if self.status[side]['timing_start'] is None:
                    self.status[side]['timing_start'] = timestamp
            else:
                self.status[side]['timing_start'] = None
            
            # 计算最终得分
            self._calculate_score(side, timestamp)

    def _update_movement_status(self, side, angle):
        """检测有效运动"""
        angle_change = abs(angle - self.angle_history[side][-1]) if self.angle_history[side] else 0
        if angle_change > self.config.MIN_MOVEMENT_ANGLE:
            self.movement_status[side] = True
            
    def _is_touching_support(self, side):
        """支撑物接触检测"""
        return self.min_angles[side] >= self.config.SUPPORT_THRESHOLD
    
    def _has_anti_gravity_effort(self, side):
        """抗重力判断"""
        return self.fall_velocity[side] < self.config.FALL_VELOCITY_THRESH
    
    def _is_full_duration_achieved(self, side, timestamp):
        """持续10秒检测"""
        return timestamp - self.start_time[side] >= self.config.REQUIRED_TIME

    def _start_timing(self, side, timestamp):
        self.status[side] = {
            'timing': True,
            'start_time': timestamp,
            'max_drift': 0,
            'last_valid_time': timestamp
        }

    def calculate_fall_velocity(self, side):
        """坠落速度计算"""
        if len(self.angle_history[side]) < 2:
            return 0
        delta_angle = self.angle_history[side][-1] - self.angle_history[side][-2]
        delta_time = self.timestamps[side][-1] - self.timestamps[side][-2]
        return abs(delta_angle) / delta_time if delta_time != 0 else 0

    def update_min_angles(self, side, current_angle):
        """记录最低角度"""
        self.min_angles[side] = min(self.min_angles[side], current_angle)

    def _update_timing(self, side, timestamp):
        current_drift = abs(np.mean(self.angle_history[side]) - self.config.TARGET_ANGLE)
        self.status[side]['max_drift'] = max(self.status[side]['max_drift'], current_drift)
        self.status[side]['last_valid_time'] = timestamp

    def _stop_timing(self, side):
        self.status[side]['timing'] = False

        # nih_utils.py
    def check_posture(self, landmarks):
        """改进的姿势验证"""
        # 计算肩髋连线与垂直轴的夹角
        shoulder_mid = (landmarks[11] + landmarks[12])/2
        hip_mid = (landmarks[23] + landmarks[24])/2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle = np.degrees(np.arctan2(dx, dy))
        
        # 同时验证头部位置
        nose_y = landmarks[0][1]
        return abs(angle) < 8 and nose_y < shoulder_mid[1]
    
        # nih_utils.py
    def get_arm_angles(self, landmarks):
        """计算手臂与躯干夹角"""
        def calc_abduction(a, b, c):
            """a:手腕, b:肩部, c:髋部"""
            vec_arm = a - b      # 手臂向量（手腕到肩）
            vec_body = c - b     # 躯干向量（髋到肩）
            return self.calculate_angle(vec_arm, vec_body)
        
        return {
            'left': calc_abduction(landmarks[15], landmarks[11], landmarks[23]),
            'right': calc_abduction(landmarks[16], landmarks[12], landmarks[24])
        }

    def calculate_angle(self, vec1, vec2):
        """向量夹角计算"""
        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    
    
    def _calculate_score(self, side, timestamp):
        """新版评分逻辑"""
        status = self.status[side]
        angle_data = list(self.angle_history[side])
        
        # 评分条件判断
        if not angle_data:  # 无动作
            self.scores[side] = 4
            return
        
        # 计算下落速度（度/秒）
        drop_speed = 0
        if len(status['last_angles']) >= 2:
            time_diff = 1/30  # 假设30fps
            angle_diff = status['last_angles'][-1] - status['last_angles'][0]
            drop_speed = abs(angle_diff) / (len(status['last_angles'])*time_diff)
        
        # 条件判断顺序按评分等级从高到低
        if status['timing_start'] and (timestamp - status['timing_start']) >= self.config.REQUIRED_TIME:
            if status['max_angle'] >= 85 and not status['support_contact']:
                self.scores[side] = 0  # 完美保持
        elif status['support_contact']:
            if drop_speed > self.drop_speed_threshold:
                self.scores[side] = 3  # 快速下落
            else:
                self.scores[side] = 2  # 缓慢接触支撑
        elif status['timing_start']:
            duration = timestamp - status['timing_start']
            if duration > 2 and drop_speed < 5:
                self.scores[side] = 1  # 有效保持但时间不足
            else:
                self.scores[side] = 2  # 短暂保持
        else:
            self.scores[side] = 4  # 无有效动作