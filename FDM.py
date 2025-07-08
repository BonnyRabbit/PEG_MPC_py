"""
Aircraft Dynamics Module
包含飞机动力学模型和敌机机动库
"""

import numpy as np
import casadi as ca
from casadi import sin, cos, pi, tan, norm_2, acos, dot, fmax, fmin, if_else, vertcat
from config import *
from config import output_config

class FlightDynamicsModel:
    """飞机动力学模型类"""
    
    def __init__(self):
        self.dt = DT
        self.g = GRAVITY
        self.maneuver_controller = EnemyManeuverController()
    
    def run(self, state, ctrl, current_time=0):
        """
        运行飞机动力学模型一步
        
        Args:
            state: 状态向量 [psi, phi, v, x, y, psi_t, phi_t, v_t, x_t, y_t, ata, rel_dist]
            ctrl: 控制向量 [dv, dphi]
            current_time: 当前时间
            
        Returns:
            下一时刻的状态向量
        """
        psi, phi, v, x, y, psi_t, phi_t, v_t, x_t, y_t, ata, rel_dist = state
        dv, dphi = ctrl
        
        # 检查是否应该触发敌机机动
        if self.maneuver_controller.should_trigger_maneuver(current_time, rel_dist):
            self.maneuver_controller.start_maneuver(current_time, psi_t)
        
        # 追击者动力学更新
        dx = v * np.sin(psi)
        dy = v * np.cos(psi)
        v += dv * self.dt
        v = np.clip(v, V_MIN, V_MAX)
        dpsi = self.g / v * np.tan(phi)
        phi += dphi * self.dt
        phi = np.clip(phi, PHI_MIN, PHI_MAX)
        psi += dpsi * self.dt
        psi = self.check_heading(psi)
        x += dx * self.dt
        y += dy * self.dt
        
        # 目标动力学更新（包含机动控制）
        maneuver_phi = self.maneuver_controller.get_maneuver_control(psi_t, self.dt)
        dphi_t = (maneuver_phi - phi_t) / self.dt
        
        dx_t = v_t * np.sin(psi_t)
        dy_t = v_t * np.cos(psi_t)
        dpsi_t = self.g / v_t * np.tan(phi_t)
        phi_t += dphi_t * self.dt
        phi_t = np.clip(phi_t, PHI_MIN, PHI_MAX)
        psi_t += dpsi_t * self.dt
        psi_t = self.check_heading(psi_t)
        x_t += dx_t * self.dt
        y_t += dy_t * self.dt

        # 计算新的ATA和相对距离
        pursuer_pos = np.array([x, y])
        evader_pos = np.array([x_t, y_t])
        rel_dist = np.linalg.norm(pursuer_pos - evader_pos)
        ata = self.compute_ata(x, y, x_t, y_t, psi)

        return np.array([
            psi, phi, v, x, y,
            psi_t, phi_t, v_t, x_t, y_t,
            ata, rel_dist
        ])
    
    @staticmethod
    def compute_ata(x, y, x_t, y_t, psi):
        """计算接近角 (Angle To Approach)"""
        LOS = np.array([x_t - x, y_t - y])
        LOS_norm = np.linalg.norm(LOS)
        if LOS_norm == 0:
            if output_config.enable_print_output:
                print(f"Warning: LOS vector has zero length! x={x}, y={y}, x_t={x_t}, y_t={y_t}")
            return 0
        LOS = LOS / max(LOS_norm, 1e-6)
        v = np.array([np.sin(psi), np.cos(psi)])
        dot_LOS_v = np.clip(np.dot(LOS, v), -1.0-1e-8, 1.0+1e-8)
        ata = np.arccos(dot_LOS_v)
        return ata
        
    @staticmethod
    def check_heading(psi):
        """航向角归一化到 [-π, π]"""
        if psi > np.pi:
            psi -= 2 * np.pi
        elif psi < -np.pi:
            psi += 2 * np.pi
        return psi


class EnemyManeuverController:
    """敌机机动库"""
    
    def __init__(self):
        # 机动状态
        self.current_maneuver = MANEUVER_NONE
        self.maneuver_progress = 0.0
        self.maneuver_start_time = 0.0
        self.last_maneuver_time = -MANEUVER_COOLDOWN
        
        # S转弯状态
        self.s_turn_phase = 0
        self.s_turn_direction = 1
        
        # 单向转弯状态
        self.single_turn_direction = 1
        self.initial_heading = 0.0
        self.target_heading = 0.0
        
    def should_trigger_maneuver(self, current_time, rel_distance):
        """判断是否应该触发机动"""
        time_since_last = current_time - self.last_maneuver_time
        
        return (rel_distance < MANEUVER_TRIGGER_DISTANCE and 
                time_since_last >= MANEUVER_COOLDOWN and 
                self.current_maneuver == MANEUVER_NONE)
    
    def start_maneuver(self, current_time, current_heading):
        """开始新的机动"""
        maneuver_types = [MANEUVER_SINGLE_TURN, MANEUVER_S_TURN]
        self.current_maneuver = np.random.choice(maneuver_types)
        
        self.maneuver_start_time = current_time
        self.maneuver_progress = 0.0
        self.initial_heading = current_heading
        self.last_maneuver_time = current_time
        
        if self.current_maneuver == MANEUVER_SINGLE_TURN:
            self.single_turn_direction = np.random.choice([-1, 1])
            self.target_heading = current_heading + self.single_turn_direction * SINGLE_TURN_ANGLE
            self.target_heading = self._normalize_heading(self.target_heading)
            if output_config.enable_print_output:
                print(f"敌机开始单向转弯: 方向={'右' if self.single_turn_direction > 0 else '左'}")
            
        elif self.current_maneuver == MANEUVER_S_TURN:
            self.s_turn_direction = np.random.choice([-1, 1])
            self.s_turn_phase = 0
            first_turn_dir = self.s_turn_direction
            self.target_heading = current_heading + first_turn_dir * S_TURN_ANGLE
            self.target_heading = self._normalize_heading(self.target_heading)
            pattern = "右-左-右" if self.s_turn_direction > 0 else "左-右-左"
            if output_config.enable_print_output:
                print(f"敌机开始S转弯: 模式={pattern}")
    
    def get_maneuver_control(self, current_heading, dt):
        """获取机动控制输入"""
        if self.current_maneuver == MANEUVER_NONE:
            return 0.0
        
        heading_error = self._normalize_heading(self.target_heading - current_heading)
        
        if abs(heading_error) < 0.1:
            if self.current_maneuver == MANEUVER_SINGLE_TURN:
                self.current_maneuver = MANEUVER_NONE
                if output_config.enable_print_output:
                    print("敌机完成单向转弯")
                return 0.0
                
            elif self.current_maneuver == MANEUVER_S_TURN:
                self.s_turn_phase += 1
                
                if self.s_turn_phase >= 3:
                    self.current_maneuver = MANEUVER_NONE
                    print("敌机完成S转弯")
                    return 0.0
                else:
                    if self.s_turn_phase == 1:
                        turn_dir = -self.s_turn_direction
                    else:
                        turn_dir = self.s_turn_direction
                    
                    self.target_heading = current_heading + turn_dir * S_TURN_ANGLE
                    self.target_heading = self._normalize_heading(self.target_heading)
                    
                    phase_names = ["第一段", "第二段", "第三段"]
                    print(f"敌机S转弯进入{phase_names[self.s_turn_phase]}")
        
        if self.current_maneuver == MANEUVER_SINGLE_TURN:
            turn_rate = SINGLE_TURN_RATE
        elif self.current_maneuver == MANEUVER_S_TURN:
            turn_rate = S_TURN_RATE
        else:
            turn_rate = SINGLE_TURN_RATE
            
        desired_turn_rate = turn_rate * np.sign(heading_error)
        v_target = 70.0
        desired_phi = np.arctan(desired_turn_rate * v_target / GRAVITY)
        desired_phi = np.clip(desired_phi, PHI_MIN, PHI_MAX)
        
        return desired_phi
    
    def _normalize_heading(self, heading):
        """航向角归一化到 [-π, π]"""
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi
        return heading
    
    def get_status_info(self):
        """获取机动状态信息"""
        if self.current_maneuver == MANEUVER_NONE:
            return "无机动"
        elif self.current_maneuver == MANEUVER_SINGLE_TURN:
            direction = "右转" if self.single_turn_direction > 0 else "左转"
            return f"单向转弯({direction})"
        elif self.current_maneuver == MANEUVER_S_TURN:
            pattern = "右-左-右" if self.s_turn_direction > 0 else "左-右-左"
            phase_names = ["第一段", "第二段", "第三段"]
            return f"S转弯({pattern})-{phase_names[self.s_turn_phase]}"
        else:
            return "未知机动"


def check_heading_symbolic(psi):
    """CasADi符号版本的航向角归一化函数"""
    psi = if_else(psi > pi, psi - 2*pi, psi)
    psi = if_else(psi < -pi, psi + 2*pi, psi)
    return psi


def compute_ata_symbolic(x, y, x_t, y_t, psi):
    """CasADi符号版本的ATA计算函数"""
    LOS = vertcat(x_t - x, y_t - y)
    LOS_norm = fmax(norm_2(LOS), 1e-6)
    LOS_unit = LOS / LOS_norm
    v = vertcat(sin(psi), cos(psi))
    dot_LOS_v = dot(LOS_unit, v)
    dot_LOS_v = fmax(fmin(dot_LOS_v, 1.0-1e-8), -1.0+1e-8)
    ata = acos(dot_LOS_v)
    return ata


def fdm_step_symbolic(state, ctrl):
    """
    飞行动力学一步积分 - CasADi符号版本用于MPC预测
    
    Args:
        state: CasADi符号状态向量
        ctrl: CasADi符号控制向量
        
    Returns:
        下一时刻的状态向量（符号形式）
    """
    # 提取状态
    psi_cur = state[0]
    phi_cur = state[1]
    v_cur = state[2]
    x_cur = state[3]
    y_cur = state[4]
    psi_t_cur = state[5]
    phi_t_cur = state[6]
    v_t_cur = state[7]
    x_t_cur = state[8]
    y_t_cur = state[9]
    
    # 提取控制
    dv_cur = ctrl[0]
    dphi_cur = ctrl[1]
    
    # 追击者动力学
    dx = v_cur * sin(psi_cur)
    dy = v_cur * cos(psi_cur)
    v_new = v_cur + dv_cur * STEP_HORIZON
    v_new = fmax(fmin(v_new, V_MAX), V_MIN)
    dpsi = GRAVITY / v_new * tan(phi_cur)
    phi_new = phi_cur + dphi_cur * STEP_HORIZON
    phi_new = fmax(fmin(phi_new, PHI_MAX), PHI_MIN)
    psi_new = psi_cur + dpsi * STEP_HORIZON
    psi_new = check_heading_symbolic(psi_new)
    x_new = x_cur + dx * STEP_HORIZON
    y_new = y_cur + dy * STEP_HORIZON
    
    # 目标动力学（匀速直线运动）
    dphi_t = 0
    dx_t = v_t_cur * sin(psi_t_cur)
    dy_t = v_t_cur * cos(psi_t_cur)
    dpsi_t = GRAVITY / v_t_cur * tan(phi_t_cur)
    phi_t_new = phi_t_cur + dphi_t * STEP_HORIZON
    phi_t_new = fmax(fmin(phi_t_new, PHI_MAX), PHI_MIN)
    psi_t_new = psi_t_cur + dpsi_t * STEP_HORIZON
    psi_t_new = check_heading_symbolic(psi_t_new)
    x_t_new = x_t_cur + dx_t * STEP_HORIZON
    y_t_new = y_t_cur + dy_t * STEP_HORIZON
    
    # 计算新的ATA和相对距离
    pursuer_pos = vertcat(x_new, y_new)
    evader_pos = vertcat(x_t_new, y_t_new)
    rel_dist_new = norm_2(pursuer_pos - evader_pos)
    ata_new = compute_ata_symbolic(x_new, y_new, x_t_new, y_t_new, psi_new)
    
    return vertcat(
        psi_new, phi_new, v_new, x_new, y_new,
        psi_t_new, phi_t_new, v_t_cur, x_t_new, y_t_new,
        ata_new, rel_dist_new
    )
