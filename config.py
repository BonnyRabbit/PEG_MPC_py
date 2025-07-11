"""
Configuration file for MPC Flight Dynamics System
包含所有系统参数、权重设置和常量定义
"""

import numpy as np

# ========== 系统基本参数 ==========
GRAVITY = 9.81  # 重力加速度 (m/s²)
DT = 0.25       # 时间步长 (s)

# ========== MPC 参数 ==========
STEP_HORIZON = 0.25  # MPC 时间步长 (s)
N = 80              # 预测步数
SIM_TIME = 300      # 仿真时间 (s)

# ========== 任务参数 ==========
THRESHOLD_DISTANCE = 2000  # 距离阈值 (m)
TARGET_DISTANCE = 250   # 目标跟随距离 (m)
TRACK_HOLD_TIME = 20    # 尾追保持时间 (s)

# ========== 任务阶段标志 ==========
PHASE_APPROACH = 1   # 接近阶段
PHASE_TRACK = 2      # 尾追保持阶段

# ========== 输出控制开关 ==========
ENABLE_PRINT_OUTPUT = True      # 是否打印详细输出信息
ENABLE_ITERATION_PRINT = True   # 是否打印每步迭代信息
ENABLE_DETAILED_STATUS = True   # 是否打印详细状态信息
ENABLE_VISUALIZATION = True     # 是否启用画图功能

# ========== 输出控制配置类 ==========
class OutputConfig:
    """输出控制配置类，便于运行时动态修改设置"""
    
    def __init__(self):
        self.enable_print_output = ENABLE_PRINT_OUTPUT
        self.enable_iteration_print = ENABLE_ITERATION_PRINT
        self.enable_detailed_status = ENABLE_DETAILED_STATUS
        self.enable_visualization = ENABLE_VISUALIZATION
    
    def set_silent_mode(self):
        """设置静默模式（适用于强化学习等自动化场景）"""
        self.enable_print_output = False
        self.enable_iteration_print = False
        self.enable_detailed_status = False
        self.enable_visualization = False
    
    def set_debug_mode(self):
        """设置调试模式（启用所有输出）"""
        self.enable_print_output = True
        self.enable_iteration_print = True
        self.enable_detailed_status = True
        self.enable_visualization = True
    
    def set_minimal_mode(self):
        """设置最小输出模式（只保留重要信息）"""
        self.enable_print_output = True
        self.enable_iteration_print = False
        self.enable_detailed_status = False
        self.enable_visualization = False

# 创建全局输出配置实例
output_config = OutputConfig()

# ========== 敌机机动类型 ==========
MANEUVER_NONE = 0        # 无机动
MANEUVER_SINGLE_TURN = 1 # 单向转弯
MANEUVER_S_TURN = 2      # S转弯

# ========== 机动参数 ==========
MANEUVER_TRIGGER_DISTANCE = 2000   # 机动触发距离 (m)
MANEUVER_COOLDOWN = 25           # 机动冷却时间 (s)
SINGLE_TURN_RATE = np.deg2rad(8)  # 单向转弯角速度 (rad/s)
S_TURN_RATE = np.deg2rad(15)     # S转弯角速度 (rad/s)
SINGLE_TURN_ANGLE = np.pi        # 单向转弯目标角度 (rad)
S_TURN_ANGLE = np.pi/2          # S转弯每段角度 (rad)

# ========== 控制限制 ==========
# 速度控制限制
DV_MAX = 4   # 最大加速度 (m/s²)
DV_MIN = -4  # 最小加速度 (m/s²)

# 滚转角控制限制
DPHI_MAX = np.deg2rad(40)  # 最大滚转角速度 (rad/s)
DPHI_MIN = np.deg2rad(-40) # 最小滚转角速度 (rad/s)

# 状态变量限制
V_MIN = 60    # 最小速度 (m/s)
V_MAX = 130   # 最大速度 (m/s)
PHI_MAX = np.pi/6   # 最大滚转角 (rad, ±30°)
PHI_MIN = -np.pi/6  # 最小滚转角 (rad)

# ========== 状态归一化系数 ==========
ANGLE_SCALE = np.pi          # 角度归一化 (rad)
VELOCITY_SCALE = 100.0       # 速度归一化 (m/s)
POSITION_SCALE = 1000.0      # 位置归一化 (m)
DISTANCE_SCALE = 500.0       # 距离归一化 (m)

# ========== MPC 权重参数 ==========
# 归一化后的权重设置
Q_ATA_NORM = 1000.0          # ATA权重 (归一化后)
Q_POS_NORM = 20.0           # 距离误差权重 (归一化后)
Q_PSI_NORM = 800.0            # 航向误差权重 (归一化后)
R_DV_NORM = 0.1             # 加速度控制权重 (归一化后)
R_DPHI_NORM = 1.0           # 转向控制权重 (归一化后)

# 计算实际权重 (考虑归一化)
Q_ATA = Q_ATA_NORM / (ANGLE_SCALE**2)      # ATA权重
Q_POS = Q_POS_NORM / (DISTANCE_SCALE**2)   # 距离权重
R_DV = R_DV_NORM                           # 加速度控制权重
R_DPHI = R_DPHI_NORM / (ANGLE_SCALE**2)    # 转向控制权重

# ========== 初始状态配置 ==========
class InitialConditions:
    """初始条件配置类"""
    
    def __init__(self, random_target=True):
        # 追击者初始状态
        self.psi_init = 0          # 追击者初始航向角 (rad)
        self.phi_init = 0          # 追击者初始倾斜角 (rad)
        self.v_init = 80           # 追击者初始速度 (m/s)
        self.x_init = 0            # 追击者初始x位置 (m)
        self.y_init = 0            # 追击者初始y位置 (m)
        
        # 目标初始状态
        if random_target:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(4000, 5000)
            self.x_t_init = radius * np.cos(angle)       # 目标初始x位置 (m)
            self.y_t_init = radius * np.sin(angle)       # 目标初始y位置 (m)
            self.psi_t_init = np.arctan2(self.x_t_init, self.y_t_init)  # 目标初始航向角 (rad)
        else:
            self.x_t_init = 4000       # 目标初始x位置 (m)
            self.y_t_init = 0          # 目标初始y位置 (m)
            self.psi_t_init = 0        # 目标初始航向角 (rad)
            
        self.phi_t_init = 0        # 目标初始倾斜角 (rad)
        self.v_t_init = 70         # 目标初始速度 (m/s)
        
        # 计算初始ATA和距离
        self.initial_dist = np.sqrt((self.x_t_init - self.x_init)**2 + 
                                   (self.y_t_init - self.y_init)**2)
        
        los_vector = np.array([self.x_t_init - self.x_init, self.y_t_init - self.y_init])
        pursuer_velocity = np.array([np.sin(self.psi_init), np.cos(self.psi_init)])
        dot_product = np.dot(los_vector, pursuer_velocity) / self.initial_dist
        self.ata_init = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        self.rel_dist_init = self.initial_dist

# ========== IPOPT 求解器配置 ==========
IPOPT_OPTIONS = {
    'ipopt': {
        'max_iter': 100,
        'print_level': 0,
        'acceptable_tol': 1e-4,
        'warm_start_init_point': 'yes'
    },
    'print_time': 0
}

# ========== 边界条件 ==========
class BoundaryConditions:
    """系统边界条件配置"""
    
    @staticmethod
    def get_state_bounds():
        """获取状态变量边界"""
        return {
            'lower': [-np.pi, -PHI_MAX, V_MIN, -2e4, -2e4, 
                     -np.pi, -PHI_MAX, 0, -2e4, -2e4, 
                     -np.pi, 0],
            'upper': [np.pi, PHI_MAX, V_MAX, 2e4, 2e4,
                     np.pi, PHI_MAX, 70, 2e4, 2e4,
                     np.pi, 2e4]
        }
    
    @staticmethod
    def get_control_bounds():
        """获取控制变量边界"""
        return {
            'lower': [DV_MIN, DPHI_MIN],
            'upper': [DV_MAX, DPHI_MAX]
        }
