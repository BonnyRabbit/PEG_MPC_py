"""
Visualization Module
可视化和绘图功能
"""

import numpy as np
import matplotlib.pyplot as plt
from config import *
from config import output_config


class FlightVisualization:
    """飞行数据可视化类"""
    
    def __init__(self):
        self.figure_size = (15, 12)
        
    def plot_simulation_results(self, cat_states, cat_controls, track_start_time):
        """
        绘制仿真结果
        
        Args:
            cat_states: 状态历史数据
            cat_controls: 控制历史数据
            track_start_time: 尾追开始时间
        """
        if not output_config.enable_visualization:
            return
            
        # 提取轨迹数据
        x_traj = cat_states[3, :, :]
        y_traj = cat_states[4, :, :]
        x_t_traj = cat_states[8, :, :]
        y_t_traj = cat_states[9, :, :]
        
        plt.figure(figsize=self.figure_size)
        
        # 绘制轨迹
        self._plot_trajectories(x_traj, y_traj, x_t_traj, y_t_traj, track_start_time)
        
        # 绘制距离变化
        self._plot_distance_history(cat_states, track_start_time)
        
        # 绘制ATA和航向角差
        self._plot_angles(cat_states, track_start_time)
        
        # 绘制控制输入
        self._plot_control_inputs(cat_controls)
        
        # 绘制速度
        self._plot_velocities(cat_states, track_start_time)
        
        # 绘制滚转角
        self._plot_roll_angles(cat_states, track_start_time)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_trajectories(self, x_traj, y_traj, x_t_traj, y_t_traj, track_start_time):
        """绘制飞行轨迹"""
        plt.subplot(3, 2, 1)
        plt.plot(x_traj[0, :], y_traj[0, :], 'b-', linewidth=2, label='Pursuer trajectory')
        plt.plot(x_t_traj[0, :], y_t_traj[0, :], 'r-', linewidth=2, label='Target trajectory')
        
        # 标记起始和结束位置
        plt.plot(x_traj[0, 0], y_traj[0, 0], 'bo', markersize=10, label='Pursuer start')
        plt.plot(x_traj[0, -1], y_traj[0, -1], 'bs', markersize=10, label='Pursuer end')
        plt.plot(x_t_traj[0, 0], y_t_traj[0, 0], 'ro', markersize=10, label='Target start')
        plt.plot(x_t_traj[0, -1], y_t_traj[0, -1], 'rs', markersize=10, label='Target end')
        
        # 标记尾追保持开始点
        if track_start_time is not None:
            track_start_idx = int(track_start_time / STEP_HORIZON)
            if track_start_idx < len(x_traj[0, :]):
                plt.plot(x_traj[0, track_start_idx], y_traj[0, track_start_idx], 
                        'g*', markersize=15, label='Track start')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Aircraft Trajectories')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    def _plot_distance_history(self, cat_states, track_start_time):
        """绘制距离变化历史"""
        plt.subplot(3, 2, 2)
        distance_history = cat_states[11, 0, :]
        time_history = np.arange(len(distance_history)) * STEP_HORIZON
        
        plt.plot(time_history, distance_history, 'g-', linewidth=2, label='Actual distance')
        plt.axhline(y=TARGET_DISTANCE, color='r', linestyle='--', linewidth=2, 
                   label=f'Target distance: {TARGET_DISTANCE}m')
        
        # 标记尾追保持阶段
        self._add_track_phase_markers(plt, track_start_time, time_history)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.title('Relative Distance')
        plt.legend()
        plt.grid(True)
    
    def _plot_angles(self, cat_states, track_start_time):
        """绘制ATA和航向角差"""
        plt.subplot(3, 2, 3)
        time_history = np.arange(cat_states.shape[2]) * STEP_HORIZON
        ata_history = np.rad2deg(cat_states[10, 0, :])
        pursuer_heading = np.rad2deg(cat_states[0, 0, :])
        target_heading = np.rad2deg(cat_states[5, 0, :])
        heading_diff = pursuer_heading - target_heading
        
        plt.plot(time_history, ata_history, 'm-', linewidth=2, label='ATA (degrees)')
        plt.plot(time_history, heading_diff, 'c-', linewidth=2, label='Heading difference')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 标记尾追保持阶段
        self._add_track_phase_markers(plt, track_start_time, time_history)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('ATA and Heading Difference')
        plt.legend()
        plt.grid(True)
    
    def _plot_control_inputs(self, cat_controls):
        """绘制控制输入"""
        plt.subplot(3, 2, 4)
        
        if cat_controls.shape[0] > 1:
            dv_history = cat_controls[1:, 0]  # 跳过第一个点（初始值）
            dphi_history = np.rad2deg(cat_controls[1:, 1]) if cat_controls.shape[1] > 1 else np.zeros_like(dv_history)
            control_time = np.arange(len(dv_history)) * STEP_HORIZON
            
            plt.plot(control_time, dv_history, 'b-', linewidth=2, label='dv (m/s²)')
            plt.plot(control_time, dphi_history, 'r-', linewidth=2, label='dphi (deg/s)')
        else:
            print("Warning: Not enough control data to plot")
            
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Inputs')
        plt.legend()
        plt.grid(True)
    
    def _plot_velocities(self, cat_states, track_start_time):
        """绘制速度历史"""
        plt.subplot(3, 2, 5)
        time_history = np.arange(cat_states.shape[2]) * STEP_HORIZON
        pursuer_velocity = cat_states[2, 0, :]
        target_velocity = cat_states[7, 0, :]
        
        plt.plot(time_history, pursuer_velocity, 'b-', linewidth=2, label='Pursuer velocity')
        plt.plot(time_history, target_velocity, 'r-', linewidth=2, label='Target velocity')
        
        # 标记尾追保持阶段
        self._add_track_phase_markers(plt, track_start_time, time_history)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs Time')
        plt.legend()
        plt.grid(True)
    
    def _plot_roll_angles(self, cat_states, track_start_time):
        """绘制滚转角"""
        plt.subplot(3, 2, 6)
        time_history = np.arange(cat_states.shape[2]) * STEP_HORIZON
        pursuer_phi = np.rad2deg(cat_states[1, 0, :])
        target_phi = np.rad2deg(cat_states[6, 0, :])
        
        plt.plot(time_history, pursuer_phi, 'b-', linewidth=2, label='Pursuer roll angle')
        plt.plot(time_history, target_phi, 'r-', linewidth=2, label='Target roll angle')
        plt.axhline(y=np.rad2deg(PHI_MAX), color='k', linestyle='--', alpha=0.5, label='Roll limit (±30°)')
        plt.axhline(y=np.rad2deg(PHI_MIN), color='k', linestyle='--', alpha=0.5)
        
        # 标记尾追保持阶段
        self._add_track_phase_markers(plt, track_start_time, time_history)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Roll Angle (degrees)')
        plt.title('Roll Angle vs Time')
        plt.legend()
        plt.grid(True)
    
    def _add_track_phase_markers(self, plt_obj, track_start_time, time_history):
        """添加尾追保持阶段标记"""
        if track_start_time is not None:
            plt_obj.axvline(x=track_start_time, color='orange', linestyle=':', linewidth=2)
            track_end_time = min(track_start_time + TRACK_HOLD_TIME, time_history[-1])
            plt_obj.axvspan(track_start_time, track_end_time, alpha=0.2, color='orange')


class RealTimeMonitor:
    """实时监控显示类"""
    
    def __init__(self):
        pass
    
    def print_status_header(self):
        """打印状态监控头部信息"""
        if not output_config.enable_print_output:
            return
        print("\n" + "="*80)
        print("MPC Flight Dynamics Simulation Started")
        print("="*80)
    
    def print_system_info(self, initial_conditions):
        """打印系统信息"""
        if not output_config.enable_print_output:
            return
        print(f"Initial Conditions:")
        print(f"  Pursuer: ({initial_conditions.x_init:.1f}, {initial_conditions.y_init:.1f})")
        print(f"  Target:  ({initial_conditions.x_t_init:.1f}, {initial_conditions.y_t_init:.1f})")
        print(f"  Initial distance: {initial_conditions.initial_dist:.1f}m")
        print(f"  Initial ATA: {np.rad2deg(initial_conditions.ata_init):.1f}°")
        print(f"  Target distance: {TARGET_DISTANCE}m")
        print(f"  Track hold time: {TRACK_HOLD_TIME}s")
        print("-"*80)


class DataLogger:
    """数据记录器类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置数据记录"""
        self.cat_states = None
        self.cat_controls = None
        self.times = np.array([[0]])
        
    def initialize(self, X0, u0):
        """初始化数据记录"""
        self.cat_states = self._DM2Arr(X0)
        self.cat_controls = self._DM2Arr(u0[:, 0]).reshape(1, -1)
        
    def log_data(self, X0, u, iteration_time):
        """记录数据"""
        self.cat_states = np.dstack((self.cat_states, self._DM2Arr(X0)))
        self.cat_controls = np.vstack((self.cat_controls, self._DM2Arr(u[:, 0]).reshape(1, -1)))
        self.times = np.vstack((self.times, iteration_time))
        
    def get_logged_data(self):
        """获取记录的数据"""
        return {
            'states': self.cat_states,
            'controls': self.cat_controls,
            'times': self.times
        }
    
    @staticmethod
    def _DM2Arr(dm):
        """将CasADi DM矩阵转换为numpy数组"""
        return np.array(dm.full())
