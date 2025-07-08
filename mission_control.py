"""
Mission Control Module
任务控制逻辑，包含任务阶段管理和状态监控
"""

import numpy as np
from config import *
from config import output_config


class MissionController:
    """任务控制器类"""
    
    def __init__(self):
        self.current_phase = PHASE_APPROACH
        self.track_start_time = None
        self.mission_complete = False
        
    def update_phase(self, current_time, current_ata_rad, current_distance):
        """
        更新任务阶段
        
        Args:
            current_time: 当前时间
            current_ata_rad: 当前ATA角度（弧度）
            current_distance: 当前距离
            
        Returns:
            阶段是否发生变化
        """
        phase_changed = False
        
        if self.current_phase == PHASE_APPROACH:
            # 检查是否可以切换到尾追保持阶段
            if abs(current_distance - TARGET_DISTANCE) < 20 and current_ata_rad < 0.5:
                self.current_phase = PHASE_TRACK
                self.track_start_time = current_time
                phase_changed = True
                if output_config.enable_print_output:
                    print(f"\n=== 切换到尾追保持阶段 (时间: {current_time:.1f}s) ===\n")
        
        elif self.current_phase == PHASE_TRACK:
            # 检查尾追保持时间是否完成
            if current_time - self.track_start_time >= TRACK_HOLD_TIME:
                self.mission_complete = True
                if output_config.enable_print_output:
                    print(f"\n=== 尾追保持任务完成 (持续时间: {current_time - self.track_start_time:.1f}s) ===\n")
                return phase_changed
            
            # 检查是否需要重新进入接近阶段
            if current_ata_rad > 0.5 or current_distance > TARGET_DISTANCE + 50:
                self.current_phase = PHASE_APPROACH
                phase_changed = True
                if output_config.enable_print_output:
                    print(f"\n=== 重新进入接近阶段 (时间: {current_time:.1f}s) ===\n")
        
        return phase_changed
    
    def is_mission_complete(self):
        """检查任务是否完成"""
        return self.mission_complete
    
    def get_phase_string(self):
        """获取当前阶段的字符串描述"""
        return "接近" if self.current_phase == PHASE_APPROACH else "尾追保持"
    
    def get_track_duration(self, current_time):
        """获取尾追保持持续时间"""
        if self.track_start_time is not None:
            return current_time - self.track_start_time
        return 0
    
    def should_terminate(self, current_time):
        """检查是否应该终止仿真"""
        return (self.mission_complete or 
                current_time >= SIM_TIME)


class StateAnalyzer:
    """状态分析器类"""
    
    @staticmethod
    def analyze_state(state_init, mpc_iter):
        """
        分析当前状态
        
        Args:
            state_init: 当前状态
            mpc_iter: MPC迭代次数
            
        Returns:
            状态分析结果字典
        """
        current_time = mpc_iter * STEP_HORIZON
        current_ata_rad = float(state_init[10])
        current_ata_deg = np.degrees(abs(current_ata_rad))
        current_dist = float(state_init[11])
        pursuer_heading = np.degrees(float(state_init[0]))
        target_heading = np.degrees(float(state_init[5]))
        
        # 计算归一化误差值
        ata_magnitude = abs(current_ata_rad)
        distance_weight_factor = 1 / (1 + np.exp(-10 * (0.5 - ata_magnitude)))
        ata_error_norm = current_ata_rad / ANGLE_SCALE
        pos_error_norm = (current_dist - TARGET_DISTANCE) / DISTANCE_SCALE
        
        return {
            'current_time': current_time,
            'current_ata_rad': current_ata_rad,
            'current_ata_deg': current_ata_deg,
            'current_distance': current_dist,
            'pursuer_heading': pursuer_heading,
            'target_heading': target_heading,
            'distance_weight_factor': distance_weight_factor,
            'ata_error_norm': ata_error_norm,
            'pos_error_norm': pos_error_norm
        }
    
    @staticmethod
    def print_detailed_status(state_analysis, u, real_model, mpc_iter):
        """
        打印详细状态信息
        
        Args:
            state_analysis: 状态分析结果
            u: 控制输入
            real_model: 真实模型
            mpc_iter: MPC迭代次数
        """
        if not output_config.enable_detailed_status:
            return
            
        if mpc_iter % 10 == 0:  # 每10步打印一次详细信息
            current_time = state_analysis['current_time']
            current_ata_deg = state_analysis['current_ata_deg']
            current_dist = state_analysis['current_distance']
            pursuer_heading = state_analysis['pursuer_heading']
            target_heading = state_analysis['target_heading']
            ata_error_norm = state_analysis['ata_error_norm']
            pos_error_norm = state_analysis['pos_error_norm']
            distance_weight_factor = state_analysis['distance_weight_factor']
            
            print(f"\n时间: {current_time:.1f}s | ATA: {current_ata_deg:.1f}° | 距离: {current_dist:.0f}m")
            print(f"追击者航向: {pursuer_heading:.1f}° | 目标航向: {target_heading:.1f}°")
            print(f"控制输入: 加速度={float(u[0,0]):.2f}, 转向={np.degrees(float(u[1,0])):.1f}°/s")
            print(f"归一化误差: ATA_norm={ata_error_norm:.3f}, Pos_norm={pos_error_norm:.3f}")
            print(f"归一化权重: ATA={Q_ATA_NORM:.1f}, Pos={Q_POS_NORM:.1f}(×{distance_weight_factor:.2f})")
            print(f"敌机机动状态: {real_model.maneuver_controller.get_status_info()}")
            
            if current_ata_deg > 30:
                print(">> 当前策略：优先建立尾追状态（ATA→0）")
            else:
                print(">> 当前策略：保持尾追并控制距离")
    
    @staticmethod
    def print_iteration_info(state_init, u, mpc_iter, iteration_time, mission_controller):
        """
        打印迭代信息
        
        Args:
            state_init: 当前状态
            u: 控制输入
            mpc_iter: MPC迭代次数
            iteration_time: 迭代时间
            mission_controller: 任务控制器
        """
        if not output_config.enable_iteration_print:
            return
            
        current_time = mpc_iter * STEP_HORIZON
        current_distance = float(state_init[11])
        pursuer_pos = [float(state_init[3]), float(state_init[4])]
        target_pos = [float(state_init[8]), float(state_init[9])]
        control_applied = [float(u[0, 0]), float(u[1, 0])]
        pursuer_heading = np.rad2deg(float(state_init[0]))
        target_heading = np.rad2deg(float(state_init[5]))
        heading_diff = pursuer_heading - target_heading
        
        phase_str = mission_controller.get_phase_string()
        
        print(f"Iteration {mpc_iter}: Time={iteration_time:.4f}s [{phase_str}]")
        print(f"  Distance={current_distance:.2f}, ATA={np.rad2deg(float(state_init[10])):.1f}°")
        print(f"  relative_velocity={float(state_init[2] - state_init[7]):.1f} m/s")
        print(f"  Pursuer: pos=({pursuer_pos[0]:.1f}, {pursuer_pos[1]:.1f}), heading={pursuer_heading:.1f}°")
        print(f"  Target:  pos=({target_pos[0]:.1f}, {target_pos[1]:.1f}), heading={target_heading:.1f}°")
        print(f"  Heading diff: {heading_diff:.1f}°")
        print(f"  Control: dv={control_applied[0]:.3f}, dphi={np.rad2deg(control_applied[1]):.1f}°")
        
        if mission_controller.current_phase == PHASE_TRACK and mission_controller.track_start_time is not None:
            track_duration = mission_controller.get_track_duration(current_time)
            print(f"  尾追保持时间: {track_duration:.1f}/{TRACK_HOLD_TIME}s")
        print()


class ResultsAnalyzer:
    """结果分析器类"""
    
    @staticmethod
    def print_final_summary(main_loop_time, mpc_iter, state_init, times, mission_controller):
        """
        打印最终结果总结
        
        Args:
            main_loop_time: 主循环总时间
            mpc_iter: 总MPC迭代次数
            state_init: 最终状态
            times: 时间记录
            mission_controller: 任务控制器
        """
        if not output_config.enable_print_output:
            return
            
        final_distance = float(state_init[11])
        final_ata = np.rad2deg(float(state_init[10]))
        final_heading_diff = np.rad2deg(float(state_init[0]) - float(state_init[5]))
        
        print('\n' + '='*50)
        print('任务完成总结:')
        print('='*50)
        print('Total time: ', main_loop_time, 's')
        print('simulation time: ', mpc_iter * STEP_HORIZON, 's')
        print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
        print('final distance: ', final_distance)
        print('final ATA: ', final_ata, '°')
        print('final heading difference: ', final_heading_diff, '°')
        print('target distance: ', TARGET_DISTANCE)
        print('distance error: ', abs(final_distance - TARGET_DISTANCE))
        
        if mission_controller.track_start_time is not None:
            total_track_time = (mpc_iter * STEP_HORIZON) - mission_controller.track_start_time
            print(f'尾追保持时间: {total_track_time:.1f}s / {TRACK_HOLD_TIME}s')
            if total_track_time >= TRACK_HOLD_TIME:
                print('尾追保持任务成功完成！')
            else:
                print('尾追保持时间不足')
        
        print('='*50)
