"""
Main Application Entry Point
主程序入口，整合所有模块完成MPC飞行动力学仿真
"""

from time import time
import casadi as ca
import numpy as np

# 导入自定义模块
from config import *
from FDM import FlightDynamicsModel
from mpc_controller import MPCController
from mission_control import MissionController, StateAnalyzer, ResultsAnalyzer
from visualization import FlightVisualization, RealTimeMonitor, DataLogger


class MPCFlightSimulation:
    """MPC飞行仿真主类"""
    
    def __init__(self, random_target=True, output_mode='debug'):
        """
        初始化仿真系统
        
        Args:
            random_target: 是否随机生成目标位置
            output_mode: 输出模式 ('debug', 'silent', 'minimal')
                - 'debug': 启用所有输出（适合调试）
                - 'silent': 禁用所有输出（适合强化学习）
                - 'minimal': 只保留重要信息（适合正常运行）
        """
        # 设置输出模式
        from config import output_config
        if output_mode == 'silent':
            output_config.set_silent_mode()
        elif output_mode == 'debug':
            output_config.set_debug_mode()
        elif output_mode == 'minimal':
            output_config.set_minimal_mode()
        
        # 初始化各个组件
        self.initial_conditions = InitialConditions(random_target)
        self.real_model = FlightDynamicsModel()
        self.mpc_controller = MPCController()
        self.mission_controller = MissionController()
        self.state_analyzer = StateAnalyzer()
        self.results_analyzer = ResultsAnalyzer()
        self.visualization = FlightVisualization()
        self.monitor = RealTimeMonitor()
        self.data_logger = DataLogger()
        
        # 初始化仿真状态
        self.setup_simulation()
        
    def setup_simulation(self):
        """设置仿真初始状态"""
        # 初始状态
        self.t0 = 0
        self.state_init = ca.DM([
            self.initial_conditions.psi_init, 
            self.initial_conditions.phi_init, 
            self.initial_conditions.v_init, 
            self.initial_conditions.x_init, 
            self.initial_conditions.y_init,
            self.initial_conditions.psi_t_init, 
            self.initial_conditions.phi_t_init, 
            self.initial_conditions.v_t_init, 
            self.initial_conditions.x_t_init, 
            self.initial_conditions.y_t_init,
            self.initial_conditions.ata_init, 
            self.initial_conditions.rel_dist_init
        ])
        
        # 初始控制和状态轨迹
        self.u0 = ca.DM.zeros((self.mpc_controller.n_controls, N))
        self.X0 = ca.repmat(self.state_init, 1, N+1)
        
        # 初始化计数器
        self.mpc_iter = 0
        
        # 初始化数据记录
        self.data_logger.initialize(self.X0, self.u0)
        
    def run_simulation(self):
        """运行主仿真循环"""
        self.monitor.print_status_header()
        self.monitor.print_system_info(self.initial_conditions)
        
        main_loop_start = time()
        
        # 主仿真循环
        while not self.mission_controller.should_terminate(self.mpc_iter * STEP_HORIZON):
            t1 = time()
            
            # 状态分析
            state_analysis = self.state_analyzer.analyze_state(self.state_init, self.mpc_iter)
            
            # 更新任务阶段
            self.mission_controller.update_phase(
                state_analysis['current_time'],
                state_analysis['current_ata_rad'],
                state_analysis['current_distance']
            )
            
            # 检查任务是否完成
            if self.mission_controller.is_mission_complete():
                break
            
            # MPC优化求解
            u, self.X0 = self.mpc_controller.solve(self.state_init, self.X0, self.u0)
            
            # 打印详细状态信息
            self.state_analyzer.print_detailed_status(
                state_analysis, u, self.real_model, self.mpc_iter
            )
            
            # 记录数据
            t2 = time()
            iteration_time = t2 - t1
            self.data_logger.log_data(self.X0, u, iteration_time)
            
            # 状态推进
            self.t0, self.state_init, self.u0 = self.mpc_controller.shift_timestep(
                STEP_HORIZON, self.t0, self.state_init, u, self.real_model
            )
            
            # 更新状态轨迹初始猜测
            self.X0 = ca.horzcat(self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1))
            
            # 打印迭代信息
            self.state_analyzer.print_iteration_info(
                self.state_init, u, self.mpc_iter, iteration_time, self.mission_controller
            )
            
            self.mpc_iter += 1
        
        # 仿真完成，分析结果
        main_loop_time = time() - main_loop_start
        self.analyze_and_visualize_results(main_loop_time)
        
    def analyze_and_visualize_results(self, main_loop_time):
        """分析和可视化结果"""
        # 获取记录的数据
        logged_data = self.data_logger.get_logged_data()
        
        # 打印最终结果总结
        self.results_analyzer.print_final_summary(
            main_loop_time, self.mpc_iter, self.state_init, 
            logged_data['times'], self.mission_controller
        )
        
        # 可视化结果
        self.visualization.plot_simulation_results(
            logged_data['states'], 
            logged_data['controls'], 
            self.mission_controller.track_start_time
        )


def main():
    """主函数"""
    # 创建并运行仿真
    # 可以通过output_mode参数控制输出级别：
    # simulation = MPCFlightSimulation(random_target=True, output_mode='silent')    # 强化学习模式（无输出）
    # simulation = MPCFlightSimulation(random_target=True, output_mode='minimal')   # 最小输出模式
    simulation = MPCFlightSimulation(random_target=True, output_mode='debug')     # 调试模式（全部输出）
    simulation.run_simulation()


if __name__ == '__main__':
    main()
