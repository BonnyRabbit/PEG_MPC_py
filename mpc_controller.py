"""
MPC Controller Module
模型预测控制器
"""

import casadi as ca
import numpy as np
from config import *
from FDM import fdm_step_symbolic


class MPCController:
    """模型预测控制器类"""
    
    def __init__(self):
        self.setup_symbolic_variables()
        self.setup_optimization_problem()
        self.setup_solver()
        
    def setup_symbolic_variables(self):
        """设置符号变量"""
        # 状态符号变量 [psi, phi, v, x, y, psi_t, phi_t, v_t, x_t, y_t, ata, rel_dist]
        self.psi = ca.SX.sym('psi')
        self.phi = ca.SX.sym('phi')
        self.v = ca.SX.sym('v')
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.psi_t = ca.SX.sym('psi_t')
        self.phi_t = ca.SX.sym('phi_t')
        self.v_t = ca.SX.sym('v_t')
        self.x_t = ca.SX.sym('x_t')
        self.y_t = ca.SX.sym('y_t')
        self.ata = ca.SX.sym('ata')
        self.rel_dist = ca.SX.sym('rel_dist')
        
        self.states = ca.vertcat(
            self.psi, self.phi, self.v, self.x, self.y,
            self.psi_t, self.phi_t, self.v_t, self.x_t, self.y_t,
            self.ata, self.rel_dist
        )
        self.n_states = self.states.numel()  # 12个状态
        
        # 控制符号变量
        self.dv = ca.SX.sym('dv')        # 速度变化率
        self.dphi = ca.SX.sym('dphi')    # 滚转角速率
        self.controls = ca.vertcat(self.dv, self.dphi)
        self.n_controls = self.controls.numel()  # 2个控制输入
        
        # 优化变量矩阵
        self.X = ca.SX.sym('X', self.n_states, N + 1)     # 状态轨迹矩阵
        self.U = ca.SX.sym('U', self.n_controls, N)       # 控制轨迹矩阵
        self.P = ca.SX.sym('P', self.n_states)            # 初始状态参数向量
        
    def setup_optimization_problem(self):
        """设置优化问题"""
        # 创建预测模型函数
        self.f_predict = ca.Function('f_predict', 
                                    [self.states, self.controls], 
                                    [fdm_step_symbolic(self.states, self.controls)])
        
        # 成本函数和约束
        self.cost_fn = 0
        self.g = self.X[:, 0] - self.P  # 初始状态约束
        
        # 构建成本函数和动力学约束
        for k in range(N):
            st = self.X[:, k]
            con = self.U[:, k]
            
            # 归一化的成本函数项
            ata_error_norm = st[10] / ANGLE_SCALE
            pos_error_norm = (st[11] - TARGET_DISTANCE) / DISTANCE_SCALE
            acceleration_penalty_norm = con[0]**2
            steering_penalty_norm = (con[1] / ANGLE_SCALE)**2
            
            # 动态权重：当ATA较大时，优先建立尾追；当ATA较小时，同时控制距离
            ata_magnitude = ca.fabs(st[10])
            distance_weight_factor = 1 / (1 + ca.exp(-10 * (0.5 - ata_magnitude)))
            
            # 组合归一化成本函数
            self.cost_fn = self.cost_fn + \
                Q_ATA_NORM * ata_error_norm**2 + \
                Q_POS_NORM * distance_weight_factor * pos_error_norm**2 + \
                R_DV_NORM * acceleration_penalty_norm + \
                R_DPHI_NORM * steering_penalty_norm
            
            # 动力学约束
            st_next = self.X[:, k+1]
            st_next_predicted = self.f_predict(st, con)
            self.g = ca.vertcat(self.g, st_next - st_next_predicted)
        
        # 优化变量
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # 状态变量
            self.U.reshape((-1, 1))    # 控制变量
        )
        
    def setup_solver(self):
        """设置求解器"""
        nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, IPOPT_OPTIONS)
        
        # 设置变量边界
        self.setup_bounds()
        
    def setup_bounds(self):
        """设置变量边界"""
        bounds = BoundaryConditions()
        state_bounds = bounds.get_state_bounds()
        control_bounds = bounds.get_control_bounds()
        
        self.lbx = ca.DM.zeros((self.n_states*(N+1) + self.n_controls*N, 1))
        self.ubx = ca.DM.zeros((self.n_states*(N+1) + self.n_controls*N, 1))
        
        # 状态变量边界
        for k in range(N+1):
            idx = k * self.n_states
            self.lbx[idx:idx+self.n_states] = state_bounds['lower']
            self.ubx[idx:idx+self.n_states] = state_bounds['upper']
        
        # 控制变量边界
        for k in range(N):
            idx = self.n_states*(N+1) + k*self.n_controls
            self.lbx[idx:idx+self.n_controls] = control_bounds['lower']
            self.ubx[idx:idx+self.n_controls] = control_bounds['upper']
        
        self.args = {
            'lbg': ca.DM.zeros((self.n_states*(N+1), 1)),
            'ubg': ca.DM.zeros((self.n_states*(N+1), 1)),
            'lbx': self.lbx,
            'ubx': self.ubx
        }
        
    def solve(self, state_init, X0, u0):
        """
        求解MPC优化问题
        
        Args:
            state_init: 当前状态
            X0: 状态轨迹初始猜测
            u0: 控制序列初始猜测
            
        Returns:
            优化的控制序列和状态轨迹
        """
        # 设置参数
        self.args['p'] = state_init
        
        # 设置初始猜测
        self.args['x0'] = ca.vertcat(
            ca.reshape(X0, self.n_states*(N+1), 1),
            ca.reshape(u0, self.n_controls*N, 1)
        )
        
        # 求解优化问题
        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        
        # 提取解
        u = ca.reshape(sol['x'][self.n_states * (N + 1):], self.n_controls, N)
        X0_new = ca.reshape(sol['x'][: self.n_states * (N+1)], self.n_states, N+1)
        
        return u, X0_new
    
    def shift_timestep(self, step_horizon, t0, state_init, u, real_model):
        """
        时间步推进函数
        
        Args:
            step_horizon: 时间步长
            t0: 当前时间
            state_init: 当前状态
            u: 控制序列
            real_model: 真实模型
            
        Returns:
            更新后的时间、状态和控制序列
        """
        # 使用真实模型进行状态更新
        current_state = np.array(state_init.full()).flatten()
        current_control = np.array(u[:, 0].full()).flatten()
        
        # 调用真实模型
        next_state = real_model.run(current_state, current_control, t0)
        next_state = ca.DM(next_state)

        t0 = t0 + step_horizon
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )

        return t0, next_state, u0
    
    @staticmethod
    def DM2Arr(dm):
        """将CasADi DM矩阵转换为numpy数组"""
        return np.array(dm.full())
