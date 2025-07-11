# MPC Pursue-Evasion Game

本项目是基于模型预测控制(MPC)的飞机追击目标仿真系统的模块化重构版本。

## 项目结构

```
├── config.py              # 配置参数和常量
├── FDM.py   # 飞机动力学模型和敌方机动库
├── mpc_controller.py      # MPC控制器
├── mission_control.py     # 任务控制逻辑
├── visualization.py       # 可视化和绘图
├── main.py                # 主程序入口
├── mpc_flight_dynamics.py # 原始单文件版本（保留作为参考）
└── README_modular.md      # 本说明文件
```

## 模块说明

### 1. config.py - 配置模块
- **功能**: 集中管理所有系统参数、权重设置和常量
- **主要内容**:
  - 系统基本参数 (重力加速度、时间步长等)
  - MPC参数 (预测步数、优化选项等)
  - 任务参数 (目标距离、追踪时间等)
  - 控制限制和边界条件
  - 状态归一化系数
  - 初始条件配置类

### 2. FDM.py - 飞机动力学模块
- **功能**: 实现飞机动力学模型和敌机机动控制
- **主要类**:
  - `FlightDynamicsModel`: 主要的飞机动力学模型
  - `EnemyManeuverController`: 敌机机动控制器
- **主要功能**:
  - 飞机状态更新计算
  - ATA (Angle To Aspect) 计算
  - 敌机智能机动 (单向转弯、S转弯)
  - 符号版本的动力学函数 (用于MPC优化)

### 3. mpc_controller.py - MPC控制器模块
- **功能**: 实现模型预测控制器
- **主要类**:
  - `MPCController`: MPC控制器主类
- **主要功能**:
  - 符号变量设置
  - 优化问题构建
  - 成本函数和约束定义
  - 求解器配置和调用
  - 时间步推进

### 4. mission_control.py - 任务控制模块
- **功能**: 管理任务阶段和状态监控
- **主要类**:
  - `MissionController`: 任务阶段控制
  - `StateAnalyzer`: 状态分析器
  - `ResultsAnalyzer`: 结果分析器
- **主要功能**:
  - 任务阶段切换 (接近阶段 ↔ 尾追保持阶段)
  - 实时状态分析和监控
  - 仿真结果统计和总结

### 5. visualization.py - 可视化模块
- **功能**: 数据可视化和绘图
- **主要类**:
  - `FlightVisualization`: 飞行数据可视化
  - `RealTimeMonitor`: 实时监控显示
  - `DataLogger`: 数据记录器
- **主要功能**:
  - 飞行轨迹绘制
  - 状态历史图表
  - 控制输入可视化
  - 实时状态监控显示

### 6. main.py - 主程序模块
- **功能**: 程序入口点，整合所有模块
- **主要类**:
  - `MPCFlightSimulation`: 仿真主控制类
- **主要功能**:
  - 初始化各个子系统
  - 协调仿真流程
  - 管理主仿真循环

## 使用方法

### 基本运行
```python
# 直接运行主程序
python main.py
```

### 自定义使用
```python
from main import MPCFlightSimulation

# 创建仿真实例 (随机目标位置)
simulation = MPCFlightSimulation(random_target=True)

# 运行仿真
simulation.run_simulation()
```

### 参数配置
在 `config.py` 中修改相应参数：
```python
# 修改MPC参数
N = 80              # 预测步数
TARGET_DISTANCE = 250  # 目标距离

# 修改权重
Q_ATA_NORM = 1000.0    # ATA权重
Q_POS_NORM = 20.0      # 距离权重

## 依赖要求

- Python 3.7+
- CasADi >= 3.5
- NumPy >= 1.19
- Matplotlib >= 3.3
