# MPC控制器输出控制功能说明

## 概述

为了适应不同的使用场景（特别是强化学习训练），我们为MPC控制器添加了灵活的输出控制功能。现在可以根据需要开启或关闭打印信息和可视化功能。

## 新增功能

### 1. 输出控制开关

在 `config.py` 中新增了以下控制开关：

```python
# 输出控制开关
ENABLE_PRINT_OUTPUT = True      # 是否打印详细输出信息
ENABLE_ITERATION_PRINT = True   # 是否打印每步迭代信息
ENABLE_DETAILED_STATUS = True   # 是否打印详细状态信息
ENABLE_VISUALIZATION = True     # 是否启用画图功能
```

### 2. 输出配置类

新增了 `OutputConfig` 类，提供三种预设模式：

- **调试模式 (debug)**: 启用所有输出，适合开发和调试
- **静默模式 (silent)**: 禁用所有输出，适合强化学习训练
- **最小模式 (minimal)**: 只显示重要信息，适合正常运行

## 使用方法

### 方法一：通过主程序参数控制

```python
from main import MPCFlightSimulation

# 调试模式（默认）- 启用所有输出
simulation = MPCFlightSimulation(random_target=True, output_mode='debug')

# 静默模式 - 适合强化学习
simulation = MPCFlightSimulation(random_target=True, output_mode='silent')

# 最小输出模式 - 只显示重要信息
simulation = MPCFlightSimulation(random_target=True, output_mode='minimal')

simulation.run_simulation()
```

### 方法二：动态控制

```python
from config import output_config

# 运行时动态切换模式
output_config.set_silent_mode()    # 切换到静默模式
output_config.set_debug_mode()     # 切换到调试模式
output_config.set_minimal_mode()   # 切换到最小模式
```

## 各模式对比

| 功能 | 调试模式 | 最小模式 | 静默模式 |
|------|----------|----------|----------|
| 任务阶段切换信息 | ✅ | ✅ | ❌ |
| 每步迭代详情 | ✅ | ❌ | ❌ |
| 详细状态信息 | ✅ | ❌ | ❌ |
| 最终结果总结 | ✅ | ✅ | ❌ |
| 轨迹可视化图表 | ✅ | ❌ | ❌ |
| 系统初始化信息 | ✅ | ✅ | ❌ |

## 强化学习应用示例

```python
def train_episode():
    """训练单个episode"""
    # 训练时使用静默模式以提高效率
    simulation = MPCFlightSimulation(
        random_target=True, 
        output_mode='silent'
    )
    simulation.run_simulation()
    return simulation  # 返回仿真结果用于学习

def evaluate_model():
    """评估模型性能"""
    # 评估时使用最小模式查看关键信息
    simulation = MPCFlightSimulation(
        random_target=True, 
        output_mode='minimal'
    )
    simulation.run_simulation()

def debug_model():
    """调试模型"""
    # 调试时使用完整输出
    simulation = MPCFlightSimulation(
        random_target=True, 
        output_mode='debug'
    )
    simulation.run_simulation()
```

## 性能优化

使用静默模式可以显著提高仿真运行速度，特别适合：

1. **强化学习训练**: 大量episode训练时无需输出信息
2. **批量仿真**: 进行参数扫描或Monte Carlo仿真时
3. **自动化测试**: 自动化测试脚本中不需要可视化输出

## 文件修改说明

### 修改的文件：

1. **config.py**: 添加输出控制开关和OutputConfig类
2. **mission_control.py**: 在打印函数中添加开关检查
3. **visualization.py**: 在可视化函数中添加开关检查
4. **main.py**: 添加输出模式参数和控制逻辑

### 新增文件：

1. **example_usage.py**: 使用示例和演示代码
2. **OUTPUT_CONTROL_README.md**: 本说明文档

## 向后兼容性

所有修改都保持了向后兼容性：

- 默认情况下所有输出都是启用的
- 现有代码无需修改即可正常运行
- 只有在明确指定静默模式时才会禁用输出

## 注意事项

1. 在静默模式下，仿真仍会正常运行并记录数据，只是不会显示输出
2. 可视化图表的生成会被跳过，但数据仍然会被收集
3. 动态切换模式会立即生效，影响后续的输出行为
4. 建议在强化学习训练循环中使用静默模式以提高训练效率
