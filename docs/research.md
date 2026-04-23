# Research

## 当前研究定位

当前仓库已经不再只是几何脚本原型，而是演化成了一个包含：

1. 程序化隧道岩壁生成
2. 4 自由度机器人运动学
3. RL 训练/评估入口
4. 经典环境与 Torch 批量环境
5. PPO / SAC 两套算法实现

的实验工程。

当前研究主线可以概括为：

**在程序化生成的湿喷场景中，研究末端轨迹跟踪任务在不同训练后端与不同算法下的样本效率、吞吐率与稳定性。**

## 当前系统结构

### 几何与任务生成

几何主线仍然来自：

- `rock_wall.py`
- `planner.py`

它们提供：

- 岩壁程序化生成
- 壁面法向估计
- 沿法向内缩后的目标点
- 训练任务采样

### RL 主线

当前 RL 主线由以下部分组成：

- 经典环境：`src/rl_env/math_env.py`
- Torch 批量环境：`src/rl_env/torch_math_env.py`
- 训练环境构建器：`src/rl_env/train_env.py`
- 批量运动学：`src/rock_3D/robot_4dof/torch_kinematics.py`
- PPO：`src/algorithm/ppo.py`
- SAC：`src/algorithm/sac.py`
- 学习率调度：`src/algorithm/lr_schedule.py`
- 训练入口：`src/train.py`
- 评估入口：`src/eval.py`

## 当前任务定义

### 状态

当前 observation 为 18 维：

- 4 维关节角归一化
- 3 维当前末端点
- 3 维目标点
- 3 维目标相对位移
- 4 维上一时刻动作
- 1 维步数比例

### 动作

当前动作为 4 维归一化关节增量：

```text
q_next = clip_joint_limits(q_current + action * max_joint_delta)
```

### 奖励

当前默认主奖励：

```text
r_t = k * [phi(d_t) - phi(d_{t+1})]
```

其中：

- `d_t = ||goal_point - current_point||_2`
- `phi(d) = log(1 + d / goal_tolerance)`

当前成功条件为：

```text
goal_distance <= goal_tolerance
```

## 当前训练后端

### `classic`

`classic` 后端保留单环境语义，便于：

- 验证 reward
- 验证 observation
- 做小规模调试

### `torch`

`torch` 后端使用 `TorchMathEnv` 批量运行多个环境。

这一版本的目标不是完全复制 Isaac Lab 风格，而是：

1. 用 Torch 张量化环境步进主路径
2. 提高 GPU 参与度
3. 提高 PPO/SAC 的总体吞吐率

当前已张量化的部分：

- 关节状态
- FK
- 末端点与目标点距离
- reward 计算
- observation 拼接

当前仍主要在 CPU 上的部分：

- reset 时的任务采样
- planner 内部的目标点生成

因此当前 `torch` 后端可理解为：

**解析式批量环境**

而不是：

**完整 GPU-native 物理仿真器**

## 当前训练配置

当前训练配置重点字段：

```yaml
train:
  device: cuda
  env_backend: torch
  num_envs: 256
  lr_schedule: cosine
  lr_min_ratio: 0.1
```

这表示：

- 优先使用 GPU 训练
- 优先使用 Torch 批量环境
- 一次并行运行 `256` 个环境
- 学习率使用余弦退火

## 当前算法实现状态

### PPO

当前 PPO 特点：

- actor-critic
- GAE
- clipped objective
- advantage normalization
- 可选 value target normalization
- 梯度裁剪
- 余弦学习率调度

当前经验现象：

- 在 `torch` 后端下，PPO 的吞吐提升通常明显
- 当 `num_envs` 提高时，wall-clock 改善通常比 SAC 更显著

### SAC

当前 SAC 特点：

- actor + twin critic + target critic
- replay buffer
- auto alpha
- 可独立控制 actor/critic/alpha 学习率
- 统一余弦学习率调度

当前经验现象：

- `torch` 后端同样能提速
- 但吞吐增益通常低于 PPO
- 一个重要原因是 SAC 在环境交互之外，还需要频繁进行 replay sampling 与多次参数更新

## 当前最重要的研究问题

### 1. reward 是否足够表达任务

当前 reward 只表达“接近目标点”，还没有显式表达：

- 贴壁距离语义
- 喷射姿态
- 法向一致性
- 轨迹平滑性

因此当前结果更应解释为：

**末端点到点跟踪性能**

而不是完整湿喷任务性能。

### 2. GPU 批量环境是否真正带来有效提速

当前 `torch` 后端已经带来了吞吐改善，但仍存在两个瓶颈：

1. reset / planner 阶段仍偏 CPU
2. SAC 的增益仍明显小于 PPO

所以后续研究可以沿两条线展开：

- 提高 `TorchMathEnv` 的环境侧吞吐
- 提高 SAC 在该环境下的总体吞吐率

### 3. 不同精度阈值下的训练表现

当前 `goal_tolerance` 可以作为任务难度开关。

一条自然的实验主线是：

```text
0.5 -> 0.2 -> 0.1 -> 0.05 -> 0.02
```

在这一主线下，比较：

- PPO vs SAC
- classic vs torch
- 不同 reward 形式
- 不同学习率调度

## 当前建议的实验维度

### 训练后端

- `classic`
- `torch`

### 算法

- PPO
- SAC

### reward 消融

- `phi(d) = log(1 + d / eps)`
- `phi(d) = d`
- `phi(d_t) - phi(d_{t+1})`
- `phi(d_t) - gamma * phi(d_{t+1})`
- `-phi(d_t)`

### 学习率调度

- `none`
- `cosine`

### 任务难度

- 不同 `goal_tolerance`
- 是否启用可达性筛选

## 目前已经失效的旧结论

以下旧结论已经不再适用：

1. `src/train.py`、`src/eval.py` 为空  
   当前两者都已有实际实现。

2. 仓库没有模型代码  
   当前已包含 PPO、SAC、MLP、学习率调度器等实现。

3. 仓库没有 RL 环境实现  
   当前已有 `MathEnv` 与 `TorchMathEnv`。

4. 仓库只有几何脚本  
   当前已经是“几何生成 + RL 训练”的一体化实验工程。

## 当前工作重点

从工程与研究优先级看，当前最值得继续推进的是：

1. 固定 reward 主线，避免频繁改动任务定义。
2. 用 `torch` 后端继续做 PPO/SAC 速度与收敛比较。
3. 用不同 `goal_tolerance` 构造难度阶梯。
4. 单独研究 SAC 吞吐率提升问题。
5. 如果后续继续追求吞吐，再把 reset / planner 也逐步 torch 化。
