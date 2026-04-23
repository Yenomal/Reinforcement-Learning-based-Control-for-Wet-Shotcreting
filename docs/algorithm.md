# 算法记录

本文档记录当前 RL 分支的真实实现状态。当前版本以 `TorchMathEnv + GPU 并行训练 + 余弦学习率调度` 为准。

## 代码入口

- 经典环境：`src/rl_env/math_env.py`
- Torch 批量环境：`src/rl_env/torch_math_env.py`
- 训练环境统一入口：`src/rl_env/train_env.py`
- Torch 批量运动学：`src/rock_3D/robot_4dof/torch_kinematics.py`
- PPO：`src/algorithm/ppo.py`
- SAC：`src/algorithm/sac.py`
- 学习率调度：`src/algorithm/lr_schedule.py`
- Buffer：`src/component/buffer.py`
- 训练入口：`src/train.py`
- 评估入口：`src/eval.py`

## 当前版本结论

当前系统可以概括为：

1. 任务是 4 自由度机械臂末端到目标点的跟踪控制。
2. 环境同时保留 `classic` 与 `torch` 两个训练后端。
3. `torch` 后端支持批量环境，主要目的是提高 GPU 利用率与训练吞吐。
4. PPO 与 SAC 共用同一套训练入口与同一套学习率调度接口。
5. reward 当前采用单主项势函数差分，不依赖成功大奖励。

## 当前环境定义

### observation

当前 observation 维度是 `18`，动作维度是 `4`。

observation 由以下部分拼接：

- `q_norm`：关节角归一化，4 维
- `current_point_norm`：当前末端点归一化坐标，3 维
- `goal_point_norm`：目标点归一化坐标，3 维
- `delta_point_norm`：目标点相对当前点的归一化位移，3 维
- `prev_action`：上一时刻动作，4 维
- `step_ratio`：步数比例，1 维

即：

```text
obs = [
  q0, q1, q2, q3,
  cur_x, cur_y, cur_z,
  goal_x, goal_y, goal_z,
  delta_x, delta_y, delta_z,
  prev_a0, prev_a1, prev_a2, prev_a3,
  step_ratio
]
```

### 动作空间

当前动作是 4 维连续动作，对应各关节的归一化增量命令。

环境执行逻辑为：

```text
bounded_action = clip(action, -1, 1)
q_next = clip_joint_limits(q_current + bounded_action * max_joint_delta)
```

其中 `max_joint_delta_deg` 来自配置文件，表示每步允许的最大关节变化量。

### 当前 reward

当前 reward 只保留主奖励项：

```text
r_t = k * [phi(d_t) - phi(d_{t+1})]
```

其中：

- `k = progress_reward_weight`
- `d_t = ||goal_point - current_point||_2`
- `phi(d) = log(1 + d / goal_tolerance)`

代码位置：

- `src/rl_env/math_env.py`
- `src/rl_env/torch_math_env.py`

当前其它 reward 参数如：

- `success_reward`
- `step_penalty`
- `action_l2_weight`
- `action_smoothness_weight`
- `boundary_penalty`

仍保留在配置中，便于后续消融，但当前默认不参与实际 reward 主体。

### 终止条件

当前 episode 终止规则：

- 成功：`goal_distance <= goal_tolerance`
- 截断：`current_step >= max_episode_steps`

## 两种训练环境后端

### `classic`

`classic` 后端直接使用 `MathEnv`。

特点：

- 逻辑最直观
- 调试方便
- 适合验证 reward、状态与训练流程正确性
- 环境步进主要依赖 Python + NumPy

### `torch`

`torch` 后端使用 `TorchMathEnv`。

特点：

- 采用批量状态张量
- FK、reward、observation 构造都在 Torch 中完成
- 更适合高 `num_envs` 并行训练
- 与 `TorchRobotKinematics` 配合，提高 GPU 利用率

当前 `TorchMathEnv` 已经张量化的部分包括：

- 关节状态
- FK
- goal distance
- reward 计算
- observation 拼接

当前仍在 CPU 上的部分主要是：

- reset 时的任务采样
- 基于现有 planner 的目标生成逻辑

因此当前版本是“环境步进主路径 GPU 化”，不是完全 GPU-native 的仿真系统。

## PPO 当前实现

### 结构

PPO 使用标准 actor-critic 结构：

- actor：输出动作分布均值
- critic：输出状态价值
- `log_std`：状态无关的可训练参数向量

动作分布：

```text
pi(a|s) = Normal(mean(s), std)
action = tanh(sample)
```

### 默认超参数

默认超参数在 `src/algorithm/ppo.py` 中定义。

当前默认设置包括：

- `lr = 3e-4`
- `gae_lambda = 0.95`
- `clip_ratio = 0.2`
- `value_coef = 0.5`
- `entropy_coef = 0.0`
- `max_grad_norm = 0.5`
- `rollout_steps = 2048`
- `update_epochs = 10`
- `minibatch_size = 64`
- `normalize_advantages = true`
- `normalize_value_targets = false`

### 训练方式

PPO 每次先收集一段 on-policy rollout，再统一更新。

在 `torch` 后端下：

- 目标总 rollout batch 大小仍由 `rollout_steps` 控制
- 每个环境实际 rollout 长度为：

```text
rollout_steps_per_env = rollout_steps / num_envs
```

所以高并行时每个 env 每轮只跑较短时间，但总 batch 大小保持不变。

### 梯度裁剪

PPO 当前保留梯度裁剪：

```text
clip_grad_norm_(actor + critic + log_std, max_grad_norm)
```

## SAC 当前实现

### 结构

SAC 当前采用：

- 一个 actor
- 两个 critic
- 两个 target critic
- 一个自适应温度参数 `alpha`

动作分布采用 tanh-squashed Gaussian。

### 默认超参数

默认超参数在 `src/algorithm/sac.py` 中定义。

当前默认设置包括：

- `actor_lr = 3e-4`
- `critic_lr = 3e-4`
- `alpha_lr = 3e-4`
- `tau = 0.005`
- `batch_size = 256`
- `buffer_size = 1_000_000`
- `learning_starts = 100`
- `updates_per_step = 1`
- `log_interval_steps = 2000`
- `alpha_init = 1.0`
- `target_entropy = auto`

### 训练方式

SAC 是 off-policy：

1. 环境交互得到 transition
2. 写入 replay buffer
3. 从 replay buffer 随机采样 batch
4. 更新 twin critic、actor、alpha

在 `torch` 后端下：

- 一次环境步进会并行得到 `num_envs` 条 transition
- 当前实现为了保持 update-to-data 比例，会执行：

```text
updates = updates_per_step * num_envs
```

也就是说，环境吞吐增加后，更新次数也随之提高。

### 梯度裁剪

SAC 当前默认没有显式梯度裁剪。

## 统一余弦学习率调度

当前 PPO 与 SAC 已统一接入 `OptimizerLRScheduler`。

配置项位于：

```yaml
train:
  lr_schedule: cosine
  lr_min_ratio: 0.1
```

含义：

- `lr_schedule = none`：不调度
- `lr_schedule = cosine`：余弦退火
- `lr_min_ratio`：最终学习率相对初始学习率的比例

即：

```text
lr_final = lr_initial * lr_min_ratio
```

当前调度覆盖：

- PPO：`lr`
- SAC：`actor_lr`、`critic_lr`、`alpha_lr`

## 当前日志

训练日志当前会输出：

- `mean_reward`
- `mean_length`
- `success_rate`
- `mean_min_goal_distance`
- `policy_loss`
- `value_loss`
- 当前学习率

当前日志仍是 episode-level 聚合为主，适合低并行或中等并行设定。

## 当前主要问题

1. `TorchMathEnv` 仍然在 reset 阶段依赖 CPU planner，GPU 并行优势还没有完全释放。
2. 高并行下 PPO 的 episode 统计会变稀疏，需要更谨慎解读日志。
3. SAC 的吞吐增益目前明显低于 PPO，后续可以单独作为研究目标。
4. 当前 reward 已经足够干净，但仍然只是点到点跟踪，还没有纳入姿态、法向或贴壁语义。
5. 当前 batch 环境的速度已经显著提升，但 GPU 利用率仍未完全吃满。

## 下一步建议

1. 继续用 `torch` 后端测试不同 `num_envs` 下的吞吐与收敛曲线。
2. 比较 PPO 与 SAC 在 `classic` / `torch` 两种后端下的 wall-clock 效率。
3. 如果后续继续追求吞吐，可优先考虑把 reset / task sampling 也 torch 化。
4. reward 侧暂时保持主奖励干净，优先观察不同精度阈值下的收敛情况。
