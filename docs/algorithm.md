# 算法记录

本文档用于记录当前项目中的强化学习算法实现状态，重点分析当前正在使用的 `PPO`，并为后续的 `SAC` 预留位置。

当前代码入口：

- 环境：`src/rl_env/math_env.py`
- PPO：`src/algorithm/ppo.py`
- SAC：`src/algorithm/sac.py`
- Buffer：`src/component/buffer.py`
- 训练入口：`src/train.py`
- 评估入口：`src/eval.py`

## 当前阶段结论

如果只看“算法基础”这一层，当前最关键的事实有三条：

1. 当前环境已经是纯 3D 点控制环境，observation、action、reward 都不再依赖 `uv` 作为真状态。
2. 当前 reward 采用“势能函数差分 + 到达奖励”的最小化设计，核心项是 `d_{t-1} - gamma * d_t`。
3. PPO 主体实现保持不变，因此这一阶段的重点是环境定义和 reward 设计，而不是算法框架本身。

换句话说，现在 PPO 本身不是空的，而且它当前学习的问题已经可以概括为：

- 输入：3D 工作点位置
- 控制：3D 点增量动作
- 目标：最小化到目标点的 3D 距离

它仍然不是最终的机械臂控制任务，但已经是一个逻辑自洽的 3D 基础环境。

## 当前环境与 reward 状态

### 当前 observation 是什么

`MathEnv` 当前的 observation 维度是 `13`，动作维度是 `3`。

当前 observation 由以下部分拼接而成：

- `current_point_norm`：当前 `retreated_point` 的归一化 3D 坐标，3 维
- `goal_point_norm`：目标 `retreated_point` 的归一化 3D 坐标，3 维
- `delta_point_norm`：当前点到目标点的归一化 3D 相对位移，3 维
- `prev_action`：上一步 3D 动作，3 维
- `step_ratio`：当前步数占最大步数的比例，1 维

即：

```text
obs = [
  current_x, current_y, current_z,
  goal_x, goal_y, goal_z,
  delta_x, delta_y, delta_z,
  prev_action_x, prev_action_y, prev_action_z,
  step_ratio
]
```

这里的 `point` 目前使用的是沿法向回退后的 `retreated_point`，原因是它比直接贴在壁面上的 `surface_point` 更接近后续机械臂末端的工作点表达。

当前 observation 的生成流程是：

1. 环境内部直接维护真实的 `current_point / goal_point`
2. 对这两个 3D 点应用 observation-side 传感器噪声
3. 计算当前点、目标点、相对位移
4. 用环境级别的 3D 包围盒做 min-max 归一化

虽然环境内部还可以查询更多 3D 几何信息，例如：

- `surface_point`
- `retreated_point`
- `normal`

但当前真正进入 policy observation 的只有 3D 点位本身，还没有加入：

- 局部法向
- joint states
- joint deltas
- 更高层的几何上下文

因此，这一版 observation 可以视为“3D 点 observation 第一版”。

### 当前动作空间是什么

当前动作是 3 维连续动作，含义是对 3D 点位的增量控制，而不是关节角控制。

环境实际执行逻辑是：

```text
proposed_point = current_point + clipped_action * action_scale
```

其中：

- 策略输出原始连续动作
- 环境内部把动作裁剪到 `[-1, 1]`
- 再乘以 `action_scale`

按当前配置，`action_scale = 0.20`，所以每一步在每个轴向上的最大位移量级约为 `0.20`。

### 当前 reward 的真实公式

环境代码中的 reward 由五部分组成：

```text
reward =
  progress_reward
  - step_penalty
  - action_penalty
  - smoothness_penalty
  - boundary_penalty

if success:
  reward += success_reward
```

各项定义如下：

- `progress_reward = progress_reward_weight * (previous_goal_distance - gamma * goal_distance)`
- `step_penalty = step_penalty`
- `action_penalty = action_l2_weight * ||action||^2`
- `smoothness_penalty = action_smoothness_weight * ||action - prev_action||^2`
- `boundary_penalty = boundary_penalty if boundary_hit else 0`
- `success = goal_distance <= goal_tolerance`

其中 `goal_distance` 目前是：

```text
goal_distance = ||goal_point - current_point||_2
```

这里已经是 3D 欧氏距离，不再经过 `uv`。

这里的 `gamma` 不是单独新增的 reward 超参数，而是直接复用算法里的折扣因子 `algorithm.gamma`。这样做有两个目的：

1. 尽量减少 reward 侧新增的手工参数。
2. 让 reward 更接近标准的 potential-based shaping 形式。

### 当前配置下，reward 实际上已经简化成什么

按 `src/config.yaml` 当前配置，参数是：

- `progress_reward_weight = 2.0`
- `success_reward = 80.0`
- `gamma = 0.99`
- `step_penalty = 0.0`
- `action_l2_weight = 0.0`
- `action_smoothness_weight = 0.0`
- `boundary_penalty = 0.0`
- `goal_tolerance = 0.10`

这意味着当前真正生效的 reward 几乎就是：

```text
reward = 2.0 * (d_prev - 0.99 * d_cur) + 80.0 * I(d_cur <= 0.10)
```

也就是说：

- 主奖励来自势能函数差分
- 当前距离项会乘上折扣因子 `gamma`
- 成功进入阈值区域时额外给一个较大的终点奖励
- 没有每步惩罚
- 没有动作幅度惩罚
- 没有动作变化惩罚
- 没有边界惩罚

### 当前 reward 的问题

这版 reward 的优点是非常克制，比较符合当前阶段的三条原则：

1. 尽量减少人工设计的 dense reward。
2. 尽量不额外引入新的手调参数。
3. 更偏向稳定和低方差，而不是追求复杂 shaping。

但它对于你现在要做的“算法基础阶段”已经暴露出明显问题：

1. 它虽然简单，但仍然只有“到达目标”这一种任务语义，还没有编码贴壁、姿态、法向等更强约束。
2. 当前默认没有控制正则项，因此终点附近的稳定性仍然主要依赖势能函数本身和 PPO 的统计稳定性。
3. 如果后续任务变得更复杂，单靠这一项 potential reward 可能不够表达任务需求。
4. 由于动作仍然是连续采样并在环境内裁剪，策略动作和执行动作仍然存在轻微不一致。

从任务定义角度说，当前 reward 已经是一个合理的 3D 基础版 reward，但还不是最终任务的完整 reward。

## PPO 当前实现

### PPO 总体结构

当前 PPO 使用的是典型的 `actor-critic` 结构，但实现上比较轻量。

- actor：输入 observation，输出动作分布均值
- critic：输入 observation，输出状态价值
- exploration：使用高斯分布采样动作
- buffer：使用 on-policy rollout buffer
- advantage：使用 GAE
- policy objective：使用 clipped surrogate objective

当前训练入口默认使用 PPO，因为 `algorithm.name = ppo`。

### PPO 网络结构

PPO 使用两个独立的 MLP：

- `actor`
- `critic`

当前配置：

- hidden sizes：`[256, 256]`
- activation：`tanh`
- action log std 初始值：`-2.0`

这意味着初始标准差约为：

```text
std = exp(-2.0) ≈ 0.1353
```

当前 actor 不是输出 `mean + log_std`，而是：

- 网络只输出 `mean`
- `log_std` 是一个独立的、与状态无关的可训练参数向量

因此，当前 PPO 的动作分布是：

```text
pi(a|s) = Normal(mean(s), std)
```

其中 `std` 在每个动作维度上是全局共享的，不依赖状态。

### 这一设计的特点

优点：

- 简单稳定
- 参数少
- 适合当前这种低维连续控制任务

局限：

- 不同状态下探索强度无法自适应
- 当任务升级成 3D 点 observation 或关节空间后，这种固定方差可能不够灵活

### PPO 采样流程

每次 PPO update 之前，训练脚本会先采集一段 on-policy rollout。

当前配置：

- `rollout_steps = 2048`
- `total_updates = 600`

所以总交互步数约为：

```text
600 * 2048 = 1,228,800
```

采样过程如下：

1. 从当前 observation 构造 tensor
2. 调用 `agent.act(obs)`
3. actor 给出高斯分布，采样出 action
4. 同时得到该 action 的 `log_prob`
5. critic 估计当前状态 value
6. 将 `(obs, action, log_prob, reward, done, value)` 写入 `OnPolicyBuffer`
7. rollout 结束后，补上 `next_observation` 和 `next_done`

这里的 buffer 是严格的 on-policy buffer，不会跨 update 复用旧数据。

### PPO 的 value 处理

当前实现保留了 `value target normalization` 的能力，但默认配置已经关闭。

代码中实现了 `RunningValueNormalizer`，会维护：

- value target 的运行均值
- value target 的运行方差

在训练时：

1. 先用原始 reward 和原始 value 计算真实 return
2. 再用运行统计量把 return 归一化
3. critic 实际拟合的是归一化后的 value target
4. 在采样和评估时，再把 critic 输出反归一化

这一点的目的，是减轻 reward 量级变化带来的 value training 不稳定。

当前默认 reward 已经比较克制，因此这套机制先作为可选项保留，而不再作为默认配置。

### PPO 的 advantage 计算

当前实现使用 GAE。

记：

- `r_t`：第 `t` 步 reward
- `V(s_t)`：critic 估计的状态价值
- `done_t`：第 `t` 步 transition 是否终止
- `gamma`：折扣因子
- `lambda`：GAE 参数

则当前实现对应：

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
R_t = A_t + V(s_t)
```

当前配置：

- `gamma = 0.99`
- `gae_lambda = 0.95`

这是 PPO 中非常常见的一组设置。

优势函数在进入 PPO loss 前还会做一次标准化：

```text
A <- (A - mean(A)) / (std(A) + 1e-8)
```

如果 reward 设计比较粗糙，优势标准化会一定程度上缓解尺度问题，但不能从根本上修复 reward 定义本身不合理的问题。

### PPO 的策略损失

当前策略损失采用标准的 clipped objective。

记：

- `old_log_prob`：采样时旧策略下动作的对数概率
- `new_log_prob`：更新时新策略下同一动作的对数概率
- `ratio = exp(new_log_prob - old_log_prob)`

则策略损失为：

```text
L_policy = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
```

当前配置中：

- `clip_ratio = 0.15`

这意味着 PPO 会限制单次更新中策略偏离旧策略过大。

### PPO 的 value loss

当前 value loss 是标准均方误差：

- 如果开启 value target normalization，则拟合归一化 return
- 否则拟合原始 return

当前配置中：

- `value_coef = 0.5`
- `normalize_value_targets = false`

### PPO 的 entropy bonus

当前实现支持 entropy regularization：

```text
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

但是当前配置：

- `entropy_coef = 0.0`

所以当前实际上没有额外的熵奖励。

这意味着探索主要来自高斯采样本身，而不是额外的 entropy encouragement。

### PPO 的优化细节

当前优化器是一个 Adam，统一优化：

- actor 参数
- critic 参数
- `log_std`

当前配置：

- `lr = 1e-5`
- `update_epochs = 5`
- `minibatch_size = 512`
- `max_grad_norm = 0.5`

每次 update 的流程是：

1. 用 rollout 数据计算 returns 和 advantages
2. 打乱样本顺序
3. 按 minibatch 训练
4. 训练 `5` 个 epoch
5. 每个 minibatch 上同时优化 policy loss 和 value loss
6. 做梯度裁剪

### 关于 KL

当前代码会记录一个 `approx_kl` 指标，但没有使用它做：

- early stopping
- adaptive clip
- adaptive learning rate

因此它目前只是监控指标，不参与训练控制。

### PPO 的训练日志与保存

训练过程中会记录：

- `mean_reward`
- `mean_length`
- `success_rate`
- `mean_min_goal_distance`
- `policy_loss`
- `value_loss`
- 以及 PPO update 返回的其他统计量

会保存：

- `final.pt`
- `metrics.csv`
- `training_curves.html`

### PPO 的评估方式

评估时不再采样，而是直接使用 actor 输出均值作为确定性动作：

```text
action = actor(observation)
```

然后把该动作送入环境执行。

所以：

- 训练：高斯采样
- 评估：均值动作

这也是 PPO 连续控制任务里的常见做法。

## 当前 PPO 与环境之间最值得注意的实现细节

这部分非常重要，因为它直接关系到你下一阶段要不要先改环境定义。

### 1. PPO 当前已经是纯 3D 点控制环境

这是当前最大的事实。

当前环境中：

- observation 用的是 3D 点位状态
- reward 用的是 3D 距离势能函数
- success 用的是 3D 距离阈值
- action 控制的是 3D 点增量

所以现在 PPO 的收敛，最多只能说明：

“在一个自由 3D 点控制环境中，这个策略能够学会逼近目标点。”

它还不能说明：

“在真实岩壁约束或机械臂关节空间中，这个策略能学会稳定控制。”

### 2. 当前 reward 与当前阶段目标是一致的

在当前阶段，我们的目标是：

- 先建立一个稳定、简单、低方差的 3D 基础环境
- 尽量少用人工 shaping
- 让 reward 更接近势能函数形式

从这个标准看，当前 reward 是和当前阶段目标一致的。

后续如果再往前走，真正需要补的不是“再加更多 dense reward”，而是：

- 岩壁约束
- 表面投影
- 关节空间动作
- 更完整的任务语义

### 3. 当前 action 存在“策略动作”和“环境执行动作”不完全一致的问题

这是当前 PPO 实现里最值得优先记录的问题。

当前流程是：

1. PPO 从高斯分布采样出原始 action
2. buffer 记录的是这个原始 action
3. PPO 训练时也是对这个原始 action 计算 log_prob
4. 但是环境执行前，会先把 action 裁剪到 `[-1, 1]`

这会导致：

- 策略认为自己执行的是 `a`
- 环境实际执行的是 `clip(a, -1, 1)`

当 action 超出范围时，policy gradient 对应的并不是环境真实执行的动作。

这会带来两个问题：

1. 梯度估计与环境真实动力学不完全匹配。
2. 当策略经常输出越界动作时，环境会进入“饱和控制”，学习信号会被扭曲。

这是一个很实在的实现问题，后续不论你继续用 PPO 还是做关节动作空间，都建议修。

更合理的做法通常有两种：

- 方案 A：策略输出后先 `tanh` 压到 `[-1, 1]`
- 方案 B：环境不再二次裁剪，而是保证策略采样空间本身合法

### 4. 当前没有独立的 observation normalization 模块

当前 observation 已经不是 `uv`，而是 3D 点位。

目前环境内部做的是一种静态归一化：

- 基于隧道点云包围盒
- 加上回退距离余量
- 将 3D 点位压到大致接近 `[-1, 1]`

这对当前版本已经够用，但它仍然不是独立的、数据驱动的 observation normalization。

但如果下一阶段 observation 变成：

- 3D current point
- 3D goal point
- 3D delta
- joint states
- normal

那么不同维度的量纲差异会明显增大。

到时候如果还不做 observation normalization，PPO 训练稳定性可能会下降。

### 5. 当前 reward 是“极简势能函数”而不是“行为雕刻”

当前这版 reward 的刻意选择是：

- 主体只保留势能函数差分
- 到达时给终点奖励
- 其余控制正则默认可以关闭

这意味着它的优先级是：

- 先保证 reward 逻辑足够干净
- 先保证训练统计稳定
- 先减少手工 shaping 对策略行为的先验绑定

它的代价也很明确：

- 不会直接鼓励漂亮轨迹
- 不会直接鼓励终点附近停稳
- 不会直接约束与岩壁的几何关系

这不是缺陷，而是当前阶段的有意取舍。

## 对下一阶段的算法建议

如果下一阶段继续做“更接近真实任务的 3D 控制”，我的建议是：

### 保持 PPO 主体暂时不动

当前 PPO 的这些部件都可以先保留：

- actor-critic 结构
- GAE
- clipped objective
- advantage normalization
- value normalization

也就是说，下一阶段的主战场不在 `src/algorithm/ppo.py`，而在 `src/rl_env/math_env.py`。

### observation 已完成基础版，后续按需增强

当前 observation 已经切成：

- current end-effector point (3)
- goal point (3)
- delta point (3)
- previous action (3)
- step ratio (1)

如果后面开始切到关节空间，再逐步加入：

- joint angles
- joint angle deltas
- local surface normal

### reward 先保持极简，再按任务升级

当前 reward 已经是一个比较符合当前阶段目标的版本：

- 主奖励：`d_{t-1} - gamma * d_t`
- 终点奖励：到达成功阈值后给 `success_reward`

如果未来确实发现训练不稳定，再优先考虑：

- 统计层面的归一化
- 学习率/clip/entropy 调度
- 轻量级正则

而不是马上堆更多人工 reward。

### 在继续做复杂任务前，优先修 action clipping 问题

这一步和 reward 无关，但很值得尽快处理，因为它会影响 PPO 学习信号的正确性。

## SAC 占位

本节为后续 `SAC` 分析预留。

后续建议补充的内容包括：

- 当前 `SACAgent` 的 actor / twin critic 结构
- tanh-squashed Gaussian policy
- replay buffer 采样流程
- target critic soft update
- alpha 自动调节
- 与 PPO 在本项目中的适用性对比
- 对 reward 稀疏性、样本效率、动作空间设计的敏感性分析

当前状态：

- `src/algorithm/sac.py` 已有基本实现
- `src/train.py` 已有训练入口
- 但本项目当前阶段仍以 PPO 为主要分析对象
