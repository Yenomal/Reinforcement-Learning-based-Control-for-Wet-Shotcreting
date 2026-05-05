# `parallel` 默认训练语义对齐设计

## 1. 背景

当前主分支已经完成包结构、Hydra 配置树、资源打包和训练入口拆分，但训练默认语义已经偏离 `parallel` 分支最新提交 `bbaadbd` 对应的实验基线。偏移主要体现在三类内容：

- 默认训练配置发生变化，例如 `train.lr_schedule` 当前默认为 `cosine`。
- 训练日志与曲线导出被精简，`det_success_rate`、`det_mean_min_goal_distance`、`ppo_std_*` 等关键观测丢失。
- 训练使用的打包岩壁 HTML 与 `parallel` 最新提交中的 `src/rock_3D/rock_environment.html` 不一致，导致可达图结果与任务分布偏离原实验。

本次整理的目标不是引入新的训练逻辑，而是让当前主分支默认行为重新对齐到 `parallel` 最新提交的实验语义，同时保留现有包结构和 Hydra 组织方式。

## 2. 目标与非目标

### 2.1 目标

- 将当前主分支默认训练配置恢复为 `parallel` 最新提交的默认实验语义。
- 恢复训练期 deterministic evaluation、PPO std 调度状态和相关日志导出。
- 恢复 `metrics.csv` 与 `training_curves.html` 的关键实验观测字段。
- 使当前打包岩壁 HTML 与 `parallel` 最新提交的岩壁快照一致。
- 保持当前 `rl_robot` 包结构、`scripts/` 入口和 Hydra 配置树不变。

### 2.2 非目标

- 本次不改变 PPO、SAC、环境动力学、reward 主体形式和网络结构。
- 本次不恢复旧目录结构，不回退包化重构。
- 本次不把旧的 checkpoint 路径重新设为默认值。
- 本次不引入新的实验预设层，直接调整当前默认值。

## 3. 已确认决策

- 当前主分支默认行为应直接等价于 `parallel` 最新提交的训练默认。
- `train.checkpoint` 默认值保持为空，训练时显式指定更稳妥。
- `disturbance.sensor_noise.enable` 恢复为 `parallel` 默认，即开启噪声。
- 当前打包资源应作为正式训练资源，内容需要与 `parallel` 最新快照同步。

## 4. 目标默认语义

本次整理后，主分支默认训练应满足以下语义：

- `train.lr_schedule = none`
- `train.deterministic_eval.enable = true`
- 默认保留训练期确定性评估频率、评估回合数和固定随机种子
- 默认保留 `ppo.std` 配置树及相关调度逻辑
- 默认保留 `action_scale_schedule`
- 默认启用传感器噪声
- 默认训练背景岩壁与 `parallel` 最新提交一致

这些默认值应直接体现在 Hydra 配置中，而不是依赖命令行覆盖。

## 5. 需要回迁的能力

### 5.1 配置层

需要将 `parallel` 最新提交中仍被使用的训练语义迁移到当前 Hydra 配置树：

- `train.deterministic_eval`
- `train.lr_schedule = none`
- `train.checkpoint = ""`
- `ppo.std`
- `disturbance.sensor_noise.enable = true`

其中：

- `train.checkpoint` 只保留空字符串默认值，不再带入旧 run 路径。
- 若当前配置树中缺失 `ppo.std`，则按 `parallel` 最新提交结构恢复。

### 5.2 训练流程层

需要在当前训练流程中恢复 `parallel` 的实验观测能力：

- 训练期 deterministic evaluation
- PPO std 调度状态记录
- 终端输出中的关键附加指标
- 写入 checkpoint 的最终指标保持完整

当前 `runner.py` 已保留 PPO rollout、学习率调度和 action scale 调度主线，因此本次重点是将 `parallel` 版本中的观测和调度状态重新接回，而不是重写训练主循环。

### 5.3 产物导出层

需要恢复 `parallel` 默认产物中的高价值字段：

- `metrics.csv` 需要包含：
  - `det_success_rate`
  - `det_mean_min_goal_distance`
  - `approx_kl`
  - `explained_variance`
  - `ppo_success_ema`
  - `ppo_std_streak`
  - `ppo_next_log_std`
  - `ppo_std_cooldown_remaining`
  - `ppo_std_phase`
  - `ppo_std_mean`
  - `ppo_log_std_mean`
- `training_curves.html` 需要恢复 deterministic eval、KL、EV、PPO std 等面板。

### 5.4 资源层

当前主分支的打包资源 `src/rl_robot/assets/html/rock_environment.html` 与 `parallel` 最新提交中的旧路径文件内容不一致，而 `kinematics.yaml` 一致。因此本次只需要同步岩壁 HTML 快照：

- `parallel:src/rock_3D/rock_environment.html`
- `main:src/rl_robot/assets/html/rock_environment.html`

同步后需要重建当前默认配置下的可达图缓存，保证训练与可达图使用同一份资源。

## 6. 具体文件与改动范围

### 6.1 配置文件

- `src/rl_robot/conf/train/default.yaml`
  - 恢复 `lr_schedule: none`
  - 补回 `deterministic_eval`
  - `checkpoint` 默认保持空字符串
- `src/rl_robot/conf/config.yaml`
  - 恢复 `ppo.std` 默认结构
- `src/rl_robot/conf/disturbance/sensor_noise.yaml`
  - 恢复 `enable: true`

### 6.2 训练与算法

- `src/rl_robot/training/runner.py`
  - 接回训练期 deterministic evaluation
  - 接回 PPO std 调度状态写入
- `src/rl_robot/training/artifacts.py`
  - 恢复完整 CSV 字段
  - 恢复完整 HTML 曲线面板
- `src/rl_robot/algorithms/lr_schedule.py`
  - 恢复 success-triggered 相关 scheduler
- `src/rl_robot/algorithms/ppo.py`
  - 恢复 `ppo.std` 配置解析与相关统计输出

### 6.3 资源与文档

- `src/rl_robot/assets/html/rock_environment.html`
  - 同步为 `parallel` 最新岩壁快照
- `README.md`
  - 同步默认训练说明与可达图重建说明
- `docs/config.md`
  - 同步当前默认配置字段说明

## 7. 改动后数据流

改动完成后，默认训练数据流为：

1. Hydra 加载当前默认配置。
2. 配置直接提供 `parallel` 默认训练语义，包括 `lr_schedule=none`、deterministic eval、PPO std 配置和噪声默认值。
3. 训练环境和可达图加载当前打包岩壁 HTML，该资源内容与 `parallel` 最新快照一致。
4. `runner.py` 在每个 update 中执行：
   - rollout 采样
   - PPO 更新
   - 学习率调度
   - action scale 调度
   - PPO std 调度状态更新
   - 按固定间隔运行 deterministic evaluation
5. `artifacts.py` 将训练窗口指标、deterministic eval 指标和 PPO std 状态完整写入 `metrics.csv` 和 `training_curves.html`。

## 8. 风险与边界

### 8.1 风险

- 回迁 `parallel` 的训练观测逻辑时，当前拆分后的模块边界可能与旧单文件训练入口存在接口差异。
- 将默认岩壁 HTML 同步到 `parallel` 快照后，当前已有的可达图缓存和历史 run 结果都不再代表最新默认环境。
- 若当前主分支还存在依赖精简 `metrics.csv` 字段的脚本，恢复字段后需确认兼容性。

### 8.2 边界

- 本次只恢复 `parallel` 默认实验语义，不继续追溯更早历史提交的配置。
- 本次不实现 best checkpoint、早停、reward 改写等后续训练改良项。
- 本次不恢复旧入口 `src/train.py`，只在当前 `scripts/` + `rl_robot.training` 架构内完成迁移。

## 9. 验证要求

完成实现后，需要至少验证以下内容：

- 默认配置下训练启动成功，且终端输出包含 deterministic eval 和 PPO std 相关指标。
- 新生成的 `metrics.csv` 包含恢复后的关键字段。
- 新生成的 `training_curves.html` 面板数量和字段与 `parallel` 目标语义一致。
- 重新生成可达图后，缓存签名与当前默认资源一致。
- 默认训练配置中 `lr_schedule` 为 `none`，`checkpoint` 为空，噪声默认开启。
