# 配置说明

## 1. 总览

当前项目使用 `Hydra` 管理配置。

入口配置文件是：

- `src/rl_robot/conf/config.yaml`

它本身主要负责两件事：

1. 选择各个 config group 的默认组合
2. 补充少量顶层参数，例如 `ppo.total_updates`、`sac.total_steps`

当前默认组合是：

```yaml
defaults:
  - algorithm: ppo
  - env: math_env
  - rl: default
  - planner: default
  - model: plain_mlp
  - train: default
  - eval: default
  - robot: robot_4dof
  - disturbance: sensor_noise
  - _self_
```

这表示当前默认配置由以下几块拼出来：

- `algorithm=ppo`
- `env=math_env`
- `rl=default`
- `planner=default`
- `model=plain_mlp`
- `train=default`
- `eval=default`
- `robot=robot_4dof`
- `disturbance=sensor_noise`

## 2. 配置目录结构

当前配置目录是：

```text
src/rl_robot/conf/
├── config.yaml
├── algorithm/
│   ├── ppo.yaml
│   └── sac.yaml
├── disturbance/
│   └── sensor_noise.yaml
├── env/
│   └── math_env.yaml
├── eval/
│   └── default.yaml
├── model/
│   ├── plain_mlp.yaml
│   └── structured_mlp.yaml
├── planner/
│   └── default.yaml
├── rl/
│   └── default.yaml
├── robot/
│   └── robot_4dof.yaml
├── simulation/
│   └── pybullet.yaml
└── train/
    └── default.yaml
```

## 3. 如何使用

### 3.1 训练

```bash
python scripts/train.py
```

带覆盖参数：

```bash
python scripts/train.py algorithm=sac train.device=cpu train.num_envs=64
```

### 3.2 评估

```bash
python scripts/eval.py eval.checkpoint=outputs/runs/<run>/final.pt
```

### 3.3 生成可达图

```bash
python scripts/build_reachability_map.py --force --device cuda
```

如果修改默认训练岩壁 `src/rl_robot/assets/html/rock_environment.html`，需要重新生成这两个缓存文件：

- `outputs/reachability/reachability_map.npz`
- `outputs/reachability/reachability_map.html`

### 3.4 岩壁可视化

```bash
python scripts/visualize_rock_env.py
```

## 4. 覆写规则

Hydra 覆写是点路径形式。

例如：

```bash
python scripts/train.py train.device=cpu
python scripts/train.py planner.use_reachability_map=false
python scripts/train.py ppo.total_updates=500
python scripts/eval.py eval.episodes=10
```

切换 config group 也是同样写法：

```bash
python scripts/train.py algorithm=sac
python scripts/train.py model=structured_mlp
```

## 5. 顶层配置 `config.yaml`

文件：`src/rl_robot/conf/config.yaml`

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ppo.total_updates` | `200` | PPO 总更新轮数 |
| `sac.total_steps` | `2_000_000` | SAC 总环境步数 |
| `hydra.job.chdir` | `false` | 运行时不切换工作目录 |

说明：

- `ppo.total_updates` 和 `sac.total_steps` 是顶层参数块，不属于 `algorithm/*.yaml`
- 运行时可直接覆写，例如 `ppo.total_updates=500`

## 6. `algorithm` 组

文件：

- `src/rl_robot/conf/algorithm/ppo.yaml`
- `src/rl_robot/conf/algorithm/sac.yaml`

### `algorithm=ppo`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `algorithm.name` | `ppo` | 当前算法名 |
| `algorithm.gamma` | `0.99` | 折扣因子 |

### `algorithm=sac`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `algorithm.name` | `sac` | 当前算法名 |
| `algorithm.gamma` | `0.99` | 折扣因子 |

说明：

- 这里放的是“算法选择”和“公共算法参数”
- PPO/SAC 各自专有的步数在顶层 `ppo`、`sac` 参数块里

## 7. `env` 组

文件：`src/rl_robot/conf/env/math_env.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `env.seed` | `42` | 环境随机种子 |
| `env.n_theta` | `200` | 岩壁周向采样数 |
| `env.n_z` | `100` | 岩壁轴向采样数 |
| `env.train_rock_env_html` | `./src/rock_3D/rock_environment.html` | 训练时固定岩壁 HTML；默认字符串会解析到打包资源 `asset:html/rock_environment.html`；为空时按种子程序化生成 |
| `env.rock_env_html` | `outputs/rock_environment.html` | 岩壁可视化输出路径 |

说明：

- 当前默认训练背景岩壁已经对齐 `parallel` 基线语义
- `train_rock_env_html` 保留旧路径风格字符串，训练与可达图构建时会解析到打包资源 `src/rl_robot/assets/html/rock_environment.html`
- 如果你想强制程序化生成岩壁，可覆写为空字符串

示例：

```bash
python scripts/train.py env.train_rock_env_html=
```

## 8. `rl` 组

文件：`src/rl_robot/conf/rl/default.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `rl.env_name` | `math_env` | RL 环境名 |
| `rl.max_episode_steps` | `200` | 单回合最大步数 |
| `rl.initial_configuration_deg` | `[0, 0, 0, 0]` | 初始关节角 |
| `rl.success_tolerance` | `0.003` | 成功阈值 |
| `rl.reward_distance_scale` | `1` | 距离奖励尺度 |
| `rl.max_joint_delta_deg` | `[4, 4, 4, 4]` | 每步最大关节变化 |
| `rl.workspace_margin` | `0.25` | 工作空间边界扩展 |
| `rl.progress_reward_weight` | `1.0` | 进展奖励权重 |
| `rl.success_reward` | `0.0` | 成功奖励 |
| `rl.step_penalty` | `0.0` | 步惩罚 |
| `rl.action_l2_weight` | `0.0` | 动作 L2 惩罚 |
| `rl.action_smoothness_weight` | `0.0` | 动作平滑惩罚 |
| `rl.boundary_penalty` | `0.0` | 越界惩罚 |

### `rl.action_scale_schedule`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `rl.action_scale_schedule.enable` | `true` | 是否启用动作尺度调度 |
| `rl.action_scale_schedule.schedule` | `cosine` | 调度类型 |
| `rl.action_scale_schedule.start_ratio` | `0.5` | 起始尺度比例 |
| `rl.action_scale_schedule.end_ratio` | `0.5` | 结束尺度比例 |

## 9. `planner` 组

文件：`src/rl_robot/conf/planner/default.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `planner.seed` | `42` | 任务采样随机种子 |
| `planner.spray_angle_range_deg` | `[-60, 60]` | 湿喷区域角度范围 |
| `planner.spray_standoff_distance` | `0.5` | 沿法向向内投影的目标距离 |
| `planner.axial_margin_ratio` | `0.1` | 轴向边缘留白比例 |
| `planner.tunnel_axial_scale` | `1.5` | 数学隧道到 PyBullet 世界坐标缩放 |
| `planner.use_reachability_map` | `true` | 是否启用可达图 |
| `planner.reachability_map_path` | `outputs/reachability/reachability_map.npz` | 可达图缓存路径 |
| `planner.reachability_map_html` | `outputs/reachability/reachability_map.html` | 可达图可视化输出 |
| `planner.reachability_map_init_samples` | `16384` | 初始采样数 |
| `planner.reachability_map_batch_size` | `256` | IK 批大小 |
| `planner.reachability_map_ik_steps` | `160` | IK 步数 |
| `planner.reachability_map_ik_lr` | `0.05` | IK 学习率 |
| `planner.reachability_map_restart_count` | `3` | 多初值重启次数 |
| `planner.reachability_tolerance` | `0.003` | 末端距离容差 |
| `planner.max_goal_sampling_trials` | `256` | 最大重采样次数 |

## 10. `model` 组

文件：

- `src/rl_robot/conf/model/plain_mlp.yaml`
- `src/rl_robot/conf/model/structured_mlp.yaml`

两组目前大部分字段相同，主要差异在：

- `model.type=plain_mlp`
- `model.type=structured_mlp`

### 公共字段

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `model.type` | `plain_mlp` 或 `structured_mlp` | 模型类型 |
| `model.hidden_sizes` | `[256, 256]` | 主隐藏层大小 |
| `model.activation` | `tanh` | 激活函数 |
| `model.joint_hidden_sizes` | `[128, 128]` | 关节分支隐藏层 |
| `model.geometry_hidden_sizes` | `[128, 128]` | 几何分支隐藏层 |
| `model.prev_action_hidden_sizes` | `[64, 64]` | 上一步动作分支隐藏层 |
| `model.time_hidden_sizes` | `[32, 32]` | 时间分支隐藏层 |
| `model.action_hidden_sizes` | `[64, 64]` | critic 动作分支隐藏层 |
| `model.fusion_hidden_sizes` | `[256, 256]` | 融合层隐藏层 |

切换示例：

```bash
python scripts/train.py model=structured_mlp
```

## 11. `train` 组

文件：`src/rl_robot/conf/train/default.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `train.seed` | `42` | 训练随机种子 |
| `train.device` | `cuda` | 训练设备 |
| `train.env_backend` | `torch` | 环境后端 |
| `train.num_envs` | `256` | 并行环境数 |
| `train.lr_schedule` | `none` | 学习率调度 |
| `train.lr_min_ratio` | `0.1` | 学习率最小比例 |
| `train.deterministic_eval.enable` | `true` | 是否开启确定性评估 |
| `train.deterministic_eval.interval_updates` | `20` | 确定性评估触发间隔 |
| `train.deterministic_eval.episodes` | `256` | 每次确定性评估回合数 |
| `train.deterministic_eval.num_envs` | `256` | 确定性评估并行环境数 |
| `train.deterministic_eval.seed` | `123` | 确定性评估随机种子 |
| `train.deterministic_eval.backend` | `torch` | 确定性评估环境后端 |
| `train.log_interval` | `1` | 终端日志间隔 |
| `train.runs_root` | `outputs/runs` | 训练输出根目录 |
| `train.resume` | `false` | 是否续训 |
| `train.checkpoint` | `""` | 初始化或续训权重路径 |

常用示例：

```bash
python scripts/train.py train.device=cpu
python scripts/train.py train.num_envs=64
python scripts/train.py train.resume=true train.checkpoint=outputs/runs/xxx/final.pt
```

## 12. `eval` 组

文件：`src/rl_robot/conf/eval/default.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `eval.checkpoint` | `./outputs/runs/sac_direct_0.1/final.pt` | 默认评估权重 |
| `eval.episodes` | `5` | 评估回合数 |
| `eval.seed` | `123` | 评估随机种子 |
| `eval.headless` | `false` | 是否关闭可视化 |
| `eval.digital_env_html` | `asset:html/rock_environment.html` | 评估场景 HTML |
| `eval.render_pause` | `0.05` | 帧刷新间隔 |
| `eval.episode_pause` | `0.8` | 回合切换停顿 |

### `eval.pybullet`

这是通过 `defaults` 注入的子树，来自 `simulation/pybullet.yaml`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `eval.pybullet.enable` | `true` | 是否启用 PyBullet |
| `eval.pybullet.headless` | `false` | PyBullet 是否无头 |
| `eval.pybullet.dt` | `0.0041666667` | 仿真步长 |
| `eval.pybullet.show_tunnel` | `true` | 是否显示隧道 |
| `eval.pybullet.show_plane` | `false` | 是否显示地面 |
| `eval.pybullet.robot_position` | `[0.0, 0.0, 0.45]` | 机器人初始位置 |
| `eval.pybullet.robot_yaw_deg` | `90.0` | 机器人初始偏航角 |
| `eval.pybullet.tunnel_position` | `[0.0, 0.0, 0.0]` | 隧道初始位置 |

示例：

```bash
python scripts/eval.py eval.checkpoint=outputs/runs/xxx/final.pt
python scripts/eval.py eval.episodes=10 eval.headless=true
python scripts/eval.py eval.pybullet.enable=false
```

## 13. `robot` 组

文件：`src/rl_robot/conf/robot/robot_4dof.yaml`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `robot.kinematics_path` | `src/rock_3D/robot_4dof/kinematics.yaml` | 机器人运动学配置路径 |

说明：

- 这个值目前仍保留旧路径字符串
- 代码内部已对默认路径做兼容，可映射到包资源

## 14. `disturbance` 组

文件：`src/rl_robot/conf/disturbance/sensor_noise.yaml`

### `disturbance.sensor_noise`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `disturbance.sensor_noise.enable` | `true` | 是否启用传感器噪声 |
| `disturbance.sensor_noise.current_point_step_std` | `0.01` | 当前点步噪声标准差 |
| `disturbance.sensor_noise.current_point_bias_std` | `0.02` | 当前点偏置标准差 |
| `disturbance.sensor_noise.goal_point_step_std` | `0.0` | 目标点步噪声标准差 |
| `disturbance.sensor_noise.goal_point_bias_std` | `0.0` | 目标点偏置标准差 |

示例：

```bash
python scripts/train.py disturbance.sensor_noise.enable=true
```

## 15. `simulation` 组

文件：`src/rl_robot/conf/simulation/pybullet.yaml`

这个组当前主要被 `eval/default.yaml` 通过：

```yaml
defaults:
  - /simulation@pybullet: pybullet
```

注入到 `eval.pybullet` 下。

如果你要覆写，就直接写：

```bash
python scripts/eval.py eval.pybullet.headless=true
python scripts/eval.py eval.pybullet.enable=false
```

## 16. 常用命令模板

### 训练 PPO

```bash
python scripts/train.py
```

### 训练 SAC

```bash
python scripts/train.py algorithm=sac
```

### CPU 训练

```bash
python scripts/train.py train.device=cpu
```

### 调小并行环境数

```bash
python scripts/train.py train.num_envs=32
```

### 关闭可达图

```bash
python scripts/train.py planner.use_reachability_map=false
```

### 指定评估权重

```bash
python scripts/eval.py eval.checkpoint=outputs/runs/<run>/final.pt
```

### 关闭 PyBullet

```bash
python scripts/eval.py eval.pybullet.enable=false
```

### 生成可达图

```bash
python scripts/build_reachability_map.py --force --device cuda
```

默认训练岩壁 HTML 发生变化后，需要重新执行该命令以刷新可达图缓存。

## 17. 使用建议

- 长期稳定的默认值，直接改对应 group 文件
- 单次实验参数，优先用命令行 overrides
- 切算法、切模型，优先用 group 切换
- 步数、设备、并行数、checkpoint 这类实验参数，优先命令行覆写

如果你后面愿意，我还可以继续给你补一份：

- “常用 override 速查表”
- “训练/评估/可达图 三套最常用命令模板”
