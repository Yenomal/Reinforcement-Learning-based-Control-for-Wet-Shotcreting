# `rl_robot` 项目重构设计

## 1. 背景

当前仓库已经具备训练、评估、环境建模、3D 资源和若干工具脚本，但存在几个持续放大阅读负担和维护成本的问题：

- 运行时代码、3D 资源、实验性脚本混放在 `src/` 下，正式边界不清楚。
- `train.py`、`eval.py` 体量过大，入口同时承担流程编排、配置装载、日志和可视化等多种职责。
- 配置集中在单个 `src/config.yaml`，组件默认值和实验覆写混在一起。
- 资源定位依赖仓库相对路径，项目安装后不能自然作为包使用。

这次重构的目标是把项目整理成一个边界清楚、支持 `pip install -e .`、支持 Hydra 配置组合与覆写、支持包内资源分发的研究代码库。

## 2. 目标与非目标

### 2.1 目标

- 将项目重构为可编辑安装的 `src` 布局 Python 包，包名固定为 `rl_robot`。
- 将训练与评估入口迁移到仓库外侧 `scripts/`，脚本只保留薄入口职责。
- 将组件默认配置下沉到包内 Hydra config tree，外侧通过 Hydra 进行组合与覆写。
- 将运行时依赖的 URDF、mesh、默认 HTML、运动学 YAML 等资源作为包数据安装。
- 一次性切换到新结构，不保留旧目录兼容层。
- 明确区分正式运行时主链路与实验性检查脚本。

### 2.2 非目标

- 本次不改变 PPO、SAC、环境动力学、奖励函数或运动学的研究逻辑。
- 本次不引入多包发布、插件机制或分仓。
- 本次不在设计阶段改写算法实现，仅规定新的模块边界和迁移目标。

## 3. 已确认的关键决策

- 包名采用 `rl_robot`。
- 训练与评估主要通过仓库外侧脚本启动：`python scripts/train.py`、`python scripts/eval.py`。
- 配置框架采用 Hydra。
- 运行时资源跟随 `rl_robot` 包一起安装。
- 迁移采用一次性切换，不保留兼容层。
- 主链路与常用工具进入 `rl_robot`，实验性检查脚本下放到仓库级 `tools/` 或 `scripts/`。

## 4. 目标目录结构

```text
.
├── pyproject.toml
├── README.md
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── build_reachability_map.py
│   └── visualize_rock_env.py
├── tools/
│   ├── check_elbow_assignments.py
│   ├── check_elbow_pivot.py
│   └── check_wrist_pivot.py
├── src/
│   └── rl_robot/
│       ├── __init__.py
│       ├── algorithms/
│       ├── envs/
│       ├── planning/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── simulation/
│       ├── assets/
│       ├── conf/
│       └── utils/
├── docs/
└── outputs/
```

这个结构的原则是：

- `src/rl_robot/` 只承载可安装、可导入、可复用的正式能力。
- `scripts/` 承载高频入口命令。
- `tools/` 承载低频、检查性、一次性脚本。
- `outputs/` 继续承载运行产物，不作为包内容。

## 5. 包内模块边界

### 5.1 `rl_robot.algorithms`

职责：

- PPO、SAC 及相关算法配置解析。
- 学习率调度、探索标准差调度。
- 算法公用的数据结构。

当前映射：

- `src/algorithm/ppo.py`
- `src/algorithm/sac.py`
- `src/algorithm/lr_schedule.py`
- `src/component/buffer.py`

说明：

- `buffer.py` 归到算法侧，是因为其直接服务于 on-policy / off-policy 训练数据流。
- 算法模块不直接依赖仓库路径，只依赖张量、环境接口和模型接口。

### 5.2 `rl_robot.envs`

职责：

- 训练环境定义。
- classic / torch 后端环境构建。
- 环境统一接口与批量封装。

当前映射：

- `src/rl_env/math_env.py`
- `src/rl_env/torch_math_env.py`
- `src/rl_env/train_env.py`

说明：

- `envs` 只负责环境状态机、观测、动作和奖励流程。
- 任务采样、可达图、扰动等外部依赖通过 `planning` 组合进来。

### 5.3 `rl_robot.planning`

职责：

- 任务采样。
- 可达图生成与加载。
- 扰动建模。
- 与表面采样、目标采样直接相关的几何辅助逻辑。

当前映射：

- `src/component/planner.py`
- `src/component/reachability_map.py`
- `src/component/disturbance.py`

说明：

- `planning` 是从“任务生成与可达性”视角组织，而不是按文件历史来源组织。
- 任何服务于“给环境提供起点/终点任务”的逻辑都归入这里。

### 5.4 `rl_robot.models`

职责：

- 策略网络、价值网络及装配函数。

当前映射：

- `src/model/mlp.py`

### 5.5 `rl_robot.training`

职责：

- 训练主流程。
- checkpoint 保存/恢复。
- metrics 汇总。
- CSV / HTML 产物导出。
- 训练期确定性评估钩子。
- 交互式暂停/退出控制。

当前映射：

- `src/train.py`

目标拆分：

- `runner.py`：训练总调度。
- `checkpoint.py`：checkpoint 与恢复逻辑。
- `metrics.py`：训练指标汇总与日志格式化。
- `artifacts.py`：CSV、曲线 HTML、运行目录管理。
- `eval_hooks.py`：训练期确定性评估。
- `control.py`：终端交互控制。

### 5.6 `rl_robot.evaluation`

职责：

- 评估主流程。
- 模型恢复。
- 评估渲染器。
- PyBullet 评估协调。

当前映射：

- `src/eval.py`

目标拆分：

- `runner.py`：评估总调度。
- `checkpoint.py`：checkpoint 恢复与 agent 构建。
- `renderer.py`：Matplotlib / 轨迹渲染。
- `pybullet_eval.py`：PyBullet 播放与场景协同。

### 5.7 `rl_robot.simulation`

职责：

- 机器人运动学与 PyBullet 运行时能力。
- 隧道/岩壁表面运行时几何能力。
- 与仿真运行直接相关的可视化辅助逻辑。

当前映射：

- `src/rock_3D/robot_4dof/kinematics.py`
- `src/rock_3D/robot_4dof/torch_kinematics.py`
- `src/rock_3D/robot_4dof/pybullet_player.py`
- `src/rock_env/rock_wall.py`
- `src/rock_3D/tools/build_tunnel_environment.py` 中可复用的运行时几何加载部分

建议子层：

- `simulation/robot/`
- `simulation/tunnel/`
- `simulation/visualization/`

说明：

- `rock_wall.py` 不继续保留在顶层 `rock_env/` 语义下，而是按“表面/隧道仿真几何”并入 `simulation.tunnel`。
- 规划侧只消费它暴露出来的表面查询接口。

### 5.8 `rl_robot.assets`

职责：

- 存放安装时一并分发的静态资源。

内容包括：

- `robot_4dof/` 下的 `urdf`、`kinematics.yaml`、mesh
- `tunnel_environment/` 下的默认资源
- 默认数字环境 HTML 模板

### 5.9 `rl_robot.conf`

职责：

- 存放 Hydra 默认配置树。
- 表达组件默认值与实验组合点。

### 5.10 `rl_robot.utils`

职责：

- 包资源读取。
- 路径规范化。
- 少量跨模块通用工具函数。

约束：

- `utils` 只放稳定、低耦合、无领域归属的小工具。
- 不允许把大段业务逻辑重新堆回 `utils`。

## 6. Hydra 配置设计

### 6.1 配置树

```text
src/rl_robot/conf/
├── config.yaml
├── algorithm/
│   ├── ppo.yaml
│   └── sac.yaml
├── env/
│   └── math_env.yaml
├── planner/
│   └── default.yaml
├── model/
│   ├── plain_mlp.yaml
│   └── structured_mlp.yaml
├── train/
│   └── default.yaml
├── eval/
│   └── default.yaml
├── robot/
│   └── robot_4dof.yaml
├── disturbance/
│   └── sensor_noise.yaml
└── simulation/
    └── pybullet.yaml
```

### 6.2 顶层组合文件

`config.yaml` 只负责组合：

```yaml
defaults:
  - algorithm: ppo
  - env: math_env
  - planner: default
  - model: plain_mlp
  - train: default
  - eval: default
  - robot: robot_4dof
  - disturbance: sensor_noise
  - simulation: pybullet
```

### 6.3 配置下沉原则

- 每个组件维护自己的默认配置文件。
- 训练脚本和评估脚本不再持有中心化默认值。
- 内部模块通过子配置消费参数，例如：
  - `build_train_env(cfg.env, cfg.planner, cfg.robot, cfg.disturbance)`
  - `build_agent(cfg.algorithm, cfg.model, env)`

### 6.4 外侧覆写方式

主要使用 Hydra 命令行覆写：

```bash
python scripts/train.py algorithm=ppo model=structured_mlp train.device=cuda train.num_envs=256 planner.seed=123
python scripts/eval.py eval.checkpoint=/path/to/final.pt simulation.pybullet.headless=true
```

同时支持外部提供新的配置目录或配置名：

```bash
python scripts/train.py --config-name config train.seed=7
```

项目内部默认配置来源固定为 `rl_robot.conf`，研究者可以在此基础上继续扩展自己的 Hydra 组合。

### 6.5 工作目录策略

Hydra 默认的切换工作目录行为会干扰资源读取和输出目录管理，因此设计中固定：

```yaml
hydra:
  job:
    chdir: false
```

这样运行时当前工作目录保持稳定，训练产物继续按项目约定写入 `outputs/`。

## 7. 入口脚本与公开 API

### 7.1 脚本入口

保留以下仓库级脚本作为正式入口：

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/build_reachability_map.py`
- `scripts/visualize_rock_env.py`

这些脚本只承担：

- Hydra 入口声明。
- 少量 CLI 参数适配。
- 调用包内公开函数。

不承担：

- 大段业务实现。
- 仓库路径拼接。
- 默认配置定义。

### 7.2 包内公开函数

建议形成以下公开调用面：

- `rl_robot.training.run_training(cfg)`
- `rl_robot.evaluation.run_evaluation(cfg)`
- `rl_robot.planning.build_and_save_reachability_map(...)`
- `rl_robot.simulation.visualize_rock_environment(cfg)`

这样仓库脚本和未来的外部 Python 调用都可以共享同一套主流程。

## 8. 资源管理设计

### 8.1 资源组织

运行时必需资源都放到 `rl_robot.assets` 下，随包安装。

典型内容：

- `assets/robot_4dof/kinematics.yaml`
- `assets/robot_4dof/shipen_4dof.urdf`
- `assets/robot_4dof/meshes/...`
- `assets/tunnel_environment/...`
- `assets/html/rock_environment.html`

### 8.2 资源读取

统一通过 `importlib.resources` 读取，禁止继续在业务代码里硬编码 `src/...` 路径。

建议提供统一辅助函数：

- `get_asset_path("robot_4dof/kinematics.yaml")`
- `get_asset_path("tunnel_environment/metadata.json")`

### 8.3 配置中的资源表达

对于包内资源，配置不再写仓库相对路径，而是写逻辑资源标识，例如：

```yaml
robot:
  kinematics_asset: robot_4dof/kinematics.yaml
```

对用户外部文件仍保留普通路径形式，例如：

- `train.checkpoint`
- `eval.checkpoint`
- `planner.reachability_map_path`

这样可以清楚地区分“包内自带资源”和“运行生成/用户提供文件”。

## 9. 打包设计

### 9.1 打包模式

项目采用标准 `src` 布局，支持：

```bash
pip install -e .
```

### 9.2 构建后端

设计采用 `setuptools`，原因是：

- 与 `src` 布局结合直接、稳定。
- 包数据声明清晰。
- editable install 行为成熟。

`pyproject.toml` 需要补齐：

- `build-system`
- `tool.setuptools.packages.find`
- `tool.setuptools.package-data`

### 9.3 依赖管理

运行依赖继续放在 `[project.dependencies]`。

建议新增至少两个可选依赖组：

- `visualization`：PyBullet、浏览/渲染相关可选依赖
- `dev`：pytest、ruff 等开发依赖

## 10. 脚本分层策略

### 10.1 `scripts/`

保留高频、面向主流程的命令：

- 训练
- 评估
- 生成可达图
- 可视化岩壁环境

### 10.2 `tools/`

下放低频、检查性、偏实验性的脚本：

- 关节轴检查
- 模型部件对齐检查
- 一次性分析脚本

判断标准：

- 日常训练/评估工作流会用到的放 `scripts/`
- 主要服务于开发验证、局部排错的放 `tools/`

## 11. 输出目录与路径约束

- `outputs/` 保持为训练、评估、可达图等运行产物目录。
- 包内代码不假设“当前工作目录必须位于仓库根目录”。
- 任何默认静态资源都通过包资源解析。
- 任何运行产物路径都通过配置显式指定，默认仍写入 `outputs/...`。

这条约束是为了让 `rl_robot` 在 editable install 后仍然可以从仓库外被正常调用。

## 12. 测试与验收标准

重构完成后至少具备以下测试面：

- 配置组合测试：
  - Hydra 默认组合可成功加载。
  - 关键命令行覆写可生效。
- 资源解析测试：
  - `kinematics.yaml`、URDF、默认 HTML 可通过包资源定位。
- 入口 smoke test：
  - 训练入口可完成最小构建并运行一个极小步数 smoke 流程。
  - 评估入口可加载 checkpoint 并完成一个最小评估 smoke 流程。
- 环境 smoke test：
  - `MathEnv.reset()` / `step()` 可在默认配置下运行。
- 可达图入口 smoke test：
  - 最小参数下能成功进入生成流程。

验收标准：

- `pip install -e .` 后可从仓库外导入 `rl_robot`。
- `python scripts/train.py` 与 `python scripts/eval.py` 可在新结构下工作。
- 默认配置不再依赖旧 `src/config.yaml`。
- 运行时资源不再依赖旧 `src/rock_3D/...` 相对路径。
- `train.py`、`eval.py` 不再维持当前单文件规模和职责混杂状态。

## 13. 迁移原则

- 一次性切换到新目录与新导入路径。
- 不保留旧模块兼容壳。
- 文档、README、脚本命令同步更新到新入口。
- 重构优先级高于保留旧命名历史。

这条原则意味着迁移完成后，以下旧中心角色将失效：

- `src/config.py`
- `src/config.yaml` 作为唯一配置中心的地位
- `python -m src.train`
- `python -m src.eval`

## 14. 附加建议

除满足本次目标外，建议顺手完成以下结构清理：

- 将训练与评估入口拆成小模块，避免只是“路径迁移”而没有“复杂度迁移”。
- 在 README 之外新增一页新目录结构说明。
- 增加 `tests/`，至少覆盖配置、资源、入口和环境 smoke test。
- 继续保持 `outputs/` 不入库。

## 15. 结论

本次重构采用“包优先的领域化重组”方案，以 `rl_robot` 作为唯一正式包边界，以 Hydra 作为唯一正式配置入口，以 `scripts/` 作为正式脚本入口，以 `assets + importlib.resources` 作为统一资源定位机制。主链路与常用工具进入包内，实验性检查脚本下放到仓库级 `tools/`。迁移采用一次性切换，并同时拆分过重的训练与评估入口文件。
