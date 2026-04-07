# Research

## 框架图

```text
                         ┌──────────────────────────────────────────┐
                         │            rock_wall.py                  │
                         │  共享“世界生成器”与参考轨迹主线          │
                         └──────────────────────────────────────────┘
                                         │
                                         ▼
               ┌──────────────────────────────────────────────────────────┐
               │ 1. 在展开平面 (u, v) 上生成 2D S 型路径                  │
               │    u: 周向展开坐标，v: 隧道轴向深度                     │
               └──────────────────────────────────────────────────────────┘
                                         │
                                         ▼
               ┌──────────────────────────────────────────────────────────┐
               │ 2. 映射到半圆柱隧道表面                                  │
               │    theta = u / R_BASE                                    │
               │    r = R_BASE + noise(u, v) * amplitude                  │
               │    P_rock_noisy = [r cos(theta), r sin(theta), v]        │
               └──────────────────────────────────────────────────────────┘
                                         │
                                         ▼
               ┌──────────────────────────────────────────────────────────┐
               │ 3. 有限差分求局部法向 normals                            │
               └──────────────────────────────────────────────────────────┘
                                         │
                                         ▼
               ┌──────────────────────────────────────────────────────────┐
               │ 4. 重力流淌补偿                                          │
               │    P_final_rock = P_rock_noisy + slump_vector            │
               └──────────────────────────────────────────────────────────┘
                                         │
                                         ▼
               ┌──────────────────────────────────────────────────────────┐
               │ 5. 沿法向内缩 D_SPRAY                                    │
               │    P_nozzle = P_final_rock - normals * D_SPRAY           │
               └──────────────────────────────────────────────────────────┘
                      │                          │                     │
          ┌───────────┘                          │                     └───────────────┐
          ▼                                      ▼                                     ▼
┌──────────────────────┐          ┌─────────────────────────┐         ┌────────────────────────────┐
│ env.py               │          │ trajectory.py           │         │ pointcloud_trajectory.py   │
│ 直接可视化共享主线   │          │ 方法 1：S-G + B 样条    │         │ 方法 2：名义点云法         │
│ 岩面/法向/喷嘴轨迹   │          │ 对 P_nozzle 再平滑      │         │ 对 P_nozzle 重新切片排序   │
└──────────────────────┘          └─────────────────────────┘         └────────────────────────────┘
                                             │                                     │
                                             ▼                                     ▼
                               trajectory_sg_bspline.html            pointcloud_slam_trajectory.html
```

```text
pointcloud_trajectory.py 的“声明流程”和“实际主线”并不一致：

声明出来的流程：
岩壁点云 -> 降采样 -> 法线估计 -> 切片覆盖 -> 法向回退 -> B 样条

main() 里真正跑的流程：
P_ref = rock_data['P_nozzle']
  ├─> voxel_downsample(P_ref) -> estimate_normals(...)       -> 结果未继续使用
  └─> generate_slicing_trajectory_from_ref(P_ref)
        -> 按 z 切片 -> 按角度排序 -> S 型拼接 -> bspline_smooth(...)
```

## 总述

### 一句话总结

这是一个“湿喷机器人在非结构化隧道岩面上生成目标喷射轨迹”的几何脚本原型库，核心不是强化学习，而是程序化生成岩壁、推导喷嘴参考轨迹、再做平滑和可视化。

### 解决问题

这个仓库想解决的问题很明确：在没有真实扫描数据、没有复杂规划器的前提下，先构造一套可重复的“凹凸隧道岩壁 + 喷射轨迹”生成流程，让后续算法都能在同一块岩壁背景上比较。

更具体地说，它试图回答三件事：

1. 如何在半圆柱隧道内壁上生成一条具有起伏的参考喷射路径。
2. 如何把这条路径处理得更平滑，更像机器人可执行轨迹。
3. 如何用统一可视化把岩壁、法向、喷嘴轨迹一起展示出来。

### 源码结构

当前仓库真正有内容的部分几乎都在 `src/rock_generate`：

1. `rock_wall.py`
   共享底座。负责生成岩壁、法向、重力补偿后的岩面轨迹，以及喷嘴轨迹。
2. `env.py`
   直接消费 `rock_wall.py` 的结果做 3D 可视化。
3. `trajectory.py`
   方法 1。把 `P_nozzle` 当成粗糙参考轨迹，做 S-G 滤波和 B 样条拟合。
4. `pointcloud_trajectory.py`
   方法 2。文件里写了不少“点云 SLAM 后端式”的处理函数，但 `main()` 真实跑的是“从参考喷嘴轨迹切片后重新拼接，再做 B 样条”。
5. `src/train.py`、`src/eval.py`、`src/model/`
   目前是空的，占位多于实现。

### 模型整体输入输出

严格来说，这个项目现在没有“学习模型”。

它的输入主要是代码顶部写死的几何和处理参数，而不是数据集或命令行参数，例如：

1. 隧道几何：`R_BASE=4.0`、`L_TUNNEL=10.0`
2. 喷射距离：`D_SPRAY=1.5`
3. 岩面噪声：`NOISE_SCALE=0.5`、`NOISE_OCTAVES=4`、`NOISE_PERSISTENCE=0.5`、`NOISE_AMPLITUDE=0.3`
4. 平滑参数：S-G 窗长、B 样条平滑因子、切片间隔、体素大小等

它的输出主要有两类：

1. 内存中的 `numpy` 数组和字典
   例如 `P_rock_noisy`、`normals`、`P_final_rock`、`P_nozzle`
2. 浏览器可打开的 HTML 可视化
   例如 `shotcrete_trajectory.html`、`trajectory_sg_bspline.html`、`pointcloud_slam_trajectory.html`

`generate_rock_wall()` 是最核心的“统一输出接口”，返回一个字典，后续脚本基本都从这里取参考数据。

## 分模块讲解

### 模块 1：`rock_wall.py`，共享岩壁与参考轨迹生成器

这是整个仓库最重要的文件，其他脚本本质上都是在消费它的输出。

#### 模块职责

1. 定义统一的隧道几何参数和噪声参数。
2. 在展开平面生成 2D S 型路径。
3. 把 2D 路径映射回半圆柱岩壁，并用分形噪声制造凹凸。
4. 计算局部法向。
5. 加入“重力流淌补偿”。
6. 沿法向向内回退，得到喷嘴参考轨迹。

#### 模块内部数据流

```text
u_coords, v_coords
   │
   ▼
[generate_2d_s_path]
   │
   ▼
P_rock_noisy
   = map_to_3d_cylinder(u, v, noise)
   │
   ▼
normals
   = compute_surface_normal(P_rock_noisy)
   │
   ▼
P_final_rock, slump_vectors
   = compute_gravity_slump(P_rock_noisy, normals)
   │
   ▼
P_nozzle
   = compute_nozzle_trajectory(P_final_rock, normals, D_SPRAY)
```

#### 关键实现细节

1. 2D S 型路径不是任意曲线，而是“线性插值 + 正弦扰动”。
   `u` 方向加入较大的 S 型摆动，`v` 方向加入较小的周期扰动，因此路径整体沿隧道深度前进，同时在周向来回摆动。
2. 3D 岩壁不是直接在笛卡尔坐标系造噪声，而是先在半圆柱参数域 `(u, v)` 上采样噪声，再映射到 `(x, y, z)`。
   这样所有方法共享的是同一块“参数化岩壁”。
3. 法向计算使用有限差分，不是解析法向，也不是点云法向。
   它通过 `(u + delta, v)` 和 `(u, v + delta)` 两个邻域点构造切向量，再叉乘得到法向。
4. 重力流淌补偿是一个经验模型。
   代码里用重力方向 `g = [0, -1, 0]` 与表面法向构造流淌方向，再按坡度估计一个位移量：

```text
steepness = sqrt(Nx^2 + Nz^2)
slump_magnitude = K_SLUMP * steepness
P_final_rock = P_rock_noisy + S_dir * slump_magnitude
```

5. 喷嘴轨迹 `P_nozzle` 不是直接贴在岩面上，而是在补偿后的岩面点基础上，沿法向往隧道内部退 `D_SPRAY` 米。

#### 这个模块真正提供了什么

从工程角度看，`rock_wall.py` 提供了两套共享资产：

1. `generate_rock_wall()`
   生成沿一条参考 S 型路径采样得到的岩面点、法向、补偿后的岩面轨迹和喷嘴轨迹。
2. `generate_dense_rock_wall()`
   生成整片高密度岩壁点云，主要给可视化当背景，不参与主线规划。

#### 这个模块的边界与小问题

1. `generate_rock_wall(n_points_u, n_points_v, seed)` 里的 `n_points_v` 参数没有被实际使用。
2. `compute_gravity_slump()` 形参里有 `u_coords`、`v_coords`，但函数体内并没有用到。
3. 这里生成的是“共享参考轨迹”和“程序化世界”，不是机器人控制器，也不是环境仿真器。

### 模块 2：`env.py`，共享主线的直接可视化

`env.py` 做的事情比较纯粹：直接调用 `generate_rock_wall()` 和 `generate_dense_rock_wall()`，把共享主线画出来。

#### 实际主线

```text
generate_rock_wall()
   ├─> P_final_rock   -> 岩面 S 型轨迹
   ├─> P_nozzle       -> 喷嘴轨迹
   └─> normals        -> 采样法向可视化

generate_dense_rock_wall()
   └─> dense points / noise_gen -> 岩壁背景网格

以上结果
   └─> create_visualization(...)
         └─> shotcrete_trajectory.html
```

#### 它展示了什么

1. Earth 色系岩壁曲面。
2. 岩面上的 S 型轨迹。
3. 与岩面保持喷射距离的喷嘴轨迹。
4. 局部法向量。
5. 起点和终点。

#### 它没有做什么

1. 没有环境交互接口。
2. 没有状态、动作、奖励定义。
3. 没有训练循环。

所以虽然文件名叫 `env.py`，但它目前更像“演示脚本”而不是 RL environment。

### 模块 3：`trajectory.py`，方法 1：S-G 滤波 + B 样条

这个文件的作用是把 `rock_wall.py` 产生的喷嘴参考轨迹 `P_nozzle` 再平滑一遍，得到一条更连续的轨迹。

#### 实际主线

```text
rock_data = generate_rock_wall()
P_ref = rock_data['P_nozzle']
   │
   ▼
apply_savitzky_golay_filter(P_ref)
   │
   ▼
P_filtered
   │
   ▼
bspline_parametric_fit(P_filtered)
   │
   ▼
P_final
   │
   ├─> compute_trajectory_metrics(P_ref, P_filtered, P_final)
   └─> create_visualization(...)
```

#### 各步骤在代码中的含义

1. `P_ref`
   来自 `rock_wall.py` 的喷嘴轨迹，不是原始传感器噪声点，也不是机器人执行后的记录轨迹。
2. S-G 滤波
   对 `x/y/z` 三个坐标轴独立做 Savitzky-Golay 滤波。
   这一步主要是局部平滑，不改变整体宏观走向。
3. B 样条拟合
   把离散点拟合成连续参数曲线，然后均匀重采样到 `N_FINAL_POINTS=1000` 个点。
4. 指标计算
   用有限差分近似速度、加速度、jerk，计算路径长度、平滑度、曲率变化率。

#### 需要注意的真实含义

1. 文件注释里提到“满足机器人动力学约束”，但代码并没有真正做动力学约束求解。
2. 这里的“平滑度”和“曲率变化率”只是基于空间离散点差分的经验指标，没有显式时间步长，也没有关节空间或速度上限约束。
3. 因为输入 `P_ref` 本身已经是程序化生成的相对平滑轨迹，所以这个文件更像“后处理平滑器”，不是从粗糙观测中恢复轨迹。

### 模块 4：`pointcloud_trajectory.py`，方法 2：名义点云法

这是仓库里最值得单独说明的文件，因为它“看起来像点云 SLAM 后端”，但 `main()` 真实执行的主线并没有完全走那条路。

#### 文件中声明出来的理想流程

```text
岩壁点云
   -> 体素降采样
   -> KD-Tree 法线估计
   -> 沿 z 轴切片
   -> 每片按角度排序
   -> 相邻切片 S 型连接
   -> 法向回退
   -> B 样条平滑
```

#### `main()` 实际执行的流程

```text
rock_data = generate_rock_wall()
P_ref = rock_data['P_nozzle']

P_ref
   ├─> voxel_downsample(P_ref)
   │     └─> estimate_normals(...)
   │           └─> 到这里就停了，结果没有接入后续
   │
   └─> generate_slicing_trajectory_from_ref(P_ref)
         ├─> 按 z 切片
         ├─> 每片按角度排序
         ├─> 相邻切片 S 型拼接
         └─> 得到 slicing_trajectory
                   │
                   ▼
            bspline_smooth(slicing_trajectory)
                   │
                   ▼
             final_trajectory
```

#### 这意味着什么

1. 这个文件并没有真正从“岩壁稠密点云”生成轨迹。
   `generate_dense_rock_wall()` 只被拿来做背景可视化。
2. 体素降采样和法线估计不是对岩壁点云做的，而是对 `P_ref` 这条喷嘴轨迹做的。
3. 即使做了 `estimate_normals(pcd_down, ...)`，结果 `pcd_normals` 也没有被后续轨迹生成使用。
4. 文件里实现了 `compute_pca_normals()` 和 `normal_retreat()`，但 `main()` 根本没有调用。
5. 文件里还实现了 `generate_slicing_trajectory(pcd)`，看起来是给点云主线准备的，但 `main()` 用的是 `generate_slicing_trajectory_from_ref(P_ref)`。

#### 额外要注意的一点

`pointcloud_trajectory.py` 里 `sort_slice_by_angle` 和 `connect_slices_s_shape` 各写了两遍。
Python 会以后出现的定义覆盖前面的定义，所以文件上半部分那两个同名函数，运行时其实会被下半部分版本替代。

#### 结论

当前“方法 2”更准确的描述不是“点云 SLAM 后端规划”，而是：

“以共享参考喷嘴轨迹为输入，做 z 向切片重排，拼出一条 S 型覆盖折线，再用 B 样条平滑。”

也就是说，它是一个“借用了点云术语的轨迹重排脚本”，而不是闭环的点云驱动轨迹生成器。

### 模块 5：仓库空壳与未实现部分

这一块虽然代码少，但对理解项目边界很重要。

#### 当前为空的部分

1. `src/train.py`
   空文件。
2. `src/eval.py`
   空文件。
3. `src/model/`
   目录存在，但没有实际模型文件。

#### 说明

这意味着仓库目前没有：

1. 训练入口。
2. 评测入口。
3. 神经网络模型定义。
4. 数据加载器。
5. RL 算法实现。

所以从现状看，它更像“轨迹生成研究草稿”而不是“完整训练工程”。

## 训练配方

### 数据使用

当前没有看到任何真实数据集接入，也没有离线数据读写逻辑。

项目使用的是程序化生成的数据：

1. 半圆柱隧道几何。
2. 参数域 `(u, v)` 上的分形噪声。
3. 由噪声岩壁推导出的参考岩面轨迹和喷嘴轨迹。

如果把它放在论文语境里，更准确的说法应当是“procedural geometry generation”，不是“dataset-driven training”。

### Setting

目前能谈的“配方”其实是几何与后处理参数，而不是训练超参数。

| 类别 | 关键参数 | 当前值 | 作用 |
| --- | --- | --- | --- |
| 隧道几何 | `R_BASE` | `4.0` | 基准半径 |
| 隧道几何 | `L_TUNNEL` | `10.0` | 隧道长度 |
| 喷射距离 | `D_SPRAY` | `1.5` | 喷嘴相对岩面的后退距离 |
| 岩壁噪声 | `NOISE_SCALE` | `0.5` | 噪声频率缩放 |
| 岩壁噪声 | `NOISE_OCTAVES` | `4` | 分形层数 |
| 岩壁噪声 | `NOISE_PERSISTENCE` | `0.5` | 各层振幅衰减 |
| 岩壁噪声 | `NOISE_AMPLITUDE` | `0.3` | 岩壁凸凹幅度 |
| 重力补偿 | `K_SLUMP` | `0.15` | 流淌补偿强度 |
| 方法 1 | `SG_WINDOW_LENGTH` | `21` | S-G 滤波窗口 |
| 方法 1 | `SG_POLY_ORDER` | `3` | S-G 多项式阶数 |
| 方法 1 | `BSPLINE_SMOOTH` | `0.5` | B 样条平滑因子 |
| 方法 1 | `N_FINAL_POINTS` | `1000` | 最终轨迹采样数 |
| 方法 2 | `VOXEL_SIZE` | `0.1` | 体素降采样分辨率 |
| 方法 2 | `NORMAL_RADIUS` | `0.5` | 法线估计邻域半径 |
| 方法 2 | `NORMAL_MAX_NN` | `30` | 法线估计最大邻居数 |
| 方法 2 | `SLICE_INTERVAL` | `0.5` | 切片间隔 |
| 方法 2 | `SLICE_THICKNESS` | `0.05` | 切片厚度 |
| 方法 2 | `BSPLINE_SMOOTH` | `0.1` | 切片轨迹平滑强度 |

可视化统一使用 Plotly，输出 HTML，脚本结束时会尝试自动打开浏览器。

## 其他

### 复用方式

这个仓库当前最适合当“共享参考世界生成器”来复用。

比较自然的复用方式是：

1. 把 `rock_wall.generate_rock_wall()` 当作统一数据源。
2. 把 `P_nozzle` 当作参考轨迹基线。
3. 在此基础上替换你自己的平滑器、覆盖规划器、控制器或 RL 环境。

### 代码风格上的现象

1. 很多函数注释写得比实际实现更宏大，阅读时要以 `main()` 的真实调用链为准。
2. 可视化辅助函数在多个文件里重复实现了一遍。
3. 配置全部写死在文件顶部，没有独立配置系统，也没有命令行参数接口。

## 注意事项

### 实现边界

这部分是最需要明确写清楚的。

1. 当前仓库没有训练代码。
   `src/train.py` 和 `src/eval.py` 都是空的。
2. 当前仓库没有模型代码。
   `src/model/` 也是空目录。
3. 当前仓库没有 RL 环境实现。
   `env.py` 是可视化脚本，不是 Gym/Gymnasium 风格环境。
4. “方法 1”没有真正验证机器人动力学约束。
   它做的是轨迹平滑与重采样。
5. “方法 2”不是严格意义上的点云驱动轨迹生成。
   它的主线实际上依赖 `rock_wall.py` 生成的 `P_nozzle`。
6. 岩壁点云目前主要是背景资产，不是规划主输入。
7. 没有测试、没有基准、没有实验脚本来证明两种方法的优劣。

### 注意事项

如果你准备继续改这个项目，下面这些点会很重要。

1. 运行入口依赖包不全时会直接失败。
   至少要注意 `numpy`、`plotly`、`scipy`，以及噪声库 `opensimplex` 或 `vnoise`；`pointcloud_trajectory.py` 还要求 `open3d`。
2. 导入方式写成了 `from src.rock_generate...`。
   这意味着最好从仓库根目录运行脚本，否则相对导入路径可能出问题。
3. 脚本会调用 `webbrowser.open(...)`。
   在无图形界面环境、远程服务器或容器里，可能会出现打开浏览器失败或无效果的情况。
4. 所有参数都写死在文件顶部。
   你一旦做实验对比，很快会遇到配置难以管理的问题。
5. `pointcloud_trajectory.py` 存在重复定义函数。
   修改时要看清楚最终生效的是后一个同名定义。
6. 如果你后面要接真实点云，请优先重构 `pointcloud_trajectory.py` 的 `main()`。
   现在最该打通的是“真实墙面点云 -> 切片 -> 法向 -> 喷嘴回退 -> 平滑”的闭环，而不是继续围绕 `P_ref` 做包装。
7. 如果你后面要接 RL，请先明确状态、动作、奖励和仿真接口。
   目前仓库只提供几何轨迹，不提供交互式环境。

