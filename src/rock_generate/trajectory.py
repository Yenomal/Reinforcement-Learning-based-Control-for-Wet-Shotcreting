#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人 3D 轨迹滤波与连续参数化拟合模块 (方法 1: S-G 滤波 + B 样条)
====================================================================
对带有高频法线抖动噪声的 3D 粗糙离散轨迹进行滤波与连续参数化拟合，
生成满足机器人动力学约束的平滑轨迹。

岩壁与 env.py 完全一致（通过 rock_wall 模块）。

核心算法：
1. Savitzky-Golay (S-G) 滤波：剔除高频锯齿毛刺，保持宏观曲率
2. B 样条 (B-Spline) 参数化拟合：生成时间参数化的连续函数
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from typing import Tuple, Optional
import webbrowser
import os

# 导入与 env.py 一致的岩壁生成模块
from src.rock_generate.rock_wall import (
    generate_rock_wall, generate_dense_rock_wall,
    NoiseGenerator, R_BASE, L_TUNNEL, D_SPRAY,
    NOISE_AMPLITUDE, NOISE_SCALE, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_SEED
)

import plotly.graph_objects as go


# =============================================================================
# 全局配置参数
# =============================================================================

# --- S-G 滤波参数 ---
SG_WINDOW_LENGTH = 21       # 滑动窗口长度 (必须为奇数)
SG_POLY_ORDER = 3           # 多项式阶数

# --- B 样条拟合参数 ---
BSPLINE_SMOOTH = 0.5        # 平滑因子 s
N_FINAL_POINTS = 1000       # 最终重采样点数

# --- 可视化参数 ---
FIG_WIDTH = 1400
FIG_HEIGHT = 900
HTML_OUTPUT = "trajectory_sg_bspline.html"


# =============================================================================
# 步骤一：直接获取 env.py 参考轨迹进行滤波
# =============================================================================

def apply_savitzky_golay_filter(P_ref: np.ndarray,
                                 window_length: int = SG_WINDOW_LENGTH,
                                 polyorder: int = SG_POLY_ORDER) -> np.ndarray:
    """
    对 env.py 参考轨迹应用 Savitzky-Golay 滤波器
    
    物理意义：
    S-G 滤波器是一种基于局部多项式最小二乘拟合的数字滤波器。
    在平滑噪声的同时能够保持信号的宏观曲率特征。
    
    Args:
        P_ref: env.py 生成的喷嘴参考轨迹 [N, 3]
        window_length: 滑动窗口长度（必须为奇数）
        polyorder: 多项式阶数（必须小于 window_length）
        
    Returns:
        P_filtered: 滤波后的轨迹数组 [N, 3]
    """
    if window_length % 2 == 0:
        raise ValueError("window_length 必须为奇数")
    if polyorder >= window_length:
        raise ValueError("polyorder 必须小于 window_length")
    if window_length > len(P_ref):
        raise ValueError("window_length 不能大于轨迹点数")
    
    P_filtered = np.zeros_like(P_ref)
    
    # 对 X, Y, Z 三个坐标轴分别进行滤波
    P_filtered[:, 0] = savgol_filter(P_ref[:, 0], window_length, polyorder)
    P_filtered[:, 1] = savgol_filter(P_ref[:, 1], window_length, polyorder)
    P_filtered[:, 2] = savgol_filter(P_ref[:, 2], window_length, polyorder)
    
    return P_filtered


# =============================================================================
# 步骤二：B 样条曲线连续参数化拟合
# =============================================================================

def bspline_parametric_fit(P_filtered: np.ndarray,
                            smooth: float = BSPLINE_SMOOTH,
                            n_final_points: int = N_FINAL_POINTS) -> Tuple[np.ndarray, tuple]:
    """
    使用 B 样条进行连续参数化拟合
    
    物理意义：
    B 样条（B-Spline）是一种分段多项式参数曲线，在机器人轨迹规划中：
    1. 将离散点集转化为连续的时间参数化函数 P(u)，u ∈ [0, 1]
    2. 保证轨迹的 C² 连续性（位置、速度、加速度均连续）
    3. 平滑因子 s 控制拟合的"松紧度"
    4. 重采样到均匀时间序列，模拟 RL 环境的固定控制频率（如 50Hz）
    
    Args:
        P_filtered: 滤波后的轨迹数组 [N, 3]
        smooth: 平滑因子 s
        n_final_points: 最终重采样点数
        
    Returns:
        P_final: 最终平滑轨迹数组 [n_final_points, 3]
        tck: 样条表示 (knots, coefficients, degree)
    """
    points_T = P_filtered.T
    
    tck, u = splprep(points_T, s=smooth, k=3)
    
    u_new = np.linspace(0, 1, n_final_points)
    P_final_T = splev(u_new, tck)
    P_final = np.column_stack(P_final_T)
    
    return P_final, tck


# =============================================================================
# 步骤四：3D 可视化对比
# =============================================================================

def create_rock_surface_mesh(rock_dense_data: dict,
                              noise_gen, grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建岩面网格表面（与 env.py 一致）

    Args:
        rock_dense_data: 岩壁数据
        noise_gen: 噪声生成器
        grid_size: 网格分辨率

    Returns:
        x_grid, y_grid, z_grid, color_grid: 网格坐标和颜色（半径）
    """
    from src.rock_generate.rock_wall import R_BASE, L_TUNNEL, NOISE_AMPLITUDE, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_SCALE

    # 在隧道表面生成网格
    theta = np.linspace(0, np.pi, grid_size)
    z = np.linspace(0, L_TUNNEL, grid_size)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = np.zeros_like(theta_grid)
    y_grid = np.zeros_like(theta_grid)
    color_grid = np.zeros_like(theta_grid)  # 使用半径作为颜色（岩面高度）

    for i in range(grid_size):
        for j in range(grid_size):
            theta_ij = theta_grid[i, j]
            z_ij = z_grid[i, j]

            u = R_BASE * theta_ij
            v = z_ij

            noise_val = noise_gen.layered_noise2d(
                u, v, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_SCALE
            )

            r_rock = R_BASE + noise_val * NOISE_AMPLITUDE

            x_grid[i, j] = r_rock * np.cos(theta_ij)
            y_grid[i, j] = r_rock * np.sin(theta_ij)
            z_grid[i, j] = z_ij
            color_grid[i, j] = r_rock  # 使用半径作为颜色（岩面高度）

    return x_grid, y_grid, z_grid, color_grid


def create_visualization(rock_dense_data: dict,
                         P_filtered: np.ndarray,
                         P_final: np.ndarray,
                         P_ref: np.ndarray) -> go.Figure:
    """
    创建交互式 3D 可视化图表
    
    包含：
    1. 凹凸岩壁面（与 env.py 一致的 Earth 色系曲面）
    2. S-G 滤波轨迹（蓝色细线）
    3. B 样条最终轨迹（红色粗实线）
    4. env.py 参考轨迹（绿色虚线）
    """
    fig = go.Figure()
    
    # 1. 凹凸岩壁面（与 env.py 一致的 Earth 色系曲面）
    x_rock, y_rock, z_rock, color_rock = create_rock_surface_mesh(
        rock_dense_data, rock_dense_data['noise_gen'], grid_size=50
    )
    
    fig.add_trace(go.Surface(
        x=x_rock, y=y_rock, z=z_rock,
        surfacecolor=color_rock,
        opacity=0.8,
        colorscale='Earth',
        showscale=True,
        colorbar=dict(title='岩面高度 (m)'),
        name='凹凸岩壁',
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra>岩壁</extra>'
    ))
    
    # 2. S-G 滤波轨迹 (P_filtered)
    fig.add_trace(go.Scatter3d(
        x=P_filtered[:, 0],
        y=P_filtered[:, 1],
        z=P_filtered[:, 2],
        mode='lines',
        line=dict(
            width=3,
            color='blue',
            dash='solid'
        ),
        name='S-G 滤波后',
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra>S-G 滤波</extra>'
    ))

    # 3. 最终 B 样条连续轨迹 (P_final)
    fig.add_trace(go.Scatter3d(
        x=P_final[:, 0],
        y=P_final[:, 1],
        z=P_final[:, 2],
        mode='lines',
        line=dict(
            width=8,
            color='red',
            dash='solid'
        ),
        name='B 样条最终轨迹',
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra>B 样条拟合</extra>'
    ))

    # 4. env.py 参考轨迹（绿色虚线）
    fig.add_trace(go.Scatter3d(
        x=P_ref[:, 0],
        y=P_ref[:, 1],
        z=P_ref[:, 2],
        mode='lines',
        line=dict(
            width=3,
            color='green',
            dash='dash'
        ),
        name='env.py 参考轨迹',
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra>env.py</extra>'
    ))
    
    # 标记起点和终点
    fig.add_trace(go.Scatter3d(
        x=[P_final[0, 0]],
        y=[P_final[0, 1]],
        z=[P_final[0, 2]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='起点',
        hovertemplate='起点<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[P_final[-1, 0]],
        y=[P_final[-1, 1]],
        z=[P_final[-1, 2]],
        mode='markers',
        marker=dict(size=10, color='darkred', symbol='circle'),
        name='终点',
        hovertemplate='终点<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))
    
    # 更新布局
    fig.update_layout(
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        scene=dict(
            xaxis=dict(title='X (m)', showgrid=True, zeroline=True),
            yaxis=dict(title='Y (m)', showgrid=True, zeroline=True),
            zaxis=dict(title='Z (m) - 隧道深度', showgrid=True, zeroline=True),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                projection=dict(type='perspective')
            )
        ),
        title=dict(
            text='🤖 机器人 3D 轨迹滤波与 B 样条拟合',
            font=dict(size=18)
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    # 添加与 env.py 一致的标题布局
    fig.update_layout(
        title=dict(
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        scene=dict(
            domain=dict(x=[0, 0.75], y=[0, 1])  # 给图例留空间
        )
    )
    
    return fig


def compute_trajectory_metrics(P_rough: np.ndarray, P_filtered: np.ndarray,
                                P_final: np.ndarray) -> dict:
    """计算轨迹质量指标"""
    def calc_path_length(P: np.ndarray) -> float:
        diffs = np.diff(P, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))
    
    def calc_smoothness(P: np.ndarray) -> float:
        velocity = np.diff(P, axis=0)
        acceleration = np.diff(velocity, axis=0)
        accel_norms = np.linalg.norm(acceleration, axis=1)
        return np.sqrt(np.mean(accel_norms ** 2))
    
    def calc_curvature_variation(P: np.ndarray) -> float:
        velocity = np.diff(P, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        jerk_norms = np.linalg.norm(jerk, axis=1)
        return np.sqrt(np.mean(jerk_norms ** 2))
    
    metrics = {
        'rough': {
            'length': calc_path_length(P_rough),
            'smoothness': calc_smoothness(P_rough),
            'curvature_variation': calc_curvature_variation(P_rough)
        },
        'filtered': {
            'length': calc_path_length(P_filtered),
            'smoothness': calc_smoothness(P_filtered),
            'curvature_variation': calc_curvature_variation(P_filtered)
        },
        'final': {
            'length': calc_path_length(P_final),
            'smoothness': calc_smoothness(P_final),
            'curvature_variation': calc_curvature_variation(P_final)
        }
    }
    
    return metrics


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主执行函数"""
    print("=" * 70)
    print("机器人 3D 轨迹滤波与连续参数化拟合模块 (S-G + B 样条)")
    print("=" * 70)

    # 打印配置信息
    print(f"\n【配置参数】")
    print(f"  隧道基准半径：{R_BASE} m")
    print(f"  隧道长度：{L_TUNNEL} m")
    print(f"  喷射距离：{D_SPRAY} m")
    print(f"  噪声参数：Scale={NOISE_SCALE}, Octaves={NOISE_OCTAVES}, Persistence={NOISE_PERSISTENCE}")
    print(f"  噪声振幅：{NOISE_AMPLITUDE} m")
    print(f"  S-G 滤波窗口长度：{SG_WINDOW_LENGTH}")
    print(f"  S-G 滤波多项式阶数：{SG_POLY_ORDER}")
    print(f"  B 样条平滑因子 s: {BSPLINE_SMOOTH}")
    print(f"  最终重采样点数：{N_FINAL_POINTS}")

    # 生成与 env.py 一致的岩壁和轨迹
    print(f"\n【岩壁生成】生成与 env.py 一致的岩壁...")
    rock_data = generate_rock_wall()
    rock_dense_data = generate_dense_rock_wall()
    print(f"  岩壁点云：{len(rock_dense_data['points'])} 点")

    # 获取 env.py 喷嘴参考轨迹
    P_ref = rock_data['P_nozzle']
    print(f"  env.py 喷嘴轨迹：{len(P_ref)} 点")

    # 步骤一：S-G 滤波
    print(f"\n【步骤一】应用 Savitzky-Golay 滤波器...")
    P_filtered = apply_savitzky_golay_filter(P_ref, SG_WINDOW_LENGTH, SG_POLY_ORDER)
    print(f"  S-G 滤波完成")

    # 步骤二：B 样条拟合
    print(f"\n【步骤二】B 样条连续参数化拟合...")
    P_final, tck = bspline_parametric_fit(P_filtered, BSPLINE_SMOOTH, N_FINAL_POINTS)
    print(f"  重采样到 {len(P_final)} 个点")

    # 计算轨迹质量指标
    print(f"\n【轨迹质量指标】")
    metrics = compute_trajectory_metrics(P_ref, P_filtered, P_final)

    print(f"  ┌{'─' * 20}┬{'─' * 15}┬{'─' * 15}┬{'─' * 15}┐")
    print(f"  │  指标          │  env.py 参考  │  S-G 滤波    │  B 样条最终  │")
    print(f"  ├{'─' * 20}┼{'─' * 15}┼{'─' * 15}┼{'─' * 15}┤")
    print(f"  │  路径长度 (m)   │  {metrics['rough']['length']:>8.2f}  │  {metrics['filtered']['length']:>8.2f}  │  {metrics['final']['length']:>8.2f}  │")
    print(f"  │  平滑度        │  {metrics['rough']['smoothness']:>8.4f}  │  {metrics['filtered']['smoothness']:>8.4f}  │  {metrics['final']['smoothness']:>8.4f}  │")
    print(f"  │  曲率变化率    │  {metrics['rough']['curvature_variation']:>8.4f}  │  {metrics['filtered']['curvature_variation']:>8.4f}  │  {metrics['final']['curvature_variation']:>8.4f}  │")
    print(f"  └{'─' * 20}┴{'─' * 15}┴{'─' * 15}┴{'─' * 15}┘")
    print(f"  注：平滑度和曲率变化率越小表示轨迹越平滑")
    
    # 创建可视化
    print(f"\n【步骤三】创建交互式 3D 可视化...")
    fig = create_visualization(
        rock_dense_data, P_filtered, P_final, P_ref
    )

    # 保存 HTML 文件
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HTML_OUTPUT)
    print(f"\n【输出】保存 HTML 文件：{output_path}")
    fig.write_html(output_path, include_plotlyjs=True, auto_open=False)

    # 在浏览器中打开
    print(f"【完成】在浏览器中打开可视化结果...")
    webbrowser.open(f'file://{output_path}')

    print("\n" + "=" * 70)
    print("轨迹滤波与拟合完成！")
    print("=" * 70)
    print(f"\n提示：")
    print(f"  - 地球色曲面：凹凸岩壁（颜色=岩面高度/半径）")
    print(f"  - 蓝色细线：S-G 滤波后轨迹")
    print(f"  - 红色粗线：B 样条最终轨迹")
    print(f"  - 绿色虚线：env.py 参考轨迹（原始输入）")
    print()


if __name__ == "__main__":
    main()
