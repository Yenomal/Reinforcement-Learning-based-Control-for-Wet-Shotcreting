#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
湿喷机器人 3D 目标轨迹生成器
============================
生成并可视化混凝土喷射机器人在非结构化岩面上的复杂 3D 目标轨迹。

使用 rock_wall 模块生成岩壁，保持所有脚本的岩壁一致性。
"""

import numpy as np
import webbrowser
import os
import plotly.graph_objects as go

# 导入共享的岩壁生成模块
from src.rock_generate.rock_wall import (
    generate_rock_wall, generate_dense_rock_wall,
    NoiseGenerator, R_BASE, L_TUNNEL, D_SPRAY,
    NOISE_AMPLITUDE, NOISE_SCALE, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_SEED
)


# =============================================================================
# 可视化函数
# =============================================================================

def create_rock_surface_mesh(noise_gen, grid_size: int = 50) -> tuple:
    """创建岩面网格表面"""
    theta = np.linspace(0, np.pi, grid_size)
    z = np.linspace(0, L_TUNNEL, grid_size)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    x_grid = np.zeros_like(theta_grid)
    y_grid = np.zeros_like(theta_grid)
    color_grid = np.zeros_like(theta_grid)
    
    for i in range(grid_size):
        for j in range(grid_size):
            theta_ij = theta_grid[i, j]
            z_ij = z_grid[i, j]
            u = R_BASE * theta_ij
            v = z_ij
            
            noise_val = noise_gen.layered_noise2d(u, v, NOISE_OCTAVES, NOISE_PERSISTENCE, NOISE_SCALE)
            r_rock = R_BASE + noise_val * NOISE_AMPLITUDE
            
            x_grid[i, j] = r_rock * np.cos(theta_ij)
            y_grid[i, j] = r_rock * np.sin(theta_ij)
            z_grid[i, j] = z_ij
            color_grid[i, j] = r_rock  # 使用半径作为颜色（岩面高度）
    
    return x_grid, y_grid, z_grid, color_grid


def create_visualization(rock_data: dict, rock_dense_data: dict) -> go.Figure:
    """创建交互式 3D 可视化图表"""
    fig = go.Figure()
    
    # 1. 凹凸岩壁面（Earth 色系，颜色表示岩面高度/半径）
    x_rock, y_rock, z_rock, color_rock = create_rock_surface_mesh(
        rock_dense_data['noise_gen'], grid_size=50
    )
    
    fig.add_trace(go.Surface(
        x=x_rock, y=y_rock, z=z_rock,
        surfacecolor=color_rock,
        opacity=0.8,
        colorscale='Earth',
        showscale=True,
        colorbar=dict(title='岩面高度 (m)'),
        name='凹凸岩壁'
    ))
    
    # 2. 岩面 S 型轨迹（按深度着色）
    fig.add_trace(go.Scatter3d(
        x=rock_data['P_final_rock'][:, 0],
        y=rock_data['P_final_rock'][:, 1],
        z=rock_data['P_final_rock'][:, 2],
        mode='lines',
        line=dict(width=6, colorscale='Viridis', color=rock_data['v_coords']),
        name='岩面 S 型轨迹'
    ))
    
    # 3. 喷嘴轨迹（红色虚线）
    fig.add_trace(go.Scatter3d(
        x=rock_data['P_nozzle'][:, 0],
        y=rock_data['P_nozzle'][:, 1],
        z=rock_data['P_nozzle'][:, 2],
        mode='lines',
        line=dict(width=4, color='red', dash='dash'),
        name='喷嘴轨迹'
    ))
    
    # 4. 法向量（采样显示）
    for i in range(0, len(rock_data['normals']), 10):
        start = rock_data['P_final_rock'][i]
        end = start + rock_data['normals'][i] * 0.5
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines', line=dict(width=2, color='green'),
            name='法向量' if i == 0 else '', showlegend=(i == 0), hoverinfo='skip'
        ))
    
    # 5. 起点和终点
    fig.add_trace(go.Scatter3d(
        x=[rock_data['P_final_rock'][0, 0]],
        y=[rock_data['P_final_rock'][0, 1]],
        z=[rock_data['P_final_rock'][0, 2]],
        mode='markers', marker=dict(size=8, color='green'),
        name='起点'
    ))
    fig.add_trace(go.Scatter3d(
        x=[rock_data['P_final_rock'][-1, 0]],
        y=[rock_data['P_final_rock'][-1, 1]],
        z=[rock_data['P_final_rock'][-1, 2]],
        mode='markers', marker=dict(size=8, color='red'),
        name='终点'
    ))
    
    # 更新布局
    fig.update_layout(
        width=1400, height=900,
        scene=dict(
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Y (m)'),
            zaxis=dict(title='Z (m) - 隧道深度'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        title=dict(text='🏗️ 湿喷机器人 3D 目标轨迹生成结果', x=0.5),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主执行函数"""
    print("=" * 60)
    print("湿喷机器人 3D 目标轨迹生成器")
    print("=" * 60)
    
    print(f"\n【配置参数】")
    print(f"  隧道基准半径：{R_BASE} m")
    print(f"  隧道长度：{L_TUNNEL} m")
    print(f"  喷射距离：{D_SPRAY} m")
    print(f"  噪声参数：Scale={NOISE_SCALE}, Octaves={NOISE_OCTAVES}, Persistence={NOISE_PERSISTENCE}")
    print(f"  噪声振幅：{NOISE_AMPLITUDE} m")
    print(f"  重力流淌系数：0.15")
    
    # 生成岩壁和轨迹（使用 rock_wall 模块）
    print(f"\n【生成轨迹】使用 rock_wall 模块...")
    rock_data = generate_rock_wall()
    rock_dense_data = generate_dense_rock_wall()
    
    print(f"  岩壁点云：{len(rock_dense_data['points'])} 点")
    print(f"  轨迹点数：{len(rock_data['P_final_rock'])}")
    
    # 计算轨迹长度
    def calc_path_length(points):
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    
    print(f"\n【轨迹统计】")
    print(f"  岩面轨迹长度：{calc_path_length(rock_data['P_final_rock']):.2f} m")
    print(f"  喷嘴轨迹长度：{calc_path_length(rock_data['P_nozzle']):.2f} m")
    
    # 创建可视化
    print(f"\n【可视化】创建交互式 3D 图表...")
    fig = create_visualization(rock_data, rock_dense_data)
    
    # 保存并打开
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shotcrete_trajectory.html")
    fig.write_html(output_path, include_plotlyjs=True, auto_open=False)
    print(f"【输出】{output_path}")
    webbrowser.open(f'file://{output_path}')
    
    print("\n轨迹生成完成！")


if __name__ == "__main__":
    main()
