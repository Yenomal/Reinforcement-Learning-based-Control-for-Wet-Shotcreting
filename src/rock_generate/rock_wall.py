#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
湿喷隧道岩壁生成共享模块
========================
与 env.py 使用完全相同的柏林噪声参数和数学逻辑，
确保所有轨迹生成算法的岩壁背景一致。

使用方法：
    from rock_wall import generate_rock_wall, NoiseGenerator, R_BASE, L_TUNNEL, ...
"""

import numpy as np
from typing import Tuple, Optional

# 尝试导入柏林噪声库（与 env.py 一致）
try:
    from opensimplex import OpenSimplex
    NOISE_LIBRARY = "opensimplex"
except ImportError:
    try:
        import vnoise
        NOISE_LIBRARY = "vnoise"
    except ImportError:
        raise ImportError(
            "错误：未找到柏林噪声库！\n"
            "请安装以下任一库:\n"
            "  pip install opensimplex  (推荐)\n"
            "  pip install vnoise\n"
        )


# =============================================================================
# 全局参数（与 env.py 完全一致）
# =============================================================================

# --- 隧道几何参数 ---
R_BASE = 4.0          # 隧道基准半径 (米)
L_TUNNEL = 10.0       # 隧道长度 (米)
D_SPRAY = 1.5         # 混凝土目标喷射距离 (米)

# --- 2D 平面轨迹起点与终点坐标 ---
U_START, V_START = 0.0, 0.0
U_END, V_END = np.pi * R_BASE, L_TUNNEL

# --- 柏林噪声参数 ---
NOISE_SCALE = 0.5         # 噪声频率缩放因子
NOISE_OCTAVES = 4         # 倍频程数量
NOISE_PERSISTENCE = 0.5   # 持久性
NOISE_AMPLITUDE = 0.3     # 噪声振幅 (米)

# --- 重力流淌参数 ---
K_SLUMP = 0.15            # 重力流淌系数

# --- 轨迹采样参数 ---
N_POINTS_U = 100
N_POINTS_V = 100
DELTA = 0.01

# --- 噪声种子 ---
NOISE_SEED = 42


# =============================================================================
# 柏林噪声生成器（与 env.py 完全一致）
# =============================================================================

class NoiseGenerator:
    """柏林噪声生成器，支持 opensimplex 和 vnoise 两种后端"""
    
    def __init__(self, seed: int = NOISE_SEED):
        self.seed = seed
        if NOISE_LIBRARY == "opensimplex":
            self.sampler = OpenSimplex(seed)
        else:  # vnoise
            self.sampler = vnoise.Noise()
            self.sampler.seed = seed
    
    def noise2d(self, x: float, y: float) -> float:
        """生成 2D 柏林噪声值"""
        if NOISE_LIBRARY == "opensimplex":
            return self.sampler.noise2(x, y)
        else:
            return self.sampler.noise2(x, y)
    
    def layered_noise2d(self, x: float, y: float,
                        octaves: int = NOISE_OCTAVES,
                        persistence: float = NOISE_PERSISTENCE,
                        scale: float = NOISE_SCALE) -> float:
        """生成多层叠加的柏林噪声（分形噪声）"""
        total = 0.0
        frequency = scale
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return total / max_value


# =============================================================================
# 2D S 型轨迹生成（与 env.py 完全一致）
# =============================================================================

def generate_2d_s_path(u_start: float = U_START, v_start: float = V_START,
                       u_end: float = U_END, v_end: float = V_END,
                       n_points: int = N_POINTS_U) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 2D 平面上的 S 型连续路径（与 env.py 完全一致）
    """
    t = np.linspace(0, 1, n_points)
    
    u_linear = u_start + (u_end - u_start) * t
    v_linear = v_start + (v_end - v_start) * t
    
    u_amplitude = (u_end - u_start) * 0.15
    u_sine = u_amplitude * np.sin(np.pi * t) * np.sin(2 * np.pi * t)
    
    v_amplitude = (v_end - v_start) * 0.05
    v_sine = v_amplitude * np.sin(2 * np.pi * t)
    
    u_coords = u_linear + u_sine
    v_coords = v_linear + v_sine
    
    return u_coords, v_coords


# =============================================================================
# 3D 岩壁点生成（与 env.py 完全一致）
# =============================================================================

def map_to_3d_cylinder(u_coords: np.ndarray, v_coords: np.ndarray,
                       r_base: float = R_BASE,
                       noise_gen: Optional[NoiseGenerator] = None,
                       noise_scale: float = NOISE_SCALE,
                       octaves: int = NOISE_OCTAVES,
                       persistence: float = NOISE_PERSISTENCE,
                       amplitude: float = NOISE_AMPLITUDE) -> np.ndarray:
    """
    将 2D 坐标映射到 3D 柱坐标系并注入径向柏林噪声（与 env.py 完全一致）
    """
    if noise_gen is None:
        noise_gen = NoiseGenerator(seed=NOISE_SEED)
    
    n_points = len(u_coords)
    P_rock_noisy = np.zeros((n_points, 3))
    
    for i in range(n_points):
        u, v = u_coords[i], v_coords[i]
        theta = u / r_base
        
        noise_val = noise_gen.layered_noise2d(
            u, v, octaves, persistence, noise_scale
        )
        
        r_rock = r_base + noise_val * amplitude
        
        x = r_rock * np.cos(theta)
        y = r_rock * np.sin(theta)
        z = v
        
        P_rock_noisy[i] = [x, y, z]
    
    return P_rock_noisy


# =============================================================================
# 法向量计算（与 env.py 完全一致）
# =============================================================================

def compute_surface_normal(P_rock_noisy: np.ndarray, u_coords: np.ndarray,
                           v_coords: np.ndarray, delta: float = DELTA,
                           noise_gen: Optional[NoiseGenerator] = None,
                           r_base: float = R_BASE,
                           noise_scale: float = NOISE_SCALE,
                           octaves: int = NOISE_OCTAVES,
                           persistence: float = NOISE_PERSISTENCE,
                           amplitude: float = NOISE_AMPLITUDE) -> np.ndarray:
    """
    使用有限差分法计算局部法向量（与 env.py 完全一致）
    """
    if noise_gen is None:
        noise_gen = NoiseGenerator(seed=NOISE_SEED)
    
    n_points = len(P_rock_noisy)
    normals = np.zeros((n_points, 3))
    
    def get_3d_point(u: float, v: float) -> np.ndarray:
        theta = u / r_base
        noise_val = noise_gen.layered_noise2d(
            u, v, octaves, persistence, noise_scale
        )
        r_rock = r_base + noise_val * amplitude
        return np.array([
            r_rock * np.cos(theta),
            r_rock * np.sin(theta),
            v
        ])
    
    for i in range(n_points):
        u, v = u_coords[i], v_coords[i]
        
        u_plus = u + delta
        v_plus = v + delta
        
        P_u = get_3d_point(u_plus, v)
        P_v = get_3d_point(u, v_plus)
        P_0 = P_rock_noisy[i]
        
        tangent_u = (P_u - P_0) / delta
        tangent_v = (P_v - P_0) / delta
        
        normal = np.cross(tangent_u, tangent_v)
        
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        else:
            theta = u / r_base
            normal = np.array([np.cos(theta), np.sin(theta), 0.0])
        
        normals[i] = normal
    
    return normals


# =============================================================================
# 重力流淌补偿（与 env.py 完全一致）
# =============================================================================

def compute_gravity_slump(P_rock_noisy: np.ndarray, normals: np.ndarray,
                          u_coords: np.ndarray, v_coords: np.ndarray,
                          k_slump: float = K_SLUMP) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算重力流淌补偿并生成最终岩面轨迹（与 env.py 完全一致）
    """
    n_points = len(P_rock_noisy)
    P_final_rock = np.zeros_like(P_rock_noisy)
    slump_vectors = np.zeros_like(P_rock_noisy)
    
    g = np.array([0.0, -1.0, 0.0])
    
    for i in range(n_points):
        N_true = normals[i]
        
        g_cross_N = np.cross(g, N_true)
        S_dir = np.cross(N_true, g_cross_N)
        
        s_norm = np.linalg.norm(S_dir)
        if s_norm > 1e-10:
            S_dir = S_dir / s_norm
        else:
            S_dir = np.array([0.0, 0.0, 0.0])
        
        steepness = np.sqrt(N_true[0]**2 + N_true[2]**2)
        
        slump_magnitude = k_slump * steepness
        slump_vector = S_dir * slump_magnitude
        
        P_final_rock[i] = P_rock_noisy[i] + slump_vector
        slump_vectors[i] = slump_vector
    
    return P_final_rock, slump_vectors


# =============================================================================
# 喷嘴轨迹计算（与 env.py 完全一致）
# =============================================================================

def compute_nozzle_trajectory(P_final_rock: np.ndarray, normals: np.ndarray,
                              d_spray: float = D_SPRAY) -> np.ndarray:
    """
    计算喷嘴最终目标轨迹（与 env.py 完全一致）
    """
    P_nozzle = P_final_rock - normals * d_spray
    return P_nozzle


# =============================================================================
# 完整岩壁生成流程
# =============================================================================

def generate_rock_wall(n_points_u: int = N_POINTS_U,
                       n_points_v: int = N_POINTS_V,
                       seed: int = NOISE_SEED) -> dict:
    """
    生成完整的岩壁和轨迹数据（与 env.py 完全一致）
    
    Returns:
        data: 包含所有中间结果的字典
    """
    noise_gen = NoiseGenerator(seed=seed)
    
    noise_params = {
        'scale': NOISE_SCALE,
        'octaves': NOISE_OCTAVES,
        'persistence': NOISE_PERSISTENCE,
        'amplitude': NOISE_AMPLITUDE
    }
    
    # 生成 2D S 型轨迹
    u_coords, v_coords = generate_2d_s_path(
        U_START, V_START, U_END, V_END, n_points_u
    )
    
    # 映射到 3D 并注入噪声
    P_rock_noisy = map_to_3d_cylinder(
        u_coords, v_coords, R_BASE,
        noise_gen, NOISE_SCALE, NOISE_OCTAVES,
        NOISE_PERSISTENCE, NOISE_AMPLITUDE
    )
    
    # 计算法向量
    normals = compute_surface_normal(
        P_rock_noisy, u_coords, v_coords, DELTA,
        noise_gen, R_BASE, NOISE_SCALE, NOISE_OCTAVES,
        NOISE_PERSISTENCE, NOISE_AMPLITUDE
    )
    
    # 重力流淌补偿
    P_final_rock, slump_vectors = compute_gravity_slump(
        P_rock_noisy, normals, u_coords, v_coords, K_SLUMP
    )
    
    # 喷嘴轨迹
    P_nozzle = compute_nozzle_trajectory(P_final_rock, normals, D_SPRAY)
    
    return {
        'noise_gen': noise_gen,
        'noise_params': noise_params,
        'u_coords': u_coords,
        'v_coords': v_coords,
        'P_rock_noisy': P_rock_noisy,
        'normals': normals,
        'P_final_rock': P_final_rock,
        'slump_vectors': slump_vectors,
        'P_nozzle': P_nozzle,
        'R_BASE': R_BASE,
        'L_TUNNEL': L_TUNNEL,
        'D_SPRAY': D_SPRAY,
        'NOISE_AMPLITUDE': NOISE_AMPLITUDE
    }


def generate_dense_rock_wall(n_theta: int = 200, n_z: int = 100,
                              seed: int = NOISE_SEED) -> dict:
    """
    生成高密度的岩壁点云（用于可视化背景）
    
    Args:
        n_theta: 角度方向采样点数
        n_z: 轴向采样点数
        seed: 随机种子
        
    Returns:
        data: 包含点云数据和半径的字典
    """
    noise_gen = NoiseGenerator(seed=seed)
    
    # 在半圆柱面上均匀采样
    theta = np.linspace(0, np.pi, n_theta)
    z = np.linspace(0, L_TUNNEL, n_z)
    
    theta_grid, z_grid = np.meshgrid(theta, z)
    theta_flat = theta_grid.flatten()
    z_flat = z_grid.flatten()
    
    n_points = len(theta_flat)
    points = np.zeros((n_points, 3))
    radius = np.zeros(n_points)  # 存储每个点的半径（岩面高度）
    
    for i in range(n_points):
        theta_i = theta_flat[i]
        z_i = z_flat[i]
        
        u = R_BASE * theta_i
        v = z_i
        
        noise_val = noise_gen.layered_noise2d(
            u, v,
            NOISE_OCTAVES,
            NOISE_PERSISTENCE,
            NOISE_SCALE
        )
        
        r_rock = R_BASE + noise_val * NOISE_AMPLITUDE
        
        x = r_rock * np.cos(theta_i)
        y = r_rock * np.sin(theta_i)
        z_coord = z_i
        
        points[i] = [x, y, z_coord]
        radius[i] = r_rock  # 记录半径
    
    return {
        'noise_gen': noise_gen,
        'points': points,
        'theta': theta_flat,
        'z': z_flat,
        'radius': radius,  # 新增：半径数组
        'R_BASE': R_BASE,
        'L_TUNNEL': L_TUNNEL,
        'NOISE_AMPLITUDE': NOISE_AMPLITUDE
    }
