#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tunnel rock-wall environment utilities.

This module only provides the environment geometry itself:
- tunnel surface generation
- surface queries in the parameter space
- finite-difference normals
- gravity-slump compensation

It does not include any planner.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ...utils.resources import asset_path
from .build_tunnel_environment import load_surface_grid

try:
    from opensimplex import OpenSimplex

    NOISE_LIBRARY = "opensimplex"
except ImportError:
    try:
        import vnoise

        NOISE_LIBRARY = "vnoise"
    except ImportError:
        NOISE_LIBRARY = "fallback"


# Tunnel geometry
R_BASE = 3.5
L_TUNNEL = 2.0

# Surface parameter bounds
U_MIN = 0.0
U_MAX = np.pi * R_BASE
V_MIN = 0.0
V_MAX = L_TUNNEL

# Noise settings
NOISE_SCALE = 0.5
NOISE_OCTAVES = 4
NOISE_PERSISTENCE = 0.5
NOISE_AMPLITUDE = 0.3

# Gravity-slump approximation
K_SLUMP = 0.15

# Sampling settings
N_THETA = 200
N_Z = 100
DELTA = 0.01
NOISE_SEED = 42
LEGACY_TRAIN_HTML_PATHS = {
    "src/rock_3D/rock_environment.html",
    "./src/rock_3D/rock_environment.html",
}
PACKAGED_TRAIN_HTML_IDENTIFIER = "asset:html/rock_environment.html"

class NoiseGenerator:
    """2D fractal noise wrapper for the tunnel wall."""

    def __init__(self, seed: int = NOISE_SEED):
        self.seed = seed
        if NOISE_LIBRARY == "opensimplex":
            self.sampler = OpenSimplex(seed)
        elif NOISE_LIBRARY == "vnoise":
            self.sampler = vnoise.Noise()
            self.sampler.seed = seed
        else:
            self.sampler = None

    def noise2d(self, x: float, y: float) -> float:
        if NOISE_LIBRARY in {"opensimplex", "vnoise"}:
            return self.sampler.noise2(x, y)
        return self._fallback_noise2d(x, y)

    def _fallback_noise2d(self, x: float, y: float) -> float:
        x0 = np.floor(x)
        y0 = np.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        sx = self._smoothstep(x - x0)
        sy = self._smoothstep(y - y0)

        n00 = self._hash_grid_value(x0, y0)
        n10 = self._hash_grid_value(x1, y0)
        n01 = self._hash_grid_value(x0, y1)
        n11 = self._hash_grid_value(x1, y1)

        nx0 = self._lerp(n00, n10, sx)
        nx1 = self._lerp(n01, n11, sx)
        return float(self._lerp(nx0, nx1, sy))

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _lerp(start: float, end: float, weight: float) -> float:
        return start + weight * (end - start)

    def _hash_grid_value(self, x: float, y: float) -> float:
        value = np.sin(
            x * 127.1 + y * 311.7 + self.seed * 74.7
        ) * 43758.5453123
        return float(2.0 * (value - np.floor(value)) - 1.0)

    def layered_noise2d(
        self,
        x: float,
        y: float,
        octaves: int = NOISE_OCTAVES,
        persistence: float = NOISE_PERSISTENCE,
        scale: float = NOISE_SCALE,
    ) -> float:
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


def _ensure_noise_gen(
    noise_gen: Optional[NoiseGenerator], seed: int = NOISE_SEED
) -> NoiseGenerator:
    if noise_gen is None:
        return NoiseGenerator(seed=seed)
    return noise_gen


def clamp_uv(u: float, v: float) -> Tuple[float, float]:
    """Clamp a query to the valid tunnel parameter range."""
    u_clamped = float(np.clip(u, U_MIN, U_MAX))
    v_clamped = float(np.clip(v, V_MIN, V_MAX))
    return u_clamped, v_clamped


def _uv_to_grid_coordinates(
    u: float,
    v: float,
    *,
    rows: int,
    cols: int,
) -> tuple[float, float]:
    u, v = clamp_uv(u, v)
    row = 0.0 if rows <= 1 else (v - V_MIN) / max(V_MAX - V_MIN, 1e-8) * (rows - 1)
    col = 0.0 if cols <= 1 else (u - U_MIN) / max(U_MAX - U_MIN, 1e-8) * (cols - 1)
    return float(row), float(col)


def _bilinear_sample(points_grid: np.ndarray, row: float, col: float) -> np.ndarray:
    rows, cols = points_grid.shape[:2]
    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row1 = min(row0 + 1, rows - 1)
    col1 = min(col0 + 1, cols - 1)
    row_weight = float(row - row0)
    col_weight = float(col - col0)

    p00 = points_grid[row0, col0]
    p01 = points_grid[row0, col1]
    p10 = points_grid[row1, col0]
    p11 = points_grid[row1, col1]

    top = (1.0 - col_weight) * p00 + col_weight * p01
    bottom = (1.0 - col_weight) * p10 + col_weight * p11
    return ((1.0 - row_weight) * top + row_weight * bottom).astype(np.float32)


def surface_radius(
    u: float,
    v: float,
    noise_gen: Optional[NoiseGenerator] = None,
    r_base: float = R_BASE,
    noise_scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    amplitude: float = NOISE_AMPLITUDE,
) -> float:
    """Return the noisy tunnel radius at parameter position (u, v)."""
    u, v = clamp_uv(u, v)
    noise_gen = _ensure_noise_gen(noise_gen)

    noise_val = noise_gen.layered_noise2d(
        u, v, octaves=octaves, persistence=persistence, scale=noise_scale
    )
    return float(r_base + noise_val * amplitude)


def surface_point(
    u: float,
    v: float,
    noise_gen: Optional[NoiseGenerator] = None,
    r_base: float = R_BASE,
    noise_scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    amplitude: float = NOISE_AMPLITUDE,
) -> np.ndarray:
    """Map a single parameter-space position to the 3D tunnel wall."""
    u, v = clamp_uv(u, v)
    radius = surface_radius(
        u,
        v,
        noise_gen=noise_gen,
        r_base=r_base,
        noise_scale=noise_scale,
        octaves=octaves,
        persistence=persistence,
        amplitude=amplitude,
    )
    theta = u / r_base

    return np.array(
        [radius * np.cos(theta), radius * np.sin(theta), v],
        dtype=float,
    )


def surface_point_from_environment(
    rock_env: Dict[str, object],
    u: float,
    v: float,
) -> np.ndarray:
    """Query one surface point from either the analytic wall or a fixed HTML grid."""
    if "noise_gen" in rock_env:
        return surface_point(u, v, noise_gen=rock_env["noise_gen"])

    points_grid = np.asarray(rock_env["points_grid"], dtype=np.float32)
    row, col = _uv_to_grid_coordinates(
        u,
        v,
        rows=int(points_grid.shape[0]),
        cols=int(points_grid.shape[1]),
    )
    return _bilinear_sample(points_grid, row, col)


def map_to_3d_cylinder(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    noise_gen: Optional[NoiseGenerator] = None,
    r_base: float = R_BASE,
    noise_scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    amplitude: float = NOISE_AMPLITUDE,
) -> np.ndarray:
    """Map arrays of parameter-space samples onto the 3D tunnel wall."""
    u_coords = np.asarray(u_coords, dtype=float)
    v_coords = np.asarray(v_coords, dtype=float)

    if u_coords.shape != v_coords.shape:
        raise ValueError("u_coords and v_coords must have the same shape.")

    noise_gen = _ensure_noise_gen(noise_gen)
    points = np.zeros((u_coords.size, 3), dtype=float)

    for idx, (u, v) in enumerate(zip(u_coords.reshape(-1), v_coords.reshape(-1))):
        points[idx] = surface_point(
            float(u),
            float(v),
            noise_gen=noise_gen,
            r_base=r_base,
            noise_scale=noise_scale,
            octaves=octaves,
            persistence=persistence,
            amplitude=amplitude,
        )

    return points


def surface_normal(
    u: float,
    v: float,
    noise_gen: Optional[NoiseGenerator] = None,
    delta: float = DELTA,
    r_base: float = R_BASE,
    noise_scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    amplitude: float = NOISE_AMPLITUDE,
) -> np.ndarray:
    """Estimate the local surface normal by finite differences."""
    u, v = clamp_uv(u, v)
    noise_gen = _ensure_noise_gen(noise_gen)

    u_minus, _ = clamp_uv(u - delta, v)
    u_plus, _ = clamp_uv(u + delta, v)
    _, v_minus = clamp_uv(u, v - delta)
    _, v_plus = clamp_uv(u, v + delta)

    point_u_minus = surface_point(
        u_minus,
        v,
        noise_gen=noise_gen,
        r_base=r_base,
        noise_scale=noise_scale,
        octaves=octaves,
        persistence=persistence,
        amplitude=amplitude,
    )
    point_u_plus = surface_point(
        u_plus,
        v,
        noise_gen=noise_gen,
        r_base=r_base,
        noise_scale=noise_scale,
        octaves=octaves,
        persistence=persistence,
        amplitude=amplitude,
    )
    point_v_minus = surface_point(
        u,
        v_minus,
        noise_gen=noise_gen,
        r_base=r_base,
        noise_scale=noise_scale,
        octaves=octaves,
        persistence=persistence,
        amplitude=amplitude,
    )
    point_v_plus = surface_point(
        u,
        v_plus,
        noise_gen=noise_gen,
        r_base=r_base,
        noise_scale=noise_scale,
        octaves=octaves,
        persistence=persistence,
        amplitude=amplitude,
    )

    du = max(u_plus - u_minus, 1e-8)
    dv = max(v_plus - v_minus, 1e-8)
    tangent_u = (point_u_plus - point_u_minus) / du
    tangent_v = (point_v_plus - point_v_minus) / dv

    normal = np.cross(tangent_u, tangent_v)
    normal_norm = np.linalg.norm(normal)

    theta = u / r_base
    radial = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    if normal_norm <= 1e-10:
        return radial

    normal = normal / normal_norm
    if np.dot(normal, radial) < 0:
        normal = -normal
    return normal


def surface_normal_from_environment(
    rock_env: Dict[str, object],
    u: float,
    v: float,
    delta: float = DELTA,
) -> np.ndarray:
    """Estimate one surface normal from either the analytic wall or a fixed HTML grid."""
    if "noise_gen" in rock_env:
        return surface_normal(u, v, noise_gen=rock_env["noise_gen"], delta=delta)

    u, v = clamp_uv(u, v)
    u_minus, _ = clamp_uv(u - delta, v)
    u_plus, _ = clamp_uv(u + delta, v)
    _, v_minus = clamp_uv(u, v - delta)
    _, v_plus = clamp_uv(u, v + delta)

    point_u_minus = surface_point_from_environment(rock_env, u_minus, v)
    point_u_plus = surface_point_from_environment(rock_env, u_plus, v)
    point_v_minus = surface_point_from_environment(rock_env, u, v_minus)
    point_v_plus = surface_point_from_environment(rock_env, u, v_plus)

    du = max(u_plus - u_minus, 1e-8)
    dv = max(v_plus - v_minus, 1e-8)
    tangent_u = (point_u_plus - point_u_minus) / du
    tangent_v = (point_v_plus - point_v_minus) / dv

    normal = np.cross(tangent_u, tangent_v)
    normal_norm = np.linalg.norm(normal)

    theta = u / R_BASE
    radial = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    if normal_norm <= 1e-10:
        return radial.astype(np.float32)

    normal = normal / normal_norm
    if np.dot(normal, radial) < 0:
        normal = -normal
    return normal.astype(np.float32)


def compute_surface_normals(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    noise_gen: Optional[NoiseGenerator] = None,
    delta: float = DELTA,
    r_base: float = R_BASE,
    noise_scale: float = NOISE_SCALE,
    octaves: int = NOISE_OCTAVES,
    persistence: float = NOISE_PERSISTENCE,
    amplitude: float = NOISE_AMPLITUDE,
) -> np.ndarray:
    """Estimate normals for multiple parameter-space samples."""
    u_coords = np.asarray(u_coords, dtype=float).reshape(-1)
    v_coords = np.asarray(v_coords, dtype=float).reshape(-1)

    if u_coords.shape != v_coords.shape:
        raise ValueError("u_coords and v_coords must have the same shape.")

    noise_gen = _ensure_noise_gen(noise_gen)
    normals = np.zeros((u_coords.size, 3), dtype=float)

    for idx, (u, v) in enumerate(zip(u_coords, v_coords)):
        normals[idx] = surface_normal(
            float(u),
            float(v),
            noise_gen=noise_gen,
            delta=delta,
            r_base=r_base,
            noise_scale=noise_scale,
            octaves=octaves,
            persistence=persistence,
            amplitude=amplitude,
        )

    return normals


def gravity_slump_vector(
    normal: np.ndarray, k_slump: float = K_SLUMP
) -> np.ndarray:
    """Approximate gravity-induced material slump on the wall."""
    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-10:
        return np.zeros(3, dtype=float)

    normal = normal / normal_norm
    gravity = np.array([0.0, -1.0, 0.0], dtype=float)

    gravity_cross_normal = np.cross(gravity, normal)
    slump_direction = np.cross(normal, gravity_cross_normal)
    direction_norm = np.linalg.norm(slump_direction)

    if direction_norm <= 1e-10:
        return np.zeros(3, dtype=float)

    slump_direction = slump_direction / direction_norm
    steepness = float(np.sqrt(normal[0] ** 2 + normal[2] ** 2))
    slump_magnitude = k_slump * steepness

    return slump_direction * slump_magnitude


def query_surface_state(
    u: float,
    v: float,
    noise_gen: Optional[NoiseGenerator] = None,
    k_slump: float = K_SLUMP,
    delta: float = DELTA,
) -> Dict[str, np.ndarray]:
    """Query a full local wall state at one parameter-space position."""
    noise_gen = _ensure_noise_gen(noise_gen)
    u, v = clamp_uv(u, v)

    raw_point = surface_point(u, v, noise_gen=noise_gen)
    normal = surface_normal(u, v, noise_gen=noise_gen, delta=delta)
    slump_vector = gravity_slump_vector(normal, k_slump=k_slump)
    compensated_point = raw_point + slump_vector

    return {
        "u": float(u),
        "v": float(v),
        "raw_point": raw_point,
        "normal": normal,
        "slump_vector": slump_vector,
        "compensated_point": compensated_point,
    }


def generate_rock_environment(
    n_theta: int = N_THETA,
    n_z: int = N_Z,
    seed: int = NOISE_SEED,
) -> Dict[str, object]:
    """Generate a dense rock-wall environment for visualization or querying."""
    noise_gen = NoiseGenerator(seed=seed)

    theta = np.linspace(0.0, np.pi, n_theta)
    v_values = np.linspace(V_MIN, V_MAX, n_z)
    theta_grid, v_grid = np.meshgrid(theta, v_values)
    u_grid = R_BASE * theta_grid

    u_flat = u_grid.reshape(-1)
    v_flat = v_grid.reshape(-1)
    points = np.zeros((u_flat.size, 3), dtype=float)
    radius = np.zeros(u_flat.size, dtype=float)

    for idx, (u, v) in enumerate(zip(u_flat, v_flat)):
        point = surface_point(float(u), float(v), noise_gen=noise_gen)
        points[idx] = point
        radius[idx] = np.linalg.norm(point[:2])

    points_grid = points.reshape(n_z, n_theta, 3)
    radius_grid = radius.reshape(n_z, n_theta)

    return {
        "noise_gen": noise_gen,
        "seed": seed,
        "points": points,
        "points_grid": points_grid,
        "radius": radius,
        "radius_grid": radius_grid,
        "theta": theta_grid.reshape(-1),
        "u": u_flat,
        "v": v_flat,
        "theta_grid": theta_grid,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "u_bounds": (U_MIN, U_MAX),
        "v_bounds": (V_MIN, V_MAX),
        "R_BASE": R_BASE,
        "L_TUNNEL": L_TUNNEL,
        "NOISE_AMPLITUDE": NOISE_AMPLITUDE,
    }


def load_rock_environment_from_html(html_path: str | Path) -> Dict[str, object]:
    """Load one fixed rock wall from a Plotly surface HTML exported by this project."""
    path = Path(html_path)
    grid = load_surface_grid(path)

    x_grid = np.asarray(grid.x, dtype=np.float32).reshape(grid.rows, grid.cols)
    y_grid = np.asarray(grid.y, dtype=np.float32).reshape(grid.rows, grid.cols)
    z_grid = np.asarray(grid.z, dtype=np.float32).reshape(grid.rows, grid.cols)
    radius_grid = np.asarray(grid.surfacecolor, dtype=np.float32).reshape(grid.rows, grid.cols)
    points_grid = np.stack([x_grid, y_grid, z_grid], axis=-1).astype(np.float32)

    theta = np.linspace(0.0, np.pi, grid.cols, dtype=np.float32)
    v_values = np.linspace(V_MIN, V_MAX, grid.rows, dtype=np.float32)
    theta_grid, v_grid = np.meshgrid(theta, v_values)
    u_grid = R_BASE * theta_grid

    return {
        "source": "html",
        "html_path": str(path),
        "seed": None,
        "points": points_grid.reshape(-1, 3).astype(np.float32),
        "points_grid": points_grid,
        "radius": radius_grid.reshape(-1).astype(np.float32),
        "radius_grid": radius_grid,
        "theta": theta_grid.reshape(-1),
        "u": u_grid.reshape(-1),
        "v": v_grid.reshape(-1),
        "theta_grid": theta_grid,
        "u_grid": u_grid,
        "v_grid": v_grid,
        "u_bounds": (U_MIN, U_MAX),
        "v_bounds": (V_MIN, V_MAX),
        "R_BASE": R_BASE,
        "L_TUNNEL": L_TUNNEL,
        "NOISE_AMPLITUDE": None,
    }


def build_training_rock_environment(env_cfg: Dict[str, object]) -> Dict[str, object]:
    """Build the training wall from a fixed HTML path or from the procedural generator."""
    html_path_raw = str(env_cfg.get("train_rock_env_html", "")).strip()
    if html_path_raw:
        candidate = Path(html_path_raw)
        normalized = candidate.as_posix()
        if normalized in LEGACY_TRAIN_HTML_PATHS or normalized == PACKAGED_TRAIN_HTML_IDENTIFIER:
            with asset_path("html/rock_environment.html") as packaged_path:
                return load_rock_environment_from_html(packaged_path)
        return load_rock_environment_from_html(candidate)

    return generate_rock_environment(
        n_theta=int(env_cfg.get("n_theta", N_THETA)),
        n_z=int(env_cfg.get("n_z", N_Z)),
        seed=int(env_cfg.get("seed", NOISE_SEED)),
    )


def generate_rock_wall(
    n_theta: int = N_THETA,
    n_z: int = N_Z,
    seed: int = NOISE_SEED,
) -> Dict[str, object]:
    """Backward-compatible alias for the pure environment generator."""
    return generate_rock_environment(n_theta=n_theta, n_z=n_z, seed=seed)


def generate_dense_rock_wall(
    n_theta: int = N_THETA,
    n_z: int = N_Z,
    seed: int = NOISE_SEED,
) -> Dict[str, object]:
    """Backward-compatible alias for dense environment generation."""
    return generate_rock_environment(n_theta=n_theta, n_z=n_z, seed=seed)


__all__ = [
    "DELTA",
    "K_SLUMP",
    "L_TUNNEL",
    "NOISE_AMPLITUDE",
    "NOISE_LIBRARY",
    "NOISE_OCTAVES",
    "NOISE_PERSISTENCE",
    "NOISE_SCALE",
    "NOISE_SEED",
    "N_THETA",
    "N_Z",
    "NoiseGenerator",
    "R_BASE",
    "U_MAX",
    "U_MIN",
    "V_MAX",
    "V_MIN",
    "clamp_uv",
    "compute_surface_normals",
    "build_training_rock_environment",
    "generate_dense_rock_wall",
    "generate_rock_environment",
    "generate_rock_wall",
    "gravity_slump_vector",
    "load_rock_environment_from_html",
    "map_to_3d_cylinder",
    "query_surface_state",
    "surface_normal",
    "surface_normal_from_environment",
    "surface_point",
    "surface_point_from_environment",
    "surface_radius",
]
