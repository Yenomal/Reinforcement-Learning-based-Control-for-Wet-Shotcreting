#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tunnel rock-wall environment utilities.

This module only provides the environment geometry itself:
- tunnel surface generation
- surface queries in the parameter space
- finite-difference normals
- gravity-slump compensation
- nozzle target queries

It does not include any planner.
"""

from typing import Dict, Optional, Tuple

import numpy as np

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
R_BASE = 4.0
L_TUNNEL = 10.0
D_SPRAY = 1.5

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
    d_spray: float = D_SPRAY,
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
    nozzle_point = compensated_point - normal * d_spray

    return {
        "u": float(u),
        "v": float(v),
        "raw_point": raw_point,
        "normal": normal,
        "slump_vector": slump_vector,
        "compensated_point": compensated_point,
        "nozzle_point": nozzle_point,
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
        "D_SPRAY": D_SPRAY,
        "NOISE_AMPLITUDE": NOISE_AMPLITUDE,
    }


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
    "D_SPRAY",
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
    "generate_dense_rock_wall",
    "generate_rock_environment",
    "generate_rock_wall",
    "gravity_slump_vector",
    "map_to_3d_cylinder",
    "query_surface_state",
    "surface_normal",
    "surface_point",
    "surface_radius",
]
