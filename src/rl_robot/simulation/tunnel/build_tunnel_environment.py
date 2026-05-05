from __future__ import annotations

import argparse
import base64
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from ...utils.resources import asset_path

DEFAULT_OUTPUT_DIR = Path("outputs/tunnel_environment")

# Design parameters requested by the user.
OUTER_WIDTH = 10.0
OUTER_HEIGHT = 5.0
EXCAVATED_LENGTH = 3.0
TOTAL_LENGTH = 5.0


@dataclass(frozen=True)
class SurfaceGrid:
    rows: int
    cols: int
    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    surfacecolor: tuple[float, ...]


@dataclass
class MeshBuilder:
    name: str
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]

    def __init__(self, name: str) -> None:
        self.name = name
        self.vertices = []
        self.faces = []

    def add_vertex(self, vertex: tuple[float, float, float]) -> int:
        self.vertices.append(vertex)
        return len(self.vertices)

    def add_triangle(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        *,
        double_sided: bool = True,
    ) -> None:
        ia = self.add_vertex(a)
        ib = self.add_vertex(b)
        ic = self.add_vertex(c)
        self.faces.append((ia, ib, ic))
        if double_sided:
            self.faces.append((ic, ib, ia))

    def add_quad(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
        c: tuple[float, float, float],
        d: tuple[float, float, float],
        *,
        double_sided: bool = True,
    ) -> None:
        self.add_triangle(a, b, c, double_sided=double_sided)
        self.add_triangle(a, c, d, double_sided=double_sided)

    def add_polygon_fan(
        self,
        boundary: list[tuple[float, float, float]],
        center: tuple[float, float, float],
        *,
        double_sided: bool = True,
    ) -> None:
        if len(boundary) < 3:
            return
        for idx in range(len(boundary)):
            p0 = boundary[idx]
            p1 = boundary[(idx + 1) % len(boundary)]
            self.add_triangle(center, p0, p1, double_sided=double_sided)

    def bbox(self) -> tuple[list[float], list[float]]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]

    def metadata(self) -> dict[str, object]:
        bbox_min, bbox_max = self.bbox()
        return {
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }

    def write_obj(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"o {self.name}"]
        for vx, vy, vz in self.vertices:
            lines.append(f"v {vx:.8f} {vy:.8f} {vz:.8f}")
        for a, b, c in self.faces:
            lines.append(f"f {a} {b} {c}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_plotly_args(text: str) -> list[str]:
    needle = "Plotly.newPlot("
    start = text.rfind(needle)
    if start < 0:
        raise RuntimeError("Could not find Plotly.newPlot in rock_environment.html")
    start = text.index("(", start) + 1

    def extract_value(src: str, i: int) -> tuple[str, int]:
        while i < len(src) and src[i].isspace():
            i += 1
        if src[i] in "[{":
            open_ch = src[i]
            close_ch = "]" if open_ch == "[" else "}"
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < len(src):
                ch = src[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            return src[i : j + 1], j + 1
                j += 1
            raise RuntimeError("Unterminated Plotly JSON value")

        if src[i] == '"':
            j = i + 1
            escape = False
            while j < len(src):
                ch = src[j]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    return src[i : j + 1], j + 1
                j += 1
            raise RuntimeError("Unterminated Plotly string value")

        j = i
        while j < len(src) and src[j] not in ",)":
            j += 1
        return src[i:j], j

    args: list[str] = []
    i = start
    for _ in range(4):
        value, i = extract_value(text, i)
        args.append(value)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == ",":
            i += 1
        else:
            break
    return args


def decode_plotly_ndarray(payload: dict[str, str]) -> tuple[int, int, tuple[float, ...]]:
    rows, cols = (int(part.strip()) for part in payload["shape"].split(","))
    if payload["dtype"] != "f8":
        raise RuntimeError(f"Unsupported Plotly dtype: {payload['dtype']}")
    raw = base64.b64decode(payload["bdata"])
    values = struct.unpack("<" + "d" * (len(raw) // 8), raw)
    if len(values) != rows * cols:
        raise RuntimeError("Decoded array length does not match Plotly shape")
    return rows, cols, values


def load_surface_grid(html_path: Path) -> SurfaceGrid:
    text = html_path.read_text(encoding="utf-8")
    args = extract_plotly_args(text)
    data = json.loads(args[1])
    if not data:
        raise RuntimeError("Plotly data array is empty")
    trace = data[0]
    if trace.get("type") != "surface":
        raise RuntimeError(f"Expected surface trace, got {trace.get('type')}")

    x_rows, x_cols, x_vals = decode_plotly_ndarray(trace["x"])
    y_rows, y_cols, y_vals = decode_plotly_ndarray(trace["y"])
    z_rows, z_cols, z_vals = decode_plotly_ndarray(trace["z"])
    c_rows, c_cols, c_vals = decode_plotly_ndarray(trace["surfacecolor"])
    if len({(x_rows, x_cols), (y_rows, y_cols), (z_rows, z_cols), (c_rows, c_cols)}) != 1:
        raise RuntimeError("Surface arrays do not share the same shape")

    return SurfaceGrid(
        rows=x_rows,
        cols=x_cols,
        x=x_vals,
        y=y_vals,
        z=z_vals,
        surfacecolor=c_vals,
    )


def plotly_to_pybullet(x_val: float, y_val: float, z_val: float) -> tuple[float, float, float]:
    return (z_val, x_val, y_val)


def build_scaled_wall(grid: SurfaceGrid) -> tuple[MeshBuilder, list[list[tuple[float, float, float]]], float]:
    raw_vertices = [plotly_to_pybullet(grid.x[i], grid.y[i], grid.z[i]) for i in range(grid.rows * grid.cols)]
    raw_xs = [v[0] for v in raw_vertices]
    min_x = min(raw_xs)
    max_x = max(raw_xs)
    raw_length = max_x - min_x
    scale_x = EXCAVATED_LENGTH / raw_length

    mesh = MeshBuilder("tunnel_wall")
    rows: list[list[tuple[float, float, float]]] = []
    for row in range(grid.rows):
        row_points = []
        for col in range(grid.cols):
            x, y, z = raw_vertices[row * grid.cols + col]
            scaled = ((x - min_x) * scale_x, y, z)
            row_points.append(scaled)
            mesh.add_vertex(scaled)
        rows.append(row_points)

    for row in range(grid.rows - 1):
        for col in range(grid.cols - 1):
            v00 = row * grid.cols + col + 1
            v01 = v00 + 1
            v10 = (row + 1) * grid.cols + col + 1
            v11 = v10 + 1
            mesh.faces.append((v00, v10, v11))
            mesh.faces.append((v00, v11, v01))
            mesh.faces.append((v11, v10, v00))
            mesh.faces.append((v01, v11, v00))

    return mesh, rows, scale_x


def build_shell(rows: list[list[tuple[float, float, float]]]) -> MeshBuilder:
    front = sorted(rows[0], key=lambda p: p[1])
    back = sorted(rows[-1], key=lambda p: p[1])

    wall_bbox_y = [p[1] for row in rows for p in row]
    wall_center_y = 0.5 * (min(wall_bbox_y) + max(wall_bbox_y))
    outer_y_min = wall_center_y - OUTER_WIDTH * 0.5
    outer_y_max = wall_center_y + OUTER_WIDTH * 0.5
    outer_z_min = 0.0
    outer_z_max = OUTER_HEIGHT

    x_front = 0.0
    x_exc = EXCAVATED_LENGTH
    x_back = TOTAL_LENGTH

    shell = MeshBuilder("tunnel_shell")

    # Outer box skin.
    shell.add_quad(
        (x_front, outer_y_min, outer_z_max),
        (x_back, outer_y_min, outer_z_max),
        (x_back, outer_y_max, outer_z_max),
        (x_front, outer_y_max, outer_z_max),
    )
    shell.add_quad(
        (x_front, outer_y_min, outer_z_min),
        (x_back, outer_y_min, outer_z_min),
        (x_back, outer_y_min, outer_z_max),
        (x_front, outer_y_min, outer_z_max),
    )
    shell.add_quad(
        (x_front, outer_y_max, outer_z_min),
        (x_back, outer_y_max, outer_z_min),
        (x_back, outer_y_max, outer_z_max),
        (x_front, outer_y_max, outer_z_max),
    )
    shell.add_quad(
        (x_front, outer_y_min, outer_z_min),
        (x_back, outer_y_min, outer_z_min),
        (x_back, outer_y_max, outer_z_min),
        (x_front, outer_y_max, outer_z_min),
    )
    shell.add_quad(
        (x_back, outer_y_min, outer_z_min),
        (x_back, outer_y_min, outer_z_max),
        (x_back, outer_y_max, outer_z_max),
        (x_back, outer_y_max, outer_z_min),
    )

    # Portal frame at x = 0: top strips from the rough wall to the rectangle roof.
    for p0, p1 in zip(front[:-1], front[1:]):
        shell.add_quad(
            (x_front, p0[1], outer_z_max),
            (x_front, p1[1], outer_z_max),
            p1,
            p0,
        )

    left_foot = front[0]
    right_foot = front[-1]
    shell.add_quad(
        (x_front, outer_y_min, outer_z_min),
        (x_front, outer_y_min, outer_z_max),
        (x_front, left_foot[1], outer_z_max),
        (x_front, left_foot[1], outer_z_min),
    )
    shell.add_quad(
        (x_front, right_foot[1], outer_z_min),
        (x_front, right_foot[1], outer_z_max),
        (x_front, outer_y_max, outer_z_max),
        (x_front, outer_y_max, outer_z_min),
    )

    # Excavation face at x = EXCAVATED_LENGTH.
    cap_center = (x_exc, wall_center_y, max(0.8, min(OUTER_HEIGHT - 0.8, 0.5 * max(p[2] for p in back))))
    shell.add_polygon_fan(back, cap_center)

    return shell


def combine_bbox(*metas: dict[str, object]) -> dict[str, object]:
    mins = [1e18, 1e18, 1e18]
    maxs = [-1e18, -1e18, -1e18]
    vertex_count = 0
    face_count = 0
    for meta in metas:
        bbox_min = meta["bbox_min"]
        bbox_max = meta["bbox_max"]
        vertex_count += int(meta["vertex_count"])
        face_count += int(meta["face_count"])
        for i in range(3):
            mins[i] = min(mins[i], bbox_min[i])
            maxs[i] = max(maxs[i], bbox_max[i])
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bbox_min": mins,
        "bbox_max": maxs,
    }


def write_urdf(urdf_path: Path) -> None:
    urdf = """<?xml version="1.0"?>
<robot name="tunnel_environment">
  <material name="rock_wall">
    <color rgba="0.63 0.56 0.48 1.0"/>
  </material>
  <material name="tunnel_shell">
    <color rgba="0.55 0.50 0.46 1.0"/>
  </material>
  <link name="tunnel_environment">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/tunnel_shell.obj"/>
      </geometry>
      <material name="tunnel_shell"/>
    </visual>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="meshes/tunnel_wall.obj"/>
      </geometry>
      <material name="rock_wall"/>
    </visual>
  </link>
</robot>
"""
    urdf_path.write_text(urdf, encoding="utf-8")


def write_metadata(
    metadata_path: Path,
    grid: SurfaceGrid,
    wall_meta: dict[str, object],
    shell_meta: dict[str, object],
) -> None:
    metadata = {
        "source_html": "asset:html/rock_environment.html",
        "plotly_shape": [grid.rows, grid.cols],
        "plotly_bbox": {
            "x": [min(grid.x), max(grid.x)],
            "y": [min(grid.y), max(grid.y)],
            "z": [min(grid.z), max(grid.z)],
        },
        "pybullet_axis_mapping": {
            "x": "plotly_z",
            "y": "plotly_x",
            "z": "plotly_y",
        },
        "design": {
            "outer_width": OUTER_WIDTH,
            "outer_height": OUTER_HEIGHT,
            "excavated_length": EXCAVATED_LENGTH,
            "total_length": TOTAL_LENGTH,
            "solid_length": TOTAL_LENGTH - EXCAVATED_LENGTH,
        },
        "wall_mesh": wall_meta,
        "shell_mesh": shell_meta,
        "mesh": combine_bbox(wall_meta, shell_meta),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tunnel environment meshes and URDF from the packaged HTML surface.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated tunnel assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if TOTAL_LENGTH <= EXCAVATED_LENGTH:
        raise SystemExit("TOTAL_LENGTH must be greater than EXCAVATED_LENGTH")

    output_dir = args.output_dir.resolve()
    mesh_dir = output_dir / "meshes"
    wall_mesh_path = mesh_dir / "tunnel_wall.obj"
    shell_mesh_path = mesh_dir / "tunnel_shell.obj"
    urdf_path = output_dir / "tunnel_environment.urdf"
    metadata_path = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    with asset_path("html/rock_environment.html") as html_path:
        grid = load_surface_grid(html_path)
    wall_mesh, rows, scale_x = build_scaled_wall(grid)
    shell_mesh = build_shell(rows)

    wall_mesh.write_obj(wall_mesh_path)
    shell_mesh.write_obj(shell_mesh_path)
    write_urdf(urdf_path)
    write_metadata(metadata_path, grid, wall_mesh.metadata(), shell_mesh.metadata())

    print(f"[done] tunnel wall written to {wall_mesh_path}")
    print(f"[done] tunnel shell written to {shell_mesh_path}")
    print(f"[done] urdf written to {urdf_path}")
    print(f"[done] metadata written to {metadata_path}")
    print(
        "[summary] "
        f"rows={grid.rows} cols={grid.cols} "
        f"wall_vertices={wall_mesh.metadata()['vertex_count']} "
        f"wall_faces={wall_mesh.metadata()['face_count']} "
        f"shell_vertices={shell_mesh.metadata()['vertex_count']} "
        f"shell_faces={shell_mesh.metadata()['face_count']} "
        f"scale_x={scale_x:.4f}"
    )


if __name__ == "__main__":
    main()
