from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


ROOT_GROUP_NAME = "3d66Group004"

# The user accepted a uniform normalization based on the current end arm length.
# This preserves the original proportions while bringing the robot into a sane
# meter-scale range for PyBullet visualization.
GLOBAL_SCALE = 0.003686347088285246

# Joint centers were estimated from the user's annotated side view and then
# mapped back into Blender world coordinates.
JOINTS_WORLD = {
    "yaw": Vector((422.5776, -1788.987485324258, -129.2250460946849)),
    "shoulder": Vector((422.5776, -1821.3497415037132, -106.78730962671605)),
    # Refined onto the forearm-side endpoint near the proximal connection,
    # so the forearm rotates around its own base instead of the boom-side end.
    "elbow": Vector((422.5776, -2068.202393, 62.970108)),
    # Refined from the nearest contact region between the forearm-side connector
    # and the end-arm root, so the last link rotates around the proximal end.
    "wrist": Vector((422.5776, -2265.193359, -50.970345)),
}

LIMITS_DEG = {
    "turret_yaw": (-90.0, 90.0),
    "shoulder_pitch": (-20.0, 60.0),
    "elbow_pitch": (-100.0, 100.0),
    "wrist_pitch": (-110.0, 70.0),
}


@dataclass(frozen=True)
class LinkSpec:
    name: str
    frame_name: str
    object_names: tuple[str, ...]


LINK_SPECS = (
    LinkSpec(
        name="base_link",
        frame_name="yaw",
        object_names=(),
    ),
    LinkSpec(
        name="yaw_link",
        frame_name="yaw",
        object_names=(
            "3d66-Editable_Poly-23002546-044",
            "3d66-Editable_Poly-23002546-045",
            "3d66-Editable_Poly-23002546-046",
            "3d66-Editable_Poly-23002546-047",
            "3d66-Editable_Poly-23002546-048",
            "3d66-Editable_Poly-23002546-049",
        ),
    ),
    LinkSpec(
        name="boom1_link",
        frame_name="shoulder",
        object_names=(
            "3d66-Editable_Poly-23002546-050",
            "3d66-Editable_Poly-23002546-054",
            "3d66-Editable_Poly-23002546-067",
            "3d66-Editable_Poly-23002546-055",
            "3d66-Editable_Poly-23002546-068",
        ),
    ),
    LinkSpec(
        name="boom2_link",
        frame_name="elbow",
        object_names=(
            "3d66-Editable_Poly-23002546-051",
            "3d66-Editable_Poly-23002546-056",
            "3d66-Editable_Poly-23002546-069",
        ),
    ),
    LinkSpec(
        name="boom3_link",
        frame_name="wrist",
        object_names=(
            "3d66-Editable_Poly-23002546-052",
        ),
    ),
)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in name)


def deg_to_rad(value: float) -> float:
    return value * 3.141592653589793 / 180.0


def ensure_root_group() -> bpy.types.Object:
    root = bpy.data.objects.get(ROOT_GROUP_NAME)
    if root is None:
        raise RuntimeError(f"Missing expected root object: {ROOT_GROUP_NAME}")
    return root


def iter_root_meshes(root: bpy.types.Object) -> Iterable[bpy.types.Object]:
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent and obj.parent.name == root.name:
            yield obj


def obj_vertices_in_link_frame(
    obj: bpy.types.Object,
    link_origin_world: Vector,
) -> tuple[list[Vector], list[tuple[int, int, int]]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    if mesh is None:
        return [], []

    mesh.calc_loop_triangles()
    if not mesh.loop_triangles:
        obj_eval.to_mesh_clear()
        return [], []

    vertices: list[Vector] = []
    for vertex in mesh.vertices:
        world = obj.matrix_world @ vertex.co
        local = (world - link_origin_world) * GLOBAL_SCALE
        vertices.append(local)

    triangles = []
    for tri in mesh.loop_triangles:
        triangles.append((tri.vertices[0], tri.vertices[1], tri.vertices[2]))

    obj_eval.to_mesh_clear()
    return vertices, triangles


def write_obj(
    dst: Path,
    object_name: str,
    vertices: list[Vector],
    triangles: list[tuple[int, int, int]],
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"o {sanitize(object_name)}"]
    for vertex in vertices:
        lines.append(f"v {vertex.x:.8f} {vertex.y:.8f} {vertex.z:.8f}")
    for a, b, c in triangles:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_base_spec(root: bpy.types.Object) -> LinkSpec:
    moving_names = {
        object_name
        for link_spec in LINK_SPECS
        if link_spec.name != "base_link"
        for object_name in link_spec.object_names
    }
    base_names = tuple(
        obj.name
        for obj in iter_root_meshes(root)
        if obj.name not in moving_names
    )
    return LinkSpec(
        name="base_link",
        frame_name="yaw",
        object_names=base_names,
    )


def export_link_meshes(output_root: Path) -> dict[str, list[str]]:
    root = ensure_root_group()
    specs = [build_base_spec(root), *[spec for spec in LINK_SPECS if spec.name != "base_link"]]
    exported: dict[str, list[str]] = {}

    for spec in specs:
        link_origin = JOINTS_WORLD[spec.frame_name]
        link_dir = output_root / "meshes" / spec.name
        exported_files: list[str] = []
        for object_name in spec.object_names:
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                print(f"[warn] missing object: {object_name}")
                continue
            vertices, triangles = obj_vertices_in_link_frame(obj, link_origin)
            if not vertices or not triangles:
                print(f"[skip] empty geometry: {object_name}")
                continue
            obj_path = link_dir / f"{sanitize(object_name)}.obj"
            write_obj(obj_path, object_name, vertices, triangles)
            exported_files.append(str(obj_path.relative_to(output_root)))
            print(f"[ok] exported {spec.name}: {object_name} -> {obj_path}")
        exported[spec.name] = exported_files
    return exported


def write_urdf(output_root: Path, exported: dict[str, list[str]]) -> None:
    yaw_origin = Vector((0.0, 0.0, 0.0))
    shoulder_offset = (JOINTS_WORLD["shoulder"] - JOINTS_WORLD["yaw"]) * GLOBAL_SCALE
    elbow_offset = (JOINTS_WORLD["elbow"] - JOINTS_WORLD["shoulder"]) * GLOBAL_SCALE
    wrist_offset = (JOINTS_WORLD["wrist"] - JOINTS_WORLD["elbow"]) * GLOBAL_SCALE

    def inertial_xml(mass: float) -> str:
        return (
            f'    <inertial>\n'
            f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'      <mass value="{mass:.6f}"/>\n'
            f'      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>\n'
            f'    </inertial>\n'
        )

    def visuals_xml(link_name: str) -> str:
        blocks = []
        for rel_path in exported.get(link_name, []):
            blocks.append(
                "    <visual>\n"
                '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
                "      <geometry>\n"
                f'        <mesh filename="{rel_path}"/>\n'
                "      </geometry>\n"
                '      <material name="robot_gray">\n'
                '        <color rgba="0.85 0.85 0.85 1.0"/>\n'
                "      </material>\n"
                "    </visual>\n"
            )
        return "".join(blocks)

    urdf = [
        '<?xml version="1.0"?>\n',
        '<robot name="shipen_4dof">\n',
        '  <material name="robot_gray">\n',
        '    <color rgba="0.85 0.85 0.85 1.0"/>\n',
        '  </material>\n',
        '  <material name="robot_black">\n',
        '    <color rgba="0.12 0.12 0.12 1.0"/>\n',
        '  </material>\n',
        '  <link name="base_link">\n',
        inertial_xml(20.0),
        visuals_xml("base_link"),
        '  </link>\n',
        '  <link name="yaw_link">\n',
        inertial_xml(0.2),
        visuals_xml("yaw_link"),
        '  </link>\n',
        '  <link name="boom1_link">\n',
        inertial_xml(2.0),
        visuals_xml("boom1_link"),
        '  </link>\n',
        '  <link name="boom2_link">\n',
        inertial_xml(1.0),
        visuals_xml("boom2_link"),
        '  </link>\n',
        '  <link name="boom3_link">\n',
        inertial_xml(1.0),
        visuals_xml("boom3_link"),
        '  </link>\n',
        '  <joint name="turret_yaw" type="revolute">\n',
        '    <parent link="base_link"/>\n',
        '    <child link="yaw_link"/>\n',
        f'    <origin xyz="{yaw_origin.x:.6f} {yaw_origin.y:.6f} {yaw_origin.z:.6f}" rpy="0 0 0"/>\n',
        '    <axis xyz="0 0 1"/>\n',
        f'    <limit lower="{deg_to_rad(LIMITS_DEG["turret_yaw"][0]):.8f}" upper="{deg_to_rad(LIMITS_DEG["turret_yaw"][1]):.8f}" effort="100" velocity="2.0"/>\n',
        '    <dynamics damping="0.5" friction="0.0"/>\n',
        '  </joint>\n',
        '  <joint name="shoulder_pitch" type="revolute">\n',
        '    <parent link="yaw_link"/>\n',
        '    <child link="boom1_link"/>\n',
        f'    <origin xyz="{shoulder_offset.x:.6f} {shoulder_offset.y:.6f} {shoulder_offset.z:.6f}" rpy="0 0 0"/>\n',
        '    <axis xyz="-1 0 0"/>\n',
        f'    <limit lower="{deg_to_rad(LIMITS_DEG["shoulder_pitch"][0]):.8f}" upper="{deg_to_rad(LIMITS_DEG["shoulder_pitch"][1]):.8f}" effort="80" velocity="2.0"/>\n',
        '    <dynamics damping="0.5" friction="0.0"/>\n',
        '  </joint>\n',
        '  <joint name="elbow_pitch" type="revolute">\n',
        '    <parent link="boom1_link"/>\n',
        '    <child link="boom2_link"/>\n',
        f'    <origin xyz="{elbow_offset.x:.6f} {elbow_offset.y:.6f} {elbow_offset.z:.6f}" rpy="0 0 0"/>\n',
        '    <axis xyz="-1 0 0"/>\n',
        f'    <limit lower="{deg_to_rad(LIMITS_DEG["elbow_pitch"][0]):.8f}" upper="{deg_to_rad(LIMITS_DEG["elbow_pitch"][1]):.8f}" effort="60" velocity="2.0"/>\n',
        '    <dynamics damping="0.5" friction="0.0"/>\n',
        '  </joint>\n',
        '  <joint name="wrist_pitch" type="revolute">\n',
        '    <parent link="boom2_link"/>\n',
        '    <child link="boom3_link"/>\n',
        f'    <origin xyz="{wrist_offset.x:.6f} {wrist_offset.y:.6f} {wrist_offset.z:.6f}" rpy="0 0 0"/>\n',
        '    <axis xyz="-1 0 0"/>\n',
        f'    <limit lower="{deg_to_rad(LIMITS_DEG["wrist_pitch"][0]):.8f}" upper="{deg_to_rad(LIMITS_DEG["wrist_pitch"][1]):.8f}" effort="40" velocity="2.0"/>\n',
        '    <dynamics damping="0.5" friction="0.0"/>\n',
        '  </joint>\n',
        '</robot>\n',
    ]

    (output_root / "shipen_4dof.urdf").write_text("".join(urdf), encoding="utf-8")


def write_metadata(output_root: Path, exported: dict[str, list[str]]) -> None:
    metadata = {
        "global_scale": GLOBAL_SCALE,
        "joints_world_unscaled": {
            name: [round(value, 6) for value in vector]
            for name, vector in JOINTS_WORLD.items()
        },
        "joint_limits_deg": LIMITS_DEG,
        "exported_meshes": exported,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[1]
    output_root = workspace_root / "robot_4dof"
    output_root.mkdir(parents=True, exist_ok=True)

    exported = export_link_meshes(output_root)
    write_urdf(output_root, exported)
    write_metadata(output_root, exported)
    print(f"[done] assets written to {output_root}")


if __name__ == "__main__":
    main()
