from __future__ import annotations

from pathlib import Path

import bpy
from mathutils import Vector


NAMES = [
    "3d66-Editable_Poly-23002546-055",
    "3d66-Editable_Poly-23002546-068",
    "3d66-Editable_Poly-23002546-056",
    "3d66-Editable_Poly-23002546-069",
    "3d66-Editable_Poly-23002546-050",
    "3d66-Editable_Poly-23002546-052",
]

ELBOW = Vector((422.577606, -2054.056396, 28.610777))

COLORS = {
    "3d66-Editable_Poly-23002546-055": (0.95, 0.30, 0.30, 1.0),
    "3d66-Editable_Poly-23002546-068": (1.00, 0.75, 0.20, 1.0),
    "3d66-Editable_Poly-23002546-056": (0.30, 0.85, 0.40, 1.0),
    "3d66-Editable_Poly-23002546-069": (0.25, 0.70, 1.00, 1.0),
    "3d66-Editable_Poly-23002546-050": (0.85, 0.85, 0.85, 1.0),
    "3d66-Editable_Poly-23002546-052": (0.85, 0.85, 0.85, 1.0),
}


def main() -> None:
    print("ELBOW_BEGIN")
    objs = []
    for name in NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        objs.append(obj)
        bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        mins = [min(p[i] for p in bb) for i in range(3)]
        maxs = [max(p[i] for p in bb) for i in range(3)]
        center = sum(bb, Vector((0.0, 0.0, 0.0))) / 8.0
        print(name)
        print(" loc", tuple(round(v, 4) for v in obj.matrix_world.to_translation()))
        print(" dims", tuple(round(v, 4) for v in obj.dimensions))
        print(" aabb_min", tuple(round(v, 4) for v in mins))
        print(" aabb_max", tuple(round(v, 4) for v in maxs))
        print(" center_to_elbow", tuple(round(v, 4) for v in (center - ELBOW)))
    print("ELBOW_END")

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1000
    scene.world.color = (0.02, 0.02, 0.02)

    for obj in bpy.data.objects:
        obj.hide_render = True

    mins = [1e18, 1e18, 1e18]
    maxs = [-1e18, -1e18, -1e18]
    for obj in objs:
        obj.hide_render = False
        for v in obj.bound_box:
            w = obj.matrix_world @ Vector(v)
            for i in range(3):
                mins[i] = min(mins[i], w[i])
                maxs[i] = max(maxs[i], w[i])

    center = Vector(
        (
            (mins[0] + maxs[0]) / 2,
            (mins[1] + maxs[1]) / 2,
            (mins[2] + maxs[2]) / 2,
        )
    )
    size = max(maxs[i] - mins[i] for i in range(3))

    camera = bpy.data.cameras.new("elbow_assignments_camera")
    camera.type = "ORTHO"
    camera.ortho_scale = size * 1.4
    camera.clip_start = 0.01
    camera.clip_end = size * 10.0
    camera_obj = bpy.data.objects.new("elbow_assignments_camera", camera)
    scene.collection.objects.link(camera_obj)
    camera_obj.location = center + Vector((-size * 2.0, 0.0, 0.0))
    direction = (center - camera_obj.location).normalized()
    camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera_obj

    for obj in objs:
        material = bpy.data.materials.new(f"elbow_{obj.name[-6:]}")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in list(nodes):
            nodes.remove(node)
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = COLORS[obj.name]
        emission.inputs["Strength"].default_value = 2.0
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(material)

    out_path = Path("/tmp/elbow_parts_check.png")
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    print(f"rendered={out_path}")


if __name__ == "__main__":
    main()
