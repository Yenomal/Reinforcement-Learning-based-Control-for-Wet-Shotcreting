from __future__ import annotations

import bpy
from mathutils import Vector


TARGETS = [
    ("3d66-Editable_Poly-23002546-051", "3d66-Editable_Poly-23002546-069"),
    ("3d66-Editable_Poly-23002546-051", "3d66-Editable_Poly-23002546-056"),
    ("3d66-Editable_Poly-23002546-052", "3d66-Editable_Poly-23002546-069"),
]


def world_vertices(obj: bpy.types.Object) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    vertices = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    obj_eval.to_mesh_clear()
    return vertices


def nearest_midpoint(a_name: str, b_name: str) -> None:
    a_obj = bpy.data.objects[a_name]
    b_obj = bpy.data.objects[b_name]
    a_vertices = world_vertices(a_obj)
    b_vertices = world_vertices(b_obj)

    best_dist_sq = float("inf")
    best_a = None
    best_b = None
    for a_vertex in a_vertices:
        for b_vertex in b_vertices:
            dist_sq = (a_vertex - b_vertex).length_squared
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_a = a_vertex
                best_b = b_vertex

    midpoint = (best_a + best_b) / 2.0
    print(f"{a_name} <-> {b_name}")
    print("  best_a", tuple(round(v, 6) for v in best_a))
    print("  best_b", tuple(round(v, 6) for v in best_b))
    print("  midpoint", tuple(round(v, 6) for v in midpoint))
    print("  distance", round(best_dist_sq**0.5, 6))


def main() -> None:
    print("WRIST_PIVOT_BEGIN")
    for pair in TARGETS:
        nearest_midpoint(*pair)
    print("WRIST_PIVOT_END")


if __name__ == "__main__":
    main()
