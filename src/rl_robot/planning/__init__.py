__all__ = [
    "SensorNoise",
    "sample_planner_task",
    "sample_planner_task_from_environment",
    "build_and_save_reachability_map",
    "load_reachability_map",
]


def __getattr__(name: str) -> object:
    if name == "SensorNoise":
        from .disturbance import SensorNoise

        return SensorNoise
    if name in {"sample_planner_task", "sample_planner_task_from_environment"}:
        from .planner import sample_planner_task, sample_planner_task_from_environment

        exports = {
            "sample_planner_task": sample_planner_task,
            "sample_planner_task_from_environment": sample_planner_task_from_environment,
        }
        return exports[name]
    if name in {"build_and_save_reachability_map", "load_reachability_map"}:
        from .reachability_map import build_and_save_reachability_map, load_reachability_map

        exports = {
            "build_and_save_reachability_map": build_and_save_reachability_map,
            "load_reachability_map": load_reachability_map,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
