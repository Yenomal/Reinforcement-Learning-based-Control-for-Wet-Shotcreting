from .train_env import BaseTrainEnv, build_train_env

__all__ = ["MathEnv", "BaseTrainEnv", "build_train_env"]


def __getattr__(name: str) -> object:
    if name == "MathEnv":
        from .math_env import MathEnv

        return MathEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
