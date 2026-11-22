"Construct mock or robosuite environments from the backend configuration."

from __future__ import annotations

from .mock import MockRepairEnv
from .robosuite import RobosuiteRepairEnv


def make_env(task_id: str, env_cfg: dict):
    """Construct the RepairEnv backend named by ``env_cfg["backend"]`` for ``task_id``."""
    backend = env_cfg.get("backend", "mock")
    if backend == "mock":
        return MockRepairEnv(task_id)
    if backend == "robosuite":
        return RobosuiteRepairEnv(
            task_id,
            sim_version=env_cfg.get("sim_version", "UNPINNED"),
            asset_version=env_cfg.get("asset_version", "UNPINNED"),
            api_tier=env_cfg.get("api_tier", "low_level"),
            pose_noise_sigma=float(env_cfg.get("pose_noise_sigma", 0.0)),
        )
    raise ValueError(f"unknown env backend {backend!r}; expected 'mock' or 'robosuite'")
