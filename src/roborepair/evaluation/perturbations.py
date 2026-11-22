"""Perturbation-transfer perturbations: pose-shift / paraphrase / distractor.

Each perturbation changes ONE aspect while preserving task identity + the API vocab:
  * pose_shift  : shift object xy within the workspace bounds, off the training seed grid.
  * paraphrase  : reword the instruction string only (API names untouched).
  * distractor  : add exactly one irrelevant object (never referenced by the task API).
"""

from __future__ import annotations

WORKSPACE_LO, WORKSPACE_HI = -0.28, 0.28  # stay strictly inside the (-0.3, 0.3) bounds

_PARAPHRASES = {
    "Lift the red cube off the table.": "Pick up the red cube from the table surface.",
    "Stack the red cube on top of the green cube.": "Place the red cube onto the green cube.",
}


def apply(pert, instruction, object_names, poses, rng):
    """Return (instruction, object_names, poses) after applying ``pert``."""
    family = pert.family
    instruction = instruction
    object_names = list(object_names)
    poses = dict(poses)

    if family == "pose_shift":
        for name in list(poses):
            x, y, z = poses[name]
            dx = float(rng.uniform(-0.05, 0.05))
            dy = float(rng.uniform(-0.05, 0.05))
            nx = min(max(x + dx, WORKSPACE_LO), WORKSPACE_HI)
            ny = min(max(y + dy, WORKSPACE_LO), WORKSPACE_HI)
            poses[name] = (nx, ny, z)
    elif family == "paraphrase":
        instruction = _PARAPHRASES.get(instruction, instruction)
    elif family == "distractor":
        # exactly one irrelevant object, not in the task API references
        dx = float(rng.uniform(WORKSPACE_LO, WORKSPACE_HI))
        dy = float(rng.uniform(WORKSPACE_LO, WORKSPACE_HI))
        poses["blue_distractor"] = (dx, dy, 0.02)
        object_names.append("blue_distractor")
    else:
        raise ValueError(f"unknown perturbation family {family!r}")

    return instruction, object_names, poses
