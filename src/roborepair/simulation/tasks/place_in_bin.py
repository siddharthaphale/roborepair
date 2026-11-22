"""Place the can inside the target bin quadrant.

Grasp and lift subgoals use event history; success requires a released object
resting inside the bin.
"""

from __future__ import annotations

from .cube_lift import FaultSpec

TASK_ID = "place_in_bin"
INSTRUCTION = "Pick up the can and place it into the bin."
OBJECT_NAMES = ["can", "bin"]
BIN_XY_TOL = (0.08, 0.10)  # half-extents of the target bin quadrant (walls excluded)
BIN_Z_MAX = 0.12  # metres above the bin floor: resting inside, not held above

SUBGOALS = [
    "source_localized",
    "target_localized",
    "source_grasped",
    "source_lifted",
    "source_placed",
]


def check_success(state) -> bool:
    """True iff the can rests inside the target bin quadrant and nothing is held."""
    can = state.object_poses.get("can")
    tgt = state.object_poses.get("bin")
    if can is None or tgt is None:
        return False
    in_bin_xy = abs(can[0] - tgt[0]) <= BIN_XY_TOL[0] and abs(can[1] - tgt[1]) <= BIN_XY_TOL[1]
    in_bin_z = -0.02 <= (can[2] - tgt[2]) <= BIN_Z_MAX
    return state.grasped is None and in_bin_xy and in_bin_z


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log."""
    localized_names = {
        e.args[0] for e in events if e.name in ("get_object_pose", "goto_pose") and e.ok and e.args
    }
    localized_names |= {
        e.args[1] for e in events if e.name == "place_on" and e.ok and len(e.args) > 1
    }
    grasped_ever = any(e.name == "grasp" and e.ok and e.args and e.args[0] == "can" for e in events)
    lifted_ever = any(e.name == "lift" and e.ok and e.args and e.args[0] == "can" for e in events)
    return {
        "source_localized": "can" in localized_names,
        "target_localized": "bin" in localized_names,
        "source_grasped": grasped_ever,
        "source_lifted": lifted_ever and grasped_ever,
        "source_placed": check_success(state),
    }


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "p = get_object_pose('can')\n"
        "b = get_object_pose('bin')\n"
        "goto_pose('can')\n"
        "grasp('can')\n"
        "lift('can', height=0.1)\n"
        "place_on('can', 'bin')\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    # Simulator fault programs are defined in robosuite.HIGH_LEVEL_FAULT_PROGRAMS.
    return {
        "invalid_api": FaultSpec(
            code="put_in_bin('can')\n",
            expected_primary_label="invalid_api",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
    }
