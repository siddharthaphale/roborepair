"""Place the round nut over the peg using the ring-wall grip point.

The round hole permits insertion at any yaw. The scene exposes the nut centre,
a reset-computed grip point, and the peg top. Grasp and placement failures use
the shared source-object diagnosis labels.
"""

from __future__ import annotations

from .cube_lift import FaultSpec

TASK_ID = "nut_on_peg"
INSTRUCTION = "Pick up the round nut and place it over the peg."
OBJECT_NAMES = ["round_nut", "nut_grip", "peg"]
PEG_XY_TOL = 0.03  # metres: |dx|,|dy| nut centre to peg axis (mirrors robosuite on_peg)
REST_Z_TOL = 0.05  # metres above the table: threaded nut has slid down the shaft

_NUT_NAMES = ("round_nut", "nut_grip")

SUBGOALS = [
    "source_localized",
    "target_localized",
    "source_grasped",
    "source_lifted",
    "source_placed",
]


def check_success(state) -> bool:
    """True iff the nut rests threaded on the peg axis and nothing is held."""
    nut = state.object_poses.get("round_nut")
    peg = state.object_poses.get("peg")
    if nut is None or peg is None:
        return False
    on_axis = abs(nut[0] - peg[0]) <= PEG_XY_TOL and abs(nut[1] - peg[1]) <= PEG_XY_TOL
    threaded = (nut[2] - state.table_z) < REST_Z_TOL
    return state.grasped is None and on_axis and threaded


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log."""
    localized_names = {
        e.args[0] for e in events if e.name in ("get_object_pose", "goto_pose") and e.ok and e.args
    }
    localized_names |= {
        e.args[1] for e in events if e.name == "place_on" and e.ok and len(e.args) > 1
    }
    grasped_ever = any(
        e.name == "grasp" and e.ok and e.args and e.args[0] in _NUT_NAMES for e in events
    )
    lifted_ever = any(
        e.name == "lift" and e.ok and e.args and e.args[0] == "round_nut" for e in events
    )
    return {
        "source_localized": bool(localized_names.intersection(_NUT_NAMES)),
        "target_localized": "peg" in localized_names,
        "source_grasped": grasped_ever,
        "source_lifted": lifted_ever and grasped_ever,
        "source_placed": check_success(state),
    }


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "g = get_object_pose('nut_grip')\n"
        "goto_pose('nut_grip')\n"
        "grasp('round_nut')\n"
        "lift('round_nut', height=0.15)\n"
        "place_on('round_nut', 'peg')\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    # Simulator fault programs are defined in robosuite.HIGH_LEVEL_FAULT_PROGRAMS.
    return {
        "invalid_api": FaultSpec(
            code="screw_on('round_nut', 'peg')\n",
            expected_primary_label="invalid_api",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
    }
