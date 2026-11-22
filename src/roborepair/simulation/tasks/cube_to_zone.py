"""Move the red cube into a reset-seeded target zone.

Grasp and lift subgoals use event history so a misplaced drop is diagnosed
as bad placement rather than an early release.
"""

from __future__ import annotations

from .cube_lift import FaultSpec

TASK_ID = "cube_to_zone"
INSTRUCTION = "Move the red cube into the target zone marked on the table."
OBJECT_NAMES = ["red_cube", "target_zone"]
ZONE_XY_TOL = 0.05  # metres: |dx|,|dy| to the zone centre to count as inside
REST_Z_TOL = 0.05  # metres above the table: the cube must be resting, not held aloft

SUBGOALS = [
    "source_localized",
    "target_localized",
    "source_grasped",
    "source_lifted",
    "source_placed",
]


def check_success(state) -> bool:
    """True iff the red cube rests inside the target zone and nothing is held."""
    cube = state.object_poses.get("red_cube")
    zone = state.object_poses.get("target_zone")
    if cube is None or zone is None:
        return False
    in_zone = abs(cube[0] - zone[0]) <= ZONE_XY_TOL and abs(cube[1] - zone[1]) <= ZONE_XY_TOL
    resting = (cube[2] - state.table_z) < REST_Z_TOL
    return state.grasped is None and in_zone and resting


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log."""
    localized_names = {
        e.args[0] for e in events if e.name in ("get_object_pose", "goto_pose") and e.ok and e.args
    }
    localized_names |= {
        e.args[1] for e in events if e.name == "place_on" and e.ok and len(e.args) > 1
    }
    grasped_ever = any(
        e.name == "grasp" and e.ok and e.args and e.args[0] == "red_cube" for e in events
    )
    lifted_ever = any(
        e.name == "lift" and e.ok and e.args and e.args[0] == "red_cube" for e in events
    )
    return {
        "source_localized": "red_cube" in localized_names,
        "target_localized": "target_zone" in localized_names,
        "source_grasped": grasped_ever,
        "source_lifted": lifted_ever and grasped_ever,
        "source_placed": check_success(state),
    }


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "p = get_object_pose('red_cube')\n"
        "z = get_object_pose('target_zone')\n"
        "goto_pose('red_cube')\n"
        "grasp('red_cube')\n"
        "lift('red_cube', height=0.1)\n"
        "place_on('red_cube', 'target_zone')\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    # Simulator fault programs are defined in robosuite.HIGH_LEVEL_FAULT_PROGRAMS.
    return {
        "invalid_api": FaultSpec(
            code="push_to_zone('red_cube')\n",
            expected_primary_label="invalid_api",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
    }
