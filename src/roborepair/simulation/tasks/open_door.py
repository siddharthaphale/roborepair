"""Open an unlatched door by engaging and pulling its handle.

Success requires at least 0.095 m of handle displacement from its reset pose,
approximating a 0.3 rad hinge rotation for the fixed door geometry.
"""

from __future__ import annotations

from .cube_lift import FaultSpec

TASK_ID = "open_door"
INSTRUCTION = "Open the door by pulling its handle."
OBJECT_NAMES = ["door_handle", "door_hinge"]  # handle_start is state-only bookkeeping
OPEN_DISPLACEMENT = 0.095  # handle displacement in metres, approximately 0.3 rad of hinge rotation

SUBGOALS = [
    "object_localized",
    "object_grasped",
    "door_opened",
]


def check_success(state) -> bool:
    """True iff the handle travelled at least OPEN_DISPLACEMENT in xy from its start."""
    handle = state.object_poses.get("door_handle")
    start = state.object_poses.get("handle_start")
    if handle is None or start is None:
        return False
    dx, dy = handle[0] - start[0], handle[1] - start[1]
    return (dx * dx + dy * dy) ** 0.5 >= OPEN_DISPLACEMENT


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log."""
    localized_names = {
        e.args[0] for e in events if e.name in ("get_object_pose", "goto_pose") and e.ok and e.args
    }
    engaged_ever = any(
        e.name in ("grasp", "pull") and e.ok and e.args and e.args[0] == "door_handle"
        for e in events
    )
    return {
        "object_localized": "door_handle" in localized_names,
        "object_grasped": engaged_ever,
        "door_opened": check_success(state),
    }


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "h = get_object_pose('door_handle')\n"
        "goto_pose('door_handle')\n"
        "grasp('door_handle')\n"
        "pull('door_handle', -0.11, 0.16)\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    # Simulator fault programs are defined in robosuite.HIGH_LEVEL_FAULT_PROGRAMS.
    return {
        "invalid_api": FaultSpec(
            code="open_the_door('door_handle')\n",
            expected_primary_label="invalid_api",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
    }
