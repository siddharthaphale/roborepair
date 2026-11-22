"""Cube lifting: success requires the cube to remain grasped and raised."""

from __future__ import annotations

from dataclasses import dataclass

TASK_ID = "cube_lift"
INSTRUCTION = "Lift the red cube off the table."
OBJECT_NAMES = ["red_cube"]
LIFT_THRESHOLD = 0.05  # metres above table to count as "lifted"

# Ordered subgoal names (frozen). Progress = fraction achieved, in order.
SUBGOALS = ["object_localized", "object_grasped", "object_lifted"]


@dataclass
class FaultSpec:
    """Scripted fault program with its expected diagnosis label and subgoal outcomes."""

    code: str
    expected_primary_label: str
    expected_subgoals: dict  # subgoal name -> achieved bool


def check_success(state) -> bool:
    """True iff the red cube is grasped AND lifted above the threshold."""
    pose = state.object_poses.get("red_cube")
    if pose is None:
        return False
    return state.grasped == "red_cube" and (pose[2] - state.table_z) >= LIFT_THRESHOLD


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log.

    "Localized" counts a successful get_object_pose OR goto_pose on the target: a
    program that navigates to the object has demonstrably located it.
    """
    localized = any(e.name in ("get_object_pose", "goto_pose") and e.ok for e in events)
    grasped = state.grasped == "red_cube"
    pose = state.object_poses.get("red_cube")
    lifted = grasped and pose is not None and (pose[2] - state.table_z) >= LIFT_THRESHOLD
    return {"object_localized": localized, "object_grasped": grasped, "object_lifted": lifted}


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "p = get_object_pose('red_cube')\n"
        "goto_pose('red_cube')\n"
        "grasp('red_cube')\n"
        "lift('red_cube', height=0.1)\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    return {
        "syntax": FaultSpec(
            code="grasp('red_cube'\n",  # missing close paren
            expected_primary_label="syntax_error",
            expected_subgoals={
                "object_localized": False,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "invalid_api": FaultSpec(
            code="teleport_cube('red_cube')\n",  # not in API namespace -> NameError
            expected_primary_label="invalid_api",
            expected_subgoals={
                "object_localized": False,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "skipped_grasp": FaultSpec(
            code=(
                "p = get_object_pose('red_cube')\ngoto_pose('red_cube')\n"
                "lift('red_cube', height=0.1)\n"
            ),
            expected_primary_label="missing_grasp",
            expected_subgoals={
                "object_localized": True,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "failed_grasp": FaultSpec(
            # grasp without going to the object first -> grasp fails (api_ok stays True, no grasp)
            code=(
                "p = get_object_pose('red_cube')\ngrasp('red_cube')\nlift('red_cube', height=0.1)\n"
            ),
            expected_primary_label="grasp_without_goto",
            expected_subgoals={
                "object_localized": True,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "released_after_lift": FaultSpec(
            # correct lift then releases the object -> no longer held in the final state
            code=(
                "p = get_object_pose('red_cube')\ngoto_pose('red_cube')\n"
                "grasp('red_cube')\nlift('red_cube', height=0.1)\nopen_gripper()\n"
            ),
            expected_primary_label="released_after_lift",
            expected_subgoals={
                "object_localized": True,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "not_lifted": FaultSpec(
            code=(  # never lift
                "p = get_object_pose('red_cube')\ngoto_pose('red_cube')\ngrasp('red_cube')\n"
            ),
            expected_primary_label="object_not_lifted",
            expected_subgoals={
                "object_localized": True,
                "object_grasped": True,
                "object_lifted": False,
            },
        ),
        "off_table": FaultSpec(
            # knock the cube off the table (unsafe)
            code="p = get_object_pose('red_cube')\ngoto_pose('red_cube')\nknock_off('red_cube')\n",
            expected_primary_label="unsafe_path",
            expected_subgoals={
                "object_localized": True,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
        "defined_not_called": FaultSpec(
            # defines a correct routine but never calls it -> nothing executes
            code=(
                "def do_task():\n"
                "    goto_pose('red_cube')\n"
                "    grasp('red_cube')\n"
                "    lift('red_cube', height=0.1)\n"
            ),
            expected_primary_label="no_api_calls",
            expected_subgoals={
                "object_localized": False,
                "object_grasped": False,
                "object_lifted": False,
            },
        ),
    }
