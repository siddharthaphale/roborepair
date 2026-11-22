"""Cube stacking with ordered localization, grasp, lift, and placement subgoals."""

from __future__ import annotations

from .cube_lift import FaultSpec

TASK_ID = "cube_stack"
INSTRUCTION = "Stack the red cube on top of the green cube."
OBJECT_NAMES = ["red_cube", "green_cube"]
PLACE_XY_TOL = 0.03  # metres: how close in xy to count as "on top of"
LIFT_THRESHOLD = 0.05

SUBGOALS = [
    "source_localized",
    "target_localized",
    "source_grasped",
    "source_lifted",
    "source_placed_on_target",
]


def check_success(state) -> bool:
    """True iff the red cube rests on top of the green cube (xy tolerance) and is released."""
    src = state.object_poses.get("red_cube")
    tgt = state.object_poses.get("green_cube")
    if src is None or tgt is None:
        return False
    on_top_xy = abs(src[0] - tgt[0]) <= PLACE_XY_TOL and abs(src[1] - tgt[1]) <= PLACE_XY_TOL
    above = src[2] > tgt[2]
    return state.grasped is None and on_top_xy and above


def subgoal_status(state, events) -> dict:
    """Ordered subgoal achievement from final state + event log."""
    # "Localized" counts a successful get_object_pose OR goto_pose on that object
    # (navigating to an object demonstrates locating it); a successful place_on
    # additionally localizes its target.
    localized_names = {
        e.args[0] for e in events if e.name in ("get_object_pose", "goto_pose") and e.ok and e.args
    }
    localized_names |= {
        e.args[1] for e in events if e.name == "place_on" and e.ok and len(e.args) > 1
    }
    src_loc = "red_cube" in localized_names
    tgt_loc = "green_cube" in localized_names
    grasped_ever = any(
        e.name == "grasp" and e.ok and e.args and e.args[0] == "red_cube" for e in events
    )
    lifted_ever = any(
        e.name == "lift" and e.ok and e.args and e.args[0] == "red_cube" for e in events
    )
    placed = check_success(state)
    return {
        "source_localized": src_loc,
        "target_localized": tgt_loc,
        "source_grasped": grasped_ever,
        "source_lifted": lifted_ever and grasped_ever,
        "source_placed_on_target": placed,
    }


def gold_program() -> str:
    """Return the scripted reference program that solves the task."""
    return (
        "get_object_pose('red_cube')\n"
        "get_object_pose('green_cube')\n"
        "goto_pose('red_cube')\n"
        "grasp('red_cube')\n"
        "lift('red_cube', height=0.1)\n"
        "place_on('red_cube', 'green_cube')\n"
    )


def fault_programs() -> dict:
    """Return the frozen fault-name -> FaultSpec map for this task."""
    return {
        "syntax": FaultSpec(
            code="place_on('red_cube', 'green_cube'\n",
            expected_primary_label="syntax_error",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
        "invalid_api": FaultSpec(
            code="stack_cubes('red_cube', 'green_cube')\n",
            expected_primary_label="invalid_api",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
        "skipped_grasp": FaultSpec(
            code=(
                "get_object_pose('red_cube')\nget_object_pose('green_cube')\n"
                "goto_pose('red_cube')\nplace_on('red_cube', 'green_cube')\n"
            ),
            expected_primary_label="missing_grasp",
            expected_subgoals={
                "source_localized": True,
                "target_localized": True,
                "source_grasped": False,
                "source_lifted": False,
                "source_placed_on_target": False,
            },
        ),
        "failed_grasp": FaultSpec(
            code=(
                "get_object_pose('red_cube')\nget_object_pose('green_cube')\n"
                "grasp('red_cube')\nlift('red_cube', height=0.1)\n"
                "place_on('red_cube', 'green_cube')\n"
            ),
            expected_primary_label="grasp_without_goto",
            expected_subgoals={
                "source_localized": True,
                "target_localized": True,
                "source_grasped": False,
                "source_lifted": False,
                "source_placed_on_target": False,
            },
        ),
        "grasp_while_holding": FaultSpec(
            # grasps the target cube first, then tries to grasp the source while holding it
            code=(
                "goto_pose('green_cube')\ngrasp('green_cube')\n"
                "goto_pose('red_cube')\ngrasp('red_cube')\n"
                "lift('red_cube', height=0.1)\nplace_on('red_cube', 'green_cube')\n"
            ),
            expected_primary_label="grasp_while_holding",
            expected_subgoals={
                "source_localized": True,
                "target_localized": True,
                "source_grasped": False,
                "source_lifted": False,
                "source_placed_on_target": False,
            },
        ),
        "not_lifted": FaultSpec(
            code=(
                "get_object_pose('red_cube')\nget_object_pose('green_cube')\n"
                "goto_pose('red_cube')\ngrasp('red_cube')\nplace_on('red_cube', 'green_cube')\n"
            ),
            expected_primary_label="object_not_lifted",
            expected_subgoals={
                "source_localized": True,
                "target_localized": True,
                "source_grasped": True,
                "source_lifted": False,
                "source_placed_on_target": False,
            },
        ),
        "defined_not_called": FaultSpec(
            code=(
                "def do_task():\n"
                "    goto_pose('red_cube')\n"
                "    grasp('red_cube')\n"
                "    lift('red_cube', height=0.1)\n"
                "    place_on('red_cube', 'green_cube')\n"
            ),
            expected_primary_label="no_api_calls",
            expected_subgoals={k: False for k in SUBGOALS},
        ),
        "bad_placement": FaultSpec(
            code=(
                "get_object_pose('red_cube')\nget_object_pose('green_cube')\n"
                "goto_pose('red_cube')\ngrasp('red_cube')\nlift('red_cube', height=0.1)\n"
                "open_gripper()\n"  # drop in place instead of placing on target
            ),
            expected_primary_label="bad_placement",
            expected_subgoals={
                "source_localized": True,
                "target_localized": True,
                "source_grasped": True,
                "source_lifted": True,
                "source_placed_on_target": False,
            },
        ),
    }
