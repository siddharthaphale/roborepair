"""Rule-based failure diagnosis from execution events and simulator state.

Maps an ``ExecResult`` + subgoal status + metrics to ``(failure_labels, diagnosis)``.
Labels (primary first): ``syntax_error``, ``invalid_api``, ``timeout``,
``target_not_localized``, ``missing_grasp``, ``failed_grasp``, ``object_not_lifted``,
``bad_placement``, ``unsafe_path``. A successful program yields no labels.

The rules are intentionally a strict precedence cascade so each scripted fault maps
to exactly one primary label (the env_trace_correctness stage requires >=95% label accuracy).
"""

from __future__ import annotations


def failure_labels(task_id, exec_result, status, metrics, *, success: bool):
    """Return ``(labels, diagnosis)``. ``labels[0]`` is the primary cause."""
    if success:
        return [], "task_success"

    labels: list[str] = []

    # 1. Hard execution failures take precedence.
    if not exec_result.compile_ok:
        labels.append("syntax_error")
        return labels, "Program failed to compile (syntax error)."
    if exec_result.timed_out:
        labels.append("timeout")
        return labels, "Program exceeded the execution time budget."
    if not exec_result.api_ok:
        # An unhandled exception escaped model code (e.g. unknown API name / bad object).
        raised = exec_result.raised
        if isinstance(raised, (KeyError,)):
            labels.append("target_not_localized")
            return labels, "Referenced an object that does not exist in the scene."
        labels.append("invalid_api")
        return labels, f"Invalid API usage: {type(raised).__name__ if raised else 'error'}."

    # 2. Safety: object knocked off the table.
    if metrics.get("off_table", 0.0) >= 1.0:
        labels.append("unsafe_path")
        return labels, "Object was knocked off the table (unsafe path)."

    # 2b. Program ran cleanly but never touched the API (e.g. defined a function and
    # never called it). Distinct from localization failures so the trace tells the
    # repair model exactly what went wrong.
    if len(exec_result.events) == 0:
        labels.append("no_api_calls")
        return (
            labels,
            "The program executed but made no API calls "
            "(e.g. a function was defined but never called).",
        )

    # 3. Subgoal cascade (program ran cleanly but did not succeed).
    localized = _any_localized(task_id, status)
    grasped = _grasped(task_id, status)
    lifted = _lifted(task_id, status)

    if not localized:
        labels.append("target_not_localized")
        return labels, "Target object was never localized."
    # Door-opening terminal failure. The task has no lift/place
    # stages; once the handle is engaged the only remaining failure is that the door
    # never opened. Additive branch keyed on subgoal PRESENCE (like source_*/object_*
    # below); no existing task's mapping changes.
    if "door_opened" in status and grasped:
        labels.append("door_not_opened")
        return labels, "The handle was grasped but the door was never pulled open."

    if not grasped:
        mode, detail = _grasp_failure_mode(exec_result.events)
        if mode == "released_after_lift":
            labels.append("released_after_lift")
            return labels, (
                "The object was grasped and lifted successfully but then RELEASED "
                "(open_gripper), so it is no longer held in the final state."
            )
        if mode == "grasp_while_holding":
            labels.append("grasp_while_holding")
            return labels, (
                f"Grasp of {detail[0]!r} failed because the gripper was still holding "
                f"{detail[1]!r} (release or place the held object first)."
            )
        if mode == "grasp_without_goto":
            labels.append("grasp_without_goto")
            return labels, (
                f"Grasp of {detail[0]!r} failed because the gripper was not positioned "
                "above it (call goto_pose on the object first)."
            )
        if mode == "failed_grasp":
            labels.append("failed_grasp")
            return labels, "Grasp was attempted but did not secure the object."
        labels.append("missing_grasp")
        return labels, "The object was never grasped."
    if not lifted:
        labels.append("object_not_lifted")
        return labels, "Object was grasped but never lifted off the table."

    # Grasped + lifted but still not success -> placement problem (stack task).
    labels.append("bad_placement")
    return labels, "Object was lifted but not placed correctly on the target."


def _grasp_failure_mode(events):
    """Classify WHY the object is not held, from the event log.

    Returns (mode, detail):
      * ("released_after_lift", (obj,))       grasp ok + lift ok, later released
      * ("grasp_while_holding", (tgt, held))  a grasp FAILED while holding another object
      * ("grasp_without_goto", (tgt,))        a grasp FAILED with the gripper elsewhere
      * ("failed_grasp", ())                  attempted grasp, other cause
      * ("missing_grasp", ())                 grasp never attempted
    """
    held = None
    last_goto = None
    lifted_ok = set()
    grasped_ok = set()
    released_after_lift = None
    first_failed = None  # (target, held_at_time, last_goto_at_time)
    for e in events:
        if e.name == "goto_pose" and e.ok and e.args:
            last_goto = e.args[0]
        elif e.name == "grasp" and e.args:
            if e.ok:
                held = e.args[0]
                grasped_ok.add(e.args[0])
            elif first_failed is None:
                first_failed = (e.args[0], held, last_goto)
        elif e.name == "lift" and e.ok and e.args:
            lifted_ok.add(e.args[0])
        elif e.name == "open_gripper" and e.ok:
            if held is not None and held in lifted_ok:
                released_after_lift = held
            held = None
        elif e.name == "place_on" and e.ok:
            held = None

    if released_after_lift is not None:
        return "released_after_lift", (released_after_lift,)
    if first_failed is not None:
        tgt, held_at, goto_at = first_failed
        if held_at is not None and held_at != tgt:
            return "grasp_while_holding", (tgt, held_at)
        if goto_at != tgt:
            return "grasp_without_goto", (tgt,)
        return "failed_grasp", ()
    if grasped_ok or any(e.name == "grasp" for e in events):
        return "failed_grasp", ()
    return "missing_grasp", ()


# Key lookup is by PRESENCE, not task name: placement-style tasks (cube_stack,
# cube_to_zone, place_in_bin) use the source_* vocabulary; single-object tasks use
# object_*. Both map to the shared diagnosis rules.


def _any_localized(task_id, status) -> bool:
    if "source_localized" in status:
        return status.get("source_localized", False)
    return status.get("object_localized", False)


def _grasped(task_id, status) -> bool:
    if "source_grasped" in status:
        return status.get("source_grasped", False)
    return status.get("object_grasped", False)


def _lifted(task_id, status) -> bool:
    if "source_lifted" in status:
        return status.get("source_lifted", False)
    return status.get("object_lifted", False)
