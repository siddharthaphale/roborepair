"Select deterministic feedback templates from execution failure labels and robot skills."

from __future__ import annotations

CONSTRAINT_SCHEMA_VERSION = "3.0"

_GRASP_HINT = "goto_pose to the object's centre position, then close_gripper()"


def api_action_hints(tier: str) -> dict:
    if tier == "high_level":
        return {
            "lift_hint": "call lift(name, height=0.1) after the grasp",
            "place_hint": "use place_on(source, target) as the final action (it releases)",
        }
    return {
        "lift_hint": "after closing the gripper, goto_pose to a position 0.15 m ABOVE "
        "the grasp point with z_approach=0 (the object moves with the gripper)",
        "place_hint": "while holding, goto_pose above the target position, then open_gripper()",
    }


_TEMPLATES = {
    "syntax_error": {
        "mc": ["write a complete syntactically valid program"],
        "fb": ["truncated statements", "markdown or prose inside the code"],
        "dx": "The previous program did not compile.",
    },
    "timeout": {
        "mc": [
            "remove unbounded loops and retry/polling logic; write a short direct "
            "sequence of API calls"
        ],
        "fb": ["while-loops that wait or retry", "sleep or polling"],
        "dx": "The previous program exceeded the execution time budget.",
    },
    "invalid_api": {
        "mc": [
            "use ONLY the documented API functions, already defined (no imports, no redefinitions)"
        ],
        "fb": [
            "import statements",
            "calling functions not in the API doc",
            "redefining API functions",
        ],
        "dx": "The previous program called functions that do not exist.",
    },
    "no_api_calls": {
        "mc": ["make the API calls execute at top level; call any helper function you define"],
        "fb": ["defining functions without calling them"],
        "dx": "The previous program executed but made no API calls.",
    },
    "target_not_localized": {
        "mc": [
            "look up each object with get_object_pose(name) and move using the returned position"
        ],
        "fb": ["referencing objects not in the scene", "guessing positions"],
        "dx": "The previous program never located the target object.",
    },
    "missing_grasp": {
        "mc": ["grasp the object before lifting/placing: " + _GRASP_HINT],
        "fb": ["lifting or placing before a successful grasp"],
        "dx": "The previous program never attempted a grasp.",
    },
    "failed_grasp": {
        "mc": ["grasp at the object's CENTRE: " + _GRASP_HINT + ", with the gripper empty"],
        "fb": ["closing the gripper away from the object's centre"],
        "dx": "The grasp was attempted but did not secure the object.",
    },
    "grasp_without_goto": {
        "mc": ["move the gripper to the object's centre BEFORE closing: " + _GRASP_HINT],
        "fb": ["close_gripper without first moving to the object"],
        "dx": "The gripper closed while positioned away from the object.",
    },
    "grasp_while_holding": {
        "mc": [
            "grasp only ONE object; move the held object to its destination instead of "
            "grasping another"
        ],
        "fb": ["closing the gripper while already holding an object"],
        "dx": "A grasp failed because the gripper was still holding another object.",
    },
    "object_not_lifted": {
        "mc": ["after the grasp, LIFT the object: {lift_hint}"],
        "fb": ["ending the program with the object still on the table"],
        "dx": "The object was grasped but never lifted.",
    },
    "released_after_lift": {
        "mc": [
            "do NOT open the gripper after lifting; keep holding (or place on the "
            "target first): {place_hint}"
        ],
        "fb": ["open_gripper immediately after lifting"],
        "dx": "The object was lifted but then released, so it was no longer held at the end.",
    },
    "bad_placement": {
        "mc": ["place the object ON the target: {place_hint}"],
        "fb": ["releasing the object away from the target"],
        "dx": "The object was lifted but not placed correctly on the target.",
    },
    "unsafe_path": {
        "mc": ["approach objects from above with z_approach; avoid sweeping motions near objects"],
        "fb": ["motions that knock objects off the table"],
        "dx": "An object was knocked off the table.",
    },
    "door_not_opened": {
        "mc": [
            "after gripping the handle, PULL it sideways away from the door frame: "
            "pull('door_handle', dx, dy) with a horizontal vector of at least 0.15 m"
        ],
        "fb": [
            "releasing the handle without pulling it",
            "lifting the handle instead of dragging it sideways",
        ],
        "dx": "The handle was grasped but the door was never pulled open.",
    },
}

_FALLBACK = {
    "mc": ["fix the cause of failure and complete every task step"],
    "fb": [],
    "dx": "The task did not succeed.",
}


def compile_constraints(
    trace: dict, *, episode_id: str, attempt: int, tier: str = "low_level"
) -> dict:
    """Select deterministic feedback constraints for the failure label and robot API."""
    labels = list(trace.get("failure_labels") or [])
    primary = labels[0] if labels else "unknown_failure"
    t = _TEMPLATES.get(primary, _FALLBACK)
    h = api_action_hints(tier)
    out = {
        "constraint_schema_version": CONSTRAINT_SCHEMA_VERSION,
        "source_episode_id": episode_id,
        "attempt": int(attempt),
        "task_id": trace.get("task_id", ""),
        "primary_failure": primary,
        "must_change": [m.format(**h) for m in t["mc"]],
        "forbidden_patterns": [f.format(**h) for f in t["fb"]],
        "one_line_diagnosis": t["dx"],
        "compiler_rule_ids": [f"{trace.get('task_id', '?')}.{primary}"],
    }
    validate_constraints(out)
    return out


def validate_constraints(c: dict) -> None:
    assert c["constraint_schema_version"] == CONSTRAINT_SCHEMA_VERSION
    assert isinstance(c["must_change"], list) and c["must_change"]
    assert isinstance(c["one_line_diagnosis"], str) and c["one_line_diagnosis"]


def render_constraints_text(constraint_sets, *, cap_tokens: int = 768) -> str:
    """Render accumulated constraint sets: newest first, deduplicated lines, capped."""
    seen = set()
    lines = []
    for _i, c in enumerate(reversed(list(constraint_sets))):  # newest first
        header = (
            f"attempt {c['attempt']} failed: {c['primary_failure']} - {c['one_line_diagnosis']}"
        )
        if header not in seen:
            seen.add(header)
            lines.append(header)
        for m in c["must_change"]:
            line = f"  MUST: {m}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
        for f in c["forbidden_patterns"]:
            line = f"  NEVER: {f}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    text = "\n".join(lines)
    toks = text.split()
    if len(toks) > cap_tokens:
        text = " ".join(toks[:cap_tokens])
    return text
