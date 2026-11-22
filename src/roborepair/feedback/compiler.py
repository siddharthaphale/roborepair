"Compile structured feedback plans and choose editing or regeneration."

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema", "repair_plan_schema.json")

PLAN_SCHEMA_VERSION = "2.0"

REGENERATE_LABELS = {"no_api_calls", "invalid_api", "syntax_error", "target_not_localized"}

# Structured plans support the two mock cube tasks used by the editing control.
_TASK_INFO = {
    "cube_lift": {"source": "red_cube", "target": None},
    "cube_stack": {"source": "red_cube", "target": "green_cube"},
}

SUPPORTED_TASKS = tuple(_TASK_INFO)

_CHECKLISTS = {
    "cube_lift": [
        "get_object_pose('red_cube') is called before moving",
        "goto_pose('red_cube') comes before grasp('red_cube')",
        "lift('red_cube', ...) is called after a successful grasp",
        "the gripper is NOT opened after lifting (the cube must still be held at the end)",
        "all code executes at top level (any helper function is called)",
    ],
    "cube_stack": [
        "get_object_pose is called for both 'red_cube' and 'green_cube'",
        "goto_pose('red_cube') comes before grasp('red_cube')",
        "only 'red_cube' is ever grasped, and only while holding nothing",
        "lift('red_cube', ...) is called after the grasp",
        "place_on('red_cube', 'green_cube') is the final placement (it releases the cube)",
        "open_gripper() is not called before place_on",
        "all code executes at top level (any helper function is called)",
    ],
}

# Templates use the mock API; source and target names are filled per task.
_TEMPLATES = {
    "syntax_error": {
        "intent": "The previous program did not compile. Write a complete syntactically "
        "valid program using only the documented API.",
        "must_change": [
            "fix the syntax so the program compiles",
            "use only the documented API functions",
        ],
        "forbidden": [
            "unclosed parentheses or truncated statements",
            "markdown or prose inside the code",
        ],
    },
    "timeout": {
        "intent": "The previous program exceeded the execution time budget (an unbounded "
        "loop or polling). Write a direct straight-line program with no loops that wait "
        "or retry.",
        "must_change": [
            "remove unbounded loops and retry/polling logic",
            "execute the task as a short direct sequence of API calls",
        ],
        "forbidden": ["while True loops", "sleep or busy-wait polling"],
    },
    "invalid_api": {
        "intent": "The previous program called functions that do not exist. Use ONLY the "
        "documented API functions, already defined in the environment (no imports, no "
        "redefinitions).",
        "must_change": [
            "remove unknown function calls and imports",
            "use only the documented API signatures",
        ],
        "forbidden": [
            "import statements",
            "calling any function not in the API doc",
            "redefining API functions",
        ],
    },
    "no_api_calls": {
        "intent": "The previous program executed but made no API calls (it only defined "
        "functions). Write top-level executable code; if you define a helper, call it.",
        "must_change": [
            "make the API calls execute at top level",
            "call any helper function you define",
        ],
        "forbidden": [
            "defining functions without calling them",
            "if __name__ guards that skip execution",
        ],
    },
    "target_not_localized": {
        "intent": "The previous program never located the target object. Look up the "
        "object with get_object_pose and use goto_pose(name) with the OBJECT NAME "
        "before manipulating it.",
        "must_change": [
            "add get_object_pose('{src}') before any motion",
            "call goto_pose with the object NAME, not a coordinate tuple",
        ],
        "forbidden": [
            "referencing objects that are not in the scene",
            "passing coordinate tuples to goto_pose",
        ],
    },
    "missing_grasp": {
        "intent": "The previous program never attempted to grasp {src}. Add the approach "
        "and grasp: goto_pose('{src}') then grasp('{src}') before lifting or placing.",
        "must_change": ["add goto_pose('{src}') followed by grasp('{src}') before lift/place"],
        "forbidden": ["lift or place_on before a successful grasp"],
    },
    "failed_grasp": {
        "intent": "The grasp of {src} was attempted but did not secure the object. "
        "Approach it first with goto_pose('{src}'), grasp while holding nothing, "
        "then continue.",
        "must_change": [
            "ensure goto_pose('{src}') immediately precedes grasp('{src}')",
            "ensure the gripper is empty when grasping",
        ],
        "forbidden": ["grasp without a preceding goto_pose on the same object"],
    },
    "grasp_without_goto": {
        "intent": "The grasp of {src} failed because the gripper was not positioned "
        "above it. Insert goto_pose('{src}') immediately before grasp('{src}').",
        "must_change": ["insert goto_pose('{src}') immediately before grasp('{src}')"],
        "forbidden": ["grasp without a preceding goto_pose on the same object"],
    },
    "grasp_while_holding": {
        "intent": "A grasp failed because the gripper was still holding another object. "
        "Grasp ONLY {src}; never grasp a second object while holding one. Move the held "
        "object to its destination instead.",
        "must_change": [
            "remove the second grasp attempt",
            "grasp only '{src}', then lift it and place it with place_on('{src}', '{tgt}')",
        ],
        "forbidden": ["grasp while already holding an object", "grasping '{tgt}'"],
    },
    "object_not_lifted": {
        "intent": "{src} was grasped but never lifted. After the grasp, call "
        "lift('{src}', height=0.1) before any placement or completion.",
        "must_change": ["add lift('{src}', height=0.1) after the successful grasp"],
        "forbidden": ["placing or releasing before lifting"],
    },
    "released_after_lift": {
        "intent": "{src} was grasped and lifted successfully but then RELEASED, so it "
        "was no longer held at the end. Do not call open_gripper after "
        "lifting{release_rule}.",
        "must_change": ["remove the open_gripper call after lift{release_change}"],
        "forbidden": ["open_gripper immediately after lift"],
    },
    "bad_placement": {
        "intent": "{src} was lifted but not placed correctly on {tgt}. Use "
        "place_on('{src}', '{tgt}') as the final action; it positions the object on "
        "the target and releases it.",
        "must_change": ["replace the manual release/drop with place_on('{src}', '{tgt}')"],
        "forbidden": ["open_gripper instead of place_on", "dropping the object away from '{tgt}'"],
    },
    "unsafe_path": {
        "intent": "An object was knocked off the table. Move only with "
        "goto_pose/lift/place_on; never knock objects.",
        "must_change": ["remove the unsafe motion", "use goto_pose + grasp + lift + place_on only"],
        "forbidden": ["knock_off", "motions that push objects off the table"],
    },
}

_FALLBACK_TEMPLATE = {
    "intent": "The task did not succeed. Follow the verification checklist and produce "
    "a complete correct program using only the documented API.",
    "must_change": ["satisfy every item of the verification checklist"],
    "forbidden": [],
}

# Achieved-subgoal -> code aspect worth preserving (per task).
_PRESERVE_MAP = {
    "cube_lift": {
        "object_localized": "object pose lookup / approach (get_object_pose, goto_pose "
        "on 'red_cube')",
        "object_grasped": "the successful grasp of 'red_cube'",
        "object_lifted": "the lift of 'red_cube'",
    },
    "cube_stack": {
        "source_localized": "localization/approach of 'red_cube'",
        "target_localized": "localization of 'green_cube'",
        "source_grasped": "the successful grasp of 'red_cube'",
        "source_lifted": "the lift of 'red_cube'",
        "source_placed_on_target": "the placement on 'green_cube'",
    },
}


class PlanValidationError(ValueError):
    """Raised when a compiled plan violates the plan schema."""


@lru_cache(maxsize=1)
def load_plan_schema() -> dict:
    """Load and cache ``repair_plan_schema.json``."""
    with open(os.path.normpath(_SCHEMA_PATH)) as fh:
        return json.load(fh)


def validate_plan(plan: dict) -> None:
    """jsonschema-validate a compiled plan; raise ``PlanValidationError`` on violation."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - jsonschema is a core dep
        raise PlanValidationError(f"jsonschema unavailable: {exc}") from exc
    try:
        jsonschema.validate(instance=plan, schema=load_plan_schema())
    except jsonschema.ValidationError as exc:
        raise PlanValidationError(str(exc)) from exc


def route_repair_mode(trace: dict) -> str:
    """Select editing or regeneration from the execution trace."""
    labels = trace.get("failure_labels") or []
    primary = labels[0] if labels else None
    if not trace.get("compile_ok", False) or not trace.get("api_ok", False):
        return "regenerate"
    if float(trace.get("progress", 0.0)) < 0.25:
        return "regenerate"
    if primary in REGENERATE_LABELS:
        return "regenerate"
    return "patch"


_EVENT_RE = re.compile(r"^(\w+)\((.*)\) -> (ok|FAILED)$")


def parse_execution_events(event_strings):
    out = []
    for s in event_strings or []:
        m = _EVENT_RE.match(s.strip())
        if not m:
            continue
        name, argstr, status = m.groups()
        args = [a.strip().strip("'\"") for a in argstr.split(",")] if argstr.strip() else []
        out.append((name, args, status == "ok"))
    return out


def infer_task_state(task_id: str, trace: dict) -> dict:
    events = parse_execution_events(trace.get("events"))
    held = None
    grasp_ok = set()
    lifted_ok = set()
    attempted_second_grasp = False
    released_after_lift = False
    grasp_failures = 0
    for name, args, ok in events:
        obj = args[0] if args else None
        if name == "grasp":
            if ok:
                held = obj
                grasp_ok.add(obj)
            else:
                grasp_failures += 1
                if held is not None:
                    attempted_second_grasp = True
        elif name == "lift" and ok:
            lifted_ok.add(obj)
        elif name == "open_gripper" and ok:
            if held is not None and held in lifted_ok:
                released_after_lift = True
            held = None
        elif name == "place_on" and ok:
            held = None
    localized = {a[0] for n, a, ok in events if n in ("get_object_pose", "goto_pose") and ok and a}
    info = _TASK_INFO[task_id]
    state = {
        "made_api_calls": bool(events),
        "compile_ok": bool(trace.get("compile_ok", False)),
        "api_ok": bool(trace.get("api_ok", False)),
        "timed_out": bool(trace.get("timed_out", False)),
        "source_localized": info["source"] in localized,
        "holding_object_at_end": held is not None,
        "grasp_succeeded_at_some_point": bool(grasp_ok),
        "grasp_failure_count": grasp_failures,
        "attempted_second_grasp": attempted_second_grasp,
        "released_after_lift": released_after_lift,
        "object_off_table": float((trace.get("metrics") or {}).get("off_table", 0.0)) >= 1.0,
    }
    if info["target"]:
        state["target_localized"] = info["target"] in localized
    return state


def format_task_instruction(template: str, task_id: str) -> str:
    info = _TASK_INFO[task_id]
    release_rule = (
        "; place_on releases it at the right moment"
        if task_id == "cube_stack"
        else " (the cube must still be held at the end)"
    )
    release_change = (
        " (use place_on('red_cube', 'green_cube') to place and release)"
        if task_id == "cube_stack"
        else ""
    )
    return template.format(
        src=info["source"],
        tgt=info["target"] or "the target",
        release_rule=release_rule,
        release_change=release_change,
    )


def compile_plan(trace: dict, *, episode_id: str) -> dict:
    """Compile a validated execution trace into a deterministic repair plan."""
    from ..simulation.tasks import TASKS

    task_id = trace["task_id"]
    if task_id not in _TASK_INFO:
        raise ValueError(f"unknown task {task_id!r}")
    labels = list(trace.get("failure_labels") or [])
    primary = labels[0] if labels else "unknown_failure"
    template = _TEMPLATES.get(primary, _FALLBACK_TEMPLATE)

    order = list(TASKS[task_id].SUBGOALS)
    subgoals = trace.get("subgoals") or {}
    achieved = [s for s in order if subgoals.get(s, False)]
    missing = [s for s in order if not subgoals.get(s, False)]

    mode = route_repair_mode(trace)
    preserve_map = _PRESERVE_MAP[task_id]
    must_preserve = (
        [preserve_map[s] for s in achieved if s in preserve_map] if mode == "patch" else []
    )

    plan = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "source_episode_id": episode_id,
        "task_id": task_id,
        "primary_failure": primary,
        "secondary_failures": labels[1:],
        "initial_progress": float(trace.get("progress", 0.0)),
        "achieved_subgoals": achieved,
        "missing_subgoals": missing,
        "causal_state": infer_task_state(task_id, trace),
        "repair_mode": mode,
        "repair_intent": format_task_instruction(template["intent"], task_id),
        "must_preserve": must_preserve,
        "must_change": [format_task_instruction(c, task_id) for c in template["must_change"]],
        "forbidden_patterns": [format_task_instruction(f, task_id) for f in template["forbidden"]],
        "verification_checklist": list(_CHECKLISTS[task_id]),
        "compiler_rule_ids": [f"{task_id}.{primary}"],
        "confidence": 0.9 if primary in _TEMPLATES else 0.5,
    }
    validate_plan(plan)
    return plan


def render_plan_text(plan: dict) -> str:
    """Render a repair plan with stable section ordering."""
    lines = [
        f"primary_failure: {plan['primary_failure']}",
        f"repair_mode: {plan['repair_mode']}",
        f"initial_progress: {plan['initial_progress']:.3f}",
        "achieved_subgoals: " + (", ".join(plan["achieved_subgoals"]) or "(none)"),
        "missing_subgoals: " + (", ".join(plan["missing_subgoals"]) or "(none)"),
    ]
    # Skip empty sections so feedback-field ablations retain the same renderer.
    if plan["causal_state"]:
        lines.append(
            "causal_state: "
            + ", ".join(f"{k}={v}" for k, v in sorted(plan["causal_state"].items()))
        )
    lines.append(f"repair_intent: {plan['repair_intent']}")
    if plan["must_preserve"]:
        lines.append("must_preserve:")
        lines.extend(f"  - {p}" for p in plan["must_preserve"])
    if plan["must_change"]:
        lines.append("must_change:")
        lines.extend(f"  - {c}" for c in plan["must_change"])
    if plan["forbidden_patterns"]:
        lines.append("forbidden_patterns:")
        lines.extend(f"  - {f}" for f in plan["forbidden_patterns"])
    if plan["verification_checklist"]:
        lines.append("verification_checklist:")
        lines.extend(f"  - {c}" for c in plan["verification_checklist"])
    return "\n".join(lines)
