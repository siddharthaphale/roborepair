"""Structured feedback compilation, routing, schema, and prompt controls."""

import copy

import pytest

from roborepair.feedback import compile_plan, render_plan_text, route_repair_mode, validate_plan
from roborepair.feedback.compiler import REGENERATE_LABELS, SUPPORTED_TASKS, PlanValidationError
from roborepair.prompts.builder import build_feedback_prompt, build_repair_prompt
from roborepair.simulation import MockRepairEnv
from roborepair.simulation.tasks import TASKS
from roborepair.tracing.instrumentation import build_trace


class TaskObservation:
    instruction = "Stack the red cube on top of the green cube."


def trace_for_fault(task_id, fault):
    env = MockRepairEnv(task_id)
    env.reset(sim_seed=0)
    code = TASKS[task_id].fault_programs()[fault].code
    res = env.run_program(code, timeout_s=10)
    return build_trace(task_id, res, sim_seed=0, code=code).to_dict()


def test_compiler_deterministic():
    """Same trace -> byte-identical plan."""
    t = trace_for_fault("cube_stack", "grasp_while_holding")
    p1 = compile_plan(t, episode_id="e1")
    p2 = compile_plan(t, episode_id="e1")
    assert p1 == p2
    assert render_plan_text(p1) == render_plan_text(p2)


def test_every_fault_compiles_to_valid_plan():
    """All scripted faults produce schema-valid plans."""
    for task_id in SUPPORTED_TASKS:  # Structured plans cover the two mock cube tasks.
        for fault in TASKS[task_id].fault_programs():
            plan = compile_plan(trace_for_fault(task_id, fault), episode_id=f"{task_id}-{fault}")
            validate_plan(plan)  # raises on violation
            assert plan["must_change"], (task_id, fault)
            assert plan["verification_checklist"], (task_id, fault)


def test_router_verbatim_rules():
    """Preserve routing for each failure label and execution state."""
    base = {
        "compile_ok": True,
        "api_ok": True,
        "progress": 0.5,
        "failure_labels": ["bad_placement"],
    }
    assert route_repair_mode(base) == "patch"
    assert route_repair_mode({**base, "compile_ok": False}) == "regenerate"
    assert route_repair_mode({**base, "api_ok": False}) == "regenerate"
    assert route_repair_mode({**base, "progress": 0.2}) == "regenerate"
    for label in REGENERATE_LABELS:
        assert route_repair_mode({**base, "failure_labels": [label]}) == "regenerate", label


def test_regenerate_plan_has_no_must_preserve():
    """Regenerate-mode plans carry no must_preserve (no anchoring on discarded code)."""
    plan = compile_plan(trace_for_fault("cube_lift", "defined_not_called"), episode_id="e")
    assert plan["repair_mode"] == "regenerate"
    assert plan["must_preserve"] == []


def test_plan_schema_rejects_bad_plan():
    """Schema enforcement is real."""
    plan = compile_plan(trace_for_fault("cube_lift", "not_lifted"), episode_id="e")
    bad = copy.deepcopy(plan)
    bad["repair_mode"] = "rewrite"  # not in enum
    with pytest.raises(PlanValidationError):
        validate_plan(bad)
    bad2 = copy.deepcopy(plan)
    bad2.pop("must_change")
    with pytest.raises(PlanValidationError):
        validate_plan(bad2)


def test_patch_prompt_includes_prev_code_regen_hides_it():
    """Editing includes previous code; regeneration omits it."""
    prev = "goto_pose('red_cube')\n"
    plan = compile_plan(trace_for_fault("cube_stack", "grasp_while_holding"), episode_id="e")
    txt = render_plan_text(plan)
    patch = build_feedback_prompt(
        plan_text=txt,
        mode="patch",
        prev_code=prev,
        api_doc="APIDOC",
        env_obs=TaskObservation(),
        token_budget=768,
    )
    regen = build_feedback_prompt(
        plan_text=txt,
        mode="regenerate",
        prev_code=prev,
        api_doc="APIDOC",
        env_obs=TaskObservation(),
        token_budget=768,
    )
    assert prev in patch
    assert prev not in regen
    assert "# Repair plan" in patch and "# Repair plan" in regen
    assert "APIDOC" in patch and "APIDOC" in regen


def test_plan_prompt_snapshot_stable():
    """Preserve the recorded prompt layout."""
    plan = compile_plan(trace_for_fault("cube_lift", "released_after_lift"), episode_id="snap")
    prompt = build_feedback_prompt(
        plan_text=render_plan_text(plan),
        mode=plan["repair_mode"],
        prev_code="X\n",
        api_doc="A",
        env_obs=TaskObservation(),
        token_budget=768,
    )
    assert prompt.startswith("# API\nA\n# Task\n")
    assert "primary_failure: released_after_lift" in prompt
    assert "must_change:" in prompt and "verification_checklist:" in prompt
    assert prompt.rstrip().endswith("Output only code.")


def test_label_only_prompt_excludes_structured_fields():
    """Label-only feedback excludes structured-plan fields."""
    p = build_repair_prompt(
        "failure_label_feedback",
        prev_code="PREV\n",
        api_doc="A",
        env_obs=TaskObservation(),
        token_budget=768,
        primary_label="grasp_while_holding",
    )
    assert "primary_failure: grasp_while_holding" in p
    assert "must_change" not in p and "verification_checklist" not in p
    assert "execution_events" not in p
    assert "PREV" in p


def test_plan_text_fits_budget():
    """Rendered plans stay under the shared 768-token feedback cap."""
    for task_id in SUPPORTED_TASKS:  # Structured plans cover the two mock cube tasks.
        for fault in TASKS[task_id].fault_programs():
            plan = compile_plan(trace_for_fault(task_id, fault), episode_id="e")
            assert len(render_plan_text(plan).split()) < 768


def test_neutral_prompt_study_suffix_selection():
    """Select the suffix from config while retaining the historical default."""
    kw = dict(
        plan_text="PLANTEXT",
        prev_code="PREV\n",
        api_doc="APIDOC",
        env_obs=TaskObservation(),
        token_budget=768,
    )
    hist_regen = build_feedback_prompt(mode="regenerate", **kw)
    hist_patch = build_feedback_prompt(mode="patch", **kw)
    # The historical default preserves the recorded feedback-control prompts.
    assert build_feedback_prompt(mode="regenerate", suffix="historical", **kw) == hist_regen
    assert "must_change" in hist_regen and "verification checklist" in hist_regen
    assert "must_preserve" in hist_patch
    neu_regen = build_feedback_prompt(mode="regenerate", suffix="neutral", **kw)
    neu_patch = build_feedback_prompt(mode="patch", suffix="neutral", **kw)
    # Neutral suffix references nothing outside the prompt, in either mode.
    for p in (neu_regen, neu_patch):
        assert "must_change" not in p and "must_preserve" not in p
        assert "verification checklist" not in p and "repair plan above" not in p
        assert "Follow every instruction present in this prompt" in p
    # Only the tail instruction differs from the historical scaffold.
    assert neu_regen.splitlines()[:-2] == hist_regen.splitlines()[:-2]
    try:
        build_feedback_prompt(mode="regenerate", suffix="bogus", **kw)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown suffix must raise")
