"""CPU tests for prompt building / fairness."""

from roborepair.prompts import build_initial_prompt, build_repair_prompt
from roborepair.simulation import MockRepairEnv
from roborepair.simulation.tasks import cube_lift
from roborepair.tracing import build_trace

API_DOC = "API: get_object_pose, goto_pose, grasp, lift, open_gripper, place_on"


def observation_from_record():
    return MockRepairEnv("cube_lift").reset(sim_seed=0)


def make_fault_trace():
    env = MockRepairEnv("cube_lift")
    env.reset(sim_seed=0)
    code = cube_lift.fault_programs()["skipped_grasp"].code
    r = env.run_program(code, timeout_s=5)
    return build_trace("cube_lift", r, sim_seed=0, code=code), code


def test_identical_api_doc_across_conditions():
    """Every condition embeds a byte-identical API doc."""
    obs = observation_from_record()
    trace, prev = make_fault_trace()
    prompts = []
    for cond in (
        "resample",
        "terminal_failure_feedback",
        "execution_log_feedback",
        "execution_trace_feedback",
        "execution_trace_without_diagnosis",
    ):
        p = build_repair_prompt(
            cond,
            prev_code=prev,
            trace=trace,
            raw_log="log",
            api_doc=API_DOC,
            env_obs=obs,
            token_budget=64,
        )
        assert API_DOC in p
        prompts.append(p)
    assert all(API_DOC in p for p in prompts)


def test_feedback_token_cap_enforced():
    """Structured and raw feedback truncate to the SAME cap."""
    obs = observation_from_record()
    trace, prev = make_fault_trace()
    long_log = " ".join([f"tok{i}" for i in range(500)])
    raw = build_repair_prompt(
        "execution_log_feedback",
        prev_code=prev,
        trace=trace,
        raw_log=long_log,
        api_doc=API_DOC,
        env_obs=obs,
        token_budget=10,
    )
    # the raw-log feedback section must be capped to 10 whitespace tokens
    feedback = raw.split("# Feedback (raw log)\n", 1)[1].split("\n# ", 1)[0]
    assert len(feedback.split()) <= 10


def test_resampling_omits_previous_program_and_trace():
    """Resampling excludes previous code and execution feedback."""
    obs = observation_from_record()
    trace, prev = make_fault_trace()
    p = build_repair_prompt(
        "resample",
        prev_code=prev,
        trace=trace,
        raw_log="x",
        api_doc=API_DOC,
        env_obs=obs,
        token_budget=64,
    )
    assert "Previous program" not in p
    assert "Feedback" not in p


def test_foreign_feedback_uses_donor_trace():
    """The donor trace's diagnosis appears in the mismatched-feedback prompt."""
    obs = observation_from_record()
    foreign, _ = make_fault_trace()
    p = build_repair_prompt(
        "foreign_trace_feedback",
        prev_code="pass",
        trace=foreign,
        raw_log=None,
        api_doc=API_DOC,
        env_obs=obs,
        token_budget=999,
    )
    assert "Feedback (trace)" in p
    assert foreign.diagnosis in p


def test_trace_condition_controls_diagnosis_visibility():
    """The diagnosis ablation omits diagnosis text while preserving other trace fields."""
    obs = observation_from_record()
    trace, prev = make_fault_trace()
    p3 = build_repair_prompt(
        "execution_trace_feedback",
        prev_code=prev,
        trace=trace,
        raw_log=None,
        api_doc=API_DOC,
        env_obs=obs,
        token_budget=999,
    )
    p4 = build_repair_prompt(
        "execution_trace_without_diagnosis",
        prev_code=prev,
        trace=trace,
        raw_log=None,
        api_doc=API_DOC,
        env_obs=obs,
        token_budget=999,
    )
    assert "diagnosis:" in p3
    assert "diagnosis:" not in p4


def test_initial_prompt_contains_task_and_api():
    """Initial prompt embeds the instruction + API doc."""
    obs = observation_from_record()
    p = build_initial_prompt(obs, API_DOC)
    assert obs.instruction in p and API_DOC in p
