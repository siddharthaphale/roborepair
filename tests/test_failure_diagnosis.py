"""CPU tests for the rule-based diagnosis."""

from roborepair.simulation import MockRepairEnv
from roborepair.simulation.tasks import cube_lift, cube_stack
from roborepair.tracing import build_trace


def trace_for_fault(task_id, code, sim_seed=2):
    env = MockRepairEnv(task_id)
    env.reset(sim_seed=sim_seed)
    r = env.run_program(code, timeout_s=5)
    return build_trace(task_id, r, sim_seed=sim_seed, code=code)


def test_each_fault_maps_to_expected_label():
    """Each scripted fault produces its expected primary label (env_trace_correctness needs >=95%)."""
    for task_id, mod in (("cube_lift", cube_lift), ("cube_stack", cube_stack)):
        for fname, spec in mod.fault_programs().items():
            t = trace_for_fault(task_id, spec.code)
            primary = t.failure_labels[0] if t.failure_labels else None
            assert primary == spec.expected_primary_label, (task_id, fname, primary)


def test_no_label_on_success():
    """Gold programs yield empty failure_labels."""
    for task_id, mod in (("cube_lift", cube_lift), ("cube_stack", cube_stack)):
        t = trace_for_fault(task_id, mod.gold_program())
        assert t.task_success is True
        assert t.failure_labels == []


def test_subgoal_status_matches_expected_per_fault():
    """Each fault's subgoal vector matches the task spec."""
    for task_id, mod in (("cube_lift", cube_lift), ("cube_stack", cube_stack)):
        for fname, spec in mod.fault_programs().items():
            t = trace_for_fault(task_id, spec.code)
            assert t.subgoals == spec.expected_subgoals, (task_id, fname)
