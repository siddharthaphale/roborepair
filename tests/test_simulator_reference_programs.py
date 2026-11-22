"""Optional robosuite checks for the task interfaces used in the paper."""

import pytest

pytestmark = pytest.mark.simulator


@pytest.mark.parametrize(
    "task", ["cube_lift", "cube_stack", "cube_to_zone", "place_in_bin", "nut_on_peg", "open_door"]
)
def test_reference_program_executes_and_produces_a_valid_trace(task):
    """Exercise real task setup, the robot API, subprocess execution, and trace validation."""
    from roborepair.simulation.factory import make_env
    from roborepair.simulation.robosuite import HIGH_LEVEL_REFERENCE_PROGRAMS
    from roborepair.tracing import build_trace

    environment = make_env(
        task,
        {
            "backend": "robosuite",
            "api_tier": "high_level",
            "sim_version": "robosuite-1.5.2+mujoco-3.3.5",
            "asset_version": "robosuite-1.5.2",
        },
    )
    try:
        environment.reset(sim_seed=0)
        code = HIGH_LEVEL_REFERENCE_PROGRAMS[task]
        outcome = environment.run_program(code, timeout_s=60)
        trace = build_trace(task, outcome, sim_seed=0, code=code)
        trace.validate()
        assert trace.compile_ok and trace.api_ok and not trace.timed_out
        assert trace.task_success
    finally:
        environment.close()
