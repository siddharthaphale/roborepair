"""CPU tests for the RepairEnv reset/run_program contract."""

import numpy as np
import pytest

from roborepair.simulation import MockRepairEnv
from roborepair.simulation.base import InfraError
from roborepair.simulation.tasks import cube_lift


def test_reset_returns_observation_shapes_dtypes():
    """Reset returns documented obs shapes/dtypes."""
    env = MockRepairEnv("cube_lift")
    obs = env.reset(sim_seed=0)
    assert isinstance(obs.instruction, str) and obs.instruction
    assert isinstance(obs.object_names, list) and all(isinstance(n, str) for n in obs.object_names)
    wb = obs.workspace_bounds
    assert wb.dtype == np.float64 and wb.ndim == 2 and wb.shape[1] == 2
    assert np.all(wb[:, 0] < wb[:, 1])  # lo < hi
    assert isinstance(obs.initial_state_hash, str)


def test_run_program_never_raises_on_model_error():
    """Syntax / invalid-API / timeout are captured into ExecResult, not raised."""
    env = MockRepairEnv("cube_lift")
    env.reset(sim_seed=0)
    # syntax error
    r = env.run_program("grasp('x'\n", timeout_s=5)
    assert r.compile_ok is False and r.raised is None
    # invalid API name -> NameError captured (api_ok False), not raised
    r = env.run_program("teleport('x')\n", timeout_s=5)
    assert r.compile_ok is True and r.api_ok is False and r.raised is not None


def test_run_program_reraises_infra_only():
    """A simulated sim crash DOES propagate as InfraError."""
    env = MockRepairEnv("cube_lift")
    env.reset(sim_seed=0)
    env.trigger_infra_crash()
    with pytest.raises(InfraError):
        env.run_program("get_object_pose('red_cube')\n", timeout_s=5)


def test_timeout_enforced():
    """An infinite-loop program returns timed_out=True."""
    env = MockRepairEnv("cube_lift")
    env.reset(sim_seed=0)
    r = env.run_program("while True:\n    pass\n", timeout_s=0.3)
    assert r.timed_out is True


def test_reset_is_deterministic_for_fixed_seed():
    """Same sim_seed -> identical initial state hash."""
    e1 = MockRepairEnv("cube_stack").reset(sim_seed=7)
    e2 = MockRepairEnv("cube_stack").reset(sim_seed=7)
    e3 = MockRepairEnv("cube_stack").reset(sim_seed=8)
    assert e1.initial_state_hash == e2.initial_state_hash
    assert e1.initial_state_hash != e3.initial_state_hash


def test_gold_program_succeeds_each_task():
    """The committed gold programs must succeed."""
    for task_id, mod in (
        ("cube_lift", cube_lift),
        ("cube_stack", __import__("roborepair.simulation.tasks.cube_stack", fromlist=["x"])),
    ):
        env = MockRepairEnv(task_id)
        env.reset(sim_seed=3)
        r = env.run_program(mod.gold_program(), timeout_s=5)
        assert r.compile_ok and r.api_ok and not r.timed_out
        assert mod.check_success(r.final_state) is True
