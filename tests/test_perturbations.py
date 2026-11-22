"""CPU tests for perturbation-transfer perturbations."""

from roborepair.simulation import MockRepairEnv
from roborepair.simulation.base import Perturbation


def _reset_with(pert, sim_seed=0, task="cube_stack"):
    env = MockRepairEnv(task)
    return env.reset(sim_seed=sim_seed, perturbation=pert)


def test_pose_shift_within_bounds():
    """Shifted poses stay inside the workspace bounds."""
    base = MockRepairEnv("cube_stack")
    base_obs = base.reset(sim_seed=0)
    obs = _reset_with(Perturbation("pose_shift"))
    wb = obs.workspace_bounds
    # internal poses are derived from the reset; verify via the success-check API path
    env = MockRepairEnv("cube_stack")
    env.reset(sim_seed=0, perturbation=Perturbation("pose_shift"))
    r = env.run_program("get_object_pose('red_cube')\n", timeout_s=5)
    x, y, z = r.final_state.object_poses["red_cube"]
    assert wb[0, 0] <= x <= wb[0, 1]
    assert wb[1, 0] <= y <= wb[1, 1]
    # the state hash changes vs the unperturbed reset
    assert obs.initial_state_hash != base_obs.initial_state_hash


def test_paraphrase_preserves_api_vocab():
    """Paraphrase changes the instruction text only, not the object/API names."""
    base = MockRepairEnv("cube_stack").reset(sim_seed=0)
    obs = _reset_with(Perturbation("paraphrase"))
    assert obs.instruction != base.instruction
    assert obs.object_names == base.object_names  # API/object vocab unchanged


def test_distractor_adds_one_object_only():
    """Exactly one irrelevant object added; original objects preserved."""
    base = MockRepairEnv("cube_stack").reset(sim_seed=0)
    obs = _reset_with(Perturbation("distractor"))
    added = set(obs.object_names) - set(base.object_names)
    assert len(added) == 1
    assert set(base.object_names) <= set(obs.object_names)
