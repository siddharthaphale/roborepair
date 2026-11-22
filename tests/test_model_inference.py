"""Optional checks against the pinned frozen model used in the main study."""

import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def frozen_policy(gpu_feedback_smoke_config):
    """Load the recorded model only when the GPU suite is explicitly enabled."""
    import torch

    from roborepair.policies.language_model import RepairPolicy

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return RepairPolicy(gpu_feedback_smoke_config["model"])


def test_frozen_model_generation_is_bounded_and_repeatable(frozen_policy):
    """Check token limits, frozen weights, and repeated greedy decoding."""
    prompt = "Write a Python program that prints the integer 1. Output only code."
    first = frozen_policy.generate([prompt], max_new_tokens=16, greedy=True, seed=0)[0]
    second = frozen_policy.generate([prompt], max_new_tokens=16, greedy=True, seed=0)[0]
    assert 0 < first.n_completion_tokens <= 16
    assert first.n_completion_tokens == len(first.completion_token_ids)
    assert first.completion_token_ids == second.completion_token_ids
    assert next(frozen_policy.model.parameters()).is_cuda
    assert not frozen_policy.model.training


def test_batch_generation_returns_one_record_per_prompt(frozen_policy):
    """Check batch cardinality and token accounting for differently sized prompts."""
    outputs = frozen_policy.generate_batch(
        ["Write Python: x = 1", "Write a short Python program assigning y = 2."],
        max_new_tokens=8,
        greedy=True,
        seed=0,
    )
    assert len(outputs) == 2
    assert all(0 < item.n_completion_tokens <= 8 for item in outputs)
    assert all(item.n_prompt_tokens == len(item.prompt_token_ids) for item in outputs)
