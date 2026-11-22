"""CPU and optional GPU smoke checks for the feedback comparison runner."""

import json
from pathlib import Path

import pytest


def assert_feedback_run(config, run_configured_script, *, timeout_s):
    bank_path = Path(__file__).resolve().parents[1] / "data/failure_banks/pilot_failures.jsonl"
    output_dir = run_configured_script(
        "feedback_comparison.py",
        config,
        timeout_s=timeout_s,
    )
    experiment = config["experiment"]
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["bank"]["n"] == experiment["n_episodes"]
    assert set(metrics["per_arm"]) == set(experiment["arms"])
    bank = [json.loads(line) for line in bank_path.read_text().splitlines() if line.strip()]
    expected_episodes = {row["episode_id"] for row in bank[: experiment["n_episodes"]]}
    for arm in experiment["arms"]:
        attempts = [
            json.loads(line)
            for line in (output_dir / f"attempts_{arm}.jsonl").read_text().splitlines()
        ]
        assert {row["episode_id"] for row in attempts} == expected_episodes
        assert all(1 <= row["attempt"] <= experiment["max_retries"] for row in attempts)
        assert not any(row["infra_excluded"] for row in attempts)
        assert 0 <= metrics["per_arm"][arm]["recovery_at_retry_limit"] <= 1


def test_cpu_feedback_pipeline(cpu_feedback_smoke_config, run_configured_script):
    assert_feedback_run(cpu_feedback_smoke_config, run_configured_script, timeout_s=300)


@pytest.mark.gpu
@pytest.mark.simulator
def test_gpu_feedback_pipeline(gpu_feedback_smoke_config, run_configured_script):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert_feedback_run(gpu_feedback_smoke_config, run_configured_script, timeout_s=1800)
