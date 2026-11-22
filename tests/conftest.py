"""Shared test configurations, runner fixtures, and optional hardware selection."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def pytest_collection_modifyitems(items):
    """Keep hardware requirements explicit in test reports."""
    requirements = {
        "gpu": (os.environ.get("ROBOREPAIR_RUN_GPU") == "1", "Set ROBOREPAIR_RUN_GPU=1"),
        "simulator": (
            os.environ.get("ROBOREPAIR_RUN_SIMULATOR") == "1",
            "Set ROBOREPAIR_RUN_SIMULATOR=1",
        ),
    }
    for item in items:
        for marker, (available, reason) in requirements.items():
            if marker in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="module")
def cpu_feedback_smoke_config():
    return {
        "stage": "feedback_controls_cpu_smoke",
        "run": {"run_id": "cpu_smoke", "output_dir": "runs/cpu_smoke", "resume": False},
        "seeds": {"sim": 0, "gen": 0, "train": 0, "perturb": 0},
        "model": {"name": "mock-code", "seed": 0},
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
            "code_timeout_s": 5,
            "trace_input_token_cap": 768,
            "rawlog_input_token_cap": 768,
        },
        "env": {
            "backend": "mock",
            "tasks": ["cube_lift", "cube_stack"],
            "sim_version": "mock",
            "asset_version": "mock",
        },
        "experiment": {
            "failure_bank": "data/failure_banks/pilot_failures.jsonl",
            "experiment_id": "cpu_smoke",
            "tier": "high_level",
            "max_retries": 2,
            "wave_size": 4,
            "n_episodes": 8,
            "arms": [
                "template_feedback",
                "resample",
                "execution_log_regeneration",
                "editing_with_execution_log",
                "editing_with_failure_label",
                "editing_with_template_feedback",
                "editing_with_reflection",
                "reflection_regeneration_control",
                "compatible_template_union",
                "fixed_task_template",
                "three_task_templates",
                "initial_failure_feedback",
                "latest_failure_feedback",
                "permuted_label_feedback",
                "random_compatible_feedback",
                "generic_instructions",
                "different_label_feedback",
            ],
        },
    }


@pytest.fixture(scope="module")
def gpu_feedback_smoke_config():
    return {
        "stage": "feedback_controls_gpu_smoke",
        "run": {"run_id": "gpu_smoke", "output_dir": "runs/gpu_smoke", "resume": False},
        "seeds": {"sim": 0, "gen": 0, "train": 0, "perturb": 0},
        "model": {
            "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
            "tokenizer_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
            "quantization": "none",
        },
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
            "code_timeout_s": 60,
            "trace_input_token_cap": 768,
            "rawlog_input_token_cap": 768,
        },
        "env": {
            "backend": "robosuite",
            "tasks": ["cube_lift", "cube_stack"],
            "sim_version": "robosuite-1.5.2+mujoco-3.3.5",
            "asset_version": "robosuite-1.5.2",
        },
        "experiment": {
            "failure_bank": "data/failure_banks/pilot_failures.jsonl",
            "experiment_id": "gpu_smoke",
            "tier": "high_level",
            "max_retries": 1,
            "wave_size": 8,
            "n_episodes": 8,
            "arms": ["resample"],
        },
    }


@pytest.fixture(scope="module")
def manipulation_task_check_config():
    return {
        "stage": "manipulation_task_checks",
        "run": {
            "run_id": "manipulation_task_checks_seed0",
            "output_dir": "runs/manipulation_task_checks",
            "resume": False,
        },
        "seeds": {"sim": 0, "gen": 0, "train": 0, "perturb": 0},
        "decoding": {"code_timeout_s": 60},
        "env": {
            "backend": "robosuite",
            "tasks": ["cube_lift", "cube_stack", "cube_to_zone", "place_in_bin"],
            "sim_version": "robosuite-1.5.2+mujoco-3.3.5",
            "asset_version": "robosuite-1.5.2",
        },
        "task_checks": {
            "tier": "high_level",
            "new_tasks": ["cube_to_zone", "place_in_bin"],
            "perturbation_tasks": ["cube_lift", "cube_stack", "cube_to_zone"],
            "pass_criteria": {
                "fault_label_min": 0.95,
                "success_agreement_min": 0.99,
                "label_agreement_min": 0.95,
                "gold_min": 0.9,
            },
        },
    }


@pytest.fixture(scope="module")
def low_level_task_check_config():
    return {
        "stage": "manipulation_task_checks",
        "run": {
            "run_id": "low_level_task_checks_seed0",
            "output_dir": "runs/low_level_task_checks",
            "resume": False,
        },
        "seeds": {"sim": 0, "gen": 0, "train": 0, "perturb": 0},
        "decoding": {"code_timeout_s": 60},
        "env": {
            "backend": "robosuite",
            "tasks": ["cube_to_zone", "place_in_bin"],
            "sim_version": "robosuite-1.5.2+mujoco-3.3.5",
            "asset_version": "robosuite-1.5.2",
        },
        "task_checks": {
            "tier": "low_level",
            "new_tasks": ["cube_to_zone", "place_in_bin"],
            "perturbation_tasks": [],
            "pass_criteria": {
                "fault_label_min": 0.95,
                "success_agreement_min": 0.99,
                "label_agreement_min": 0.95,
                "gold_min": 0.9,
            },
        },
    }


@pytest.fixture(scope="module")
def nut_and_door_check_config():
    return {
        "stage": "manipulation_task_checks",
        "run": {
            "run_id": "additional_scene_checks_seed0",
            "output_dir": "runs/additional_scene_checks",
            "resume": False,
        },
        "seeds": {"sim": 0, "gen": 0, "train": 0, "perturb": 0},
        "decoding": {"code_timeout_s": 60},
        "env": {
            "backend": "robosuite",
            "tasks": ["nut_on_peg", "open_door"],
            "sim_version": "robosuite-1.5.2+mujoco-3.3.5",
            "asset_version": "robosuite-1.5.2",
        },
        "task_checks": {
            "tier": "high_level",
            "new_tasks": ["nut_on_peg", "open_door"],
            "perturbation_tasks": [],
            "pass_criteria": {
                "fault_label_min": 0.95,
                "success_agreement_min": 0.99,
                "label_agreement_min": 0.95,
                "gold_min": 0.85,
            },
        },
    }


@pytest.fixture
def run_configured_script(tmp_path):
    def run(script_name, config, *extra_args, timeout_s=300):
        root = Path(__file__).resolve().parents[1]
        run_id = config["run"]["run_id"]
        config_path = tmp_path / f"{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        output_dir = tmp_path / run_id
        result = subprocess.run(
            [
                sys.executable,
                str(root / "experiments" / script_name),
                "--config",
                str(config_path),
                "--seed",
                "0",
                "--output-dir",
                str(output_dir),
                *extra_args,
            ],
            cwd=root,
            env={**os.environ, "MUJOCO_GL": "disable"},
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        return output_dir

    return run
