"""Optional task, failure-label, and reset checks in the simulator."""

import json
from argparse import Namespace

import pytest
import yaml
from simulator_checks import check_manipulation_tasks

from roborepair.execution.run_context import bootstrap

pytestmark = pytest.mark.simulator


@pytest.mark.parametrize(
    "config_fixture",
    [
        "manipulation_task_check_config",
        "low_level_task_check_config",
        "nut_and_door_check_config",
    ],
    ids=["high_level_manipulation", "low_level_manipulation", "nut_insertion_and_door_opening"],
)
def test_simulator_task_criteria(request, config_fixture, tmp_path):
    config = request.getfixturevalue(config_fixture)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_dir = tmp_path / "simulator_checks"
    context = bootstrap(Namespace(config=config_path, output_dir=output_dir, seed=0, resume=False))
    check_manipulation_tasks(context)
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["status"] == "PASS", metrics["criteria"]
    if "open_door" in config["env"]["tasks"]:
        assert metrics["environment_correctness"]["gold_policy_success"]["open_door"] >= 0.90
