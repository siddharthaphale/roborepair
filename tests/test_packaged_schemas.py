"""Schema resources must be available from an installed distribution."""

import json
from importlib.resources import files

from roborepair.feedback.compiler import load_plan_schema
from roborepair.schema.execution_trace import load_trace_schema


def test_installed_schema_resources_are_available():
    """Catch wheels that omit either schema even though editable imports work."""
    resources = files("roborepair.schema")
    trace = json.loads(resources.joinpath("trace_schema.json").read_text())
    plan = json.loads(resources.joinpath("repair_plan_schema.json").read_text())
    assert load_trace_schema() == trace
    assert load_plan_schema() == plan
    assert trace["type"] == plan["type"] == "object"
