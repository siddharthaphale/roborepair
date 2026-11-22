"""CPU tests for the trace schema."""

import pytest

from roborepair.schema.execution_trace import (
    SCHEMA_VERSION,
    Trace,
    TraceValidationError,
    load_trace_schema,
)


def _valid_trace() -> Trace:
    return Trace(
        task_id="cube_lift",
        compile_ok=True,
        api_ok=True,
        timed_out=False,
        task_success=True,
        progress=1.0,
        subgoals={"object_localized": True},
        events=["grasp('red_cube') -> ok"],
        metrics={"n_events": 1.0},
        failure_labels=[],
        diagnosis="task_success",
        sim_seed=0,
        program_hash="abc",
    )


def test_valid_trace_passes_schema():
    """A complete trace validates."""
    _valid_trace().validate()


def test_each_missing_field_fails():
    """Every required field is enforced."""
    from roborepair.schema.execution_trace import validate_trace_dict

    full = _valid_trace().to_dict()
    for field in list(full):
        broken = dict(full)
        del broken[field]
        with pytest.raises(TraceValidationError):
            validate_trace_dict(broken)


def test_progress_in_unit_interval():
    """Progress must be in [0, 1]."""
    t = _valid_trace()
    t.progress = 1.5
    with pytest.raises(TraceValidationError):
        t.validate()
    t.progress = -0.1
    with pytest.raises(TraceValidationError):
        t.validate()


def test_schema_version_pinned():
    """The version constant matches the schema file."""
    assert load_trace_schema()["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    t = _valid_trace()
    t.schema_version = "9.9"
    with pytest.raises(TraceValidationError):
        t.validate()


def test_unexpected_field_rejected():
    """Extra fields are rejected."""
    from roborepair.schema.execution_trace import validate_trace_dict

    d = _valid_trace().to_dict()
    d["surprise"] = 1
    with pytest.raises(TraceValidationError):
        validate_trace_dict(d)
