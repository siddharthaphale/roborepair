"""Execution-trace records and schema validation.

Use jsonschema when available; otherwise validate required fields, types,
and ranges with the built-in validator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

SCHEMA_VERSION = "1.0"
_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "trace_schema.json")


class TraceValidationError(ValueError):
    """Raised when a trace dict violates the trace schema."""


@lru_cache(maxsize=1)
def load_trace_schema() -> dict:
    """Load and cache ``trace_schema.json``."""
    with open(_SCHEMA_PATH) as fh:
        return json.load(fh)


_REQUIRED = (
    "schema_version",
    "task_id",
    "compile_ok",
    "api_ok",
    "timed_out",
    "task_success",
    "progress",
    "subgoals",
    "events",
    "metrics",
    "failure_labels",
    "diagnosis",
    "sim_seed",
    "program_hash",
)


def _builtin_validate(d: dict) -> None:
    schema = load_trace_schema()
    allowed = set(schema["properties"])
    extra = set(d) - allowed
    if extra:
        raise TraceValidationError(f"unexpected trace fields: {sorted(extra)}")
    for key in _REQUIRED:
        if key not in d:
            raise TraceValidationError(f"missing required trace field {key!r}")
    if d["schema_version"] != SCHEMA_VERSION:
        raise TraceValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {d['schema_version']!r}"
        )
    for key in ("compile_ok", "api_ok", "timed_out", "task_success"):
        if not isinstance(d[key], bool):
            raise TraceValidationError(f"{key} must be bool, got {type(d[key]).__name__}")
    if not isinstance(d["task_id"], str):
        raise TraceValidationError("task_id must be str")
    if not isinstance(d["diagnosis"], str):
        raise TraceValidationError("diagnosis must be str")
    if not isinstance(d["sim_seed"], int) or isinstance(d["sim_seed"], bool):
        raise TraceValidationError("sim_seed must be int")
    prog = d["progress"]
    if not isinstance(prog, (int, float)) or isinstance(prog, bool):
        raise TraceValidationError("progress must be a number")
    if not (0.0 <= prog <= 1.0):
        raise TraceValidationError(f"progress must be in [0, 1], got {prog}")
    if not isinstance(d["subgoals"], dict) or not all(
        isinstance(v, bool) for v in d["subgoals"].values()
    ):
        raise TraceValidationError("subgoals must be dict[str, bool]")
    if not isinstance(d["events"], list) or not all(isinstance(e, str) for e in d["events"]):
        raise TraceValidationError("events must be list[str]")
    if not isinstance(d["failure_labels"], list) or not all(
        isinstance(e, str) for e in d["failure_labels"]
    ):
        raise TraceValidationError("failure_labels must be list[str]")
    if not isinstance(d["metrics"], dict) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in d["metrics"].values()
    ):
        raise TraceValidationError("metrics must be dict[str, number]")


def validate_trace_dict(d: dict) -> None:
    """Validate a trace dict against the schema. Raises ``TraceValidationError``."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _builtin_validate(d)
        return
    try:
        jsonschema.validate(instance=d, schema=load_trace_schema())
    except jsonschema.ValidationError as exc:  # pragma: no cover - exercised only with the lib
        raise TraceValidationError(str(exc)) from exc


@dataclass
class Trace:
    """Execution-trace record matching the JSON schema."""

    schema_version: str = SCHEMA_VERSION
    task_id: str = ""
    compile_ok: bool = True
    api_ok: bool = True
    timed_out: bool = False
    task_success: bool = False
    progress: float = 0.0
    subgoals: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    failure_labels: list = field(default_factory=list)
    diagnosis: str = ""
    sim_seed: int = 0
    program_hash: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict matching the trace schema."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "compile_ok": self.compile_ok,
            "api_ok": self.api_ok,
            "timed_out": self.timed_out,
            "task_success": self.task_success,
            "progress": float(self.progress),
            "subgoals": dict(self.subgoals),
            "events": list(self.events),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "failure_labels": list(self.failure_labels),
            "diagnosis": self.diagnosis,
            "sim_seed": int(self.sim_seed),
            "program_hash": self.program_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        """Build a ``Trace`` from a dict, validating first.

        Unexpected/extra keys raise a clean ``TraceValidationError`` instead of a raw
        ``TypeError``.
        """
        validate_trace_dict(d)
        return cls(**{k: d[k] for k in d})

    def validate(self) -> None:
        """jsonschema-validate against trace_schema.json; raise on failure."""
        validate_trace_dict(self.to_dict())
