"""Trace schema package (versioned JSON schema + Trace dataclass)."""

from .execution_trace import (
    SCHEMA_VERSION,
    Trace,
    TraceValidationError,
    load_trace_schema,
    validate_trace_dict,
)

__all__ = [
    "SCHEMA_VERSION",
    "Trace",
    "TraceValidationError",
    "load_trace_schema",
    "validate_trace_dict",
]
