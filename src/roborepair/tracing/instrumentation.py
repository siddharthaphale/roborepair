"""Build a validated ``Trace`` from an execution result."""

from __future__ import annotations

import hashlib

from ..schema.execution_trace import Trace
from ..simulation.tasks import TASKS
from . import diagnosis as failure_diagnosis
from .progress import progress_score, subgoal_status


def program_hash(code: str) -> str:
    """First 16 hex chars of the sha256 of ``code``."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def format_execution_events(events) -> list:
    out = []
    for e in events:
        status = "ok" if e.ok else "FAILED"
        out.append(f"{e.name}({', '.join(map(repr, e.args))}) -> {status}")
    return out


def build_trace(task_id: str, exec_result, *, sim_seed: int, code: str) -> Trace:
    """Compute subgoals + metrics + diagnosis and return a validated ``Trace``."""
    task = TASKS[task_id]
    state = exec_result.final_state
    status = subgoal_status(task_id, state, exec_result.events)
    success = (
        exec_result.compile_ok
        and exec_result.api_ok
        and not exec_result.timed_out
        and task.check_success(state)
    )
    progress = progress_score(task_id, status)

    metrics = {
        # Number of API calls that returned not-ok (failed grasp, bad goto, etc.) -- this is
        # an action-failure count, NOT specifically a physical collision count.
        "failed_event_count": float(sum(1 for e in exec_result.events if not e.ok)),
        "off_table": 1.0 if state.off_table else 0.0,
        "n_events": float(len(exec_result.events)),
    }
    # Object heights help downstream analysis.
    for name, pose in state.object_poses.items():
        metrics[f"height_{name}"] = float(pose[2] - state.table_z)

    labels, diagnosis = failure_diagnosis.failure_labels(
        task_id, exec_result, status, metrics, success=success
    )

    trace = Trace(
        task_id=task_id,
        compile_ok=exec_result.compile_ok,
        api_ok=exec_result.api_ok,
        timed_out=exec_result.timed_out,
        task_success=success,
        progress=progress,
        subgoals=status,
        events=format_execution_events(exec_result.events),
        metrics=metrics,
        failure_labels=labels,
        diagnosis=diagnosis,
        sim_seed=sim_seed,
        program_hash=program_hash(code),
    )
    trace.validate()
    return trace
