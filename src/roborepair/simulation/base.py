"Environment interfaces and execution-result records for mock and real simulation."

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


class InfraError(RuntimeError):
    """Infrastructure failure (sim crash / asset corruption). MUST propagate out of run_program."""


@dataclass
class Observation:
    """What the policy sees at reset: instruction, object names, bounds, state hash."""

    instruction: str
    object_names: list
    workspace_bounds: Any  # np.ndarray of shape (n, 2): columns [lo, hi], lo < hi
    initial_state_hash: str


@dataclass
class Event:
    """One code-as-policy API call appended to the execution event log."""

    name: str
    args: tuple
    ok: bool
    ts: float


@dataclass
class SimState:
    """Privileged simulator state used by success checks + progress (no perception)."""

    object_poses: dict = field(default_factory=dict)  # name -> (x, y, z) tuple
    grasped: str | None = None  # name of currently grasped object, or None
    table_z: float = 0.0
    off_table: bool = False  # any object knocked off the table


@dataclass
class ExecResult:
    """Outcome of one run_program call: compile/api/timeout flags, event log, final state."""

    compile_ok: bool
    api_ok: bool
    timed_out: bool
    raised: BaseException | None
    events: list  # list[Event]
    final_state: SimState


@dataclass
class Perturbation:
    """Episode-level perturbation spec: family name plus a free-form payload."""

    family: str  # "pose_shift" | "paraphrase" | "distractor"
    payload: dict = field(default_factory=dict)


class RepairEnv(abc.ABC):
    """Abstract reset/step contract for a code-as-policy repair environment."""

    task_id: str
    api_doc: str

    @abc.abstractmethod
    def reset(self, *, sim_seed: int, perturbation: Perturbation | None = None) -> Observation:
        """Reset to a sim_seed-determined episode and return the initial Observation."""

    @abc.abstractmethod
    def run_program(self, code: str, *, timeout_s: float) -> ExecResult:
        """Execute generated code, capturing model failures; only InfraError propagates."""

    def close(self) -> None:  # noqa: B027 - intentional no-op default, optional override
        """Release backend resources (no-op by default)."""
