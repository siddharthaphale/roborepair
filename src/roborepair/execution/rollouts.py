"""Shared program execution, trace serialization, and mock-policy helpers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from roborepair.data.failure_bank import FailureEntry
from roborepair.schema.execution_trace import Trace
from roborepair.seeding import derive_seed
from roborepair.simulation.base import InfraError
from roborepair.tracing.instrumentation import build_trace
from roborepair.tracing.instrumentation import program_hash as program_hash

_FENCE_RE = re.compile("```(?:python|py)?\\s*\\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile("```(?:python|py)?\\s*\\n(.*)\\Z", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract closed code fences in order, or use the unfinished fence or raw completion."""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return "\n".join(b.strip() for b in blocks) + "\n"
    m = _OPEN_FENCE_RE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def render_raw_log(code: str, exec_result) -> str:
    """Verbose, unstructured simulator-style log built from the execution record.

    Deliberately noisy/low-level (timestamps, raw call lines, exception dump) --
    the RAW baseline against the structured trace.
    """
    lines = ["[sim] session start", f"[sim] program bytes={len(code)}"]
    if not exec_result.compile_ok:
        lines.append("[sim] E: program failed to compile")
    for e in exec_result.events:
        status = "OK" if e.ok else "ERR"
        args = ", ".join(repr(a) for a in e.args)
        lines.append(f"[sim t={e.ts:07.2f}] call {e.name}({args}) status={status}")
    if exec_result.timed_out:
        lines.append("[sim] E: watchdog: execution deadline exceeded")
    if exec_result.raised is not None:
        lines.append(
            f"[sim] E: unhandled exception {type(exec_result.raised).__name__}: {exec_result.raised}"
        )
    st = exec_result.final_state
    for name, pose in st.object_poses.items():
        lines.append(f"[sim] final pose {name}: ({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})")
    lines.append(f"[sim] final grasped={st.grasped!r} off_table={st.off_table}")
    lines.append("[sim] session end")
    return "\n".join(lines)


def run_and_trace(env, task_id: str, code: str, *, sim_seed: int, timeout_s: float) -> dict:
    """Run ``code`` in ``env`` (already reset), build a validated trace.

    Returns a dict with exec/trace/raw-log fields. Infra failures (InfraError) are
    caught and flagged -- they are EXCLUDED episodes, never model outcomes.
    """
    t0 = time.monotonic()
    try:
        exec_result = env.run_program(code, timeout_s=timeout_s)
    except InfraError as exc:
        return {
            "infra_excluded": True,
            "infra_reason": str(exc),
            "trace": None,
            "raw_log": "",
            "sim_wall_time_s": time.monotonic() - t0,
        }
    wall = time.monotonic() - t0
    trace = build_trace(task_id, exec_result, sim_seed=sim_seed, code=code)
    return {
        "infra_excluded": False,
        "infra_reason": None,
        "trace": trace,
        "raw_log": render_raw_log(code, exec_result),
        "sim_wall_time_s": wall,
    }


def entry_to_row(entry: FailureEntry, *, raw_log: str = "", extra: dict | None = None) -> dict:
    row = {
        "episode_id": entry.episode_id,
        "task": entry.task,
        "instruction": entry.instruction,
        "reset_seed": entry.reset_seed,
        "init_code": entry.init_code,
        "init_trace": entry.init_trace.to_dict(),
        "init_success": entry.init_success,
        "init_progress": entry.init_progress,
        "failure_labels": list(entry.failure_labels),
        "gen_seed": entry.gen_seed,
        "sim_version": entry.sim_version,
        "program_hash": entry.program_hash,
        "infra_excluded": entry.infra_excluded,
        "raw_log": raw_log,
    }
    if extra:
        row.update(extra)
    return row


def row_to_entry(row: dict) -> FailureEntry:
    return FailureEntry(
        episode_id=row["episode_id"],
        task=row["task"],
        instruction=row["instruction"],
        reset_seed=int(row["reset_seed"]),
        init_code=row["init_code"],
        init_trace=Trace.from_dict(row["init_trace"]),
        init_success=bool(row["init_success"]),
        init_progress=float(row["init_progress"]),
        failure_labels=list(row["failure_labels"]),
        gen_seed=int(row["gen_seed"]),
        sim_version=row["sim_version"],
        program_hash=row["program_hash"],
        infra_excluded=bool(row["infra_excluded"]),
    )


def paired_generation_seed(gen_namespace_seed: int, episode_id: str) -> int:
    """Model seed derived from the initial episode ID -- identical across conditions."""
    return derive_seed(gen_namespace_seed, "paired", episode_id)


def build_policy(config: dict, *, device: str = "cuda"):
    """Instantiate the generation policy from ``config['model']``.

    ``model.name == 'mock-code'`` selects the canned CPU smoke policy (pipeline
    validation only -- never for reported results).
    """
    model_cfg = dict(config.get("model", {}))
    if model_cfg.get("name") == "mock-code":
        return MockCodePolicy(seed=int(model_cfg.get("seed", 0)))
    from roborepair.policies.language_model import RepairPolicy

    return RepairPolicy(model_cfg, device=device)


class MockGeneration:
    def __init__(self, text: str, n_prompt_tokens: int):
        self.text = text
        self.prompt_token_ids: list[int] = []
        self.completion_token_ids: list[int] = []
        self.per_token_logprobs = None
        self.n_prompt_tokens = n_prompt_tokens
        self.n_completion_tokens = len(text.split())


class MockCodePolicy:
    """Seeded reference and fault-program generator for CPU pipeline checks."""

    is_mock = True
    _VARIANTS = ("gold", "skipped_grasp", "failed_grasp", "not_lifted", "syntax", "invalid_api")

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _program_for(self, task_id: str, variant: str) -> str:
        from roborepair.simulation.tasks import TASKS

        task = TASKS[task_id]
        if variant == "gold":
            return task.gold_program()
        faults = task.fault_programs()
        if variant in faults:
            return faults[variant].code
        return next(iter(faults.values())).code

    def generate(
        self,
        prompts,
        *,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        seed=0,
        greedy=False,
        num_return_sequences=1,
    ):
        import numpy as np

        out = []
        for i, prompt in enumerate(prompts):
            task_id = "cube_stack" if "green cube" in prompt else "cube_lift"
            rng = np.random.default_rng(derive_seed(self.seed, seed, i))
            for _j in range(num_return_sequences):
                variant = self._VARIANTS[int(rng.integers(0, len(self._VARIANTS)))]
                code = self._program_for(task_id, variant)
                out.append(MockGeneration(f"```python\n{code}```", len(prompt.split())))
        return out

    def generate_batch(self, prompts, **kwargs):
        """Generate one deterministic mock completion for each prompt in a batch."""
        return self.generate(prompts, **kwargs)

    def close(self):
        """Release policy resources; the mock policy owns no external resources."""
        return None


def evaluate_criterion(name: str, observed, required, op: str) -> dict:
    """One pass-criterion row: observed vs required under comparison ``op``."""
    ok: bool | None
    if observed is None:
        ok = None
    elif op == ">=":
        ok = observed >= required
    elif op == "<=":
        ok = observed <= required
    elif op == ">":
        ok = observed > required
    elif op == "<":
        ok = observed < required
    elif op == "==":
        ok = observed == required
    elif op == "in":
        ok = required[0] <= observed <= required[1]
    else:
        raise ValueError(f"unknown criterion op {op!r}")
    return {"name": name, "observed": observed, "required": required, "op": op, "pass": ok}


def overall_status(criteria: list[dict]) -> str:
    if any(c["pass"] is None for c in criteria):
        return "INCOMPLETE"
    return "PASS" if all(c["pass"] for c in criteria) else "FAIL"


def load_evaluation_cell(run_dir: Path, cell_id: str, eval_cap: int):
    """Reproduce the task_evaluation eval-failure selection + stored outcomes for one cell."""
    bank = []
    for line in (run_dir / "bank.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("cell") == cell_id and (not r["infra_excluded"]) and (not r.get("init_success")):
            bank.append(r)
    bank = bank[:eval_cap]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    stored = metrics["cells"][cell_id]["per_arm"]
    ids = {r["episode_id"] for r in bank}
    assert ids == set(stored["template_feedback"]["first_successful_attempt"]), (
        f"{cell_id}: bank selection does not reproduce the task_evaluation eval set"
    )
    return (bank, stored)
