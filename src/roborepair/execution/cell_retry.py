"""Task-cell feedback and original-prompt ablation protocol."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roborepair.execution.rollouts import (
    extract_code,
    paired_generation_seed,
    program_hash,
    run_and_trace,
)
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.feedback_ablation_v3")
ARMS = [
    "resample",
    "template_feedback",
    "donor_feedback",
    "failure_label_feedback",
    "reflection_regeneration",
    "structured_plan_with_editing",
]
_PRESERVE_MAP = {
    "object_localized": "the object pose lookup",
    "source_localized": "the source pose lookup",
    "target_localized": "the target pose lookup",
    "object_grasped": "the approach and close_gripper",
    "source_grasped": "the approach and close_gripper",
    "object_lifted": "the lift",
    "source_lifted": "the lift",
}


def load_failure_bank_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observation_from_record(row):
    return SimpleNamespace(instruction=row["instruction"], object_names=row.get("object_names"))


class EpisodeState:
    """Mutable per-episode retry state: accumulated feedback, attempts, outcome."""

    def __init__(self, row):
        """Initialize the state from a frozen failure-bank row."""
        self.row = row
        self.eid = row["episode_id"]
        self.done = False
        self.first_successful_attempt = None
        self.constraints: list = []
        self.labels: list = []
        self.reflections: list = []
        self.last_trace = row["init_trace"]
        self.last_code = row["init_code"]
        self.last_raw_log = row.get("raw_log", "")
        self.attempt_rows: list = []
        self.valid_attempts = 0
        self.n_attempts = 0
        self.safety = False
        self.tokens = 0


def build_prompt(arm, ep, ctx, api_doc, tier, donor_constraint_history=None, static_text=None):
    """Build the arm-specific repair prompt for one episode's next attempt."""
    from roborepair.feedback.constraints import render_constraints_text
    from roborepair.prompts.builder import build_feedback_prompt

    cap = int(ctx.config["decoding"]["trace_input_token_cap"])
    obs = observation_from_record(ep.row)
    if arm == "resample":
        return build_feedback_prompt(
            plan_text=None,
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm == "template_feedback":
        return build_feedback_prompt(
            plan_text=render_constraints_text(ep.constraints, cap_tokens=cap),
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm in ("donor_feedback", "cross_task_feedback"):
        donor_sets = donor_constraint_history or []
        use = donor_sets[: max(1, len(ep.constraints))]
        text = render_constraints_text(use, cap_tokens=cap) if use else "(no constraints available)"
        return build_feedback_prompt(
            plan_text=text,
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm == "fixed_task_guidance":
        return build_feedback_prompt(
            plan_text=static_text or "(no static prior)",
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm == "failure_label_feedback":
        lines = [f"attempt {i + 1} failed: {lab}" for i, lab in enumerate(ep.labels)]
        return build_feedback_prompt(
            plan_text="\n".join(lines) + "\nFix these failures.",
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm == "reflection_regeneration":
        text = "\n".join(
            (f"Reflection after attempt {i + 1}: {t}" for i, t in enumerate(ep.reflections))
        )
        return build_feedback_prompt(
            plan_text=text or "(no reflections yet)",
            mode="regenerate",
            prev_code=None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    if arm == "structured_plan_with_editing":
        from roborepair.feedback.compiler import route_repair_mode

        mode = route_repair_mode(ep.last_trace)
        text = render_constraints_text(ep.constraints, cap_tokens=cap)
        subgoals = ep.last_trace.get("subgoals") or {}
        preserve = [_PRESERVE_MAP[s] for s, ok in subgoals.items() if ok and s in _PRESERVE_MAP]
        if mode == "patch" and preserve:
            text += "\nmust_preserve:\n" + "\n".join(f"  - {p}" for p in preserve)
        return build_feedback_prompt(
            plan_text=text,
            mode=mode,
            prev_code=ep.last_code if mode == "patch" else None,
            api_doc=api_doc,
            env_obs=obs,
            token_budget=cap,
        )
    raise ValueError(arm)


def perturbation_from_record(row):
    """Restore the banked perturbation so each retry uses the same scene."""
    fam = row.get("perturbation")
    if not fam:
        return None
    from roborepair.simulation.base import Perturbation

    return Perturbation(family=fam)


def run_condition(
    ctx,
    policy,
    ns,
    arm,
    bank,
    tier,
    K,
    wave,
    donor_feedback=None,
    donor_of=None,
    seed_salt="",
    static_text=None,
):
    """Run one arm over the bank for K wave-batched attempts; return episode states."""
    from roborepair.feedback.constraints import compile_constraints

    dec = ctx.config["decoding"]
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    envs = {t: make_env(t, env_cfg) for t in sorted({r["task"] for r in bank})}
    eps = {r["episode_id"]: EpisodeState(r) for r in bank}
    order = [r["episode_id"] for r in bank]
    for e in eps.values():
        c0 = compile_constraints(e.last_trace, episode_id=e.eid, attempt=0, tier=tier)
        e.constraints.append(c0)
        e.labels.append(c0["primary_failure"])
    t_start = time.monotonic()
    for attempt in range(1, K + 1):
        alive = [eid for eid in order if not eps[eid].done]
        if not alive:
            break
        if arm == "reflection_regeneration":
            for w0 in range(0, len(alive), wave):
                chunk = alive[w0 : w0 + wave]
                prompts = []
                for eid in chunk:
                    e = eps[eid]
                    prompts.append(
                        f"# Task\n{e.row['instruction']}\n# Your previous program\n{e.last_code}\n# Raw execution log (truncated)\n{' '.join(e.last_raw_log.split()[:400])}\n# In at most 6 sentences, reflect on why the program failed the task and what to do differently next time. Output only the reflection."
                    )
                gens = policy.generate_batch(
                    prompts,
                    max_new_tokens=256,
                    temperature=float(dec["temperature"]),
                    top_p=float(dec["top_p"]),
                    seed=paired_generation_seed(
                        ns["gen"], f"{seed_salt}{arm}-reflect-a{attempt}-w{w0 // wave}"
                    ),
                )
                for eid, g in zip(chunk, gens, strict=False):
                    eps[eid].reflections.append(" ".join(g.text.split()[:120]))
                    eps[eid].tokens += g.n_completion_tokens
        for w0 in range(0, len(alive), wave):
            chunk = alive[w0 : w0 + wave]
            prompts = [
                build_prompt(
                    arm,
                    eps[eid],
                    ctx,
                    envs[eps[eid].row["task"]].api_doc,
                    tier,
                    donor_constraint_history=(donor_feedback or {}).get((donor_of or {}).get(eid)),
                    static_text=static_text,
                )
                for eid in chunk
            ]
            gens = policy.generate_batch(
                prompts,
                max_new_tokens=int(dec["max_output_tokens"]),
                temperature=float(dec["temperature"]),
                top_p=float(dec["top_p"]),
                seed=paired_generation_seed(
                    ns["gen"], f"{seed_salt}{arm}-a{attempt}-w{w0 // wave}"
                ),
            )
            for eid, g in zip(chunk, gens, strict=False):
                e = eps[eid]
                code = extract_code(g.text)
                env = envs[e.row["task"]]
                env.reset(
                    sim_seed=e.row["reset_seed"], perturbation=perturbation_from_record(e.row)
                )
                res = run_and_trace(
                    env,
                    e.row["task"],
                    code,
                    sim_seed=e.row["reset_seed"],
                    timeout_s=float(dec["code_timeout_s"]),
                )
                e.n_attempts += 1
                e.tokens += g.n_completion_tokens
                t = res["trace"]
                row = {
                    "arm": arm,
                    "episode_id": eid,
                    "attempt": attempt,
                    "task": e.row["task"],
                    "code_hash": program_hash(code),
                    "infra_excluded": res["infra_excluded"],
                    "n_completion_tokens": g.n_completion_tokens,
                }
                if t is not None:
                    td = t.to_dict()
                    ok_valid = t.compile_ok and t.api_ok and (not t.timed_out)
                    e.valid_attempts += ok_valid
                    e.safety = e.safety or t.metrics.get("off_table", 0.0) >= 1.0
                    row.update(
                        {
                            "success": t.task_success,
                            "valid": ok_valid,
                            "progress": t.progress,
                            "primary_label": t.failure_labels[0] if t.failure_labels else None,
                            "prior_labels": list(dict.fromkeys(e.labels)),
                        }
                    )
                    if t.task_success:
                        e.done = True
                        e.first_successful_attempt = attempt
                    else:
                        c = compile_constraints(td, episode_id=eid, attempt=attempt, tier=tier)
                        e.constraints.append(c)
                        e.labels.append(c["primary_failure"])
                        e.last_trace = td
                        e.last_code = code
                        e.last_raw_log = res["raw_log"]
                e.attempt_rows.append(row)
        LOGGER.info(
            "%s attempt %d: alive %d -> %d (%.1f min)",
            arm,
            attempt,
            len(alive),
            sum(1 for e in eps.values() if not e.done),
            (time.monotonic() - t_start) / 60,
        )
    for env in envs.values():
        env.close()
    return eps


def recovery_at_retry_limit(eps, k):
    """Return (pass@k mean, per-episode 0/1 success vector) for episode states."""
    vals = [
        1.0 if e.first_successful_attempt is not None and e.first_successful_attempt <= k else 0.0
        for e in eps
    ]
    return (float(np.mean(vals)), np.array(vals))
