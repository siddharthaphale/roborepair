"""Frozen-bank feedback comparison protocol."""

from __future__ import annotations

import hashlib
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

LOGGER = logging.getLogger("roborepair.feedback_controls")
STUDY_ID = "feedback_controls"
REFLECTION_ARMS = ("editing_with_reflection", "reflection_regeneration_control")
DIRECT_ARMS = (
    "editing_with_execution_log",
    "editing_with_failure_label",
    "editing_with_template_feedback",
    "editing_with_reflection",
)
STATIC_ARMS = (
    "compatible_template_union",
    "fixed_task_template",
    "three_task_templates",
    "generic_instructions",
    "static_guidance_from_start",
)


def load_failure_bank_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observation_from_record(row):
    return SimpleNamespace(instruction=row["instruction"], object_names=row.get("object_names"))


def progress_bucket(progress: float) -> int:
    return min(3, int(progress * 4))


def feedback_text_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EpisodeState:
    """Mutable per-episode retry state."""

    def __init__(self, row):
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
        self.prompt_tokens = 0
        self.donor_constraint_sets: list = []
        self.donor_episode_ids: list = []
        self.different_label_matches: list = []
        self.cur: dict = {}


def extend_different_label_feedback(ep, donor_feedback, ns):
    """Ensure ep.donor_constraint_sets has one donor set per known own label (strict different-label).

    Selection per index j: same task, same initial-progress bucket, donor has a
    constraint set at index j, donor label != own label at j whenever any candidate
    exists (bucket relaxed before the different-label requirement is dropped).
    Deterministic via derive_seed(ns_gen, 's6', eid, j). Frozen once chosen.
    """
    from roborepair.seeding import derive_seed

    row = ep.row
    task, bucket = (row["task"], progress_bucket(row["init_progress"]))
    while len(ep.donor_constraint_sets) < len(ep.labels):
        j = len(ep.donor_constraint_sets)
        own = ep.labels[j]
        pool = [
            (deid, d["sets"][j])
            for deid, d in donor_feedback.items()
            if deid != ep.eid and d["task"] == task and (len(d["sets"]) > j)
        ]
        same_bucket = [(e, s) for e, s in pool if donor_feedback[e]["bucket"] == bucket] or pool
        diff = [(e, s) for e, s in same_bucket if s["primary_failure"] != own]
        if not diff:
            diff = [(e, s) for e, s in pool if s["primary_failure"] != own]
        cands = diff or same_bucket or pool
        if not cands:
            if not ep.donor_constraint_sets:
                raise RuntimeError(
                    f"different_label_feedback: empty donor pool at index 0 for {ep.eid}"
                )
            ep.donor_constraint_sets.append(ep.donor_constraint_sets[-1])
            ep.donor_episode_ids.append(ep.donor_episode_ids[-1])
            ep.different_label_matches.append(False)
            continue
        rng = np.random.default_rng(derive_seed(ns["gen"], "different_label_donor", ep.eid, j))
        deid, dset = cands[int(rng.integers(0, len(cands)))]
        ep.donor_constraint_sets.append(dset)
        ep.donor_episode_ids.append(deid)
        ep.different_label_matches.append(dset["primary_failure"] != own)


def build_prompt(
    arm, ep, ctx, api_doc, tier, *, ns, statics, label_permutation, compat, donor_feedback
):
    """Return (prompt, provenance-meta) for one episode's next attempt."""
    from roborepair.feedback.constraints import compile_constraints, render_constraints_text
    from roborepair.prompts.builder import build_feedback_prompt
    from roborepair.seeding import derive_seed

    cap = int(ctx.config["decoding"]["trace_input_token_cap"])
    obs = observation_from_record(ep.row)
    task = ep.row["task"]
    suffix = ctx.config["experiment"].get("suffix", "historical")
    meta = {
        "previous_code_in_prompt": False,
        "raw_log_in_prompt": False,
        "reflection_in_prompt": False,
        "compiled_constraints_in_prompt": False,
        "feedback_source_episode": ep.eid,
        "feedback_labels": [],
        "suffix": suffix,
    }

    def render_prompt_with_metadata(plan_text, mode="regenerate", prev_code=None):
        meta["feedback_text_hash"] = feedback_text_hash(plan_text)
        return (
            build_feedback_prompt(
                plan_text=plan_text,
                mode=mode,
                prev_code=prev_code,
                api_doc=api_doc,
                env_obs=obs,
                token_budget=cap,
                suffix=suffix,
            ),
            meta,
        )

    def previous_execution_log_text():
        head = " ".join(ep.last_raw_log.split()[:cap])
        return "raw execution log of the previous failed attempt:\n" + head

    if arm == "resample":
        meta["feedback_source_episode"] = None
        return render_prompt_with_metadata(None)
    if arm == "template_feedback":
        meta["compiled_constraints_in_prompt"] = True
        meta["feedback_labels"] = list(ep.labels)
        return render_prompt_with_metadata(render_constraints_text(ep.constraints, cap_tokens=cap))
    if arm == "execution_log_regeneration":
        meta["raw_log_in_prompt"] = True
        meta["feedback_labels"] = [ep.labels[-1]]
        return render_prompt_with_metadata(previous_execution_log_text())
    if arm == "editing_with_execution_log":
        meta.update(previous_code_in_prompt=True, raw_log_in_prompt=True)
        meta["feedback_labels"] = [ep.labels[-1]]
        return render_prompt_with_metadata(
            previous_execution_log_text(), mode="patch", prev_code=ep.last_code
        )
    if arm == "editing_with_failure_label":
        meta["previous_code_in_prompt"] = True
        lab = ep.labels[-1]
        diag = (ep.last_trace or {}).get("diagnosis", "")
        meta["feedback_labels"] = [lab]
        return render_prompt_with_metadata(
            f"previous attempt failed: {lab} - {diag}", mode="patch", prev_code=ep.last_code
        )
    if arm == "editing_with_template_feedback":
        meta.update(previous_code_in_prompt=True, compiled_constraints_in_prompt=True)
        meta["feedback_labels"] = list(ep.labels)
        return render_prompt_with_metadata(
            render_constraints_text(ep.constraints, cap_tokens=cap),
            mode="patch",
            prev_code=ep.last_code,
        )
    if arm == "editing_with_reflection":
        meta.update(previous_code_in_prompt=True, raw_log_in_prompt=True, reflection_in_prompt=True)
        meta["feedback_labels"] = [ep.labels[-1]]
        refl = ep.reflections[-1] if ep.reflections else "(no reflection available)"
        loghead = " ".join(ep.last_raw_log.split()[: max(64, cap - 160)])
        plan = (
            f"Reflection on the previous failed attempt: {refl}\nraw execution log of the previous failed attempt:\n"
            + loghead
        )
        return render_prompt_with_metadata(plan, mode="patch", prev_code=ep.last_code)
    if arm == "reflection_regeneration_control":
        meta["reflection_in_prompt"] = True
        text = "\n".join(
            (f"Reflection after attempt {i + 1}: {t}" for i, t in enumerate(ep.reflections))
        )
        return render_prompt_with_metadata(text or "(no reflections yet)")
    if arm in STATIC_ARMS:
        meta["feedback_source_episode"] = "static"
        return render_prompt_with_metadata(statics[arm][task])
    if arm == "initial_failure_feedback":
        meta["compiled_constraints_in_prompt"] = True
        meta["feedback_labels"] = [ep.labels[0]]
        return render_prompt_with_metadata(
            render_constraints_text(ep.constraints[:1], cap_tokens=cap)
        )
    if arm == "latest_failure_feedback":
        meta["compiled_constraints_in_prompt"] = True
        meta["feedback_labels"] = [ep.labels[-1]]
        return render_prompt_with_metadata(
            render_constraints_text(ep.constraints[-1:], cap_tokens=cap)
        )
    if arm == "different_label_feedback":
        extend_different_label_feedback(ep, donor_feedback, ns)
        meta["compiled_constraints_in_prompt"] = True
        meta["feedback_source_episode"] = list(ep.donor_episode_ids)
        meta["feedback_labels"] = [s["primary_failure"] for s in ep.donor_constraint_sets]
        return render_prompt_with_metadata(
            render_constraints_text(ep.donor_constraint_sets, cap_tokens=cap)
        )
    if arm == "permuted_label_feedback":
        meta["compiled_constraints_in_prompt"] = True
        sets = [
            compile_constraints(
                {"failure_labels": [label_permutation.get(lab, lab)], "task_id": task},
                episode_id=ep.eid,
                attempt=i,
                tier=tier,
            )
            for i, lab in enumerate(ep.labels)
        ]
        meta["feedback_labels"] = [s["primary_failure"] for s in sets]
        return render_prompt_with_metadata(render_constraints_text(sets, cap_tokens=cap))
    if arm == "random_compatible_feedback":
        meta["compiled_constraints_in_prompt"] = True
        attempt = len(ep.labels)
        rng = np.random.default_rng(
            derive_seed(ns["perturb"], "random_compatible_label", ep.eid, attempt)
        )
        lab = compat[task][int(rng.integers(0, len(compat[task])))]
        s = compile_constraints(
            {"failure_labels": [lab], "task_id": task},
            episode_id=ep.eid,
            attempt=attempt - 1,
            tier=tier,
        )
        meta["feedback_labels"] = [lab]
        return render_prompt_with_metadata(render_constraints_text([s], cap_tokens=cap))
    if arm == "repeated_api_rules":
        meta["feedback_source_episode"] = "static"
        return render_prompt_with_metadata(
            "Execution rules:" + api_doc.split("\nExecution rules:", 1)[1]
        )
    raise ValueError(arm)


def perturbation_from_record(row):
    fam = row.get("perturbation")
    if not fam:
        return None
    from roborepair.simulation.base import Perturbation

    return Perturbation(family=fam)


def generate_batch_with_timing(policy, prompts, **kw):
    """generate_batch wrapped with wall + GPU timing; returns (gens, wall_s, gpu_s)."""
    import torch

    cuda = torch.cuda.is_available() and (not getattr(policy, "is_mock", False))
    if cuda:
        torch.cuda.synchronize()
        ev0, ev1 = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        ev0.record()
    t0 = time.monotonic()
    gens = policy.generate_batch(prompts, **kw)
    if cuda:
        ev1.record()
        torch.cuda.synchronize()
        gpu_s = ev0.elapsed_time(ev1) / 1000.0
    else:
        gpu_s = 0.0
    return (gens, time.monotonic() - t0, gpu_s)


def token_proportional_shares(gens):
    tot = sum(g.n_prompt_tokens + g.n_completion_tokens for g in gens) or 1
    return [(g.n_prompt_tokens + g.n_completion_tokens) / tot for g in gens]


def run_condition(
    ctx, policy, ns, arm, bank, tier, K, wave, *, statics, label_permutation, compat, donor_feedback
):
    """Run one arm over the bank for K wave-batched attempts; return episode states."""
    from roborepair.evaluation.program_structure import normalized_ast_hash
    from roborepair.feedback.constraints import compile_constraints

    dec = ctx.config["decoding"]
    model_cfg = ctx.config.get("model", {})
    exp_id = ctx.config["experiment"]["experiment_id"]
    study_id = ctx.config["experiment"].get("study_id", STUDY_ID)
    log_prompts = bool(ctx.config["experiment"].get("log_prompt_text", False))
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
        for e in eps.values():
            e.cur = {}
        if arm in REFLECTION_ARMS:
            for w0 in range(0, len(alive), wave):
                chunk = alive[w0 : w0 + wave]
                prompts = [
                    f"# Task\n{eps[eid].row['instruction']}\n# Your previous program\n{eps[eid].last_code}\n# Raw execution log (truncated)\n{' '.join(eps[eid].last_raw_log.split()[:400])}\n# In at most 6 sentences, reflect on why the program failed the task and what to do differently next time. Output only the reflection."
                    for eid in chunk
                ]
                gens, wall, gpu = generate_batch_with_timing(
                    policy,
                    prompts,
                    max_new_tokens=256,
                    temperature=float(dec["temperature"]),
                    top_p=float(dec["top_p"]),
                    seed=paired_generation_seed(
                        ns["gen"], f"{arm}-reflect-a{attempt}-w{w0 // wave}"
                    ),
                )
                for eid, g, sh in zip(chunk, gens, token_proportional_shares(gens), strict=False):
                    e = eps[eid]
                    e.reflections.append(" ".join(g.text.split()[:120]))
                    e.tokens += g.n_completion_tokens
                    e.prompt_tokens += g.n_prompt_tokens
                    e.cur = {
                        "reflection_prompt_tokens": g.n_prompt_tokens,
                        "reflection_completion_tokens": g.n_completion_tokens,
                        "reflection_model_wall_seconds": wall * sh,
                        "reflection_gpu_seconds": gpu * sh,
                    }
        for w0 in range(0, len(alive), wave):
            chunk = alive[w0 : w0 + wave]
            built = [
                build_prompt(
                    arm,
                    eps[eid],
                    ctx,
                    envs[eps[eid].row["task"]].api_doc,
                    tier,
                    ns=ns,
                    statics=statics,
                    label_permutation=label_permutation,
                    compat=compat,
                    donor_feedback=donor_feedback,
                )
                for eid in chunk
            ]
            prompts = [p for p, _ in built]
            metas = [m for _, m in built]
            wave_seed = paired_generation_seed(ns["gen"], f"{arm}-a{attempt}-w{w0 // wave}")
            gens, wall, gpu = generate_batch_with_timing(
                policy,
                prompts,
                max_new_tokens=int(dec["max_output_tokens"]),
                temperature=float(dec["temperature"]),
                top_p=float(dec["top_p"]),
                seed=wave_seed,
            )
            shares = token_proportional_shares(gens)
            for eid, prompt, g, m, sh in zip(chunk, prompts, gens, metas, shares, strict=False):
                e = eps[eid]
                code = extract_code(g.text)
                env = envs[e.row["task"]]
                env.reset(
                    sim_seed=e.row["reset_seed"], perturbation=perturbation_from_record(e.row)
                )
                t_exec = time.monotonic()
                res = run_and_trace(
                    env,
                    e.row["task"],
                    code,
                    sim_seed=e.row["reset_seed"],
                    timeout_s=float(dec["code_timeout_s"]),
                )
                exec_wall = time.monotonic() - t_exec
                e.n_attempts += 1
                e.tokens += g.n_completion_tokens
                e.prompt_tokens += g.n_prompt_tokens
                t = res["trace"]
                row = {
                    "study_id": study_id,
                    "experiment_id": exp_id,
                    "condition_id": arm,
                    "arm": arm,
                    "model_id": model_cfg.get("name"),
                    "model_revision": model_cfg.get("revision"),
                    "task_id": e.row["task"],
                    "api_tier": tier,
                    "environment_seed": e.row["reset_seed"],
                    "generation_seed": wave_seed,
                    "bank_row_id": eid,
                    "episode_id": eid,
                    "cell": e.row.get("cell"),
                    "attempt": attempt,
                    "attempt_index": attempt,
                    "task": e.row["task"],
                    "initial_program_hash": e.row["program_hash"],
                    "code": code,
                    "code_hash": program_hash(code),
                    "program_hash": program_hash(code),
                    "normalized_ast_hash": normalized_ast_hash(code),
                    "prompt_tokens": g.n_prompt_tokens,
                    "completion_tokens": g.n_completion_tokens,
                    "n_completion_tokens": g.n_completion_tokens,
                    "model_wall_seconds": wall * sh,
                    "gpu_seconds": gpu * sh,
                    "execution_wall_seconds": exec_wall,
                    "reflection_prompt_tokens": e.cur.get("reflection_prompt_tokens", 0),
                    "reflection_completion_tokens": e.cur.get("reflection_completion_tokens", 0),
                    "reflection_model_wall_seconds": e.cur.get(
                        "reflection_model_wall_seconds", 0.0
                    ),
                    "reflection_gpu_seconds": e.cur.get("reflection_gpu_seconds", 0.0),
                    "infra_excluded": res["infra_excluded"],
                    **m,
                }
                if log_prompts:
                    row["prompt_text"] = prompt
                if t is not None:
                    td = t.to_dict()
                    ok_valid = t.compile_ok and t.api_ok and (not t.timed_out)
                    e.valid_attempts += ok_valid
                    e.safety = e.safety or t.metrics.get("off_table", 0.0) >= 1.0
                    row.update(
                        {
                            "success": t.task_success,
                            "valid": ok_valid,
                            "valid_program": ok_valid,
                            "progress": t.progress,
                            "primary_label": t.failure_labels[0] if t.failure_labels else None,
                            "primary_failure_label": t.failure_labels[0]
                            if t.failure_labels
                            else None,
                            "all_failure_labels": list(t.failure_labels),
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
    vals = [
        1.0 if e.first_successful_attempt is not None and e.first_successful_attempt <= k else 0.0
        for e in eps
    ]
    return (float(np.mean(vals)), np.array(vals))


class CompletedCondition:
    """Minimal per-episode view reconstructed from a finished arm's attempt rows."""

    def __init__(self, eid, rows):
        self.eid = eid
        self.attempt_rows = rows
        succ = [r["attempt"] for r in rows if r.get("success")]
        self.first_successful_attempt = min(succ) if succ else None
        self.n_attempts = len(rows)
        self.valid_attempts = sum(1 for r in rows if r.get("valid"))
        self.safety = False
        self.tokens = sum(
            r.get("n_completion_tokens", 0) + r.get("reflection_completion_tokens", 0) for r in rows
        )
        self.prompt_tokens = sum(
            r.get("prompt_tokens", 0) + r.get("reflection_prompt_tokens", 0) for r in rows
        )


def load_completed_condition(path: Path, bank):
    rows_by = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows_by.setdefault(r["episode_id"], []).append(r)
    return {
        b["episode_id"]: CompletedCondition(b["episode_id"], rows_by.get(b["episode_id"], []))
        for b in bank
    }


def wait_for_artifact(path: Path, timeout_s: float = 7200.0) -> None:
    t0 = time.monotonic()
    while not path.exists():
        if time.monotonic() - t0 > timeout_s:
            raise SystemExit(f"timed out waiting for {path}")
        LOGGER.info("waiting for %s ...", path)
        time.sleep(60)
