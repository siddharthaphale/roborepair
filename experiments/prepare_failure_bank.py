"""Generate a failure bank or select pilot failures from completed task evaluations."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from roborepair.execution.environment_checks import audit_environment_correctness
from roborepair.execution.rollouts import (
    build_policy,
    evaluate_criterion,
    extract_code,
    load_evaluation_cell,
    overall_status,
    paired_generation_seed,
    program_hash,
    run_and_trace,
)
from roborepair.execution.run_context import (
    bootstrap,
    build_arg_parser,
    load_config,
    write_jsonl,
    write_metrics,
)
from roborepair.sampling_identity import sampling_identity
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.input_preparation")


def generate_failure_bank(ctx, policy, ns, tier: str):
    """Generate + execute wave-batched initial programs; return (rows, gen_seconds, gen_tokens)."""
    from roborepair.prompts.builder import build_initial_prompt

    dec = ctx.config["decoding"]
    g = ctx.config["generation"]
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    wave = int(g.get("wave_size", 16))
    rows = []
    gen_seconds = 0.0
    gen_tokens = 0
    for task_id in ctx.config["env"]["tasks"]:
        env = make_env(task_id, env_cfg)
        seeds = [int(g["sim_seed_start"]) + i for i in range(int(g["sim_seeds_per_task"]))]
        episodes = [(s, gidx) for s in seeds for gidx in range(int(g["gen_seeds_per_sim_seed"]))]
        obs = env.reset(sim_seed=seeds[0])
        prompt = build_initial_prompt(obs, env.api_doc)
        for w0 in range(0, len(episodes), wave):
            chunk = episodes[w0 : w0 + wave]
            t0 = time.monotonic()
            gens = policy.generate_batch(
                [prompt] * len(chunk),
                max_new_tokens=int(dec["max_output_tokens"]),
                temperature=float(dec["temperature"]),
                top_p=float(dec["top_p"]),
                seed=paired_generation_seed(ns["gen"], f"{tier}-{task_id}-wave{w0 // wave}"),
            )
            gen_seconds += time.monotonic() - t0
            gen_tokens += sum(x.n_completion_tokens for x in gens)
            for (sim_seed, gidx), gen in zip(chunk, gens, strict=False):
                code = extract_code(gen.text)
                obs_i = env.reset(sim_seed=sim_seed)
                res = run_and_trace(
                    env, task_id, code, sim_seed=sim_seed, timeout_s=float(dec["code_timeout_s"])
                )
                row = {
                    "episode_id": sampling_identity(
                        f"failure_bank_episode-{tier}-{task_id}-sim{sim_seed}-g{gidx}"
                    ),
                    "tier": tier,
                    "task": task_id,
                    "instruction": obs_i.instruction,
                    "object_names": list(obs_i.object_names),
                    "reset_seed": sim_seed,
                    "init_code": code,
                    "program_hash": program_hash(code),
                    "gen_seed": 0,
                    "sim_version": ctx.config["env"]["sim_version"],
                    "infra_excluded": res["infra_excluded"],
                    "raw_log": res["raw_log"],
                }
                if res["trace"] is not None:
                    t = res["trace"]
                    row.update(
                        {
                            "init_trace": t.to_dict(),
                            "init_success": t.task_success,
                            "init_progress": t.progress,
                            "failure_labels": t.failure_labels,
                            "compile_ok": t.compile_ok,
                            "api_ok": t.api_ok,
                            "timed_out": t.timed_out,
                        }
                    )
                rows.append(row)
            LOGGER.info(
                "%s: %d/%d programs (succ %d)",
                task_id,
                len([r for r in rows if r["task"] == task_id]),
                len(episodes),
                sum(1 for r in rows if r["task"] == task_id and r.get("init_success")),
            )
        env.close()
    return (rows, gen_seconds, gen_tokens)


def prepare_failure_bank(ctx) -> None:
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    tier = ctx.config["failure_bank"]["primary_tier"]
    corr = audit_environment_correctness(ctx, tier)
    LOGGER.info(
        "correctness: labels=%.2f det=%s gold=%s",
        corr["fault_label_accuracy"],
        corr["determinism"],
        corr["gold_policy_success"],
    )
    policy = build_policy(ctx.config)
    rows, gen_seconds, gen_tokens = generate_failure_bank(ctx, policy, ns, tier)
    if hasattr(policy, "close"):
        policy.close()
    write_jsonl(ctx.output_dir / "episodes.jsonl", rows)
    ev = [r for r in rows if not r["infra_excluded"]]
    fails = [r for r in ev if not r.get("init_success")]
    succ_rate = sum(r.get("init_success", False) for r in ev) / len(ev) if ev else None
    dist: dict = {}
    for r in fails:
        lab = r["failure_labels"][0] if r.get("failure_labels") else "none"
        dist[lab] = dist.get(lab, 0) + 1
    cats10 = sum(1 for v in dist.values() if v >= 10)
    maxfrac = max(dist.values()) / len(fails) if fails else None
    rng = np.random.default_rng(ctx.seed)
    groups: dict = {}
    for r in fails:
        groups.setdefault(r["program_hash"], []).append(r)
    glist = [groups[h] for h in sorted(groups)]
    rng.shuffle(glist)
    dev, evalb = ([], [])
    for grp in glist:
        if len(dev) + len(grp) <= int(ctx.config["failure_bank"]["dev_max"]):
            dev.extend(grp)
        else:
            evalb.extend(grp)
    dev.sort(key=lambda r: r["episode_id"])
    evalb.sort(key=lambda r: r["episode_id"])
    write_jsonl(ctx.output_dir / "development_failures.jsonl", dev)
    write_jsonl(ctx.output_dir / "evaluation_failures.jsonl", evalb)
    tok_s = gen_tokens / gen_seconds if gen_seconds else None
    mean_tok = gen_tokens / len(rows) if rows else 0
    K = 5
    worst_gens = 6 * len(evalb) * K + len(evalb) * K
    proj_h = worst_gens * mean_tok / tok_s / 3600 if tok_s else None
    ladder = []
    if proj_h is not None and proj_h > 5.0:
        ladder.append("structured_plan_with_editing -> 60-episode subsample")
        proj_h2 = (5 * len(evalb) * K + 60 * K + len(evalb) * K) * mean_tok / tok_s / 3600
        if proj_h2 > 5.0:
            ladder.append("K := 4")
    pc = ctx.config["failure_bank"]["pass_criteria"]
    criteria = [
        evaluate_criterion(
            "fault_label_accuracy", corr["fault_label_accuracy"], pc["fault_label_min"], ">="
        ),
        evaluate_criterion("schema_validity", corr["schema_validity"], 1.0, ">="),
        evaluate_criterion(
            "determinism_success_agreement",
            corr["determinism"]["success_agreement"],
            pc["success_agreement_min"],
            ">=",
        ),
        evaluate_criterion(
            "determinism_label_agreement",
            corr["determinism"]["label_set_agreement"],
            pc["label_agreement_min"],
            ">=",
        ),
        evaluate_criterion(
            "gold_min_per_task", min(corr["gold_policy_success"].values()), pc["gold_min"], ">="
        ),
        evaluate_criterion(
            "initial_success_band", succ_rate, (pc["success_band"][0], pc["success_band"][1]), "in"
        ),
        evaluate_criterion("eval_failures", len(evalb), pc["eval_min"], ">="),
        evaluate_criterion("categories_ge_10", cats10, 3, ">="),
        evaluate_criterion("max_category_fraction", maxfrac, 0.7, "<="),
        evaluate_criterion(
            "infra_rate", (len(rows) - len(ev)) / len(rows) if rows else None, 0.02, "<"
        ),
    ]
    metrics = {
        "stage": "manipulation_failure_bank",
        "tier": tier,
        "environment_correctness": corr,
        "bank": {
            "n": len(rows),
            "initial_success": succ_rate,
            "failures": len(fails),
            "distribution": dict(sorted(dist.items())),
            "dev_n": len(dev),
            "eval_n": len(evalb),
        },
        "throughput": {
            "batched_tok_s": tok_s,
            "mean_completion_tokens": mean_tok,
            "projected_feedback_ablation_hours_worst": proj_h,
            "fallback_ladder_applied": ladder,
        },
        "criteria": criteria,
        "status": overall_status(criteria),
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "manipulation failure-bank build %s | success=%.3f eval=%d tok/s=%.0f projected_feedback_ablation_hours=%.1f ladder=%s",
        metrics["status"],
        succ_rate,
        len(evalb),
        tok_s or 0,
        proj_h or -1,
        ladder,
    )


def prepare_pilot_bank(ctx) -> None:
    source_run_dir = Path(ctx.config["pilot_bank"]["source_run_dir"])
    cells = ctx.config["pilot_bank"]["cells"]
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell, cap in cells.items():
        bank, _stored = load_evaluation_cell(source_run_dir, cell, cap)
        rows.extend(bank)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path = output_dir / "pilot_failures.jsonl"
    path.write_text(text)
    manifest = {
        "source": str(source_run_dir),
        "cells": cells,
        "n": len(rows),
        "n_per_cell": {c: sum(1 for r in rows if r["cell"] == c) for c in cells},
        "tasks": sorted({r["task"] for r in rows}),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    (output_dir / "pilot_failures.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if ("failure_bank" in config) == ("pilot_bank" in config):
        parser.error("Specify either failure_bank or pilot_bank in the configuration.")
    context = bootstrap(args)
    if "pilot_bank" in config:
        prepare_pilot_bank(context)
    else:
        prepare_failure_bank(context)


if __name__ == "__main__":
    main()
