"""Evaluate retry recovery across tasks, model sizes, and perturbations."""

from __future__ import annotations

import logging
import time

import numpy as np

from roborepair.execution.cell_retry import recovery_at_retry_limit, run_condition
from roborepair.execution.rollouts import (
    build_policy,
    extract_code,
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
from roborepair.execution.task_evaluation import assign_feedback_donors, compute_cell_contrast
from roborepair.sampling_identity import sampling_identity
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.task_evaluation")


def build_cell_bank(ctx, policy, ns, cell, tier: str):
    """Wave-batched initial-program generation for one cell; returns all rows."""
    from roborepair.prompts.builder import build_initial_prompt
    from roborepair.simulation.base import Perturbation

    dec = ctx.config["decoding"]
    g = ctx.config["task_evaluation"]
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    task, fam = (cell["task"], cell.get("perturbation"))
    pert = Perturbation(family=fam) if fam else None
    env = make_env(task, env_cfg)
    wave = int(g["wave_size"])
    seeds = [int(cell["sim_seed_start"]) + i for i in range(int(g["programs_per_cell"]))]
    obs0 = env.reset(sim_seed=seeds[0], perturbation=pert)
    prompt = build_initial_prompt(obs0, env.api_doc)
    rows = []
    for w0 in range(0, len(seeds), wave):
        chunk = seeds[w0 : w0 + wave]
        gens = policy.generate_batch(
            [prompt] * len(chunk),
            max_new_tokens=int(dec["max_output_tokens"]),
            temperature=float(dec["temperature"]),
            top_p=float(dec["top_p"]),
            seed=paired_generation_seed(ns["gen"], f"{cell['id']}-bank-wave{w0 // wave}"),
        )
        for sim_seed, gen in zip(chunk, gens, strict=False):
            code = extract_code(gen.text)
            obs_i = env.reset(sim_seed=sim_seed, perturbation=pert)
            res = run_and_trace(
                env, task, code, sim_seed=sim_seed, timeout_s=float(dec["code_timeout_s"])
            )
            row = {
                "episode_id": sampling_identity(
                    f"task_evaluation_episode-{cell['id']}-sim{sim_seed}"
                ),
                "cell": cell["id"],
                "tier": tier,
                "task": task,
                "perturbation": fam,
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
                    }
                )
            rows.append(row)
    env.close()
    return rows


def classify_initial_success_rate(succ_rate, band_conf, band_rep):
    """Classify a cell's initial success rate: confirmatory, reported, or dropped."""
    if succ_rate is None:
        return "dropped"
    if band_conf[0] <= succ_rate <= band_conf[1]:
        return "confirmatory"
    if band_rep[0] <= succ_rate <= band_rep[1]:
        return "reported"
    return "dropped"


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("task_evaluation"), dict):
        parser.error("This experiment requires a task_evaluation configuration.")
    ctx = bootstrap(args)
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    g = ctx.config["task_evaluation"]
    K, wave, tier = (int(g["max_retries"]), int(g["wave_size"]), g["tier"])
    band_conf = tuple(g["band_confirmatory"])
    band_rep = tuple(g["band_reported"])
    policy = build_policy(ctx.config)
    cells_out = {}
    t_run = time.monotonic()
    for cell in g["cells"]:
        cid = cell["id"]
        t0 = time.monotonic()
        rows = build_cell_bank(ctx, policy, ns, cell, tier)
        write_jsonl(ctx.output_dir / "bank.jsonl", rows)
        ev = [r for r in rows if not r["infra_excluded"]]
        fails = [r for r in ev if not r.get("init_success")][: int(g["eval_cap"])]
        succ_rate = sum(r.get("init_success", False) for r in ev) / len(ev) if ev else None
        band = classify_initial_success_rate(succ_rate, band_conf, band_rep)
        dist: dict = {}
        for r in fails:
            lab = r["failure_labels"][0] if r.get("failure_labels") else "none"
            dist[lab] = dist.get(lab, 0) + 1
        info = {
            "cell": cid,
            "task": cell["task"],
            "perturbation": cell.get("perturbation"),
            "n_programs": len(rows),
            "n_infra": len(rows) - len(ev),
            "initial_success": succ_rate,
            "initial_success_band": band,
            "n_eval_failures": len(fails),
            "label_distribution": dict(sorted(dist.items())),
            "min_eval_ok": len(fails) >= int(g["min_eval_failures"]),
        }
        LOGGER.info(
            "cell %s: success=%.3f band=%s failures=%d dist=%s",
            cid,
            succ_rate if succ_rate is not None else -1,
            band,
            len(fails),
            dist,
        )
        if band == "dropped" or not info["min_eval_ok"]:
            info["arms"] = None
            cells_out[cid] = info
            continue
        donor_of, donor_kind = assign_feedback_donors(fails, ctx.seed)
        info["donor_audit"] = {
            "different_label": sum(1 for v in donor_kind.values() if v == "different_label"),
            "same_label": sum(1 for v in donor_kind.values() if v == "same_label"),
            "self_no_pool": sum(1 for v in donor_kind.values() if v == "self_no_pool"),
        }
        results = {}
        donor_feedback = None
        for arm in cell["arms"]:
            eps = run_condition(
                ctx,
                policy,
                ns,
                arm,
                fails,
                tier,
                K,
                wave,
                donor_feedback=donor_feedback,
                donor_of=donor_of if arm == "donor_feedback" else None,
                seed_salt=f"{cid}-",
            )
            results[arm] = eps
            if arm == "template_feedback":
                donor_feedback = {eid: e.constraints for eid, e in eps.items()}
            write_jsonl(
                ctx.output_dir / "attempts.jsonl",
                [dict(r, cell=cid) for e in eps.values() for r in e.attempt_rows],
            )
            LOGGER.info(
                "cell %s arm %s pass@%d = %.3f",
                cid,
                arm,
                K,
                recovery_at_retry_limit(list(eps.values()), K)[0],
            )
        ids = [r["episode_id"] for r in fails]
        info["per_arm"] = {}
        for arm, eps in results.items():
            elist = [eps[i] for i in ids if i in eps]
            info["per_arm"][arm] = {
                "pass_at_k": {k: recovery_at_retry_limit(elist, k)[0] for k in range(1, K + 1)},
                "attempt_validity": float(
                    np.mean([e.valid_attempts / e.n_attempts for e in elist if e.n_attempts])
                ),
                "safety_any": float(np.mean([e.safety for e in elist])),
                "mean_tokens_per_episode": float(np.mean([e.tokens for e in elist])),
                "first_successful_attempt": {e.eid: e.first_successful_attempt for e in elist},
            }
        info["contrasts"] = {
            "template_feedback_minus_resample_template_feedback_minus_resample": compute_cell_contrast(
                results, "template_feedback", "resample", ids, K
            ),
            "template_feedback_minus_donor_feedback_template_feedback_minus_donor_feedback": compute_cell_contrast(
                results, "template_feedback", "donor_feedback", ids, K
            ),
        }
        if "reflection_regeneration" in results:
            info["contrasts"][
                "template_feedback_minus_reflection_template_feedback_minus_reflection_regeneration"
            ] = compute_cell_contrast(
                results, "template_feedback", "reflection_regeneration", ids, K
            )
        info["wall_clock_min"] = (time.monotonic() - t0) / 60
        cells_out[cid] = info
    if hasattr(policy, "close"):
        policy.close()
    metrics = {
        "stage": ctx.config["task_evaluation"]["stage_name"],
        "model": ctx.config["model"]["name"],
        "tier": tier,
        "max_retries": K,
        "cells": cells_out,
        "total_wall_clock_hours": (time.monotonic() - t_run) / 3600,
        "criteria": [],
        "status": "DONE",
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "task and model evaluation cells run complete: %s",
        {
            c: (
                v["initial_success_band"],
                (v.get("contrasts") or {}).get(
                    "template_feedback_minus_resample_template_feedback_minus_resample", {}
                )
                or {},
            )
            for c, v in cells_out.items()
        },
    )


if __name__ == "__main__":
    main()
