"""Repeat task-cell comparisons with different seeds or sampling temperatures."""

from __future__ import annotations

import logging
from pathlib import Path

from roborepair.execution.cell_retry import recovery_at_retry_limit, run_condition
from roborepair.execution.rollouts import (
    build_policy,
    load_evaluation_cell,
)
from roborepair.execution.run_context import (
    bootstrap,
    build_arg_parser,
    load_config,
    write_jsonl,
    write_metrics,
)
from roborepair.execution.task_evaluation import (
    _StoredEpisodeOutcomes,
    assign_feedback_donors,
    compute_cell_contrast,
)

LOGGER = logging.getLogger("roborepair.task_evaluation")


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("replication"), dict):
        parser.error("This experiment requires a replication configuration.")
    ctx = bootstrap(args)
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    g = ctx.config["replication"]
    K, wave, tier = (int(g["max_retries"]), int(g["wave_size"]), g["tier"])
    run_dir = Path(g["task_evaluation_run_dir"])
    policy = build_policy(ctx.config)
    out_cells = {}
    for c in g["cells"]:
        cid = c["id"]
        bank, stored = load_evaluation_cell(run_dir, cid, int(g["eval_cap"]))
        donor_of = None
        if "donor_feedback" in c["arms"]:
            donor_of, donor_kind = assign_feedback_donors(bank, ctx.seed)
        results = {}
        donor_feedback = None
        for arm in c["arms"]:
            eps = run_condition(
                ctx,
                policy,
                ns,
                arm,
                bank,
                tier,
                K,
                wave,
                donor_feedback=donor_feedback,
                donor_of=donor_of if arm == "donor_feedback" else None,
                seed_salt=f"{g['stage_name']}-s{ctx.seed}-{cid}-",
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
        ids = [r["episode_id"] for r in bank]
        contrasts = {}
        if "template_feedback" in results and "resample" in results:
            contrasts["template_feedback_minus_resample_rerun"] = compute_cell_contrast(
                results, "template_feedback", "resample", ids, K
            )
        if "template_feedback" in results and "donor_feedback" in results:
            contrasts["template_feedback_minus_donor_feedback_rerun"] = compute_cell_contrast(
                results, "template_feedback", "donor_feedback", ids, K
            )
        out_cells[cid] = {
            "cell": cid,
            "task": c["task"],
            "n": len(bank),
            "cli_seed": ctx.seed,
            "decoding_temperature": float(ctx.config["decoding"]["temperature"]),
            "pass_at_5": {
                arm: recovery_at_retry_limit([results[arm][i] for i in ids], K)[0]
                for arm in results
            },
            "stored_pass_at_5": {
                arm: recovery_at_retry_limit(
                    [
                        _StoredEpisodeOutcomes(stored[arm]["first_successful_attempt"])[i]
                        for i in ids
                    ],
                    K,
                )[0]
                for arm in stored
                if arm in ("resample", "template_feedback", "donor_feedback")
            },
            "contrasts": contrasts,
            "first_successful_attempt": {
                arm: {e.eid: e.first_successful_attempt for e in results[arm].values()}
                for arm in results
            },
        }
    if hasattr(policy, "close"):
        policy.close()
    metrics = {
        "stage": g["stage_name"],
        "model": ctx.config["model"]["name"],
        "tier": tier,
        "max_retries": K,
        "cli_seed": ctx.seed,
        "task_evaluation_run_dir": str(run_dir),
        "cells": out_cells,
        "criteria": [],
        "status": "DONE",
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "%s complete: %s", g["stage_name"], {cid: v["pass_at_5"] for cid, v in out_cells.items()}
    )


if __name__ == "__main__":
    main()
