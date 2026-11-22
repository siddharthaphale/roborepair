"""Compare episode feedback with cross-task feedback and fixed task guidance."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

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
from roborepair.execution.task_evaluation import _StoredEpisodeOutcomes, compute_cell_contrast

LOGGER = logging.getLogger("roborepair.task_evaluation")


def reconstruct_template_feedback(run_dir: Path, cell_id: str, bank, tier: str):
    """Rebuild each episode's accumulated template_feedback constraint sets from task_evaluation artifacts.

    compile_constraints depends only on (primary label, task_id, tier, attempt), so
    the reconstruction is exact: attempt 0 from the banked initial failure, attempts
    1..K from the template_feedback attempt rows (non-success attempts appended a constraint).
    """
    from roborepair.feedback.constraints import compile_constraints

    rows_by_ep: dict = {}
    for line in (run_dir / "attempts.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("cell") == cell_id and r["arm"] == "template_feedback":
            rows_by_ep.setdefault(r["episode_id"], []).append(r)
    store = {}
    for b in bank:
        eid = b["episode_id"]
        sets = [compile_constraints(b["init_trace"], episode_id=eid, attempt=0, tier=tier)]
        for r in sorted(rows_by_ep.get(eid, []), key=lambda x: x["attempt"]):
            if not r.get("success") and r.get("primary_label"):
                sets.append(
                    compile_constraints(
                        {"failure_labels": [r["primary_label"]], "task_id": b["task"]},
                        episode_id=eid,
                        attempt=r["attempt"],
                        tier=tier,
                    )
                )
        store[eid] = sets
    return store


def assign_cross_task_donors(bank_a, bank_b, seed: int):
    """Donor map: each episode of cell A gets a donor from cell B (different task)."""
    rng = np.random.default_rng(seed)
    donor_of = {}
    for r in bank_a:
        bucket = min(3, int(r["init_progress"] * 4))
        lab = r["failure_labels"][0] if r.get("failure_labels") else None
        same_b = [d for d in bank_b if min(3, int(d["init_progress"] * 4)) == bucket] or bank_b
        diff = [
            d
            for d in same_b
            if (d["failure_labels"][0] if d.get("failure_labels") else None) != lab
        ]
        donor_of[r["episode_id"]] = (diff or same_b)[int(rng.integers(0, len(diff or same_b)))][
            "episode_id"
        ]
    return donor_of


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("feedback_granularity"), dict):
        parser.error("This experiment requires a feedback_granularity configuration.")
    ctx = bootstrap(args)
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    g = ctx.config["feedback_granularity"]
    K, wave, tier = (int(g["max_retries"]), int(g["wave_size"]), g["tier"])
    run_dir = Path(g["task_evaluation_run_dir"])
    static_text = Path(g["fixed_guidance_path"]).read_text().strip()
    cells = {}
    banks = {}
    for c in g["cells"]:
        banks[c["id"]], cells[c["id"]] = ({}, {})
        bank, stored = load_evaluation_cell(run_dir, c["id"], int(g["eval_cap"]))
        banks[c["id"]] = {"bank": bank, "stored": stored, "cfg": c}
    policy = build_policy(ctx.config)
    out_cells = {}
    for cid, entry in banks.items():
        bank, stored, ccfg = (entry["bank"], entry["stored"], entry["cfg"])
        donor_cell = ccfg["cross_donor_cell"]
        donor_bank = banks[donor_cell]["bank"]
        donor_feedback_store = reconstruct_template_feedback(run_dir, donor_cell, donor_bank, tier)
        donor_of = assign_cross_task_donors(bank, donor_bank, ctx.seed)
        results = {}
        for arm, kwargs in (
            ("cross_task_feedback", {"donor_feedback": donor_feedback_store, "donor_of": donor_of}),
            ("fixed_task_guidance", {"static_text": static_text}),
        ):
            eps = run_condition(
                ctx, policy, ns, arm, bank, tier, K, wave, seed_salt=f"G-{cid}-", **kwargs
            )
            results[arm] = eps
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

        merged = dict(results)
        for arm in ("resample", "template_feedback", "donor_feedback"):
            merged[arm] = _StoredEpisodeOutcomes(stored[arm]["first_successful_attempt"])
        info = {
            "cell": cid,
            "task": ccfg["task"],
            "n": len(bank),
            "cross_donor_cell": donor_cell,
            "pass_at_5": {
                arm: recovery_at_retry_limit([merged[arm][i] for i in ids], K)[0] for arm in merged
            },
            "contrasts": {
                "template_feedback_minus_cross_task_feedback": compute_cell_contrast(
                    merged, "template_feedback", "cross_task_feedback", ids, K
                ),
                "template_feedback_minus_fixed_task_guidance": compute_cell_contrast(
                    merged, "template_feedback", "fixed_task_guidance", ids, K
                ),
                "fixed_task_guidance_minus_resample": compute_cell_contrast(
                    merged, "fixed_task_guidance", "resample", ids, K
                ),
                "cross_task_feedback_minus_resample": compute_cell_contrast(
                    merged, "cross_task_feedback", "resample", ids, K
                ),
            },
            "first_successful_attempt": {
                arm: {e.eid: e.first_successful_attempt for e in results[arm].values()}
                for arm in results
            },
        }
        out_cells[cid] = info
    if hasattr(policy, "close"):
        policy.close()
    metrics = {
        "stage": ctx.config["feedback_granularity"]["stage_name"],
        "model": ctx.config["model"]["name"],
        "tier": tier,
        "max_retries": K,
        "fixed_guidance_path": g["fixed_guidance_path"],
        "task_evaluation_run_dir": str(run_dir),
        "cells": out_cells,
        "criteria": [],
        "status": "DONE",
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "feedback granularity run complete: %s",
        {cid: c["pass_at_5"] for cid, c in out_cells.items()},
    )


if __name__ == "__main__":
    main()
