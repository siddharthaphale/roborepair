"""Evaluate feedback, reflection, and structured-editing baselines on frozen failures."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from roborepair.execution.cell_retry import (
    load_failure_bank_rows,
    recovery_at_retry_limit,
    run_condition,
)
from roborepair.execution.rollouts import (
    build_policy,
    evaluate_criterion,
    overall_status,
)
from roborepair.execution.run_context import (
    bootstrap,
    build_arg_parser,
    load_config,
    write_jsonl,
    write_metrics,
)

LOGGER = logging.getLogger("roborepair.feedback_ablation_v3")


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("feedback_ablation"), dict):
        parser.error("This experiment requires a feedback_ablation configuration.")
    ctx = bootstrap(args)
    from roborepair import seeding
    from roborepair.evaluation.metrics import mcnemar_exact, paired_bootstrap

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    cfg = ctx.config["feedback_ablation"]
    K, wave, tier = (int(cfg["max_retries"]), int(cfg["wave_size"]), cfg["tier"])
    bank = load_failure_bank_rows(Path(ctx.config["feedback_ablation"]["failure_bank"]))[
        : int(cfg.get("n_episodes", 10**9))
    ]
    structured_plan_sample_size = int(cfg.get("structured_plan_sample_size", len(bank)))
    LOGGER.info(
        "bank=%d K=%d tier=%s arms=%s structured_plan_sample_size=%d",
        len(bank),
        K,
        tier,
        cfg["arms"],
        structured_plan_sample_size,
    )
    rng = np.random.default_rng(ctx.seed)
    donor_of, donor_kind = ({}, {})
    for r in bank:
        lab = r["failure_labels"][0] if r.get("failure_labels") else None
        bucket = min(3, int(r["init_progress"] * 4))
        pool = [
            d
            for d in bank
            if d["episode_id"] != r["episode_id"]
            and d["task"] == r["task"]
            and (d["program_hash"] != r["program_hash"])
        ]
        same_bucket = [d for d in pool if min(3, int(d["init_progress"] * 4)) == bucket] or pool
        diff = [
            d
            for d in same_bucket
            if (d["failure_labels"][0] if d.get("failure_labels") else None) != lab
        ]
        chosen = (diff or same_bucket)[int(rng.integers(0, len(diff or same_bucket)))]
        donor_of[r["episode_id"]] = chosen["episode_id"]
        donor_kind[r["episode_id"]] = (
            "different_label"
            if (chosen["failure_labels"][0] if chosen.get("failure_labels") else None) != lab
            else "same_label"
        )
    policy = build_policy(ctx.config)
    results = {}
    donor_feedback = None
    for arm in cfg["arms"]:
        sub = bank if arm != "structured_plan_with_editing" else bank[:structured_plan_sample_size]
        eps = run_condition(
            ctx,
            policy,
            ns,
            arm,
            sub,
            tier,
            K,
            wave,
            donor_feedback=donor_feedback,
            donor_of=donor_of if arm == "donor_feedback" else None,
        )
        results[arm] = eps
        if arm == "template_feedback":
            donor_feedback = {eid: e.constraints for eid, e in eps.items()}
        write_jsonl(
            ctx.output_dir / "attempts.jsonl", [r for e in eps.values() for r in e.attempt_rows]
        )
        p5, _ = recovery_at_retry_limit(list(eps.values()), K)
        LOGGER.info("arm %s pass@%d = %.3f", arm, K, p5)
    if hasattr(policy, "close"):
        policy.close()
    write_jsonl(
        ctx.output_dir / "constraints.jsonl",
        [c for e in (results.get("template_feedback") or {}).values() for c in e.constraints],
    )
    ids = [r["episode_id"] for r in bank]
    per_arm = {}
    curves = {}
    for arm, eps in results.items():
        elist = [eps[i] for i in ids if i in eps]
        curves[arm] = {k: recovery_at_retry_limit(elist, k)[0] for k in range(1, K + 1)}
        per_arm[arm] = {
            "n": len(elist),
            "recovery_at_retry_limit": curves[arm][K],
            "median_attempts_to_success": float(
                np.median([e.first_successful_attempt for e in elist if e.first_successful_attempt])
            )
            if any(e.first_successful_attempt for e in elist)
            else None,
            "attempt_validity": float(
                np.mean([e.valid_attempts / e.n_attempts for e in elist if e.n_attempts])
            ),
            "safety_any": float(np.mean([e.safety for e in elist])),
            "mean_tokens_per_episode": float(np.mean([e.tokens for e in elist])),
        }

    def condition_success_vector(arm, k=K):
        return recovery_at_retry_limit([results[arm][i] for i in ids if i in results[arm]], k)[1]

    def paired_success_contrast(a, b):
        va, vb = (condition_success_vector(a), condition_success_vector(b))
        n = min(len(va), len(vb))
        bs = paired_bootstrap(va[:n], vb[:n], n=10000, ci=0.9, seed=0)
        return {
            "point": bs.point,
            "ci_low": bs.low,
            "ci_high": bs.high,
            "mcnemar_p": mcnemar_exact(
                int(np.sum((va[:n] == 1) & (vb[:n] == 0))),
                int(np.sum((va[:n] == 0) & (vb[:n] == 1))),
            ),
        }

    contrasts = {
        "P1_template_feedback_minus_resample": paired_success_contrast(
            "template_feedback", "resample"
        ),
        "P2_template_feedback_minus_donor_feedback": paired_success_contrast(
            "template_feedback", "donor_feedback"
        ),
        "P3_template_feedback_minus_failure_label_feedback": paired_success_contrast(
            "template_feedback", "failure_label_feedback"
        ),
        "P4_template_feedback_minus_reflection_regeneration": paired_success_contrast(
            "template_feedback", "reflection_regeneration"
        ),
    }
    noninf = min(curves["template_feedback"][k] - curves["resample"][k] for k in range(1, K + 1))
    per_task = {}
    for task in ctx.config["env"]["tasks"]:
        tids = [i for i, r in zip(ids, bank, strict=False) if r["task"] == task]
        if not tids:
            continue
        r1 = np.mean(
            [1.0 if results["template_feedback"][i].first_successful_attempt else 0.0 for i in tids]
        )
        r0 = np.mean(
            [1.0 if results["resample"][i].first_successful_attempt else 0.0 for i in tids]
        )
        per_task[task] = {
            "n": len(tids),
            "template_feedback": float(r1),
            "resample": float(r0),
            "positive": bool(r1 > r0),
        }

    def steering(arm):
        rec = tot = 0
        for e in results[arm].values():
            for row in e.attempt_rows:
                if row["attempt"] >= 2 and row.get("primary_label"):
                    tot += 1
                    rec += row["primary_label"] in row.get("prior_labels", [])
        return rec / tot if tot else None

    steer = {"template_feedback": steering("template_feedback"), "resample": steering("resample")}
    steer["relative_reduction"] = (
        1.0 - steer["template_feedback"] / steer["resample"]
        if steer["template_feedback"] is not None and steer["resample"]
        else None
    )
    pc = cfg["pass_criteria"]
    criteria = [
        evaluate_criterion(
            "P1_pp",
            contrasts["P1_template_feedback_minus_resample"]["point"] * 100,
            pc["p1_pp"],
            ">=",
        ),
        evaluate_criterion(
            "P1_lb", contrasts["P1_template_feedback_minus_resample"]["ci_low"], 0, ">"
        ),
        evaluate_criterion(
            "P2_pp",
            contrasts["P2_template_feedback_minus_donor_feedback"]["point"] * 100,
            pc["p2_pp"],
            ">=",
        ),
        evaluate_criterion(
            "P2_lb", contrasts["P2_template_feedback_minus_donor_feedback"]["ci_low"], 0, ">"
        ),
        evaluate_criterion("noninferiority_all_k_pp", noninf * 100, -2.0, ">="),
        evaluate_criterion(
            "P1_positive_both_tasks", int(all(v["positive"] for v in per_task.values())), 1, "=="
        ),
        evaluate_criterion(
            "validity_gap_pp",
            (
                per_arm["resample"]["attempt_validity"]
                - per_arm["template_feedback"]["attempt_validity"]
            )
            * 100,
            3,
            "<=",
        ),
        evaluate_criterion(
            "safety_gap_pp",
            (per_arm["template_feedback"]["safety_any"] - per_arm["resample"]["safety_any"]) * 100,
            3,
            "<=",
        ),
        evaluate_criterion("steering_relative_reduction", steer["relative_reduction"], 0.2, ">="),
    ]
    metrics = {
        "stage": "feedback_ablation",
        "bank": {
            "n": len(bank),
            "path": str(Path(ctx.config["feedback_ablation"]["failure_bank"])),
            "max_retries": K,
            "tier": tier,
        },
        "donor_audit": {
            "different_label": sum(1 for v in donor_kind.values() if v == "different_label"),
            "same_label": sum(1 for v in donor_kind.values() if v == "same_label"),
        },
        "per_arm": per_arm,
        "pass_at_k_curves": curves,
        "contrasts": contrasts,
        "per_task_template_feedback_vs_resample": per_task,
        "steering": steer,
        "criteria": criteria,
        "pass_criteria": pc,
        "status": overall_status(criteria),
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "feedback ablation %s | pass@%d: %s",
        metrics["status"],
        K,
        {a: round(s["recovery_at_retry_limit"], 3) for a, s in per_arm.items()},
    )


if __name__ == "__main__":
    main()
