"""Compare feedback content, program editing, and resampling on frozen failures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from roborepair.execution.retry import (
    load_completed_condition,
    load_failure_bank_rows,
    progress_bucket,
    recovery_at_retry_limit,
    run_condition,
    wait_for_artifact,
)
from roborepair.execution.rollouts import (
    build_policy,
)
from roborepair.execution.run_context import (
    bootstrap,
    build_arg_parser,
    load_config,
    write_jsonl,
    write_metrics,
)


def validate_neutral_prompt(config: dict) -> None:
    experiment = config["experiment"]
    if experiment.get("suffix") != "neutral" or not experiment.get("log_prompt_text"):
        raise ValueError("Neutral-prompt runs require suffix: neutral and log_prompt_text: true.")
    store = experiment.get("donor_feedback_path")
    if "different_label_feedback" in experiment["arms"] and not (store and Path(store).is_file()):
        raise ValueError(f"Missing donor feedback file: {store!r}")


LOGGER = logging.getLogger("roborepair.feedback_controls")


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("experiment"), dict):
        parser.error("This experiment requires an experiment configuration.")
    if "fresh_episodes" in config:
        parser.error("Use fresh_scene_evaluation.py for fresh-scene experiments.")
    if config["experiment"].get("study_id") == "neutral_prompt_study":
        validate_neutral_prompt(config)
    ctx = bootstrap(args)
    from roborepair import seeding
    from roborepair.evaluation.metrics import mcnemar_exact, paired_bootstrap

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    cfg = ctx.config["experiment"]
    K, wave, tier = (int(cfg["max_retries"]), int(cfg["wave_size"]), cfg["tier"])
    bank = load_failure_bank_rows(Path(ctx.config["experiment"]["failure_bank"]))[
        : int(cfg.get("n_episodes", 10**9))
    ]
    LOGGER.info("bank=%d K=%d tier=%s arms=%s", len(bank), K, tier, cfg["arms"])
    root = Path.cwd()
    priors_dir = root / cfg.get("priors_dir", "data/feedback_templates")
    compat = json.loads((priors_dir / "compatibility.json").read_text())
    label_permutation = json.loads((priors_dir / "label_permutation.json").read_text())
    tasks = sorted({r["task"] for r in bank})
    statics = {
        "compatible_template_union": {
            t: (priors_dir / f"compatible_template_union_{t}_high_level.txt").read_text().strip()
            for t in tasks
        },
        "fixed_task_template": {
            t: (priors_dir / f"fixed_task_template_{t}_high_level.txt").read_text().strip()
            for t in tasks
        },
        "three_task_templates": {
            t: (priors_dir / f"three_task_templates_{t}_high_level.txt").read_text().strip()
            for t in tasks
        },
        "generic_instructions": {
            t: (root / "configs/task_evaluations/generic_guidance_high_level.txt")
            .read_text()
            .strip()
            for t in tasks
        },
    }
    donor_feedback: dict = {}
    donor_feedback_path = cfg.get("donor_feedback_path")
    policy = build_policy(ctx.config)
    results = {}
    for arm in cfg["arms"]:
        done_path = ctx.output_dir / f"attempts_{arm}.jsonl"
        if done_path.exists():
            LOGGER.info("arm %s already complete (resume); loading rows", arm)
            results[arm] = load_completed_condition(done_path, bank)
            if arm == "template_feedback":
                donor_feedback = json.loads((ctx.output_dir / "donor_feedback.json").read_text())
            continue
        if arm == "different_label_feedback":
            if donor_feedback_path:
                store_file = Path(donor_feedback_path)
                wait_for_artifact(store_file)
                donor_feedback = json.loads(store_file.read_text())
            elif not donor_feedback:
                store_file = ctx.output_dir / "donor_feedback.json"
                wait_for_artifact(store_file)
                donor_feedback = json.loads(store_file.read_text())
        eps = run_condition(
            ctx,
            policy,
            ns,
            arm,
            bank,
            tier,
            K,
            wave,
            statics=statics,
            label_permutation=label_permutation,
            compat=compat,
            donor_feedback=donor_feedback,
        )
        results[arm] = eps
        if arm == "template_feedback":
            donor_feedback = {
                eid: {
                    "task": e.row["task"],
                    "bucket": progress_bucket(e.row["init_progress"]),
                    "sets": e.constraints,
                }
                for eid, e in eps.items()
            }
            (ctx.output_dir / "donor_feedback.json").write_text(json.dumps(donor_feedback))
        rows = [r for e in eps.values() for r in e.attempt_rows]
        tmp = ctx.output_dir / f"attempts_{arm}.jsonl.tmp"
        if tmp.exists():
            tmp.unlink()
        write_jsonl(tmp, rows)
        tmp.rename(done_path)
        p5, _ = recovery_at_retry_limit(list(eps.values()), K)
        LOGGER.info("arm %s pass@%d = %.3f", arm, K, p5)
    if hasattr(policy, "close"):
        policy.close()
    merged = ctx.output_dir / "attempts.jsonl"
    if merged.exists():
        merged.unlink()
    all_rows = []
    for arm in cfg["arms"]:
        p = ctx.output_dir / f"attempts_{arm}.jsonl"
        all_rows.extend(json.loads(line) for line in p.read_text().splitlines())
    write_jsonl(merged, all_rows)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        safe = [{k: v for k, v in r.items() if k != "feedback_source_episode"} for r in all_rows]
        pq.write_table(pa.Table.from_pylist(safe), ctx.output_dir / "episodes.parquet")
    except Exception as exc:
        LOGGER.warning("parquet export skipped: %s", exc)
    ids = [r["episode_id"] for r in bank]
    per_arm = {}
    curves = {}
    for arm, eps in results.items():
        elist = [eps[i] for i in ids if i in eps]
        curves[arm] = {k: recovery_at_retry_limit(elist, k)[0] for k in range(1, K + 1)}
        succ = [e.first_successful_attempt for e in elist if e.first_successful_attempt]
        per_arm[arm] = {
            "n": len(elist),
            "recovery_at_retry_limit": curves[arm][K],
            "median_attempts_to_success": float(np.median(succ)) if succ else None,
            "attempt_validity": float(
                np.mean([e.valid_attempts / e.n_attempts for e in elist if e.n_attempts])
            ),
            "safety_any": float(np.mean([getattr(e, "safety", False) for e in elist])),
            "mean_completion_tokens_per_episode": float(np.mean([e.tokens for e in elist])),
            "mean_prompt_tokens_per_episode": float(np.mean([e.prompt_tokens for e in elist])),
            "mean_total_tokens_per_episode": float(
                np.mean([e.tokens + e.prompt_tokens for e in elist])
            ),
        }

    def condition_success_vector(arm, k=K):
        return recovery_at_retry_limit([results[arm][i] for i in ids if i in results[arm]], k)[1]

    def paired_success_contrast(a, b):
        va, vb = (condition_success_vector(a), condition_success_vector(b))
        n = min(len(va), len(vb))
        bs = paired_bootstrap(va[:n], vb[:n], n=10000, ci=0.95, seed=0)
        return {
            "point": bs.point,
            "ci95_low": bs.low,
            "ci95_high": bs.high,
            "mcnemar_p": mcnemar_exact(
                int(np.sum((va[:n] == 1) & (vb[:n] == 0))),
                int(np.sum((va[:n] == 0) & (vb[:n] == 1))),
            ),
        }

    contrasts = {}
    arms_run = [a for a in cfg["arms"] if a in results]
    base = "template_feedback" if "template_feedback" in results else arms_run[0]
    for arm in arms_run:
        if arm != base:
            contrasts[f"{base}_minus_{arm}"] = paired_success_contrast(base, arm)
    if "resample" in results:
        for arm in arms_run:
            if arm not in ("resample", base):
                contrasts[f"{arm}_minus_resample"] = paired_success_contrast(arm, "resample")
    different_label_audit = None
    if "different_label_feedback" in results:
        donor_feedback_episodes = results["different_label_feedback"]
        flags = [
            f
            for e in donor_feedback_episodes.values()
            if hasattr(e, "different_label_matches")
            for f in e.different_label_matches
        ]
        different_label_audit = {
            "indices": len(flags),
            "different_label": int(sum(flags)),
            "different_label_rate": float(np.mean(flags)) if flags else None,
        }
    metrics = {
        "stage": cfg["experiment_id"],
        "bank": {
            "n": len(bank),
            "path": str(Path(ctx.config["experiment"]["failure_bank"])),
            "max_retries": K,
            "tier": tier,
        },
        "per_arm": per_arm,
        "pass_at_k_curves": curves,
        "contrasts_ci95": contrasts,
        "different_label_donor_audit": different_label_audit,
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "feedback_controls %s done | pass@%d: %s",
        cfg["experiment_id"],
        K,
        {a: round(s["recovery_at_retry_limit"], 3) for a, s in per_arm.items()},
    )


if __name__ == "__main__":
    main()
