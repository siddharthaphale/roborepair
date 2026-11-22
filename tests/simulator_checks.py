from __future__ import annotations

import logging

from roborepair.execution.environment_checks import audit_environment_correctness
from roborepair.execution.rollouts import evaluate_criterion, overall_status
from roborepair.execution.run_context import write_metrics
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.task_checks")


def perturbation_determinism(ctx, tier: str):
    """Same (sim_seed, family) must produce an identical initial scene, twice."""
    from roborepair.simulation.base import Perturbation

    env_cfg = dict(ctx.config["env"], api_tier=tier)
    rows = []
    for task_id in ctx.config["task_checks"]["perturbation_tasks"]:
        env = make_env(task_id, env_cfg)
        for family in ("pose_shift", "paraphrase", "distractor"):
            ok = True
            for seed in range(3):
                p = Perturbation(family=family)
                a = env.reset(sim_seed=seed, perturbation=p)
                env.reset(sim_seed=seed + 71)
                b = env.reset(sim_seed=seed, perturbation=p)
                base = env.reset(sim_seed=seed)
                same = a.initial_state_hash == b.initial_state_hash
                differs = (
                    a.initial_state_hash != base.initial_state_hash
                    or a.instruction != base.instruction
                    or a.object_names != base.object_names
                )
                ok = ok and same and differs
            rows.append({"task": task_id, "family": family, "deterministic_and_distinct": ok})
        env.close()
    return rows


def template_coverage(audit):
    """Report which observed fault labels lack a constraint template."""
    from roborepair.feedback.constraints import _TEMPLATES

    observed = {a["got"] for a in audit if a["got"]}
    missing = sorted(lab for lab in observed if lab not in _TEMPLATES)
    return {"observed_labels": sorted(observed), "missing_templates": missing}


def check_manipulation_tasks(ctx) -> None:
    tier = ctx.config["task_checks"]["tier"]
    corr = audit_environment_correctness(ctx, tier)
    LOGGER.info(
        "correctness: labels=%.3f det=%s gold=%s",
        corr["fault_label_accuracy"],
        corr["determinism"],
        corr["gold_policy_success"],
    )
    pert = perturbation_determinism(ctx, tier)
    cov = template_coverage(corr["fault_audit"])
    new_tasks = ctx.config["task_checks"]["new_tasks"]
    pc = ctx.config["task_checks"]["pass_criteria"]
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
            "gold_min_new_tasks",
            min(corr["gold_policy_success"][t] for t in new_tasks),
            pc["gold_min"],
            ">=",
        ),
        evaluate_criterion("template_coverage_missing", len(cov["missing_templates"]), 0, "=="),
        evaluate_criterion("infra_events", corr["infra_events"], 0, "=="),
    ]
    frozen = [v for t, v in corr["gold_policy_success"].items() if t not in new_tasks]
    if frozen:
        criteria.insert(
            5, evaluate_criterion("gold_min_frozen_tasks", min(frozen), pc["gold_min"], ">=")
        )
    if pert:
        criteria.append(
            evaluate_criterion(
                "perturbation_determinism",
                int(all(r["deterministic_and_distinct"] for r in pert)),
                1,
                "==",
            )
        )
    metrics = {
        "stage": "manipulation_task_checks",
        "tier": tier,
        "environment_correctness": corr,
        "template_coverage": cov,
        "perturbation_determinism": pert,
        "criteria": criteria,
        "pass_criteria": pc,
        "status": overall_status(criteria),
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info(
        "manipulation task checks %s | labels=%.3f gold=%s",
        metrics["status"],
        corr["fault_label_accuracy"],
        corr["gold_policy_success"],
    )
