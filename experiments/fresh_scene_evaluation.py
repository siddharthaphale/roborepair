"""Paired fresh-scene evaluations with guidance before or after the initial attempt."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from roborepair.execution.retry import (
    generate_batch_with_timing,
    load_completed_condition,
    recovery_at_retry_limit,
    run_condition,
    token_proportional_shares,
)
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
from roborepair.sampling_identity import sampling_identity
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.fresh_episode_evaluation")


def generate_fresh_initial_attempts(ctx, policy, ns, tier):
    """Phase 1: shared initial generations; returns (initial_rows, failure_rows)."""
    from roborepair.prompts.builder import build_initial_prompt

    dec = ctx.config["decoding"]
    g = ctx.config["fresh_episodes"]
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    wave = int(ctx.config["experiment"]["wave_size"])
    perturb = g.get("perturbation")
    initial_rows, failures = ([], [])
    for task_id, seed_start in g["tasks"].items():
        env = make_env(task_id, env_cfg)
        seeds = [int(seed_start) + i for i in range(int(g["seeds_per_task"]))]
        pert = None
        if perturb:
            from roborepair.simulation.base import Perturbation

            pert = Perturbation(family=perturb)
        obs = env.reset(sim_seed=seeds[0], perturbation=pert)
        prompt = build_initial_prompt(obs, env.api_doc)
        for w0 in range(0, len(seeds), wave):
            chunk = seeds[w0 : w0 + wave]
            gens, wall, gpu = generate_batch_with_timing(
                policy,
                [prompt] * len(chunk),
                max_new_tokens=int(dec["max_output_tokens"]),
                temperature=float(dec["temperature"]),
                top_p=float(dec["top_p"]),
                seed=paired_generation_seed(
                    ns["gen"], f"fresh_initial_generation-{task_id}-w{w0 // wave}"
                ),
            )
            shares = token_proportional_shares(gens)
            for sim_seed, gen, sh in zip(chunk, gens, shares, strict=False):
                code = extract_code(gen.text)
                obs_i = env.reset(sim_seed=sim_seed, perturbation=pert)
                t_exec = time.monotonic()
                res = run_and_trace(
                    env, task_id, code, sim_seed=sim_seed, timeout_s=float(dec["code_timeout_s"])
                )
                exec_wall = time.monotonic() - t_exec
                eid = sampling_identity(f"fresh_episodes-{tier}-{task_id}-sim{sim_seed}-g0")
                row = {
                    "episode_id": eid,
                    "tier": tier,
                    "task": task_id,
                    "instruction": obs_i.instruction,
                    "object_names": list(obs_i.object_names),
                    "reset_seed": sim_seed,
                    "perturbation": perturb,
                    "init_code": code,
                    "program_hash": program_hash(code),
                    "gen_seed": 0,
                    "sim_version": ctx.config["env"]["sim_version"],
                    "infra_excluded": res["infra_excluded"],
                    "raw_log": res["raw_log"],
                    "prompt_tokens": gen.n_prompt_tokens,
                    "completion_tokens": gen.n_completion_tokens,
                    "model_wall_seconds": wall * sh,
                    "gpu_seconds": gpu * sh,
                    "execution_wall_seconds": exec_wall,
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
                initial_rows.append(row)
                if (
                    not row["infra_excluded"]
                    and res["trace"] is not None
                    and (not row["init_success"])
                ):
                    failures.append(row)
            LOGGER.info(
                "fresh episode generation %s: %d/%d seeds done",
                task_id,
                min(w0 + wave, len(seeds)),
                len(seeds),
            )
        env.close()
    return (initial_rows, failures)


def run_fresh_scene_evaluation(ctx) -> None:
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    cfg = ctx.config["experiment"]
    K, wave, tier = (int(cfg["max_retries"]), int(cfg["wave_size"]), cfg["tier"])
    root = Path.cwd()
    priors_dir = root / cfg.get("priors_dir", "data/feedback_templates")
    compat = json.loads((priors_dir / "compatibility.json").read_text())
    label_permutation = json.loads((priors_dir / "label_permutation.json").read_text())
    tasks = list(ctx.config["fresh_episodes"]["tasks"])
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
    policy = build_policy(ctx.config)
    init_path = ctx.output_dir / "initial.jsonl"
    bank_path = ctx.output_dir / "fresh_episode_failure_bank.jsonl"
    if init_path.exists() and bank_path.exists():
        LOGGER.info("phase 1 already complete (resume)")
        initial_rows = [json.loads(line) for line in init_path.read_text().splitlines()]
        bank = [json.loads(line) for line in bank_path.read_text().splitlines()]
    else:
        initial_rows, bank = generate_fresh_initial_attempts(ctx, policy, ns, tier)
        write_jsonl(init_path, initial_rows)
        write_jsonl(bank_path, bank)
    eligible = [r for r in initial_rows if not r["infra_excluded"] and "init_success" in r]
    one_shot = float(np.mean([r["init_success"] for r in eligible])) if eligible else 0.0
    LOGGER.info(
        "initial generation: %d episodes, %d infra-excluded, one-shot success %.3f, %d failures banked",
        len(initial_rows),
        sum(r["infra_excluded"] for r in initial_rows),
        one_shot,
        len(bank),
    )
    if one_shot > 0.7:
        write_metrics(
            ctx.output_dir,
            {"stage": cfg["experiment_id"], "one_shot_success": one_shot, "status": "SWITCH"},
        )
        raise SystemExit(
            "one-shot success exceeds the preregistered 70% ceiling; launch the pose_shift variant instead"
        )
    results = {}
    donor_feedback: dict = {}
    for arm in cfg["arms"]:
        done_path = ctx.output_dir / f"attempts_{arm}.jsonl"
        if done_path.exists():
            LOGGER.info("arm %s already complete (resume); loading rows", arm)
            results[arm] = load_completed_condition(done_path, bank)
            continue
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
        rows = [r for e in eps.values() for r in e.attempt_rows]
        tmp = ctx.output_dir / f"attempts_{arm}.jsonl.tmp"
        if tmp.exists():
            tmp.unlink()
        write_jsonl(tmp, rows)
        tmp.rename(done_path)
        p5, _ = recovery_at_retry_limit(list(eps.values()), K)
        LOGGER.info("arm %s conditional pass@%d = %.3f", arm, K, p5)
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
    ids_fail = [r["episode_id"] for r in bank]
    n_total = len(eligible)
    n_init_succ = sum(r["init_success"] for r in eligible)
    per_arm = {}
    for arm, eps in results.items():
        elist = [eps[i] for i in ids_fail if i in eps]
        cond5, _ = recovery_at_retry_limit(elist, K)
        recovered = sum(
            1
            for e in elist
            if e.first_successful_attempt is not None and e.first_successful_attempt <= K
        )
        per_arm[arm] = {
            "n_failures": len(elist),
            "conditional_pass_at_5": cond5,
            "overall_success_by_attempt_6": (n_init_succ + recovered) / n_total,
        }
    metrics = {
        "stage": cfg["experiment_id"],
        "one_shot_success": one_shot,
        "n_episodes": n_total,
        "n_infra_excluded_initial": sum(r["infra_excluded"] for r in initial_rows),
        "n_initial_success": n_init_succ,
        "n_failures": len(bank),
        "per_arm": per_arm,
        "status": "COMPLETE",
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info("Fresh episode evaluation complete: one-shot %.3f, per-arm %s", one_shot, per_arm)


STATIC_GUIDANCE_FROM_START_CONDITION = "static_guidance_from_start"


def generate_guided_initial_attempts(ctx, policy, ns, tier, statics):
    """Phase 1b: static_guidance_from_start initial attempts using the exact fixed_task_template retry prompt at attempt 0."""
    from types import SimpleNamespace

    from roborepair.prompts.builder import build_feedback_prompt

    dec = ctx.config["decoding"]
    g = ctx.config["fresh_episodes"]
    cap = int(dec["trace_input_token_cap"])
    suffix = ctx.config["experiment"].get("suffix", "historical")
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    wave = int(ctx.config["experiment"]["wave_size"])
    perturb = g.get("perturbation")
    initial_rows, failures = ([], [])
    for task_id, seed_start in g["tasks"].items():
        env = make_env(task_id, env_cfg)
        seeds = [int(seed_start) + i for i in range(int(g["seeds_per_task"]))]
        pert = None
        if perturb:
            from roborepair.simulation.base import Perturbation

            pert = Perturbation(family=perturb)
        obs = env.reset(sim_seed=seeds[0], perturbation=pert)
        prompt = build_feedback_prompt(
            plan_text=statics[STATIC_GUIDANCE_FROM_START_CONDITION][task_id],
            mode="regenerate",
            prev_code=None,
            api_doc=env.api_doc,
            env_obs=SimpleNamespace(
                instruction=obs.instruction, object_names=list(obs.object_names)
            ),
            token_budget=cap,
            suffix=suffix,
        )
        for w0 in range(0, len(seeds), wave):
            chunk = seeds[w0 : w0 + wave]
            gens, wall, gpu = generate_batch_with_timing(
                policy,
                [prompt] * len(chunk),
                max_new_tokens=int(dec["max_output_tokens"]),
                temperature=float(dec["temperature"]),
                top_p=float(dec["top_p"]),
                seed=paired_generation_seed(
                    ns["gen"], f"guided_initial_generation-{task_id}-w{w0 // wave}"
                ),
            )
            shares = token_proportional_shares(gens)
            for sim_seed, gen, sh in zip(chunk, gens, shares, strict=False):
                code = extract_code(gen.text)
                obs_i = env.reset(sim_seed=sim_seed, perturbation=pert)
                t_exec = time.monotonic()
                res = run_and_trace(
                    env, task_id, code, sim_seed=sim_seed, timeout_s=float(dec["code_timeout_s"])
                )
                exec_wall = time.monotonic() - t_exec
                eid = sampling_identity(
                    f"static_guidance_episode-{tier}-{task_id}-sim{sim_seed}-g0"
                )
                row = {
                    "episode_id": eid,
                    "tier": tier,
                    "task": task_id,
                    "instruction": obs_i.instruction,
                    "object_names": list(obs_i.object_names),
                    "reset_seed": sim_seed,
                    "perturbation": perturb,
                    "init_code": code,
                    "program_hash": program_hash(code),
                    "gen_seed": 0,
                    "sim_version": ctx.config["env"]["sim_version"],
                    "infra_excluded": res["infra_excluded"],
                    "raw_log": res["raw_log"],
                    "prompt_tokens": gen.n_prompt_tokens,
                    "completion_tokens": gen.n_completion_tokens,
                    "model_wall_seconds": wall * sh,
                    "gpu_seconds": gpu * sh,
                    "execution_wall_seconds": exec_wall,
                    "static_prompt_at_attempt0": True,
                    "init_prompt_text": prompt,
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
                initial_rows.append(row)
                if (
                    not row["infra_excluded"]
                    and res["trace"] is not None
                    and (not row["init_success"])
                ):
                    failures.append(row)
            LOGGER.info(
                "guided initial generation %s: %d/%d seeds done",
                task_id,
                min(w0 + wave, len(seeds)),
                len(seeds),
            )
        env.close()
    return (initial_rows, failures)


def load_or_generate_initial_attempts(ctx, policy, ns, tier, *, tag, builder, **kw):
    """Run-or-resume one phase-1 pass; returns (initial_rows, bank)."""
    init_path = ctx.output_dir / f"initial_{tag}.jsonl"
    bank_path = ctx.output_dir / f"bank_{tag}.jsonl"
    if init_path.exists() and bank_path.exists():
        LOGGER.info("phase 1 (%s) already complete (resume)", tag)
        initial = [json.loads(line) for line in init_path.read_text().splitlines()]
        bank = [json.loads(line) for line in bank_path.read_text().splitlines()]
        return (initial, bank)
    initial, bank = builder(ctx, policy, ns, tier, **kw)
    write_jsonl(init_path, initial)
    write_jsonl(bank_path, bank)
    return (initial, bank)


def summarize_overall_success(initial_rows, eps, K):
    """Overall success by attempt six over non-infra initial episodes."""
    eligible = [r for r in initial_rows if not r["infra_excluded"] and "init_success" in r]
    n_init_succ = sum(r["init_success"] for r in eligible)
    ids_fail = [r["episode_id"] for r in eligible if not r["init_success"]]
    elist = [eps[i] for i in ids_fail if i in eps]
    cond5, _ = recovery_at_retry_limit(elist, K) if elist else (float("nan"), None)
    recovered = sum(
        1
        for e in elist
        if e.first_successful_attempt is not None and e.first_successful_attempt <= K
    )
    return {
        "n_episodes": len(eligible),
        "n_initial_success": n_init_succ,
        "n_failures": len(elist),
        "conditional_pass_at_5": cond5,
        "overall_success_by_attempt_6": (n_init_succ + recovered) / len(eligible)
        if eligible
        else float("nan"),
    }


def run_static_guidance_evaluation(ctx) -> None:
    from roborepair import seeding

    ns = seeding.namespace_seeds(ctx.seed, ctx.config.get("seeds", {}))
    cfg = ctx.config["experiment"]
    K, wave, tier = (int(cfg["max_retries"]), int(cfg["wave_size"]), cfg["tier"])
    root = Path.cwd()
    priors_dir = root / cfg.get("priors_dir", "data/feedback_templates")
    compat = json.loads((priors_dir / "compatibility.json").read_text())
    label_permutation = json.loads((priors_dir / "label_permutation.json").read_text())
    tasks = list(ctx.config["fresh_episodes"]["tasks"])
    s2 = {
        t: (priors_dir / f"fixed_task_template_{t}_high_level.txt").read_text().strip()
        for t in tasks
    }
    statics = {"fixed_task_template": s2, STATIC_GUIDANCE_FROM_START_CONDITION: s2}
    policy = build_policy(ctx.config)
    initial_rows, bank = load_or_generate_initial_attempts(
        ctx, policy, ns, tier, tag="shared", builder=generate_fresh_initial_attempts
    )
    eligible = [r for r in initial_rows if not r["infra_excluded"] and "init_success" in r]
    one_shot = float(np.mean([r["init_success"] for r in eligible])) if eligible else 0.0
    LOGGER.info(
        "shared initial generation: %d episodes, %d infra-excluded, one-shot %.3f, %d failures",
        len(initial_rows),
        sum(r["infra_excluded"] for r in initial_rows),
        one_shot,
        len(bank),
    )
    if one_shot > 0.7:
        write_metrics(
            ctx.output_dir,
            {"stage": cfg["experiment_id"], "one_shot_success": one_shot, "status": "SWITCH"},
        )
        raise SystemExit("shared-phase one-shot success exceeds the preregistered 70% ceiling")
    initial_static_guidance, static_guidance_failure_bank = load_or_generate_initial_attempts(
        ctx,
        policy,
        ns,
        tier,
        tag="static_guidance",
        builder=generate_guided_initial_attempts,
        statics=statics,
    )
    eligible_guided_attempts = [
        r for r in initial_static_guidance if not r["infra_excluded"] and "init_success" in r
    ]
    guided_one_shot_success = (
        float(np.mean([r["init_success"] for r in eligible_guided_attempts]))
        if eligible_guided_attempts
        else 0.0
    )
    LOGGER.info(
        "guided initial generation: %d episodes, one-shot %.3f, %d failures",
        len(initial_static_guidance),
        guided_one_shot_success,
        len(static_guidance_failure_bank),
    )
    arm_banks = {arm: bank for arm in cfg["arms"] if arm != STATIC_GUIDANCE_FROM_START_CONDITION}
    if STATIC_GUIDANCE_FROM_START_CONDITION in cfg["arms"]:
        arm_banks[STATIC_GUIDANCE_FROM_START_CONDITION] = static_guidance_failure_bank
    results = {}
    donor_feedback: dict = {}
    for arm in cfg["arms"]:
        abank = arm_banks[arm]
        done_path = ctx.output_dir / f"attempts_{arm}.jsonl"
        if done_path.exists():
            LOGGER.info("arm %s already complete (resume); loading rows", arm)
            results[arm] = load_completed_condition(done_path, abank)
            continue
        if not abank:
            LOGGER.info("arm %s: empty bank, nothing to retry", arm)
            results[arm] = {}
            done_path.write_text("")
            continue
        eps = run_condition(
            ctx,
            policy,
            ns,
            arm,
            abank,
            tier,
            K,
            wave,
            statics=statics,
            label_permutation=label_permutation,
            compat=compat,
            donor_feedback=donor_feedback,
        )
        results[arm] = eps
        rows = [r for e in eps.values() for r in e.attempt_rows]
        tmp = ctx.output_dir / f"attempts_{arm}.jsonl.tmp"
        if tmp.exists():
            tmp.unlink()
        write_jsonl(tmp, rows)
        tmp.rename(done_path)
        p5, _ = recovery_at_retry_limit(list(eps.values()), K)
        LOGGER.info("arm %s conditional pass@%d = %.3f", arm, K, p5)
    if hasattr(policy, "close"):
        policy.close()
    per_arm = {}
    for arm in cfg["arms"]:
        init = (
            initial_static_guidance if arm == STATIC_GUIDANCE_FROM_START_CONDITION else initial_rows
        )
        per_arm[arm] = summarize_overall_success(init, results[arm], K)
    metrics = {
        "stage": cfg["experiment_id"],
        "one_shot_success_shared": one_shot,
        "guided_one_shot_success": guided_one_shot_success,
        "n_infra_excluded_shared": sum(r["infra_excluded"] for r in initial_rows),
        "guided_infrastructure_exclusions": sum(
            r["infra_excluded"] for r in initial_static_guidance
        ),
        "per_arm": per_arm,
        "status": "COMPLETE",
    }
    write_metrics(ctx.output_dir, metrics)
    LOGGER.info("Static guidance evaluation complete: per-arm %s", per_arm)


def main() -> None:
    parser = build_arg_parser(__doc__)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("fresh_episodes"), dict) or not isinstance(
        config.get("experiment"), dict
    ):
        parser.error("Fresh-scene evaluation requires experiment and fresh_episodes sections.")
    context = bootstrap(args)
    if "static_guidance_from_start" in config["experiment"]["arms"]:
        run_static_guidance_evaluation(context)
    else:
        run_fresh_scene_evaluation(context)


if __name__ == "__main__":
    main()
