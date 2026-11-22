"""Simulator correctness checks shared by bank preparation and simulator tests."""

from __future__ import annotations

import logging

import numpy as np

from roborepair.execution.rollouts import (
    run_and_trace,
)
from roborepair.simulation import make_env

LOGGER = logging.getLogger("roborepair.input_preparation")


def audit_environment_correctness(ctx, tier: str):
    """Audit the tier: fault-label accuracy, schema validity, determinism, gold-policy success."""
    if tier == "low_level":
        from roborepair.simulation.robosuite import (
            LOW_LEVEL_FAULT_PROGRAMS as HIGH_LEVEL_FAULT_PROGRAMS,
        )
        from roborepair.simulation.robosuite import (
            LOW_LEVEL_REFERENCE_PROGRAMS as HIGH_LEVEL_REFERENCE_PROGRAMS,
        )
    else:
        from roborepair.simulation.robosuite import (
            HIGH_LEVEL_FAULT_PROGRAMS,
            HIGH_LEVEL_REFERENCE_PROGRAMS,
        )
    HIGH_LEVEL_FAULT_PROGRAMS = {
        t: f for t, f in HIGH_LEVEL_FAULT_PROGRAMS.items() if t in ctx.config["env"]["tasks"]
    }
    env_cfg = dict(ctx.config["env"], api_tier=tier)
    timeout = float(ctx.config["decoding"]["code_timeout_s"])
    n_traces = n_valid = infra = 0
    audit = []
    for task_id, faults in HIGH_LEVEL_FAULT_PROGRAMS.items():
        env = make_env(task_id, env_cfg)
        for name, (code, expected) in faults.items():
            env.reset(sim_seed=0)
            res = run_and_trace(env, task_id, code, sim_seed=0, timeout_s=timeout)
            if res["infra_excluded"]:
                infra += 1
                continue
            t = res["trace"]
            n_traces += 1
            try:
                t.validate()
                n_valid += 1
            except Exception as exc:
                LOGGER.error("schema violation %s/%s: %s", task_id, name, exc)
            got = t.failure_labels[0] if t.failure_labels else None
            audit.append(
                {
                    "task": task_id,
                    "fault": name,
                    "expected": expected,
                    "got": got,
                    "ok": got == expected,
                }
            )
        env.close()
    total = agree_s = agree_l = 0
    for task_id in ctx.config["env"]["tasks"]:
        env = make_env(task_id, env_cfg)
        progs = [HIGH_LEVEL_REFERENCE_PROGRAMS[task_id]] + [
            c for c, _e in HIGH_LEVEL_FAULT_PROGRAMS[task_id].values()
        ]
        for code in progs:
            for seed in range(5):
                ref = None
                for _rep in range(3):
                    env.reset(sim_seed=seed)
                    res = run_and_trace(env, task_id, code, sim_seed=seed, timeout_s=timeout)
                    if res["infra_excluded"]:
                        infra += 1
                        continue
                    t = res["trace"]
                    n_traces += 1
                    n_valid += 1
                    key = (
                        t.task_success,
                        tuple(sorted(t.failure_labels)),
                        tuple(sorted(t.subgoals.items())),
                    )
                    if ref is None:
                        ref = key
                        continue
                    total += 1
                    agree_s += key[0] == ref[0]
                    agree_l += key[1:] == ref[1:]
        env.close()
    gold = {}
    for task_id in ctx.config["env"]["tasks"]:
        env = make_env(task_id, env_cfg)
        succ = 0
        for seed in range(50):
            env.reset(sim_seed=seed)
            res = run_and_trace(
                env,
                task_id,
                HIGH_LEVEL_REFERENCE_PROGRAMS[task_id],
                sim_seed=seed,
                timeout_s=timeout,
            )
            if res["infra_excluded"]:
                infra += 1
                continue
            n_traces += 1
            n_valid += 1
            succ += res["trace"].task_success
        gold[task_id] = succ / 50
        env.close()
    return {
        "fault_audit": audit,
        "fault_label_accuracy": float(np.mean([a["ok"] for a in audit])),
        "schema_validity": n_valid / n_traces if n_traces else None,
        "determinism": {
            "comparisons": total,
            "success_agreement": agree_s / total if total else None,
            "label_set_agreement": agree_l / total if total else None,
        },
        "gold_policy_success": gold,
        "infra_events": infra,
    }
