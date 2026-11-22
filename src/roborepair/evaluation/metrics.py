"""Recovery metrics, paired bootstrap intervals, and exact McNemar tests.

Recovery metrics exclude initial successes and infrastructure failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeOutcome:
    """Initial + repair outcome of one episode (the input row for all repair metrics)."""

    episode_id: str
    init_success: bool
    init_infra_excluded: bool
    init_progress: float
    repair_success: bool
    repair_progress: float
    init_primary_label: str | None = None
    repair_primary_label: str | None = None


def _study_set(outcomes):
    """Episodes whose initial program is a non-infra FAILURE (the conditioning set)."""
    return [o for o in outcomes if not o.init_infra_excluded and not o.init_success]


def repair_success_at_1(outcomes) -> float:
    """Fraction of the study set (non-infra initial failures) whose repair succeeds."""
    study = _study_set(outcomes)
    if not study:
        return 0.0
    return sum(o.repair_success for o in study) / len(study)


def progress_delta(outcomes) -> float:
    """Mean (repair_progress - init_progress) over the study set."""
    study = _study_set(outcomes)
    if not study:
        return 0.0
    return float(np.mean([o.repair_progress - o.init_progress for o in study]))


def failure_resolution(outcomes) -> dict:
    """Per-label Resolve(f) + the rate of introducing a NEW failure while 'resolving'."""
    study = _study_set(outcomes)
    by_label: dict = {}
    for o in study:
        by_label.setdefault(o.init_primary_label, []).append(o)
    resolution = {}
    for label, eps in by_label.items():
        resolved = sum(o.repair_success for o in eps)
        # 'resolved but new failure introduced': not success AND a different primary label appears
        new_failure = sum(
            1
            for o in eps
            if not o.repair_success
            and o.repair_primary_label is not None
            and o.repair_primary_label != o.init_primary_label
        )
        resolution[label] = {
            "n": len(eps),
            "resolved": resolved / len(eps),
            "new_failure_introduced": new_failure / len(eps),
        }
    return resolution


@dataclass
class BootstrapCI:
    """Point estimate with bootstrap confidence-interval bounds."""

    point: float
    low: float
    high: float


def paired_bootstrap(
    metric_a, metric_b, *, n: int = 10000, ci: float = 0.90, seed: int = 0
) -> BootstrapCI:
    """Paired bootstrap of the mean difference (a - b). Same index resampled in both arms."""
    a = np.asarray(metric_a, dtype=np.float64)
    b = np.asarray(metric_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap requires equal-length paired arrays")
    if a.size == 0:
        return BootstrapCI(0.0, 0.0, 0.0)
    diff = a - b
    point = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    boot = diff[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    hi = 1.0 - lo
    return BootstrapCI(point, float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


def mcnemar_exact(discordant_b: int, discordant_c: int) -> float:
    """Exact two-sided McNemar p-value on discordant counts.

    ``discordant_b`` = #(a wins, b loses), ``discordant_c`` = #(a loses, b wins). Uses the
    exact binomial test with p=0.5 on the n = discordant_b + discordant_c discordant pairs.
    """
    from math import comb

    n = discordant_b + discordant_c
    if n == 0:
        return 1.0
    k = min(discordant_b, discordant_c)
    # two-sided exact: 2 * P(X <= k) under Binomial(n, 0.5), capped at 1.
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)
