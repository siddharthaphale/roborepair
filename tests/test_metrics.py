"""CPU tests for metrics & stats."""

import numpy as np
import pytest

from roborepair.evaluation.metrics import (
    EpisodeOutcome,
    failure_resolution,
    mcnemar_exact,
    paired_bootstrap,
    progress_delta,
    repair_success_at_1,
)


def validated_episode_outcomes():
    return [
        # init failure, repair success
        EpisodeOutcome("e0", False, False, 0.3, True, 1.0, "missing_grasp", None),
        # init failure, repair fail (still failing same label)
        EpisodeOutcome("e1", False, False, 0.3, False, 0.3, "missing_grasp", "missing_grasp"),
        # init failure, repair fail but NEW failure introduced
        EpisodeOutcome("e2", False, False, 0.6, False, 0.3, "object_not_lifted", "missing_grasp"),
        # init SUCCESS -> excluded from the study set
        EpisodeOutcome("e3", True, False, 1.0, True, 1.0, None, None),
        # infra excluded -> excluded
        EpisodeOutcome("e4", False, True, 0.0, False, 0.0, "missing_grasp", "missing_grasp"),
    ]


def test_repair_success_at_1_known_counts():
    """Matches hand-counted success over the non-infra-failure denominator."""
    # study set = e0, e1, e2 (3 episodes); successes = e0 -> 1/3
    assert repair_success_at_1(validated_episode_outcomes()) == pytest.approx(1 / 3)


def test_progress_delta_known():
    """progress_delta = mean(after - before) over the study set."""
    # deltas: e0 +0.7, e1 0.0, e2 -0.3 -> mean = 0.4/3
    assert progress_delta(validated_episode_outcomes()) == pytest.approx((0.7 + 0.0 - 0.3) / 3)


def test_resolution_and_new_failure_metrics():
    """Resolve(f) and new-failure-introduced computed correctly, incl. regressions."""
    res = failure_resolution(validated_episode_outcomes())
    # missing_grasp stratum: e0 (resolved), e1 (not resolved, same label) -> resolved 1/2
    assert res["missing_grasp"]["resolved"] == pytest.approx(0.5)
    assert res["missing_grasp"]["new_failure_introduced"] == pytest.approx(0.0)
    # object_not_lifted stratum: e2 not resolved AND introduced a new label
    assert res["object_not_lifted"]["resolved"] == pytest.approx(0.0)
    assert res["object_not_lifted"]["new_failure_introduced"] == pytest.approx(1.0)


def test_paired_bootstrap_ci_known_distribution():
    """On synthetic paired data with a known effect, the CI brackets the true difference.

    Resamples are PAIRED (same index in both arms).
    """
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(1.0, 0.5, n)
    b = a - 0.3 + rng.normal(0.0, 0.01, n)  # a - b ~ 0.3
    ci = paired_bootstrap(a.tolist(), b.tolist(), n=2000, ci=0.90, seed=0)
    assert ci.point == pytest.approx(0.3, abs=0.05)
    assert ci.low < 0.3 < ci.high

    with pytest.raises(ValueError):
        paired_bootstrap([1, 2, 3], [1, 2], n=10)  # unequal lengths


def test_mcnemar_exact_matches_reference():
    """Exact McNemar p on a fixed discordant pair matches a reference value."""
    # b=1, c=9 discordant: two-sided exact binomial(10, .5), 2*P(X<=1) = 2*(11/1024)
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (11 / 1024))
    # symmetric counts -> p == 1
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    # no discordant pairs -> p == 1
    assert mcnemar_exact(0, 0) == 1.0
