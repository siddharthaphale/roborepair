"""Shared donor selection and paired task-cell outcomes."""

from __future__ import annotations

import numpy as np

from roborepair.execution.cell_retry import recovery_at_retry_limit


def assign_feedback_donors(bank, seed: int):
    """Assign within-cell donors with matching progress and different labels when possible."""
    rng = np.random.default_rng(seed)
    donor_of, donor_kind = ({}, {})
    for r in bank:
        lab = r["failure_labels"][0] if r.get("failure_labels") else None
        bucket = min(3, int(r["init_progress"] * 4))
        pool = [
            d
            for d in bank
            if d["episode_id"] != r["episode_id"] and d["program_hash"] != r["program_hash"]
        ]
        if not pool:
            donor_of[r["episode_id"]] = r["episode_id"]
            donor_kind[r["episode_id"]] = "self_no_pool"
            continue
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
    return (donor_of, donor_kind)


def compute_cell_contrast(results, a, b, ids, K):
    """Return the paired bootstrap and McNemar contrast of arms a and b at pass@K."""
    from roborepair.evaluation.metrics import mcnemar_exact, paired_bootstrap

    va = recovery_at_retry_limit([results[a][i] for i in ids if i in results[a]], K)[1]
    vb = recovery_at_retry_limit([results[b][i] for i in ids if i in results[b]], K)[1]
    n = min(len(va), len(vb))
    if n == 0:
        return None
    bs = paired_bootstrap(va[:n], vb[:n], n=10000, ci=0.9, seed=0)
    return {
        "point": bs.point,
        "ci_low": bs.low,
        "ci_high": bs.high,
        "mcnemar_p": mcnemar_exact(
            int(np.sum((va[:n] == 1) & (vb[:n] == 0))), int(np.sum((va[:n] == 0) & (vb[:n] == 1)))
        ),
        "n": n,
    }


class _StoredEpisodeOutcomes:
    def __init__(self, sa):
        self.by_id = {
            eid: type("E", (), {"first_successful_attempt": v})() for eid, v in sa.items()
        }

    def __getitem__(self, i):
        return self.by_id[i]

    def __contains__(self, i):
        return i in self.by_id
