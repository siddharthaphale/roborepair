"""CPU tests for the failure bank."""

from roborepair.data import (
    FailureEntry,
    build_failure_bank,
    make_mismatched,
    split_disjoint,
    stratify,
)
from roborepair.schema.execution_trace import Trace


def make_failure_entry(ep, task, label, prog, seed, phash, success=False, infra=False):
    tr = Trace(
        task_id=task,
        progress=prog,
        failure_labels=[label] if label else [],
        compile_ok=True,
        api_ok=True,
        program_hash=phash,
        sim_seed=seed,
    )
    return FailureEntry(
        episode_id=ep,
        task=task,
        instruction="x",
        reset_seed=seed,
        init_code=f"code_{ep}",
        init_trace=tr,
        init_success=success,
        init_progress=prog,
        failure_labels=[label] if label else [],
        gen_seed=seed,
        sim_version="v0",
        program_hash=phash,
        infra_excluded=infra,
    )


def make_failure_bank(n=24):
    entries = []
    labels = ["missing_grasp", "failed_grasp", "object_not_lifted"]
    for i in range(n):
        entries.append(
            make_failure_entry(
                f"ep{i}",
                "cube_lift",
                labels[i % 3],
                (i % 3) / 3.0,
                seed=i,
                phash=f"h{i}",
            )
        )
    return entries


def test_only_non_infra_failures_kept():
    """Infra-excluded + successful episodes never enter the bank."""
    entries = make_failure_bank(6)
    entries.append(
        make_failure_entry("infra", "cube_lift", "missing_grasp", 0.0, 99, "hinfra", infra=True)
    )
    entries.append(make_failure_entry("ok", "cube_lift", None, 1.0, 98, "hok", success=True))
    bank = build_failure_bank(entries)
    ids = {e.episode_id for e in bank}
    assert "infra" not in ids and "ok" not in ids
    assert len(bank) == 6


def test_split_disjoint_by_seed_and_hash():
    """train/val/test share no reset seed or program hash."""
    bank = build_failure_bank(make_failure_bank(24))
    sp = split_disjoint(bank, n_train=12, n_val=6, n_test=6, seed=0)
    seeds = {
        "train": {e.reset_seed for e in sp.train},
        "val": {e.reset_seed for e in sp.val},
        "test": {e.reset_seed for e in sp.test},
    }
    hashes = {
        "train": {e.program_hash for e in sp.train},
        "val": {e.program_hash for e in sp.val},
        "test": {e.program_hash for e in sp.test},
    }
    assert seeds["train"].isdisjoint(seeds["val"])
    assert seeds["train"].isdisjoint(seeds["test"])
    assert seeds["val"].isdisjoint(seeds["test"])
    assert hashes["train"].isdisjoint(hashes["val"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])


def test_stratification_preserves_strata():
    """Stratify keys by (task, label, progress-bucket, valid)."""
    bank = build_failure_bank(make_failure_bank(24))
    strata = stratify(bank)
    for key, group in strata.items():
        task, label, bucket, valid = key
        for e in group:
            assert e.task == task and e.primary_label == label
            assert e.progress_bucket == bucket and e.valid == valid


def test_mismatch_is_within_stratum_derangement():
    """Mismatched traces are same-stratum, different seed AND hash, and never self-matched."""
    bank = build_failure_bank(make_failure_bank(24))
    by_id = {e.episode_id: e for e in bank}
    mapping = make_mismatched(bank, seed=0)
    assert mapping, "expected at least one mismatch assignment"
    for ep_id, foreign_trace in mapping.items():
        src = by_id[ep_id]
        # the foreign trace must not be the episode's own
        assert foreign_trace is not src.init_trace
        # same task + same primary label (same stratum)
        assert foreign_trace.task_id == src.task
        assert (
            foreign_trace.failure_labels[0] if foreign_trace.failure_labels else None
        ) == src.primary_label
        # different seed and different program hash
        assert foreign_trace.sim_seed != src.reset_seed
        assert foreign_trace.program_hash != src.program_hash
        # bounded progress gap (config mismatch.max_progress_diff <= 0.25)
        assert abs(foreign_trace.progress - src.init_progress) <= 0.25


def test_mismatch_respects_max_progress_diff():
    """Within a stratum spanning a wide progress range, mismatches stay within 0.25 Δprogress."""
    # Two episodes in the SAME stratum (same task/label/bucket-3/valid). |Δprogress| = 0.21,
    # within the 0.25 tolerance, so they may be mismatched to each other.
    entries = [
        make_failure_entry("a", "cube_lift", "missing_grasp", 0.78, seed=1, phash="ha"),
        make_failure_entry("b", "cube_lift", "missing_grasp", 0.99, seed=2, phash="hb"),
    ]
    bank = build_failure_bank(entries)
    mapping = make_mismatched(bank, seed=0)
    by_id = {e.episode_id: e for e in bank}
    for ep_id, foreign in mapping.items():
        src = by_id[ep_id]
        assert abs(foreign.progress - src.init_progress) <= 0.25
