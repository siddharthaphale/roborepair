"""Failure-bank filtering, stratified splits, and compatible donor-trace assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FailureEntry:
    """One banked episode: initial program, its trace, labels, and split/stratum keys."""

    episode_id: str
    task: str
    instruction: str
    reset_seed: int
    init_code: str
    init_trace: object  # a Trace
    init_success: bool
    init_progress: float
    failure_labels: list
    gen_seed: int
    sim_version: str
    program_hash: str
    infra_excluded: bool = False

    @property
    def primary_label(self):
        """First failure label, or None when the episode has no labels."""
        return self.failure_labels[0] if self.failure_labels else None

    @property
    def progress_bucket(self) -> int:
        """Progress quartile in {0, 1, 2, 3} (4 buckets over [0, 1])."""
        return min(3, int(self.init_progress * 4))

    @property
    def valid(self) -> bool:
        """Whether the initial trace compiled and used the API correctly."""
        t = self.init_trace
        return bool(getattr(t, "compile_ok", True) and getattr(t, "api_ok", True))

    def stratum_key(self):
        """Stratification key: (task, primary_label, progress_bucket, valid)."""
        return (self.task, self.primary_label, self.progress_bucket, self.valid)


@dataclass
class Splits:
    """Train/val/test lists of failure-bank entries."""

    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)


def build_failure_bank(entries) -> list:
    """Keep failed episodes, excluding infrastructure failures."""
    return [e for e in entries if not e.infra_excluded and not e.init_success]


def stratify(entries) -> dict:
    """Group entries into a dict of lists keyed by ``stratum_key()``."""
    out: dict = {}
    for e in entries:
        out.setdefault(e.stratum_key(), []).append(e)
    return out


def split_disjoint(entries, n_train, n_val, n_test, seed) -> Splits:
    """Stratified split that is disjoint by reset_seed AND program_hash.

    Within each stratum, entries are shuffled deterministically then dealt round-robin
    into train/val/test up to the requested caps; an entry whose reset_seed OR
    program_hash already appears in another split is skipped (prevents leakage).
    """
    rng = np.random.default_rng(seed)
    strata = stratify(entries)
    splits = Splits()
    used_seeds = {"train": set(), "val": set(), "test": set()}
    used_hashes = {"train": set(), "val": set(), "test": set()}
    caps = {"train": n_train, "val": n_val, "test": n_test}

    def _claim(split_name, e):
        for other in ("train", "val", "test"):
            if other == split_name:
                continue
            if e.reset_seed in used_seeds[other] or e.program_hash in used_hashes[other]:
                return False
        if len(getattr(splits, split_name)) >= caps[split_name]:
            return False
        getattr(splits, split_name).append(e)
        used_seeds[split_name].add(e.reset_seed)
        used_hashes[split_name].add(e.program_hash)
        return True

    for _stratum, group in sorted(strata.items(), key=lambda kv: str(kv[0])):
        order = rng.permutation(len(group))
        cycle = ["train", "val", "test"]
        cursor = 0
        for i in order:
            e = group[i]
            for _ in range(3):
                if _claim(cycle[cursor % 3], e):
                    cursor += 1
                    break
                cursor += 1
    return splits


# Maximum progress gap between an episode and its donor trace.
MAX_PROGRESS_DIFF = 0.25


def _mismatch_ok(src, donor) -> bool:
    """Whether ``donor``'s trace is an admissible foreign trace for ``src``."""
    return (
        src is not donor
        and src.reset_seed != donor.reset_seed
        and src.program_hash != donor.program_hash
        and abs(src.init_progress - donor.init_progress) <= MAX_PROGRESS_DIFF
    )


def make_mismatched(entries, seed) -> dict:
    """Assign same-stratum donor traces with different seeds and program hashes.

    Progress gaps are bounded by MAX_PROGRESS_DIFF. Random derangements are tried
    first; the greedy fallback may reuse donors. Unpairable episodes are omitted.
    """
    rng = np.random.default_rng(seed)
    strata = stratify(entries)
    mapping: dict = {}
    for _stratum, group in strata.items():
        n = len(group)
        if n < 2:
            continue
        # Try random derangements until one satisfies the seed/hash/progress constraints.
        for _attempt in range(200):
            perm = rng.permutation(n)
            ok = all(_mismatch_ok(group[i], group[perm[i]]) for i in range(n))
            if ok:
                for i in range(n):
                    mapping[group[i].episode_id] = group[perm[i]].init_trace
                break
        else:
            # Fall back to valid pairs; a donor may be reused and unpairable episodes are omitted.
            for i in range(n):
                for j in range(n):
                    if _mismatch_ok(group[i], group[j]):
                        mapping[group[i].episode_id] = group[j].init_trace
                        break
    return mapping
