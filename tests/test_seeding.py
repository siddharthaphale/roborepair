"""CPU tests for seeding / determinism."""

from roborepair import seeding


def test_seed_everything_determinism():
    """seed_everything reproduces python+numpy streams."""
    import random

    import numpy as np

    seeding.seed_everything(123)
    a = [random.random() for _ in range(5)], np.random.rand(4).tolist()
    seeding.seed_everything(123)
    b = [random.random() for _ in range(5)], np.random.rand(4).tolist()
    assert a == b


def test_derive_seed_stable_and_distinct():
    """derive_seed is stable across calls and distinct for distinct tags."""
    s1 = seeding.derive_seed(0, "sim", "ep7")
    s2 = seeding.derive_seed(0, "sim", "ep7")
    s3 = seeding.derive_seed(0, "sim", "ep8")
    assert s1 == s2 and s1 != s3
    assert isinstance(s1, int)


def test_four_namespaces_independent():
    """sim/gen/train/perturb derive independent streams from the same top seed."""
    s = seeding.namespace_seeds(0, {"sim": 0, "gen": 0, "train": 0, "perturb": 0})
    vals = list(s.values())
    assert set(s) == {"sim", "gen", "train", "perturb"}
    assert len(set(vals)) == 4  # all four distinct


def test_namespace_seed_isolated_from_other_config_bases():
    """Changing one namespace's config base does not move another namespace's seed."""
    a = seeding.namespace_seeds(0, {"sim": 0, "gen": 0, "train": 0, "perturb": 0})
    b = seeding.namespace_seeds(0, {"sim": 0, "gen": 0, "train": 99, "perturb": 0})
    assert a["sim"] == b["sim"]
    assert a["gen"] == b["gen"]
    assert a["perturb"] == b["perturb"]
    assert a["train"] != b["train"]


def test_rng_reproducible_per_namespace():
    """seeding.rng gives a reproducible numpy Generator for a (namespace, tag)."""
    r1 = seeding.rng(0, "sim", "ep0").random(3).tolist()
    r2 = seeding.rng(0, "sim", "ep0").random(3).tolist()
    r3 = seeding.rng(0, "gen", "ep0").random(3).tolist()
    assert r1 == r2 and r1 != r3
