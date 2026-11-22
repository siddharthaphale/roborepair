"Independent random streams and stable sampling identifiers for paired experiments."

from __future__ import annotations

import hashlib
import os

NAMESPACES = ("sim", "gen", "train", "perturb")


def derive_seed(base: int, *tags) -> int:
    """Stable, collision-resistant 32-bit sub-seed for a (base, *tags) tuple.

    Deterministic across processes (uses sha256, NOT python ``hash()`` which is
    salted). Distinct tags -> distinct seeds with overwhelming probability.
    """
    from .sampling_identity import sampling_identity

    h = hashlib.sha256("|".join([str(base), *map(sampling_identity, tags)]).encode()).hexdigest()
    return int(h[:8], 16)


def namespace_seeds(top_seed: int, config_seeds: dict | None = None) -> dict:
    """Derive the four namespace base seeds from the CLI ``--seed`` and config.

    The top-level seed is folded into every namespace via ``derive_seed`` so the
    streams are independent (cross-contamination would silently couple the arms).
    """
    config_seeds = dict(config_seeds or {})
    out = {}
    for ns in NAMESPACES:
        cfg_base = int(config_seeds.get(ns, 0))
        out[ns] = derive_seed(top_seed, ns, cfg_base)
    return out


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch; set PYTHONHASHSEED for child interpreters."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # numpy optional for lightweight paths
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def rng(top_seed: int, namespace: str, *tags):
    """Return a numpy Generator seeded deterministically for ``(namespace, *tags)``."""
    import numpy as np

    if namespace not in NAMESPACES:
        raise ValueError(f"unknown seed namespace {namespace!r}; expected one of {NAMESPACES}")
    return np.random.default_rng(derive_seed(top_seed, namespace, *tags))
