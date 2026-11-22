"""Renaming human-facing conditions must preserve recorded generation seeds."""

import hashlib

import pytest

from roborepair.sampling_identity import sampling_identity
from roborepair.seeding import derive_seed


@pytest.mark.parametrize(
    ("descriptive", "recorded"),
    [
        (
            "sampling_temperature_0_2-s0-cube_lift_pose_shift_7b-resample-a1-w0",
            "gateN_temp_sensitivity-s0-D7-R0_resample-a1-w0",
        ),
        (
            "sampling_seed_replication-s1-cube_stack_distractor_7b-resample-a1-w0",
            "replication_seed_replication-s1-D12-R0_resample-a1-w0",
        ),
        ("template_feedback-a2-w0", "R1_constraints-a2-w0"),
        ("resample-a1-w3", "R0_resample-a1-w3"),
        ("fresh_initial_generation-cube_lift-w0", "e3-init-cube_lift-w0"),
        ("guided_initial_generation-cube_stack-w0", "e7-init-s2fs-cube_stack-w0"),
        ("cube_to_zone_7b-bank-wave0", "D1-bank-wave0"),
        ("high_level-cube_lift-wave0", "M-cube_lift-wave0"),
    ],
)
def test_descriptive_tags_preserve_seed_hashes(descriptive, recorded):
    expected = int(hashlib.sha256(f"42|paired|{recorded}".encode()).hexdigest()[:8], 16)
    assert derive_seed(42, "paired", descriptive) == expected


def test_recorded_ids_and_unrelated_words_are_unchanged():
    for value in ("v3-M-cube_lift-sim100-g0", "R0_resample", "resample_count"):
        assert sampling_identity(value) == value
