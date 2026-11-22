"""Immutable sampling tags connecting descriptive names to the recorded experiments.

These values are provenance identifiers, never command names or config options.
Changing a tag changes SHA-256-derived generation seeds. Keep them stable when
renaming experiment interfaces or moving artifacts.
"""

SAMPLING_TAGS = {
    "sampling_temperature_0_2": "gateN_temp_sensitivity",
    "sampling_seed_replication": "replication_seed_replication",
    "template_feedback_then_reflection": "R11_hybrid",
    "reflection_regeneration_control": "B6_reflect_regen",
    "editing_with_template_feedback": "B4_direct_compiled",
    "structured_plan_with_editing": "R5_v2_full",
    "execution_log_regeneration": "B1_rawlog",
    "editing_with_execution_log": "B2_direct",
    "editing_with_failure_label": "B3_direct_label",
    "random_compatible_feedback": "S8_random_template",
    "static_guidance_from_start": "S2FS_dev_top1",
    "dominant_failure_guidance": "R10_single",
    "compatible_template_union": "S1_static_union",
    "initial_failure_feedback": "S4_initial_only",
    "different_label_feedback": "S6_strict_mismatch",
    "reflection_regeneration": "R4_reflexion",
    "larger_model_reflection": "R12_xreflect",
    "editing_with_reflection": "B5_reflect_repair",
    "latest_failure_feedback": "S5_latest_only",
    "permuted_label_feedback": "S7_label_perm",
    "failure_label_feedback": "R3_label_only",
    "three_task_templates": "S3_dev_top3",
    "generic_instructions": "S9_placebo",
    "cross_task_feedback": "R6_cross_task",
    "fixed_task_guidance": "R7_static",
    "fixed_task_template": "S2_dev_top1",
    "repeated_api_rules": "N5_api_rules",
    "template_feedback": "R1_constraints",
    "generic_guidance": "R8_placebo",
    "donor_feedback": "R2_mismatched",
    "task_checklist": "R9_checklist",
    "fresh_episodes": "c2e3",
    "fresh_initial_generation": "e3-init",
    "high_level": "M",
    "low_level": "P",
    "resample": "R0_resample",
    "guided_initial_generation": "e7-init-s2fs",
    "static_guidance_episode": "c6e7",
    "task_evaluation_episode": "v3d",
    "failure_bank_episode": "v3",
}

SAMPLING_TAGS.update(
    {
        "cube_to_zone_7b": "D1",
        "place_in_bin_7b": "D2",
        "cube_lift_3b": "D3",
        "cube_stack_3b": "D4",
        "cube_to_zone_3b": "D5",
        "place_in_bin_3b": "D6",
        "cube_lift_pose_shift_7b": "D7",
        "cube_lift_paraphrase_7b": "D8",
        "cube_lift_distractor_7b": "D9",
        "cube_stack_pose_shift_7b": "D10",
        "cube_stack_paraphrase_7b": "D11",
        "cube_stack_distractor_7b": "D12",
        "cube_to_zone_low_level_7b": "I1",
        "place_in_bin_low_level_7b": "I2",
        "cube_lift_pose_shift_32b": "S1",
        "cube_stack_distractor_32b": "S2",
        "cube_lift_observation_noise_7b": "P1",
        "cube_stack_observation_noise_7b": "P2",
        "nut_insertion_7b": "Q1",
        "door_opening_7b": "R1c",
        "cube_lift_deepseek": "F1",
        "cube_stack_deepseek": "F2",
    }
)

SAMPLING_TAGS.update({"different_label_donor": "s6", "random_compatible_label": "s8"})


def sampling_identity(value: object) -> str:
    """Resolve a descriptive label to the original identity used in random seeds."""
    import re

    text = str(value)
    if text in SAMPLING_TAGS.values() or text.startswith(("v3-", "v3d-", "c2e3-", "c6e7-")):
        return text
    pattern = "|".join(re.escape(name) for name in sorted(SAMPLING_TAGS, key=len, reverse=True))
    return re.sub(
        r"(?<![A-Za-z0-9_])(?:" + pattern + r")(?![A-Za-z0-9_])",
        lambda match: SAMPLING_TAGS[match.group()],
        text,
    )
