"""Build initial and retry prompts with shared API documentation.

Feedback uses a whitespace-word cap. Resampling omits previous code and traces;
the mismatched-feedback control substitutes a donor episode's trace.
"""

from __future__ import annotations

CONDITIONS = (
    "resample",
    "terminal_failure_feedback",
    "execution_log_feedback",
    "execution_trace_feedback",
    "execution_trace_without_diagnosis",
    "foreign_trace_feedback",
    "failure_label_feedback",
)

_PLAN_PATCH_INSTRUCTION = (
    "Apply the repair plan above. Preserve the parts listed under must_preserve. Write "
    "the COMPLETE corrected Python program using ONLY the API above, following the same "
    "execution rules (no imports, no redefining API functions, call any helper you "
    "define). Satisfy every must_change item and the verification checklist. Output only code."
)
_PLAN_REGEN_INSTRUCTION = (
    "Using the repair plan above (compiled from a previous failed attempt), write a "
    "COMPLETE fresh Python program using ONLY the API above, following the same "
    "execution rules (no imports, no redefining API functions, call any helper you "
    "define). Satisfy every must_change item and the verification checklist. Output only code."
)

# Neutral suffixes avoid references to plan fields missing from some conditions.
# The historical default preserves the original feedback-control prompts.
NEUTRAL_PATCH_INSTRUCTION = (
    "Revise the previous program above into a COMPLETE corrected Python program "
    "using ONLY the API above, following the same execution rules (no imports, no "
    "redefining API functions, call any helper you define). Follow every "
    "instruction present in this prompt. Output only code."
)
NEUTRAL_REGEN_INSTRUCTION = (
    "Write a COMPLETE fresh Python program using ONLY the API above, following the "
    "same execution rules (no imports, no redefining API functions, call any helper "
    "you define). Follow every instruction present in this prompt. Output only code."
)

_PLAN_SUFFIXES = {
    "historical": {"patch": _PLAN_PATCH_INSTRUCTION, "regenerate": _PLAN_REGEN_INSTRUCTION},
    "neutral": {"patch": NEUTRAL_PATCH_INSTRUCTION, "regenerate": NEUTRAL_REGEN_INSTRUCTION},
}


def build_feedback_prompt(
    *,
    plan_text: str,
    mode: str,
    prev_code: str | None,
    api_doc: str,
    env_obs,
    token_budget: int,
    suffix: str = "historical",
) -> str:
    """Build an editing or regeneration prompt with capped feedback.

    Editing includes the previous program; regeneration omits it. The suffix
    selects the historical or condition-neutral generation instruction.
    """
    if mode not in ("patch", "regenerate"):
        raise ValueError(f"unknown plan repair mode {mode!r}")
    if suffix not in _PLAN_SUFFIXES:
        raise ValueError(f"unknown plan repair suffix {suffix!r}")
    header = f"# API\n{api_doc}\n# Task\n{env_obs.instruction}\n"
    # Keep the same scene-object list across initial and retry prompts.
    names = getattr(env_obs, "object_names", None)
    if names:
        header += f"# Objects\n{', '.join(names)}\n"
    body = [header]
    if mode == "patch" and prev_code is not None:
        body.append(f"# Previous program\n{prev_code}")
    if plan_text is not None:
        # Omitting plan_text keeps the scaffold while removing the feedback block.
        body.append("# Repair plan\n" + truncate_feedback_words(plan_text, token_budget))
    body.append("# " + _PLAN_SUFFIXES[suffix][mode] + "\n")
    return "\n".join(body)


# Shared instruction requests a complete runnable program in every repair condition.
_REPAIR_INSTRUCTION = (
    "Repair the program. Write the COMPLETE corrected Python program using ONLY the API "
    "above, following the same execution rules (no imports, no redefining API functions, "
    "call any helper you define). Keep the parts that worked; fix the cause of failure. "
    "Output only code."
)


def truncate_feedback_words(text: str, cap: int) -> str:
    toks = text.split()
    if len(toks) <= cap:
        return text
    return " ".join(toks[:cap])


def build_initial_prompt(env_obs, api_doc: str) -> str:
    """Render the initial-program prompt: API doc, task instruction, and object list."""
    objs = ", ".join(env_obs.object_names)
    return (
        f"# API\n{api_doc}\n"
        f"# Task\n{env_obs.instruction}\n"
        f"# Objects\n{objs}\n"
        f"# Write a Python program using ONLY the API above.\n"
    )


def _render_trace_feedback(trace, include_diagnosis: bool) -> str:
    # Events provide episode-specific evidence even when label and progress match.
    lines = [
        f"task_success: {trace.task_success}",
        f"progress: {trace.progress:.3f}",
        "subgoals: " + ", ".join(f"{k}={v}" for k, v in trace.subgoals.items()),
        "failure_labels: " + ", ".join(trace.failure_labels),
    ]
    if include_diagnosis:
        lines.append(f"diagnosis: {trace.diagnosis}")
    if trace.events:
        lines.append("execution_events:")
        lines.extend(f"  {e}" for e in trace.events)
    else:
        lines.append("execution_events: (none - the program made no API calls)")
    off_table = trace.metrics.get("off_table", 0.0) if trace.metrics else 0.0
    lines.append(f"object_off_table: {bool(off_table >= 1.0)}")
    return "\n".join(lines)


def render_execution_log(raw_log: str) -> str:
    return raw_log


def build_repair_prompt(
    condition: str,
    *,
    prev_code: str | None,
    trace=None,
    raw_log: str | None = None,
    api_doc: str,
    env_obs,
    token_budget: int,
    primary_label: str | None = None,
) -> str:
    """Build the repair prompt for ``condition``. API doc is identical across conditions."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown repair condition {condition!r}")

    header = f"# API\n{api_doc}\n# Task\n{env_obs.instruction}\n"

    if condition == "resample":
        return header + "# Write a fresh Python program using ONLY the API above.\n"

    body = [header]
    if prev_code is not None:
        body.append(f"# Previous program\n{prev_code}")

    if condition == "terminal_failure_feedback":
        body.append("# Feedback\ntask_success: False")
    elif condition == "execution_log_feedback":
        body.append(
            "# Feedback (raw log)\n"
            + truncate_feedback_words(render_execution_log(raw_log or ""), token_budget)
        )
    elif condition == "execution_trace_feedback":
        fb = _render_trace_feedback(trace, include_diagnosis=True)
        body.append("# Feedback (trace)\n" + truncate_feedback_words(fb, token_budget))
    elif condition == "execution_trace_without_diagnosis":
        fb = _render_trace_feedback(trace, include_diagnosis=False)
        body.append("# Feedback (trace)\n" + truncate_feedback_words(fb, token_budget))
    elif condition == "foreign_trace_feedback":
        fb = _render_trace_feedback(trace, include_diagnosis=True)
        body.append("# Feedback (trace)\n" + truncate_feedback_words(fb, token_budget))
    elif condition == "failure_label_feedback":
        body.append(f"# Feedback\nprimary_failure: {primary_label or 'unknown'}\nFix this failure.")

    body.append(f"# {_REPAIR_INSTRUCTION}\n")
    return "\n".join(body)
