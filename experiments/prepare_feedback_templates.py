"""Build fixed task guidance and label controls from the development failure bank."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from roborepair.execution.run_context import (
    bootstrap,
    build_arg_parser,
    load_config,
)
from roborepair.feedback.constraints import compile_constraints, render_constraints_text

GENERAL_FAILURE_LABELS = [
    "syntax_error",
    "timeout",
    "invalid_api",
    "no_api_calls",
    "target_not_localized",
    "unsafe_path",
]


MANIPULATION_FAILURE_LABELS = [
    "missing_grasp",
    "failed_grasp",
    "grasp_without_goto",
    "grasp_while_holding",
    "object_not_lifted",
    "released_after_lift",
]


COMPATIBLE_FAILURE_LABELS = {
    "cube_lift": GENERAL_FAILURE_LABELS + MANIPULATION_FAILURE_LABELS,
    "cube_stack": GENERAL_FAILURE_LABELS + MANIPULATION_FAILURE_LABELS + ["bad_placement"],
}


ALL_FAILURE_LABELS = GENERAL_FAILURE_LABELS + MANIPULATION_FAILURE_LABELS + ["bad_placement"]


PERMUTATION_LABELS = ALL_FAILURE_LABELS


def render_guidance_for_labels(labels: list[str], task: str) -> str:
    """Render labels (most-important first) in the frozen template_feedback/fixed_task_guidance text format."""
    sets = []
    n = len(labels)
    for i, lab in enumerate(reversed(labels)):
        sets.append(
            compile_constraints(
                {"failure_labels": [lab], "task_id": task},
                episode_id="static-prior",
                attempt=i,
                tier="high_level",
            )
        )
    assert len(sets) == n
    return render_constraints_text(sets, cap_tokens=768)


def derange_failure_labels(vocab: list[str], seed: int = 0) -> dict[str, str]:
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = list(range(len(vocab)))
    while True:
        perm = rng.permutation(idx)
        if all((int(p) != i for i, p in enumerate(perm))):
            return {vocab[i]: vocab[int(p)] for i, p in enumerate(perm)}


def prepare_feedback_templates(ctx) -> None:
    development_bank = Path(ctx.config["feedback_templates"]["development_bank"])
    output_dir = ctx.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = [json.loads(line) for line in development_bank.read_text().splitlines() if line.strip()]
    freqs = {
        task: Counter(
            r["failure_labels"][0] for r in dev if r["task"] == task and r.get("failure_labels")
        )
        for task in COMPATIBLE_FAILURE_LABELS
    }
    artifacts: dict[str, str] = {}

    def write(name: str, text: str) -> None:
        p = output_dir / name
        p.write_text(text)
        artifacts[name] = hashlib.sha256(text.encode()).hexdigest()

    write("compatibility.json", json.dumps(COMPATIBLE_FAILURE_LABELS, indent=2) + "\n")
    write(
        "development_label_frequencies.json",
        json.dumps({t: dict(c.most_common()) for t, c in freqs.items()}, indent=2) + "\n",
    )
    for task, compat in COMPATIBLE_FAILURE_LABELS.items():
        observed = [lab for lab, _ in freqs[task].most_common() if lab in compat]
        rest = [lab for lab in compat if lab not in observed]
        union_order = observed + rest
        write(
            f"compatible_template_union_{task}_high_level.txt",
            render_guidance_for_labels(union_order, task) + "\n",
        )
        write(
            f"fixed_task_template_{task}_high_level.txt",
            render_guidance_for_labels(observed[:1], task) + "\n",
        )
        write(
            f"three_task_templates_{task}_high_level.txt",
            render_guidance_for_labels(observed[:3], task) + "\n",
        )
    write(
        "label_permutation.json",
        json.dumps(derange_failure_labels(PERMUTATION_LABELS, seed=0), indent=2) + "\n",
    )
    write_manifest = {
        "dev_bank": str(development_bank),
        "dev_bank_sha256": hashlib.sha256(development_bank.read_bytes()).hexdigest(),
        "artifacts": artifacts,
    }
    (output_dir / "guidance_manifest.json").write_text(json.dumps(write_manifest, indent=2) + "\n")
    print(json.dumps(write_manifest, indent=2))


def main() -> None:
    parser = build_arg_parser(__doc__, allow_resume=False)
    args = parser.parse_args()
    config = load_config(args.config)
    if not isinstance(config.get("feedback_templates"), dict):
        parser.error("Template preparation requires a feedback_templates configuration.")
    prepare_feedback_templates(bootstrap(args))


if __name__ == "__main__":
    main()
