"""Run configuration, provenance, output directories, and random seeds."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("roborepair")
REQUIRED_ARTIFACTS = (
    "config_resolved.yaml",
    "command.txt",
    "git_commit.txt",
    "environment.txt",
    "stdout.log",
)


@dataclass
class RunContext:
    """Configuration and output state for an experiment run."""

    config: dict[str, Any]
    config_path: Path
    run_id: str
    seed: int
    output_dir: Path
    resume: bool


def build_arg_parser(description: str, *, allow_resume: bool = True) -> argparse.ArgumentParser:
    """Parse configuration, random seed, and output location."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML configuration.")
    parser.add_argument("--seed", required=True, type=int, help="Top-level seed for this run.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for run outputs.")
    parser.set_defaults(resume=False)
    if allow_resume:
        parser.add_argument("--resume", action="store_true", help="Continue an interrupted run.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Read an experiment configuration and reject invalid shared limits."""
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    for name in ("model", "decoding", "env"):
        if name in config and (not isinstance(config[name], dict)):
            raise ValueError(f"Config requires a {name!r} mapping.")
    decoding = config.get("decoding", {})
    for name in ("max_output_tokens", "code_timeout_s", "trace_input_token_cap"):
        if name in decoding and float(decoding[name]) <= 0:
            raise ValueError(f"decoding.{name} must be positive.")
    if not 0 < float(decoding.get("top_p", 1)) <= 1:
        raise ValueError("decoding.top_p must be in (0, 1].")
    if float(decoding.get("temperature", 0)) < 0:
        raise ValueError("decoding.temperature cannot be negative.")
    for section in config.values():
        if isinstance(section, dict):
            for name in ("max_retries", "wave_size"):
                if name in section and int(section[name]) <= 0:
                    raise ValueError(f"{name} must be positive.")
    return config


def current_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def environment_snapshot() -> str:
    """Record package versions without importing model or simulator modules."""
    from importlib.metadata import PackageNotFoundError, version

    lines = [f"python={sys.version.split()[0]}", f"platform={sys.platform}"]
    for package in ("torch", "transformers", "robosuite", "mujoco", "numpy"):
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "NOT_INSTALLED"
        lines.append(f"{package}={package_version}")
    return "\n".join(lines) + "\n"


def seed_everything(seed: int, config: dict[str, Any]) -> dict[str, int]:
    """Derive independent simulator, generation, training, and perturbation seeds."""
    from roborepair import seeding

    namespace_seeds = seeding.namespace_seeds(seed, config.get("seeds", {}))
    seeding.seed_everything(seed)
    LOGGER.info("Seeded namespaces: %s", namespace_seeds)
    return namespace_seeds


def bootstrap(args: argparse.Namespace) -> RunContext:
    """Create run artifacts, seed generators, and configure logging."""
    output_dir: Path = args.output_dir
    args.run_id = output_dir.name
    if output_dir.exists() and (not args.resume):
        raise SystemExit(
            f"Output directory {output_dir} already exists; use --resume or choose another directory."
        )
    config = load_config(args.config)
    if args.resume and (output_dir / "config_resolved.yaml").exists():
        import yaml

        recorded = yaml.safe_load((output_dir / "config_resolved.yaml").read_text())
        if recorded != config:
            raise ValueError("Cannot resume a run with a different resolved configuration.")
    identity = {"run_id": args.run_id, "seed": args.seed}
    identity_path = output_dir / "run_identity.json"
    if args.resume and output_dir.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("Cannot resume without a matching recorded run ID and seed.")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    (output_dir / "artifacts").mkdir(exist_ok=True)
    log_path = output_dir / "stdout.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(current_git_commit() + "\n", encoding="utf-8")
    (output_dir / "environment.txt").write_text(environment_snapshot(), encoding="utf-8")
    import yaml

    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    seed_everything(args.seed, config)
    LOGGER.info("Run %s bootstrapped at %s (resume=%s)", args.run_id, output_dir, args.resume)
    return RunContext(
        config=config,
        config_path=args.config,
        run_id=args.run_id,
        seed=args.seed,
        output_dir=output_dir,
        resume=args.resume,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSONL artifact (episodes / traces / predictions)."""
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Write metrics.json for downstream analyses."""
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    LOGGER.info("Wrote metrics.json with %d top-level keys", len(metrics))
