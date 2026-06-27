"""CLI entrypoints for CEMSPIMS pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.pipeline.orchestrator import describe_pipelines, list_pipeline_names, run_pipelines
from mss.run_copy import copy_run


def _configs_dir_from_arg(configs_dir: str | None) -> Path:
    if configs_dir:
        return Path(configs_dir).resolve()
    return (project_root() / "configs").resolve()


def _config_path_for_run(configs_dir: Path, run_id: str) -> Path:
    return (configs_dir / f"{run_id}.toml").resolve()


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        rid = sanitize_run_id(args.run)
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        config_path = _config_path_for_run(cdir, rid)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    try:
        pipelines = args.pipelines if args.pipelines else None
        run_pipelines(
            config_path,
            rid,
            pipelines=pipelines,
            overwrite_arg=args.overwrite,
            skip_validate=args.skip_validate,
            atol=args.atol,
            rtol=args.rtol,
            step_from=args.step_from,
            step_to=args.step_to,
        )
    except (FileNotFoundError, FileExistsError, ValueError, AssertionError, RuntimeError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        rid = sanitize_run_id(args.run)
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        config_path = _config_path_for_run(cdir, rid)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    try:
        cfg = load_resolved_config(config_path, rid)
    except Exception as e:
        print(f"Invalid config: {e}", file=sys.stderr)
        return 1
    print("Config OK.")
    print(f"  run_id:       {cfg.run_id}")
    print(f"  project_root: {cfg.project_root}")
    print(f"  raw_dir:      {cfg.raw_dir}")
    print(f"  interim_dir:  {cfg.interim_dir}")
    print(f"  processed_dir: {cfg.processed_dir}")
    return 0


def _cmd_copy(args: argparse.Namespace) -> int:
    try:
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        copy_run(args.src_run, args.dst_run, configs_dir=cdir)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_list_pipelines(_args: argparse.Namespace) -> int:
    print("Pipelines (steps in order):")
    print(describe_pipelines())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mss", description="CEMSPIMS pipeline CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    configs_help = "Directory containing <run_id>.toml (default: configs/ under project root)."

    p_run = sub.add_parser("run", help="Run one or more pipelines for a run id.")
    p_run.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=configs_help,
    )
    p_run.add_argument(
        "--run",
        type=str,
        default="default",
        help="Run id; config is <configs-dir>/<run_id>.toml; data under data/<run_id>/... (default: default).",
    )
    p_run.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        metavar="NAME",
        choices=list_pipeline_names(),
        help="Pipeline to run (repeatable). Omit to run all pipelines in canonical order.",
    )
    p_run.add_argument(
        "--overwrite",
        nargs="*",
        default=None,
        metavar="PIPELINE",
        choices=list_pipeline_names(),
        help="Force regeneration: without this flag, each step is skipped if its outputs already exist "
        "(resume). --overwrite alone = all selected pipelines may replace outputs; "
        "--overwrite P ... = only those pipelines. Pipeline names must be among selected.",
    )
    p_run.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip CSV vs Parquet ingest validation (ingest step only).",
    )
    p_run.add_argument("--atol", type=float, default=1e-8, help="Validation absolute tolerance.")
    p_run.add_argument("--rtol", type=float, default=1e-8, help="Validation relative tolerance.")
    p_run.add_argument(
        "--from",
        dest="step_from",
        type=str,
        default=None,
        metavar="STEP",
        help="First step id to run (inclusive); only when exactly one pipeline is selected.",
    )
    p_run.add_argument(
        "--to",
        dest="step_to",
        type=str,
        default=None,
        metavar="STEP",
        help="Last step id to run (inclusive); only when exactly one pipeline is selected.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_vc = sub.add_parser("validate-config", help="Load and print resolved paths.")
    p_vc.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_vc.add_argument(
        "--run",
        type=str,
        default="default",
        help="Run id; loads <configs-dir>/<run_id>.toml (default: default).",
    )
    p_vc.set_defaults(func=_cmd_validate_config)

    p_copy = sub.add_parser(
        "copy",
        help="Copy config + interim/processed from one run id to another; rewrite absolute paths in JSON.",
    )
    p_copy.add_argument("src_run", type=str, help="Source run id (configs/<id>.toml and data/<id>/...).")
    p_copy.add_argument("dst_run", type=str, help="Destination run id (must not already exist).")
    p_copy.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_copy.set_defaults(func=_cmd_copy)

    p_lp = sub.add_parser("list-pipelines", help="List registered pipelines and steps.")
    p_lp.set_defaults(func=_cmd_list_pipelines)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
