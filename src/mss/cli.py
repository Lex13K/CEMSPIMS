"""CLI entrypoints for CEMSPIMS pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mss.analysis.compare_config import (
    config_path_for_run,
    build_compare_run_specs,
    load_compare_runs_config,
    preset_label_overrides,
    resolve_preset,
)
from mss.analysis.compare_runs import run_compare_runs
from mss.config.validate import collect_config_warnings
from mss.io.config import load_resolved_config, sanitize_run_id
from mss.io.paths import project_root
from mss.pipeline.order import CANONICAL_PIPELINE_ORDER
from mss.pipeline.orchestrator import describe_pipelines, list_pipeline_names, run_pipelines
from mss.run.branch import branch_run
from mss.run.registry import diff_configs, format_run_tree, list_runs, summarize_config_diff
from mss.run_copy import copy_run


def _configs_dir_from_arg(configs_dir: str | None) -> Path:
    if configs_dir:
        return Path(configs_dir).resolve()
    return (project_root() / "configs").resolve()


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        rid = sanitize_run_id(args.run)
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        config_path = config_path_for_run(cdir, rid)
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
        config_path = config_path_for_run(cdir, rid)
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
    print(f"  run_id:            {cfg.run_id}")
    print(f"  project_root:      {cfg.project_root}")
    print(f"  raw_dir:           {cfg.raw_dir}")
    print(f"  shared_interim_dir: {cfg.shared_interim_dir}")
    print(f"  run_interim_dir:   {cfg.run_interim_dir}")
    print(f"  processed_dir:     {cfg.processed_dir}")
    print(f"  manifest_path:     {cfg.manifest_path}")
    warnings = collect_config_warnings(config_path, configs_dir=cdir)
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
        if getattr(args, "strict", False):
            return 1
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


def _cmd_branch(args: argparse.Namespace) -> int:
    try:
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        path = branch_run(
            args.parent,
            args.child,
            at_pipeline=args.at,
            configs_dir=cdir,
        )
        print(f"Branched {args.parent!r} -> {args.child!r} at {args.at!r}")
        print(f"  config: {path}")
        print(f"  manifest: data/{args.child}/run_manifest.json")
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
    records = list_runs(cdir)
    if not records:
        print("(no runs)")
        return 0
    print(f"{'run_id':<20} {'parent':<16} {'branch_at':<18} {'created_at'}")
    print("-" * 80)
    for r in records:
        parent = r.parent_run_id or "-"
        branch = r.branch_pipeline or "-"
        created = (r.created_at or "-")[:19]
        print(f"{r.run_id:<20} {parent:<16} {branch:<18} {created}")
    return 0


def _cmd_run_tree(args: argparse.Namespace) -> int:
    cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
    print(format_run_tree(cdir))
    return 0


def _cmd_diff_config(args: argparse.Namespace) -> int:
    try:
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        if args.summary:
            changes = summarize_config_diff(args.run_a, args.run_b, configs_dir=cdir)
            if not changes:
                print(f"No key differences between {args.run_a} and {args.run_b}.")
            else:
                print(f"Config diff: {args.run_a} vs {args.run_b}")
                print("\n".join(changes))
        else:
            print(diff_configs(args.run_a, args.run_b, configs_dir=cdir), end="")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _merge_label_overrides(
    toml_overrides: tuple,
    preset_overrides: tuple,
) -> tuple:
    by_id = {o.run_id: o for o in preset_overrides}
    for o in toml_overrides:
        by_id[o.run_id] = o
    return tuple(by_id.values())


def _cmd_compare_runs(args: argparse.Namespace) -> int:
    try:
        cdir = _configs_dir_from_arg(getattr(args, "configs_dir", None))
        preset = getattr(args, "preset", None)
        if preset:
            anchor_id, peers = resolve_preset(preset)
            preset_overrides = preset_label_overrides(preset)
        else:
            if not args.anchor:
                raise ValueError("--anchor is required unless --preset is set")
            if not args.runs:
                raise ValueError("--runs is required unless --preset is set")
            anchor_id = sanitize_run_id(args.anchor)
            peers = tuple(x.strip() for x in args.runs.split(",") if x.strip())
            preset_overrides = ()

        comparison_id = (args.comparison_id or "").strip()
        if not comparison_id and preset:
            comparison_id = preset
        if not comparison_id:
            raise ValueError("--comparison-id is required (or set [analysis.compare_runs].comparison_id)")

        config_path = config_path_for_run(cdir, anchor_id)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        anchor_cfg = load_resolved_config(config_path, anchor_id)
        cr = load_compare_runs_config(config_path)
        label_overrides = _merge_label_overrides(cr.run_labels, preset_overrides)
        specs = build_compare_run_specs(
            anchor_id,
            peers,
            configs_dir=cdir,
            label_overrides=label_overrides,
        )
        out = run_compare_runs(
            anchor_cfg,
            specs,
            comparison_id,
            overwrite=bool(args.overwrite),
        )
        print(f"Comparison written: {out}")
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.server.launch import main as app_main

    argv: list[str] = []
    if args.port != 8765:
        argv.extend(["--port", str(args.port)])
    if args.host != "127.0.0.1":
        argv.extend(["--host", args.host])
    if args.no_open:
        argv.append("--no-open")
    if args.web_dist:
        argv.extend(["--web-dist", args.web_dist])
    return app_main(argv)


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

    p_vc = sub.add_parser("validate-config", help="Load resolved paths and print config warnings.")
    p_vc.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_vc.add_argument(
        "--run",
        type=str,
        default="default",
        help="Run id; loads <configs-dir>/<run_id>.toml (default: default).",
    )
    p_vc.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any warnings are present.",
    )
    p_vc.set_defaults(func=_cmd_validate_config)

    p_copy = sub.add_parser(
        "copy",
        help="[DEPRECATED] Full copy of interim/processed; prefer branch.",
    )
    p_copy.add_argument("src_run", type=str, help="Source run id (configs/<id>.toml and data/<id>/...).")
    p_copy.add_argument("dst_run", type=str, help="Destination run id (must not already exist).")
    p_copy.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_copy.set_defaults(func=_cmd_copy)

    p_branch = sub.add_parser("branch", help="Branch a new run from a parent at a pipeline boundary.")
    p_branch.add_argument("--parent", required=True, help="Parent run id.")
    p_branch.add_argument("--child", required=True, help="New child run id (must not exist).")
    p_branch.add_argument(
        "--at",
        required=True,
        choices=list(CANONICAL_PIPELINE_ORDER),
        help="Pipeline boundary to branch from (child reruns from here).",
    )
    p_branch.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_branch.set_defaults(func=_cmd_branch)

    p_lr = sub.add_parser("list-runs", help="List configured runs and lineage metadata.")
    p_lr.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_lr.set_defaults(func=_cmd_list_runs)

    p_rt = sub.add_parser("run-tree", help="Print parent/child experiment tree.")
    p_rt.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_rt.set_defaults(func=_cmd_run_tree)

    p_dc = sub.add_parser("diff-config", help="Compare two run configs.")
    p_dc.add_argument("run_a", type=str, help="First run id.")
    p_dc.add_argument("run_b", type=str, help="Second run id.")
    p_dc.add_argument(
        "--summary",
        action="store_true",
        help="Key-level TOML diff instead of unified diff.",
    )
    p_dc.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_dc.set_defaults(func=_cmd_diff_config)

    p_lp = sub.add_parser("list-pipelines", help="List registered pipelines and steps.")
    p_lp.set_defaults(func=_cmd_list_pipelines)

    p_cr = sub.add_parser(
        "compare-runs",
        help="Build cross-run Appendix E-style comparison under anchor processed/comparisons/.",
    )
    p_cr.add_argument("--configs-dir", type=str, default=None, metavar="DIR", help=configs_help)
    p_cr.add_argument(
        "--anchor",
        type=str,
        default=None,
        help="Anchor run id (comparison output lives under data/<anchor>/processed/comparisons/).",
    )
    p_cr.add_argument(
        "--runs",
        type=str,
        default=None,
        metavar="ID,ID,...",
        help="Comma-separated peer run ids (anchor is always included).",
    )
    p_cr.add_argument(
        "--comparison-id",
        type=str,
        default=None,
        help="Output folder name under processed/comparisons/ (default: --preset name when preset set).",
    )
    p_cr.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=("v2_matrix",),
        help="Use a built-in run matrix (implies anchor + peers).",
    )
    p_cr.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild comparison even if comparison_manifest.json exists.",
    )
    p_cr.set_defaults(func=_cmd_compare_runs)

    p_app = sub.add_parser("app", help="Launch local experiment graph UI (FastAPI + React).")
    p_app.add_argument("--port", type=int, default=8765)
    p_app.add_argument("--host", type=str, default="127.0.0.1")
    p_app.add_argument("--no-open", action="store_true", help="Do not open browser automatically.")
    p_app.add_argument("--web-dist", type=str, default=None, metavar="DIR", help="Built frontend dist path.")
    p_app.set_defaults(func=_cmd_app)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
