"""Commands: `probe`, `plan`, `sample`, `assess`, `repair`.

`probe` and `plan` are safe anywhere and run no model pass. `sample` needs the
GPU and refuses without one. `assess` works on files already on disk, so it
runs on any machine. `repair` needs no GPU of its own but does need a model
endpoint to talk to, and refuses when the harness is not ready.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from moltrain import hostcheck, logs

from . import harness, repair, runner
from .config import ConfigError, load_config, resolve
from .triage import assess_all, discover, write_report

LOG = logs.get_logger("protacbench")


async def _cmd_probe(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    print("rdkit / openbabel")
    lines, problems = hostcheck.report_binaries(("obabel",))
    for line in lines:
        print(line)
    reports, import_problems = hostcheck.report_imports(("rdkit", "torch", "yaml"))
    for report in reports:
        print(
            f"  {report.name}: {report.detail if report.ok else 'FAILED - ' + report.detail}"
        )
    problems.extend(import_problems)

    print("\nPro-GAT")
    runtime = await runner.probe(cfg)
    for key, value in runtime.as_dict().items():
        print(f"  {key}: {value}")
    for problem in runtime.problems:
        print(f"  ! {problem}")

    print("\nrepair harness")
    harness_runtime = harness.probe(cfg)
    for key, value in harness_runtime.as_dict().items():
        print(f"  {key}: {value}")
    for problem in harness_runtime.problems:
        print(f"  ! {problem}")

    print("\nsamples on disk")
    for stage, key in (("generated", "output_dir"), ("repaired", None)):
        root = (
            resolve(cfg, cfg["progat"]["output_dir"])
            if key
            else resolve(cfg, cfg["assess"]["repaired_dir"])
        )
        found = discover(root, stage)
        print(f"  {stage}: {len(found)} samples under {root}")

    print()
    if problems:
        for problem in problems:
            print(f"! {problem}")
        return 1
    print("ready")
    return 0


async def _cmd_plan(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Print the Pro-GAT command without running any of it."""
    output_dir = resolve(cfg, cfg["progat"]["output_dir"])
    print(f"# cwd: {runner.repo_path(cfg)}")
    print(" ".join(runner.build_command(cfg, output_dir)))
    size = int(cfg["progat"]["linker_size"])
    if size <= 0:
        print(
            "\n# linker_size 0 means each PROTAC keeps its own reference linker\n"
            "# length. run_progat.py only installs a fixed-size sample_fn when\n"
            "# the value is positive."
        )
    else:
        print(
            f"\n# every PROTAC will be forced to a {size}-atom linker, whatever\n"
            "# its own geometry calls for."
        )
    return 0


async def _cmd_sample(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    runtime = await runner.probe(cfg)
    if not runtime.cuda_available and not cfg["progat"]["allow_cpu"]:
        print("refusing to sample: no CUDA device visible to Pro-GAT")
        for problem in runtime.problems:
            print(f"  ! {problem}")
        return 2
    output = await runner.sample(cfg, runtime=runtime, refresh=args.refresh)
    print(f"samples written under {output}")
    return 0


async def _cmd_assess(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    assessments, report = assess_all(
        cfg, limit=args.limit, disable_progress=logs.progress_disabled(cfg, args)
    )
    if not assessments:
        print(
            "no samples found. Run `python -m protacbench sample` first, or "
            "point progat.output_dir at an existing run.",
            file=sys.stderr,
        )
        return 1

    paths = write_report(cfg, assessments, report)
    summary = report.as_dict()

    print(f"\nassessed {summary['total_samples']} samples")
    print(f"  verdicts: {summary['by_verdict']}")
    for stage, validity in summary["validity_percent"].items():
        counts = summary["by_stage"][stage]
        print(f"  {stage}: {validity}% ok  {counts}")
    if summary["rescued_by_progat"]:
        print(f"  rescued by Pro-GAT repair: {summary['rescued_by_progat']}")
    if summary["broken_after_repair"]:
        print(f"  still broken after repair: {summary['broken_after_repair']}")
    print(f"  unique valid molecules: {summary['unique_valid_smiles']}")
    print(f"  recovered the reference: {summary['recovered_reference']}")

    scores = summary["synthesis_score"]
    if scores.get("scored"):
        print(
            f"\n  synthesis score over {scores['scored']} scored samples:"
            f" median {scores['median']}, mean {scores['mean']}"
            f" (range {scores['min']}-{scores['max']})"
        )
        print(
            f"    {scores['at_least_80']} at 80+, "
            f"{scores['at_least_60']} at 60+"
        )

    if summary["failure_counts"]:
        print("\n  why samples failed:")
        for code, count in list(summary["failure_counts"].items())[:12]:
            print(f"    {count:>6}  {code}")

    print()
    for label, path in paths.items():
        print(f"{label:>14}: {path}")
    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


async def _cmd_repair(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Run the rejected samples through the harness agent and verify what
    comes back."""
    tasks_file = resolve(cfg, cfg["repair"]["tasks_file"])
    if not tasks_file.is_file():
        print(
            f"no repair tasks at {tasks_file}. Run "
            "`python -m protacbench assess` first.",
            file=sys.stderr,
        )
        return 1

    runtime = harness.probe(cfg)
    if runtime.problems:
        print("refusing to repair: the harness is not ready", file=sys.stderr)
        for problem in runtime.problems:
            print(f"  ! {problem}", file=sys.stderr)
        return 2

    tasks = repair.read_tasks(tasks_file, limit=args.limit)
    if not tasks:
        print(f"no tasks in {tasks_file}", file=sys.stderr)
        return 1

    print(f"repairing {len(tasks)} sample(s) via {runtime.model} at {runtime.base_url}")
    outcomes = await repair.run_batch(
        cfg, tasks, disable_progress=logs.progress_disabled(cfg, args)
    )
    paths = repair.write_report(cfg, outcomes)
    summary = repair.summarise(outcomes)

    print(f"\nran {summary['tasks']} repair task(s)")
    print(f"  answered: {summary['answered']}")
    print(
        f"  salvaged: {summary['salvaged']} "
        f"({summary['salvage_rate_percent']}%) -- "
        f"{summary['salvaged_from_broken']} from broken, "
        f"{summary['salvaged_from_poor']} from poor"
    )
    if summary["fragments_lost"]:
        print(
            f"  ! {summary['fragments_lost']} answer(s) dropped a fragment and "
            "were rejected"
        )
    if summary["mean_score_delta"] is not None:
        print(f"  mean score change: {summary['mean_score_delta']:+}")
    print(f"  unique repaired molecules: {summary['unique_repaired_smiles']}")

    print("\n  outcomes:")
    for status, count in summary["by_status"].items():
        print(f"    {count:>6}  {status}: {repair.STATUSES.get(status, '')}")

    print()
    for label, path in paths.items():
        print(f"{label:>14}: {path}")
    if args.json:
        print(json.dumps(summary, indent=2))
    return 0


COMMANDS = {
    "probe": _cmd_probe,
    "plan": _cmd_plan,
    "sample": _cmd_sample,
    "assess": _cmd_assess,
    "repair": _cmd_repair,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protacbench",
        description="Assess and triage DiffPROTAC -> Pro-GAT linker samples.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap the sample count")
    parser.add_argument(
        "--refresh", action="store_true", help="redo work already on disk"
    )
    parser.add_argument("--json", action="store_true", help="also dump JSON")
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logs.configure_from_args(cfg, args, stem=f"protacbench-{args.command}")

    try:
        return asyncio.run(COMMANDS[args.command](cfg, args))
    except (runner.RunnerError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
