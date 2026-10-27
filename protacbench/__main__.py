"""Commands: `probe`, `plan`, `sample`, `assess`, `prompts`, `repair`, `grade`.

`probe` and `plan` are safe anywhere and run no model pass. `sample` needs the
GPU and refuses without one. `assess` works on files already on disk, so it
runs on any machine. `repair` needs no GPU of its own but does need a model
endpoint to talk to, and refuses when the harness is not ready. `prompts` and
`grade` are the two ends of a hand-run loop: one writes paste-ready tasks, the
other scores the replies that come back, and neither needs an endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

from moltrain import hostcheck, logs

from . import grade as grading
from . import harness, prompts, repair, runner
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


def _reactive_in_linker(linker_smiles: str) -> list[str]:
    """Reactive groups in a linker read back from a CSV.

    Recomputed rather than read from a column, so this works on a samples.csv
    written before the reactive-group gate existed.
    """
    if not linker_smiles:
        return []
    from rdkit import Chem, rdBase

    from .assess import DISQUALIFYING_GROUPS, groups_in_linker

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(linker_smiles)
    if mol is None:
        return []
    return groups_in_linker(mol, list(range(mol.GetNumAtoms())), DISQUALIFYING_GROUPS)


async def _cmd_prompts(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Write paste-ready repair prompts, one per molecule worth optimising."""
    settings = cfg["promptout"]
    source = Path(args.source) if args.source else (
        resolve(cfg, cfg["report"]["output_dir"]) / cfg["report"]["samples_csv"]
    )
    if not source.is_file():
        print(
            f"no samples at {source}. Run `python -m protacbench assess` "
            "first, or pass --source.",
            file=sys.stderr,
        )
        return 1

    wanted = set(settings["classes"])
    out_dir = resolve(cfg, settings["output_dir"])
    task_dir = out_dir / "tasks"

    # Clear the directory first. Without this, prompts from an earlier run
    # survive for molecules this run deliberately skipped -- a stale task is
    # indistinguishable from a current one once it is sitting in a folder you
    # are working through, so you would send exactly the molecules you had
    # decided to exclude.
    if task_dir.is_dir():
        removed = 0
        for stale in task_dir.glob("*.txt"):
            stale.unlink()
            removed += 1
        if removed:
            LOG.info("cleared %d prompt(s) from a previous run", removed)
    task_dir.mkdir(parents=True, exist_ok=True)

    system_path = out_dir / settings["system_file"]
    system_path.write_text(prompts.SYSTEM_PROMPT)

    counts: dict[str, int] = {}
    written = 0
    recovered = 0
    manifest: list[dict[str, Any]] = []
    tasks_path = out_dir / settings["tasks_jsonl"]

    with source.open(encoding="utf-8", errors="replace", newline="") as handle, \
            tasks_path.open("w") as tasks_out:
        for row in csv.DictReader(handle):
            reactive = _reactive_in_linker(row.get("linker_smiles") or "")
            group = prompts.classify_row(row, reactive)
            counts[group] = counts.get(group, 0) + 1
            if group not in wanted:
                continue

            # The generator sometimes reproduces the published molecule
            # exactly. Repairing one is not a repair task -- the input is
            # already the answer -- and acceptance test 9 would reject
            # anything close to it. The column is read to decide this and
            # still never reaches the prompt.
            if settings["skip_reference_matches"] and (
                row.get("recovers_reference") or ""
            ).strip().lower() == "true":
                recovered += 1
                continue
            if args.limit and written >= args.limit:
                continue

            text = prompts.build_prompt_from_row(row, reactive)
            name = f"{row.get('target_id','x')}-{row.get('sample_id','x')}.txt"
            (task_dir / name).write_text(text)
            tasks_out.write(
                json.dumps(
                    {
                        "task_id": f"{row.get('target_id')}-{row.get('sample_id')}",
                        "target_id": row.get("target_id"),
                        "sample_id": row.get("sample_id"),
                        "optimisation_class": group,
                        "failure_codes": row.get("failure_codes", ""),
                        "reactive_groups": ";".join(reactive),
                        "protac_smiles": row.get("protac_smiles", ""),
                        "linker_smiles": row.get("linker_smiles", ""),
                        "fragment_smiles": row.get("fragment_smiles", ""),
                        "prompt": text,
                    }
                )
                + "\n"
            )
            manifest.append(
                {
                    "task_id": f"{row.get('target_id')}-{row.get('sample_id')}",
                    "optimisation_class": group,
                    "failure_codes": row.get("failure_codes", ""),
                    "reactive_groups": ";".join(reactive),
                    "prompt_file": str(task_dir / name),
                }
            )
            written += 1

    manifest_path = out_dir / settings["manifest_csv"]
    if manifest:
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)

    print(f"read {sum(counts.values())} samples from {source}\n")
    print("  by optimisation class:")
    for group in sorted(counts, key=lambda g: -counts[g]):
        mark = "  -> prompts written" if group in wanted else "     skipped"
        print(f"    {counts[group]:>6}  {group:<12}{mark}")
    if recovered:
        print(
            f"\n  skipped {recovered} sample(s) whose input already equals the\n"
            "  published reference molecule -- those are not repair tasks."
        )
    print(f"\n  wrote {written} prompt(s)")

    # Say it plainly: the reference molecule is the published answer, and a
    # run that leaks it measures recall rather than repair.
    print(
        f"\n  withheld from every prompt: "
        f"{', '.join(sorted(prompts.WITHHELD_COLUMNS))}\n"
        "  reference_smiles is the real published PROTAC for that target. An\n"
        "  agent that sees it reads the answer off instead of repairing."
    )
    print()
    print(f"{'system prompt':>16}: {system_path}")
    print(f"{'one per task':>16}: {task_dir}/")
    print(f"{'structured':>16}: {tasks_path}")
    if manifest:
        print(f"{'manifest':>16}: {manifest_path}")
    return 0


async def _cmd_grade(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Score one or more model replies against the rubric they were given."""
    source = Path(args.source) if args.source else (
        resolve(cfg, cfg["report"]["output_dir"]) / cfg["report"]["samples_csv"]
    )
    if not source.is_file():
        print(f"no samples at {source}. Pass --source.", file=sys.stderr)
        return 1

    if args.reply_dir:
        replies = sorted(Path(args.reply_dir).glob("*.txt"))
    elif args.reply:
        replies = [Path(args.reply)]
    else:
        print("pass --reply FILE or --reply-dir DIR", file=sys.stderr)
        return 2
    if not replies:
        print("no reply files found", file=sys.stderr)
        return 1

    reports: list[grading.Report] = []
    for path in replies:
        # The task id comes from --task, or from the filename, which is how
        # `prompts` names the task files the reply answers.
        task_id = args.task or path.stem
        baseline = grading.load_baseline(source, task_id)
        if baseline is None:
            print(
                f"! {path.name}: no sample {task_id} in {source}; name the "
                "file <target>-<sample>.txt or pass --task",
                file=sys.stderr,
            )
            continue
        report = grading.grade(baseline, path.read_text(encoding="utf-8",
                                                        errors="replace"))
        reports.append(report)
        print(grading.format_report(report, verbose=args.verbose))

    if not reports:
        return 1
    if len(reports) > 1:
        counts: dict[str, int] = {}
        for report in reports:
            counts[report.verdict] = counts.get(report.verdict, 0) + 1
        print(f"graded {len(reports)} reply(ies)")
        for verdict in sorted(counts, key=lambda v: -counts[v]):
            print(f"  {counts[verdict]:>6}  {verdict}")
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    return 0 if all(r.verdict == "pass" for r in reports) else 1


COMMANDS = {
    "probe": _cmd_probe,
    "prompts": _cmd_prompts,
    "plan": _cmd_plan,
    "sample": _cmd_sample,
    "assess": _cmd_assess,
    "repair": _cmd_repair,
    "grade": _cmd_grade,
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
    parser.add_argument(
        "--source", default=None, help="read samples from this CSV instead"
    )
    parser.add_argument("--reply", default=None, help="grade: one reply file")
    parser.add_argument(
        "--reply-dir", default=None,
        help="grade: a directory of <target>-<sample>.txt replies",
    )
    parser.add_argument(
        "--task", default=None,
        help="grade: the task id, when the filename does not carry it",
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
