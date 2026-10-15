"""Stage 4: held-out evaluation.

Training reward is not a result. It is measured at exhaustiveness 4 with a warm
cache on tasks the policy has seen, and all three of those inflate it. This
stage re-measures on held-out pockets at a stronger docking setting, and
against controls, because "mean reward 0.7" means nothing without knowing what
doing nothing scores.

Two controls are reported next to the policy:

  unchanged  submit the starting molecule as-is. This is the floor a policy has
             to clear to have done anything, and it is not zero -- validity and
             threshold terms pay out for a molecule that was already fine.
  expert     the scripted move library from `experts.py`. A policy that cannot
             beat a greedy hill-climb over twelve hand-written SMARTS patterns
             has not learned chemistry, whatever its reward says.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import atem, logs, progress
from .config import ConfigError, load_config, resolve
from .episode import EpisodeResult, Task, run_episode
from .experts import ScriptedExpert
from .oracle import Oracle
from .tasks import TaskBuildError
from .toolkit import Toolkit, build_toolkit

LOG = logs.get_logger("evaluate")


def unchanged_factory(cfg: dict[str, Any]) -> Callable[[Task], Callable[[str], str]]:
    """A policy that submits its input untouched: the do-nothing floor."""

    def make(task: Task) -> Callable[[str], str]:
        def generate(_transcript: str) -> str:
            return atem.render_continuation(
                "Submit the starting molecule unchanged, as a control.",
                ("submit_molecule", {"smiles": task.smiles, "rationale": "control"}),
            )

        return generate

    return make


@dataclass
class ArmReport:
    """One policy's results over the evaluation set."""

    name: str
    episodes: int = 0
    rewards: list[float] = field(default_factory=list)
    submitted: int = 0
    invalid: int = 0
    unchanged: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    malformed_calls: int = 0
    threshold_met: dict[str, int] = field(default_factory=dict)
    threshold_scored: dict[str, int] = field(default_factory=dict)
    threshold_deltas: dict[str, list[float]] = field(default_factory=dict)
    wall_time_s: float = 0.0

    def add(self, result: EpisodeResult, start_smiles: str) -> None:
        self.episodes += 1
        self.rewards.append(result.reward)
        self.tool_calls += result.tool_calls
        self.tool_failures += result.tool_failures
        self.malformed_calls += result.malformed_calls
        self.wall_time_s += result.wall_time_s
        if result.submitted_smiles:
            self.submitted += 1
            if not (result.final_properties or {}).get("valid", False):
                self.invalid += 1
            elif result.final_properties.get("smiles") == start_smiles:
                self.unchanged += 1
        for outcome in result.breakdown.get("thresholds", []):
            name = outcome["name"]
            if outcome["delta"] is None:
                continue
            self.threshold_scored[name] = self.threshold_scored.get(name, 0) + 1
            if outcome["met"]:
                self.threshold_met[name] = self.threshold_met.get(name, 0) + 1
            self.threshold_deltas.setdefault(name, []).append(outcome["delta"])

    def as_dict(self) -> dict[str, Any]:
        def rate(met: int, scored: int) -> float | None:
            return round(met / scored, 4) if scored else None

        return {
            "name": self.name,
            "episodes": self.episodes,
            "reward_mean": (
                round(statistics.fmean(self.rewards), 4) if self.rewards else None
            ),
            "reward_median": (
                round(statistics.median(self.rewards), 4) if self.rewards else None
            ),
            "reward_min": round(min(self.rewards), 4) if self.rewards else None,
            "reward_max": round(max(self.rewards), 4) if self.rewards else None,
            "submitted": self.submitted,
            "invalid_submissions": self.invalid,
            "submitted_input_unchanged": self.unchanged,
            "tool_calls": self.tool_calls,
            "tool_failure_rate": (
                round(self.tool_failures / self.tool_calls, 4)
                if self.tool_calls
                else None
            ),
            "malformed_calls": self.malformed_calls,
            "threshold_pass_rate": {
                name: rate(self.threshold_met.get(name, 0), scored)
                for name, scored in sorted(self.threshold_scored.items())
            },
            "threshold_mean_delta": {
                name: round(statistics.fmean(values), 4)
                for name, values in sorted(self.threshold_deltas.items())
            },
            "seconds_per_episode": (
                round(self.wall_time_s / self.episodes, 2) if self.episodes else None
            ),
        }


def evaluation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Training settings, with docking turned up to something quotable."""
    import copy

    evaluated = copy.deepcopy(cfg)
    settings = evaluated["evaluate"]
    docking = evaluated["reward"]["docking"]
    docking["mode"] = settings["docking_mode"]
    docking["exhaustiveness"] = int(settings["exhaustiveness"])
    docking["num_modes"] = int(settings["num_modes"])
    if settings["separate_cache"]:
        docking["cache_path"] = settings["docking_cache_path"]
    return evaluated


async def evaluate_arm(
    cfg: dict[str, Any],
    name: str,
    factory,
    tasks: list[Task],
    toolkit: Toolkit,
    oracle: Oracle,
    *,
    episodes_per_task: int = 1,
    disable_progress: bool | None = None,
) -> tuple[ArmReport, list[dict[str, Any]]]:
    report = ArmReport(name=name)
    rows: list[dict[str, Any]] = []
    bar = progress.track(
        total=len(tasks) * episodes_per_task,
        desc=f"arm {name}",
        unit="episode",
        disable=disable_progress,
    )
    for task in tasks:
        for attempt in range(episodes_per_task):
            trial = Task.from_dict(task.as_dict())
            trial.task_id = f"{task.task_id}~{name}~{attempt}"
            bar.postfix(f"{task.task_id} ({attempt + 1}/{episodes_per_task})")
            result = await run_episode(cfg, trial, factory(trial), toolkit, oracle)
            report.add(result, trial.smiles)
            LOG.info(
                "%s %s: reward %.3f, %d tool calls, %s",
                name,
                task.task_id,
                result.reward,
                result.tool_calls,
                result.stop_reason,
            )
            rows.append(
                {
                    "arm": name,
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "target": task.target,
                    "start_smiles": trial.smiles,
                    "submitted_smiles": result.submitted_smiles,
                    "reward": round(result.reward, 4),
                    "breakdown": result.breakdown,
                    "stop_reason": result.stop_reason,
                    "tool_calls": result.tool_calls,
                    "tool_failures": result.tool_failures,
                    "wall_time_s": result.wall_time_s,
                }
            )
            bar.update(1)
    bar.close()
    return report, rows


async def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from .tasks import read_tasks

    evaluated = evaluation_config(cfg)
    tasks = read_tasks(evaluated, args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks to evaluate", file=sys.stderr)
        return 1

    toolkit = build_toolkit(evaluated)
    oracle = Oracle(evaluated, toolkit)
    thresholds = evaluated["reward"]["thresholds"]

    arms: dict[str, Any] = {}
    if "unchanged" in args.arms:
        arms["unchanged"] = unchanged_factory(evaluated)
    if "expert" in args.arms:
        dock = evaluated["reward"]["docking"]["mode"] == "every_candidate"
        arms["expert"] = lambda task: ScriptedExpert(task, thresholds, dock=dock)
    if "policy" in args.arms:
        from .policy import vllm_factory

        arms["policy"] = vllm_factory(evaluated)

    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, factory in arms.items():
        LOG.info("running arm `%s` over %d tasks", name, len(tasks))
        report, arm_rows = await evaluate_arm(
            evaluated,
            name,
            factory,
            tasks,
            toolkit,
            oracle,
            episodes_per_task=args.episodes_per_task,
            disable_progress=logs.progress_disabled(evaluated, args),
        )
        reports.append(report.as_dict())
        rows.extend(arm_rows)

    out_dir = resolve(evaluated, evaluated["evaluate"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tasks": len(tasks),
        "episodes_per_task": args.episodes_per_task,
        "docking": {
            k: evaluated["reward"]["docking"][k]
            for k in ("mode", "engine", "exhaustiveness", "num_modes")
        },
        "arms": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "episodes.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(
        f"\n{'arm':<10} {'reward':>8} {'median':>8} {'submit':>7} "
        f"{'unchanged':>10} {'invalid':>8} {'malformed':>10}"
    )
    for report in reports:
        print(
            f"{report['name']:<10} {report['reward_mean']!s:>8} "
            f"{report['reward_median']!s:>8} {report['submitted']:>7} "
            f"{report['submitted_input_unchanged']:>10} "
            f"{report['invalid_submissions']:>8} {report['malformed_calls']:>10}"
        )
    for report in reports:
        print(
            f"\n{report['name']} threshold pass rates: "
            f"{json.dumps(report['threshold_pass_rate'])}"
        )
        print(
            f"{report['name']} mean deltas: "
            f"{json.dumps(report['threshold_mean_delta'])}"
        )
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    print(f"episodes-> {out_dir / 'episodes.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m moltrain.evaluate",
        description="Evaluate on held-out tasks against controls.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", default=None, help="held-out tasks.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unchanged", "expert"],
        choices=["unchanged", "expert", "policy"],
        help="`policy` loads the model; the other two need no GPU",
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logs.configure_from_args(cfg, args, stem="evaluate")
    try:
        return asyncio.run(_run(cfg, args))
    except (TaskBuildError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
