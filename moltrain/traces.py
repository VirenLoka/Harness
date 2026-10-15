"""Collecting SFT traces from episodes that actually went well.

Each record is one assistant turn: the transcript before it as `prompt`, the
turn itself as `completion`. That shape is deliberate. A trace interleaves
model output with tool results, and training on the whole string would teach
the model to *write* tool results -- which is how a tool-using model learns to
fabricate measurements instead of calling the tool. Splitting per turn puts
every tool result in some later record's prompt and never in a completion, so
TRL's `completion_only_loss` masks them with no custom collator.

Two sources, both scored by the same reward as training:
  scripted   - the expert in `experts.py`; needs no model, so it works today
  rejection  - sample the base policy and keep the episodes that scored well
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import logs, progress
from .config import ConfigError, load_config, resolve
from .episode import EpisodeResult, Task, run_episode
from .experts import ScriptedExpert
from .oracle import Oracle
from .tasks import TaskBuildError
from .toolkit import Toolkit, build_toolkit

PolicyFactory = Callable[[Task], Callable[[str], str | Awaitable[str]]]

LOG = logs.get_logger("traces")


class RecordingPolicy:
    """Wraps a policy and keeps every (transcript_before, emitted_turn) pair.

    Recording at the policy boundary is what makes the per-turn split exact:
    there is no need to re-parse a finished transcript and guess where the
    model stopped and the environment started.
    """

    def __init__(self, inner: Callable[[str], str | Awaitable[str]]):
        self.inner = inner
        self.turns: list[tuple[str, str]] = []

    async def __call__(self, transcript: str) -> str:
        result = self.inner(transcript)
        text = await result if hasattr(result, "__await__") else result
        self.turns.append((transcript, text))
        return text


@dataclass
class TraceReport:
    """What collection produced, including the episodes it threw away."""

    episodes: int = 0
    kept: int = 0
    records: int = 0
    rewards: list[float] = field(default_factory=list)
    kept_rewards: list[float] = field(default_factory=list)
    stop_reasons: dict[str, int] = field(default_factory=dict)
    tool_failures: int = 0
    malformed_calls: int = 0
    prompt_chars: list[int] = field(default_factory=list)
    completion_chars: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def summary(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            return {
                "n": len(values),
                "mean": round(statistics.fmean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "median": round(statistics.median(values), 4),
            }

        return {
            "episodes": self.episodes,
            "kept_episodes": self.kept,
            "sft_records": self.records,
            "reward_all": summary(self.rewards),
            "reward_kept": summary(self.kept_rewards),
            "stop_reasons": self.stop_reasons,
            "tool_failures": self.tool_failures,
            "malformed_calls": self.malformed_calls,
            "prompt_chars": summary([float(v) for v in self.prompt_chars]),
            "completion_chars": summary([float(v) for v in self.completion_chars]),
            # A rough guide only: 4 characters per token is the usual ballpark
            # and the real count depends on the tokenizer.
            "longest_prompt_tokens_estimate": (
                max(self.prompt_chars) // 4 if self.prompt_chars else 0
            ),
        }


def _improved(result: EpisodeResult) -> bool:
    """Did the episode move a *goal*, not just honour a constraint?

    Constraints are satisfied by changing nothing, so counting them here would
    keep every episode that submitted its input untouched.
    """
    return any(
        outcome.get("delta") is not None
        and outcome.get("goal_delta", 0.0) > 0
        and outcome.get("progress", 0.0) > 0.0
        for outcome in result.breakdown.get("thresholds", [])
    )


def scripted_factory(cfg: dict[str, Any]) -> PolicyFactory:
    """The expert, docking only when per-candidate affinity feeds the reward.

    Under `final_only` the reward docks just the submitted molecule, so the
    expert hill-climbs on 2D properties and lets the reward pay for the one
    dock that counts. Docking every intermediate there would cost conformer
    generation and a search per move for a number nothing reads.
    """
    dock = cfg["reward"]["docking"]["mode"] == "every_candidate"
    exhaustiveness = int(cfg["reward"]["docking"]["exhaustiveness"])
    thresholds = cfg["reward"]["thresholds"]

    def make(task: Task):
        return ScriptedExpert(
            task, thresholds, dock=dock, exhaustiveness=exhaustiveness
        )

    return make


async def collect(
    cfg: dict[str, Any],
    tasks: list[Task],
    factory: PolicyFactory,
    *,
    attempts: int = 1,
    toolkit: Toolkit | None = None,
    oracle: Oracle | None = None,
    disable_progress: bool | None = None,
) -> tuple[list[dict[str, Any]], list[EpisodeResult], TraceReport]:
    """Run episodes, keep the good ones, and split them into SFT records."""
    toolkit = toolkit or build_toolkit(cfg)
    oracle = oracle or Oracle(cfg, toolkit)
    min_reward = float(cfg["sft"]["min_reward"])
    require_submission = bool(cfg["sft"]["require_submission"])
    require_improvement = bool(cfg["sft"]["require_improvement"])

    records: list[dict[str, Any]] = []
    episodes: list[EpisodeResult] = []
    report = TraceReport()

    LOG.info(
        "collecting traces from %d tasks x %d attempt(s); keeping reward >= %.2f",
        len(tasks),
        attempts,
        min_reward,
    )
    bar = progress.track(
        total=len(tasks) * attempts,
        desc="collecting traces",
        unit="episode",
        disable=disable_progress,
    )

    for task in tasks:
        for attempt in range(attempts):
            bar.postfix(f"{task.task_id} attempt {attempt + 1}/{attempts}")
            # A fresh Task per attempt: run_episode fills in `baseline`, and
            # sharing it across attempts would hide a re-measurement bug.
            trial = Task.from_dict(task.as_dict())
            trial.task_id = (
                task.task_id if attempts == 1 else f"{task.task_id}#{attempt}"
            )
            policy = RecordingPolicy(factory(trial))
            result = await run_episode(cfg, trial, policy, toolkit, oracle)

            report.episodes += 1
            report.rewards.append(result.reward)
            report.stop_reasons[result.stop_reason] = (
                report.stop_reasons.get(result.stop_reason, 0) + 1
            )
            report.tool_failures += result.tool_failures
            report.malformed_calls += result.malformed_calls
            episodes.append(result)

            reason = None
            if result.reward < min_reward:
                reason = f"reward {result.reward:.3f} below {min_reward:.2f}"
            elif require_submission and not result.submitted_smiles:
                reason = "no submission"
            elif result.malformed_calls:
                reason = "malformed turns"  # never teach a format the parser rejects
            elif require_improvement and not _improved(result):
                # An episode that measured carefully and found nothing is
                # honest work, but imitating it teaches "submit the input".
                reason = "moved no goal"
            if reason is not None:
                LOG.info(
                    "%s dropped (%s) after %d tool calls in %s",
                    trial.task_id,
                    reason,
                    result.tool_calls,
                    progress.format_seconds(result.wall_time_s),
                )
                bar.update(1)
                continue

            report.kept += 1
            report.kept_rewards.append(result.reward)
            LOG.info(
                "%s kept: reward %.3f, %d turns, %d tool calls in %s",
                trial.task_id,
                result.reward,
                len(policy.turns),
                result.tool_calls,
                progress.format_seconds(result.wall_time_s),
            )
            for index, (prompt, completion) in enumerate(policy.turns):
                records.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "task_id": trial.task_id,
                        "turn": index,
                        "reward": round(result.reward, 4),
                        "target": trial.target,
                    }
                )
                report.prompt_chars.append(len(prompt))
                report.completion_chars.append(len(completion))
            bar.update(1)

    bar.close()
    report.records = len(records)
    LOG.info(
        "kept %d/%d episodes -> %d records in %s",
        report.kept,
        report.episodes,
        len(records),
        progress.format_seconds(bar.elapsed_s),
    )
    return records, episodes, report


def write_records(cfg: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    path = resolve(cfg, cfg["sft"]["traces_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def read_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = resolve(cfg, cfg["sft"]["traces_file"])
    if not path.is_file():
        raise FileNotFoundError(
            f"SFT traces not found: {path}. Run `python -m moltrain.traces collect`."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from .tasks import read_tasks

    tasks = read_tasks(cfg, args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks to run", file=sys.stderr)
        return 1

    if args.source == "scripted":
        factory = scripted_factory(cfg)
    else:
        from .policy import vllm_factory

        factory = vllm_factory(cfg)

    records, episodes, report = await collect(
        cfg,
        tasks,
        factory,
        attempts=args.attempts,
        disable_progress=logs.progress_disabled(cfg, args),
    )
    path = write_records(cfg, records)
    report_path = path.with_name(path.stem + "_report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2))

    episodes_path = path.with_name(path.stem + "_episodes.jsonl")
    with episodes_path.open("w") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    {
                        "task_id": episode.task_id,
                        "reward": episode.reward,
                        "breakdown": episode.breakdown,
                        "submitted_smiles": episode.submitted_smiles,
                        "stop_reason": episode.stop_reason,
                        "tool_calls": episode.tool_calls,
                        "tool_failures": episode.tool_failures,
                        "wall_time_s": episode.wall_time_s,
                    }
                )
                + "\n"
            )

    summary = report.as_dict()
    print(f"{report.kept}/{report.episodes} episodes kept -> {report.records} records")
    print(f"traces  -> {path}")
    print(f"report  -> {report_path}")
    print(f"episodes-> {episodes_path}")
    print(json.dumps(summary, indent=2))
    return 0 if records else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m moltrain.traces",
        description="Collect SFT traces from scored episodes.",
    )
    parser.add_argument("command", choices=["collect"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", default=None, help="tasks.jsonl to run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--source",
        default="scripted",
        choices=["scripted", "rejection"],
        help="scripted needs no model; rejection samples the policy",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="episodes per task; >1 is rejection sampling",
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logs.configure_from_args(cfg, args, stem=f"traces-{args.command}")
    try:
        return asyncio.run(_run(cfg, args))
    except (TaskBuildError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
