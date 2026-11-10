"""Collecting SFT traces from linker episodes that actually went well.

Each record is one assistant turn: the transcript before it as `prompt`, the
turn itself as `completion`. That shape is deliberate, and it is load-bearing
twice over here.

A trace interleaves model output with tool results, and training on the whole
string would teach the model to *write* tool results -- which is exactly how a
tool-using model learns to state measurements it never made. Splitting per
turn puts every tool result in some later record's prompt and never in a
completion, so TRL's `completion_only_loss` masks them with no custom
collator.

The second reason is local. `protocol.py` scores an episode on whether every
number the reply asserted came back from a tool, and that term is the only one
in the whole reward that can go negative on its own. Teaching the model to
fabricate tool output during the warmup, and then paying it to stop during
GRPO, would be training the two stages against each other.

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

# moltrain owns the logging root, so protactrain's lines land under the same
# handlers that `logs.configure` installs. One run, one log file.
from moltrain import logs, progress

from .config import ConfigError, load_config, log_destination, resolve
from .episode import EpisodeResult, Task, run_episode
from .tasks import TaskBuildError, read_tasks
from .toolkit import Toolkit, build_toolkit

PolicyFactory = Callable[[Task], Callable[[str], str | Awaitable[str]]]

LOG = logs.get_logger("protac.traces")


class _MirroredRecordingPolicy:
    """Wraps a policy and keeps every (transcript_before, emitted_turn) pair.

    Recording at the policy boundary is what makes the per-turn split exact:
    there is no need to re-parse a finished transcript and guess where the
    model stopped and the environment started.

    This is moltrain's class, mirrored only for the case where moltrain's own
    module cannot be imported -- it pulls in a docking oracle and a CrossDocked
    toolkit that have nothing to do with a linker task.
    """

    def __init__(self, inner: Callable[[str], str | Awaitable[str]]):
        self.inner = inner
        self.turns: list[tuple[str, str]] = []

    async def __call__(self, transcript: str) -> str:
        result = self.inner(transcript)
        text = await result if hasattr(result, "__await__") else result
        self.turns.append((transcript, text))
        return text


try:  # noqa: SIM105 - the fallback needs a body, not a suppression
    from moltrain.traces import RecordingPolicy
except Exception:  # noqa: BLE001 - moltrain.traces imports its own oracle chain
    RecordingPolicy = _MirroredRecordingPolicy  # type: ignore[assignment,misc]


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read a threshold outcome whether it arrived as a dataclass or a dict.

    `EpisodeResult.breakdown` is serialised somewhere between `rewards.py` and
    here, and which side of that boundary this runs on depends on whether the
    episodes were just produced or read back from disk.
    """
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _breakdown_dict(breakdown: Any) -> dict[str, Any]:
    """`RewardBreakdown` as plain JSON-able data.

    `dataclasses.asdict` would do it, but it also walks `ThresholdOutcome`
    into nested dicts -- which `_field` above already expects on both sides of
    the disk boundary, so flattening here keeps the in-memory and read-back
    shapes identical rather than making the reader handle two.
    """
    if isinstance(breakdown, dict):
        return breakdown
    import dataclasses

    return {
        "reward": breakdown.reward,
        "thresholds": [dataclasses.asdict(o) for o in breakdown.thresholds],
        "terms": dict(breakdown.terms),
        "floor_reason": breakdown.floor_reason,
        "flags": list(breakdown.flags),
    }


@dataclass
class TraceReport:
    """What collection produced, including the episodes it threw away."""

    episodes: int = 0
    kept: int = 0
    records: int = 0
    rewards: list[float] = field(default_factory=list)
    kept_rewards: list[float] = field(default_factory=list)
    stop_reasons: dict[str, int] = field(default_factory=dict)
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    tool_failures: int = 0
    malformed_calls: int = 0
    floored: int = 0
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
            "dropped_reasons": self.dropped_reasons,
            "tool_failures": self.tool_failures,
            "malformed_calls": self.malformed_calls,
            "floored_episodes": self.floored,
            "prompt_chars": summary([float(v) for v in self.prompt_chars]),
            "completion_chars": summary([float(v) for v in self.completion_chars]),
            # A rough guide only: 4 characters per token is the usual ballpark
            # and the real count depends on the tokenizer.
            "longest_prompt_tokens_estimate": (
                max(self.prompt_chars) // 4 if self.prompt_chars else 0
            ),
        }


def _improved(breakdown: dict[str, Any]) -> bool:
    """Did the episode move a *goal*, not just hold a band it started inside?

    `rewards.score_thresholds` splits goals from holds by the starting value,
    and a hold is satisfied by changing nothing. Counting holds here would keep
    every episode that submitted its input untouched -- which, with the two
    fragments frozen and most descriptors barely moving, is the majority of
    them.
    """
    for outcome in breakdown.get("thresholds", []):
        if _field(outcome, "kind") == "goal" and float(
            _field(outcome, "progress", 0.0) or 0.0
        ) > 0.0:
            return True
    return False


def scripted_factory(cfg: dict[str, Any]) -> PolicyFactory:
    """The hand-written linker expert, which needs no model and no GPU.

    Imported here rather than at module scope so that `read_records` and the
    record shape stay usable on a machine where the expert's chemistry
    dependencies are not installed.
    """
    from .experts import ScriptedExpert

    thresholds = cfg["reward"]["thresholds"]

    def make(task: Task) -> Any:
        return ScriptedExpert(task, thresholds)

    return make


async def collect(
    cfg: dict[str, Any],
    tasks: list[Task],
    factory: PolicyFactory,
    *,
    attempts: int = 1,
    toolkit: Toolkit | None = None,
    disable_progress: bool | None = None,
) -> tuple[list[dict[str, Any]], list[EpisodeResult], TraceReport]:
    """Run episodes, keep the good ones, and split them into SFT records.

    The scoring is `grpo.score_episode`, imported here rather than at module
    scope because `grpo` imports this module for `RecordingPolicy`. Borrowing
    GRPO's scorer rather than writing a second one is the point: a warmup that
    imitated episodes scored by different arithmetic from the one GRPO
    optimises would train the two stages against each other, which is the
    failure this package cannot see from the inside.
    """
    from .grpo import score_episode

    toolkit = toolkit or build_toolkit(cfg)
    sft = cfg["sft"]
    # `min_reward` ships as null on purpose: moltrain's 0.45 was calibrated
    # against its own expert's measured spread and does not transfer to a
    # reward whose bands, holds and protocol terms are different. Until the
    # linker expert has been run and its spread looked at, no threshold is
    # better than a borrowed one.
    raw_min = sft.get("min_reward")
    min_reward = None if raw_min is None else float(raw_min)
    require_submission = bool(sft["require_submission"])
    require_improvement = bool(sft["require_improvement"])

    records: list[dict[str, Any]] = []
    episodes: list[EpisodeResult] = []
    report = TraceReport()

    LOG.info(
        "collecting traces from %d tasks x %d attempt(s); reward filter: %s",
        len(tasks),
        attempts,
        "none (sft.min_reward is unset)"
        if min_reward is None
        else f">= {min_reward:.2f}",
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
            # `check_protac_linker` is bound to one fragment pair, on purpose:
            # a policy that could nominate its own fragments could declare any
            # molecule intact. So the toolkit is derived per task rather than
            # shared, and a shared one handed in here is rebound, not reused.
            bound = toolkit.for_fragments(trial.fragment_smiles)
            result = await run_episode(cfg, trial, policy, bound)
            # `run_episode` deliberately does not score -- the evaluation arms
            # score the same transcript several different ways -- so the
            # reward is filled in here, from the episode's own parts.
            reward, breakdown_obj, _protocol = score_episode(
                cfg, trial, result, policy.turns
            )
            result.reward = reward
            breakdown = _breakdown_dict(breakdown_obj)

            report.episodes += 1
            report.rewards.append(result.reward)
            report.stop_reasons[result.stop_reason] = (
                report.stop_reasons.get(result.stop_reason, 0) + 1
            )
            report.tool_failures += result.n_tool_failures
            report.malformed_calls += result.n_malformed_calls
            if breakdown.get("floor_reason"):
                report.floored += 1
            episodes.append(result)

            reason = None
            if breakdown.get("floor_reason"):
                # An inadmissible answer is not a bad example, it is the wrong
                # compound. Never imitate one.
                reason = f"inadmissible: {breakdown['floor_reason']}"
            elif min_reward is not None and result.reward < min_reward:
                reason = f"reward {result.reward:.3f} below {min_reward:.2f}"
            elif require_submission and not result.submitted_smiles:
                reason = "no submission"
            elif result.n_malformed_calls:
                reason = "malformed turns"  # never teach a format the parser rejects
            elif require_improvement and not _improved(breakdown):
                # An episode that measured carefully and found nothing is
                # honest work, but imitating it teaches "submit the input".
                # The qed-only controls land here, which is correct: they are
                # evaluation material, not imitation material.
                reason = "moved no goal"
            if reason is not None:
                key = reason.split(":")[0]
                report.dropped_reasons[key] = report.dropped_reasons.get(key, 0) + 1
                LOG.info(
                    "%s dropped (%s) after %d tool calls in %s",
                    trial.task_id,
                    reason,
                    result.n_tool_calls,
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
                result.n_tool_calls,
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
                        "target_id": getattr(trial, "target_id", ""),
                        "failure_codes": list(getattr(trial, "failure_codes", []) or []),
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
            f"SFT traces not found: {path}. "
            "Run `python -m protactrain.traces collect`."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    tasks = read_tasks(cfg, args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks to run", file=sys.stderr)
        return 1

    if args.source == "scripted":
        factory = scripted_factory(cfg)
    else:
        from .config import trained_adapter
        from .policy import vllm_factory

        # Same refusal as `evaluate`: traces sampled from the base model are
        # prose with no parseable call in it, so every episode would be
        # dropped as inadmissible and the collection would report a clean run
        # that produced nothing usable.
        adapter, searched = trained_adapter(cfg)
        if adapter is None:
            where = ", ".join(str(path) for path in searched)
            print(
                f"--source {args.source} needs a trained adapter to sample, "
                f"and none was found in {where} (no adapter_config.json "
                "there, and none in any checkpoint-N/ under them). Collect "
                "scripted traces with `--source scripted` first, or train "
                "one.",
                file=sys.stderr,
            )
            return 1
        LOG.info("collecting model traces with the adapter at %s", adapter)
        factory = vllm_factory(cfg, adapter)

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
            handle.write(json.dumps(episode.as_dict()) + "\n")

    summary = report.as_dict()
    print(f"{report.kept}/{report.episodes} episodes kept -> {report.records} records")
    print(f"traces  -> {path}")
    print(f"report  -> {report_path}")
    print(f"episodes-> {episodes_path}")
    print(json.dumps(summary, indent=2))
    return 0 if records else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.traces",
        description="Collect SFT traces from scored linker episodes.",
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
    args.log_file = log_destination(cfg, args, f"traces-{args.command}")
    logs.configure_from_args(cfg, args, stem=f"traces-{args.command}")
    try:
        return asyncio.run(_run(cfg, args))
    except (TaskBuildError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
