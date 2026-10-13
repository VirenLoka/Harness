"""Stage 3: GRPO on the tool-using episode.

A GRPO step samples a group of rollouts for the same task, scores each with the
reward in `rewards.py`, and pushes the policy toward the rollouts that beat
their group's mean. The baseline is the group, so there is no value network --
and so every rollout in a group must be for the same task, which is what
`rollout_tasks` guarantees.

Two things here are worth reading before the first run.

**The mask.** A rollout interleaves model output with tool results. Only the
model's tokens may carry gradient, and `grpo_core.build_completion` produces
that mask from exact span arithmetic. `check` reports whether the installed TRL
accepts a `completion_mask` column, because if it does not, training would
learn to emit tool output and the whole pipeline is worse than useless.

**Degenerate groups.** If every rollout in a group scores the same, the
advantage is zero and the step teaches nothing. The rollout report counts
these. A deterministic policy makes every group degenerate, which is why the
scripted expert is for SFT only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config, resolve
from .episode import Task, run_episode
from .grpo_core import build_batch
from .oracle import Oracle
from .toolkit import Toolkit, build_toolkit
from .traces import RecordingPolicy


def rollout_tasks(tasks: list[Task], num_generations: int) -> list[Task]:
    """Each task repeated `num_generations` times, grouped and contiguous."""
    expanded: list[Task] = []
    for task in tasks:
        for generation in range(num_generations):
            copy = Task.from_dict(task.as_dict())
            copy.task_id = f"{task.task_id}@{generation}"
            expanded.append(copy)
    return expanded


async def run_group(
    cfg: dict[str, Any],
    tasks: list[Task],
    policy_factory,
    toolkit: Toolkit,
    oracle: Oracle,
    *,
    concurrency: int | None = None,
) -> list[tuple[str, list[tuple[str, str]], str, float, Any]]:
    """Run every rollout concurrently so the generator can batch their turns."""
    limit = asyncio.Semaphore(concurrency or len(tasks) or 1)

    async def one(task: Task):
        async with limit:
            recorder = RecordingPolicy(policy_factory(task))
            result = await run_episode(cfg, task, recorder, toolkit, oracle)
            return (
                task.task_id,
                recorder.turns,
                result.transcript,
                result.reward,
                result,
            )

    return list(await asyncio.gather(*(one(task) for task in tasks)))


def summarise(results: list[Any], batch: Any, elapsed: float) -> dict[str, Any]:
    rewards = [r.reward for r in results]
    return {
        "rollouts": len(results),
        "wall_time_s": round(elapsed, 2),
        "seconds_per_rollout": round(elapsed / max(len(results), 1), 2),
        "reward_mean": round(statistics.fmean(rewards), 4) if rewards else 0.0,
        "reward_min": round(min(rewards), 4) if rewards else 0.0,
        "reward_max": round(max(rewards), 4) if rewards else 0.0,
        "submitted": sum(1 for r in results if r.submitted_smiles),
        "tool_calls": sum(r.tool_calls for r in results),
        "tool_failures": sum(r.tool_failures for r in results),
        "malformed_calls": sum(r.malformed_calls for r in results),
        "docks_run": sum(r.docks_run for r in results),
        "stop_reasons": {
            reason: sum(1 for r in results if r.stop_reason == reason)
            for reason in {r.stop_reason for r in results}
        },
        **batch.diagnostics,
    }


def make_rollout_func(cfg: dict[str, Any], tasks_by_id: dict[str, Task]):
    """TRL's `rollout_func`: generate, score, and return masked token columns.

    TRL passes the prompts it built from the dataset; the task behind each one
    is recovered by id, because the reward needs the pocket and the baseline,
    not just the text.
    """
    from transformers import AutoTokenizer

    from .policy import build_policy, build_sampling_params, load_engine

    toolkit = build_toolkit(cfg)
    oracle = Oracle(cfg, toolkit)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    engine = load_engine(cfg)
    params = build_sampling_params(cfg)
    num_generations = int(cfg["grpo"]["num_generations"])

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def rollout_func(prompts: list[str], *_args: Any, **kwargs: Any) -> dict[str, Any]:
        # The task behind each prompt has to come from the dataset column, not
        # be guessed: GRPO's baseline is the group mean, so mis-grouping the
        # rollouts silently produces meaningless advantages. Better to stop.
        ids = kwargs.get("task_id")
        if not ids:
            raise RuntimeError(
                "rollout_func received no `task_id` column, so the rollouts "
                "cannot be grouped by task and the advantages would be "
                "meaningless. The train_dataset built here carries `task_id`; "
                "check whether this TRL version forwards extra dataset "
                "columns to rollout_func."
            )
        if len(ids) != len(prompts):
            raise RuntimeError(f"{len(ids)} task ids for {len(prompts)} prompts")
        tasks = [tasks_by_id[str(task_id).split("@")[0]] for task_id in ids]
        expanded = rollout_tasks(tasks, 1)  # ids already carry the generation

        policy = build_policy(cfg, engine, params, max_batch=len(expanded) or 1)
        started = time.monotonic()
        episodes = asyncio.run(
            run_group(cfg, expanded, lambda _t: policy, toolkit, oracle)
        )
        elapsed = time.monotonic() - started

        batch = build_batch(
            [
                (tid, turns, transcript, reward)
                for tid, turns, transcript, reward, _ in episodes
            ],
            encode,
            num_generations,
        )
        payload = batch.as_trl_payload()
        payload["_moltrain"] = summarise([e[4] for e in episodes], batch, elapsed)
        return payload

    return rollout_func


def training_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    grpo = cfg["grpo"]
    return {
        "output_dir": str(resolve(cfg, grpo["output_dir"])),
        "learning_rate": float(grpo["learning_rate"]),
        "num_generations": int(grpo["num_generations"]),
        "per_device_train_batch_size": int(grpo["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(grpo["gradient_accumulation_steps"]),
        "max_steps": int(grpo["max_steps"]),
        "beta": float(grpo["beta"]),
        "temperature": float(grpo["temperature"]),
        "top_p": float(grpo["top_p"]),
        "top_k": int(grpo["top_k"]),
        "max_prompt_length": int(cfg["model"]["max_prompt_tokens"]),
        "max_completion_length": int(cfg["model"]["max_completion_tokens"]),
        "use_vllm": True,
        "vllm_mode": grpo["vllm_mode"],
        "vllm_gpu_memory_utilization": float(grpo["vllm_gpu_memory_utilization"]),
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": int(cfg["run"]["seed"]),
    }


def check(cfg: dict[str, Any]) -> int:
    from .trl_compat import accepted_names, describe

    problems: list[str] = []
    print("imports")
    for module in ("torch", "transformers", "trl", "peft", "vllm", "datasets"):
        try:
            loaded = __import__(module)
            print(f"  {module}: {getattr(loaded, '__version__', 'present')}")
        except ImportError as exc:
            print(f"  {module}: MISSING ({exc})")
            problems.append(f"{module} is not installed")

    try:
        import torch

        print(f"\ncuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(
                f"  device: {properties.name} ({properties.total_memory / 1e9:.0f} GB)"
            )
        else:
            problems.append("no CUDA device; GRPO needs one")
    except ImportError:
        pass

    try:
        from trl import GRPOConfig, GRPOTrainer

        print()
        print(describe(GRPOConfig, training_settings(cfg)))

        trainer_args = accepted_names(GRPOTrainer.__init__)
        print("\nGRPOTrainer:")
        for name in ("rollout_func", "peft_config", "reward_funcs", "args"):
            present = (
                "ok      " if (not trainer_args or name in trainer_args) else "ABSENT  "
            )
            print(f"  {present} {name}")
        if trainer_args and "rollout_func" not in trainer_args:
            problems.append(
                "this TRL's GRPOTrainer has no `rollout_func`, so it cannot run "
                "this package's multi-turn tool loop; upgrade TRL"
            )

        # The mask decides whether tool output is trained on, so its support is
        # not a nice-to-have.
        import inspect

        source = ""
        try:
            source = inspect.getsource(GRPOTrainer)
        except (OSError, TypeError):
            pass
        if source:
            honoured = "completion_mask" in source
            print(f"\n  completion_mask referenced in GRPOTrainer: {honoured}")
            if not honoured:
                problems.append(
                    "this TRL's GRPOTrainer never mentions `completion_mask`; "
                    "confirm environment tokens are excluded from the loss "
                    "before training, or the policy will learn to write tool "
                    "output itself"
                )
    except ImportError:
        print("\nTRL is absent, so its settings cannot be checked")

    print("\ntasks")
    try:
        from .tasks import read_tasks

        tasks = read_tasks(cfg)
        print(f"  tasks: {len(tasks)}")
        docked = sum(1 for t in tasks if t.baseline and t.baseline.binding_affinity)
        print(f"  with a docked baseline: {docked}")
        if not tasks:
            problems.append("no tasks; run `python -m moltrain.tasks build`")
        if cfg["reward"]["thresholds"].get("binding_affinity") and not docked:
            problems.append(
                "a binding-affinity threshold is configured but no task has a "
                "docked baseline, so that threshold cannot be scored"
            )
    except Exception as exc:  # noqa: BLE001 - a check reports
        print(f"  {type(exc).__name__}: {exc}")
        problems.append("tasks could not be read")

    print()
    if problems:
        for problem in problems:
            print(f"! {problem}")
        return 1
    print("ready to train")
    return 0


async def _rollout_only(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Run one step's worth of rollouts and report cost, without training.

    Worth doing before committing to a run: it measures a real GRPO step with
    the real model, and it is the only way to see the degenerate-group rate
    that the configured temperature actually produces.
    """
    from transformers import AutoTokenizer

    from .policy import build_policy, build_sampling_params, load_engine
    from .tasks import read_tasks

    tasks = read_tasks(cfg, args.tasks)[: args.tasks_per_step]
    if not tasks:
        print("no tasks to roll out", file=sys.stderr)
        return 1

    num_generations = int(cfg["grpo"]["num_generations"])
    expanded = rollout_tasks(tasks, num_generations)
    toolkit = build_toolkit(cfg)
    oracle = Oracle(cfg, toolkit)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    policy = build_policy(
        cfg, load_engine(cfg), build_sampling_params(cfg), max_batch=len(expanded)
    )

    started = time.monotonic()
    episodes = await run_group(cfg, expanded, lambda _t: policy, toolkit, oracle)
    elapsed = time.monotonic() - started

    batch = build_batch(
        [
            (tid, turns, transcript, reward)
            for tid, turns, transcript, reward, _ in episodes
        ],
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        num_generations,
    )
    summary = summarise([e[4] for e in episodes], batch, elapsed)
    summary["generator_batches"] = policy.generator.batches
    summary["context_truncations"] = policy.truncations

    print(json.dumps(summary, indent=2))
    out = resolve(cfg, cfg["run"]["log_dir"]) / "rollout_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nreport -> {out}")
    if summary["degenerate_groups"]:
        print(
            f"warning: {summary['degenerate_groups']}/{len(tasks)} groups had no "
            "reward spread and will contribute no gradient. Raise the "
            "temperature, or pick tasks that are neither trivial nor hopeless."
        )
    return 0


def train(cfg: dict[str, Any], *, dry_run: bool = False) -> int:
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from .sft import lora_settings
    from .tasks import read_tasks
    from .trl_compat import adapt

    tasks = read_tasks(cfg)
    if not tasks:
        print("no tasks to train on", file=sys.stderr)
        return 1

    from datasets import Dataset

    from .episode import build_prompt

    dataset = Dataset.from_list(
        [{"prompt": build_prompt(cfg, task), "task_id": task.task_id} for task in tasks]
    )

    wanted = training_settings(cfg)
    settings, dropped = adapt(GRPOConfig, wanted)
    if dropped:
        print(f"note: this TRL does not accept {dropped}; proceeding without them")
    args = GRPOConfig(**settings)
    peft_config = LoraConfig(**lora_settings(cfg, "grpo"))

    model = cfg["model"]["name"]
    if cfg["grpo"]["resume_from_sft"]:
        adapter = resolve(cfg, cfg["sft"]["output_dir"])
        if adapter.is_dir():
            print(f"starting from the SFT adapter at {adapter}")
        else:
            print(f"note: grpo.resume_from_sft is set but {adapter} does not exist")

    print(f"model   : {model}")
    print(f"tasks   : {len(tasks)}  x {args.num_generations} generations")
    print(f"steps   : {args.max_steps}")
    print(f"output  : {args.output_dir}")
    if dry_run:
        print("dry run: not starting the trainer")
        return 0

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=make_rollout_func(cfg, {t.task_id: t for t in tasks}),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    Path(args.output_dir, "moltrain_grpo.json").write_text(
        json.dumps(
            {"tasks": len(tasks), "dropped_settings": dropped, "model": model}, indent=2
        )
    )
    print(f"adapter saved to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m moltrain.grpo",
        description="GRPO on the tool-using optimization episode.",
    )
    parser.add_argument("command", choices=["check", "rollout", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--tasks-per-step", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        return check(cfg)
    if args.command == "rollout":
        return asyncio.run(_rollout_only(cfg, args))
    return train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
