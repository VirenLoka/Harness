"""Stage 2: SFT warmup on the collected linker traces.

The point of this stage is not chemistry, it is protocol. A base model asked
to emit ATEM tool calls into a tool loop it has never seen will mostly produce
turns the parser rejects, and GRPO cannot learn from a malformed rollout -- it
scores badly for the wrong reason. A short warmup on traces that are known to
parse and known to have scored well gets the format out of the way, so GRPO's
gradient goes into decisions rather than syntax.

Records are prompt/completion pairs, one per assistant turn, so TRL's
completion-only loss masks every tool result. See `traces.py` for why that
matters, and note the refusal in `train` below: this stage will not start if a
single completion contains a tool-output marker. The reward's grounding term
pays the policy to state only numbers a tool returned; a warmup that taught it
to write tool output would be training the two stages against each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from moltrain import logs
from moltrain.trl_compat import adapt, describe

from . import limits
from .config import ConfigError, load_config, log_destination, resolve
from .policy import require_cuda

LOG = logs.get_logger("protac.sft")

# What `moltrain.atem.render_tool_result` opens every environment reply with.
# A completion containing this was built from the wrong span of a transcript.
TOOL_OUTPUT_MARKER = "<tool_output"


def training_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    sft = cfg["sft"]
    return {
        "output_dir": str(resolve(cfg, sft["output_dir"])),
        "learning_rate": float(sft["learning_rate"]),
        "num_train_epochs": float(sft["num_train_epochs"]),
        "per_device_train_batch_size": int(sft["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(sft["gradient_accumulation_steps"]),
        "max_seq_length": int(sft["max_seq_length"]),
        # Packing concatenates examples into one sequence, which is exactly
        # what `completion_only_loss` cannot survive: the prompt boundary it
        # masks on is per-example. Never on for this dataset.
        "packing": bool(sft.get("packing", False)),
        # These three are moltrain's values (configs/default.yaml, sft:) and
        # are schedule and bookkeeping, not objective -- carried over rather
        # than re-chosen so the two packages' warmups stay comparable.
        "warmup_ratio": float(sft.get("warmup_ratio", 0.03)),
        "logging_steps": int(sft.get("logging_steps", 5)),
        "save_steps": int(sft.get("save_steps", 100)),
        "bf16": bool(sft["bf16"]),
        "gradient_checkpointing": bool(sft["gradient_checkpointing"]),
        "completion_only_loss": True,
        "report_to": [],
        "seed": int(cfg["run"]["seed"]),
    }


def lora_settings(cfg: dict[str, Any], section: str = "sft") -> dict[str, Any]:
    """Shared with `grpo.py`, which passes section="grpo"."""
    lora = cfg[section]["lora"]
    return {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        # Attention and MLP projections, which is where LoRA earns its capacity
        # on a decoder. Narrow this if memory is tight.
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }


def build_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]
    )


def check(cfg: dict[str, Any]) -> int:
    """Report whether this host can run the stage, without starting it."""
    from moltrain import hostcheck

    problems: list[str] = []

    print("imports")
    reports, import_problems = hostcheck.report_imports(
        ("torch", "transformers", "trl", "peft", "datasets")
    )
    for report in reports:
        print(
            f"  {report.name}: {report.detail if report.ok else 'FAILED - ' + report.detail}"
        )
    problems.extend(import_problems)

    print("\ncuda")
    cuda_lines, cuda_problems = hostcheck.report_cuda()
    for line in cuda_lines:
        print(line)
    problems.extend(cuda_problems)

    # `report_cuda` prints each card's total memory, which stays reassuring on
    # a card that is already full. This stage is the one that OOMed on a shared
    # H200, so it asks the question that decides whether a run can start.
    ranks = int(cfg["grpo"].get("ddp", {}).get("world_size", 1) or 1)
    print("\nthreads")
    for line in (thread := limits.report_threads(ranks))[0]:
        print(line)
    problems.extend(thread[1])

    print("\nfree device memory")
    for line in (gpus := limits.report_gpus(ranks))[0]:
        print(line)
    problems.extend(gpus[1])

    if any(r.name == "trl" and r.ok for r in reports):
        try:
            from trl import SFTConfig

            print()
            print(describe(SFTConfig, training_settings(cfg)))
        except Exception as exc:  # noqa: BLE001 - a check reports, it does not raise
            print(f"\nTRL settings could not be checked: {type(exc).__name__}: {exc}")
    else:
        print("\nTRL is not usable, so its settings cannot be checked")

    print("\ntraces")
    try:
        from .traces import read_records

        records = read_records(cfg)
        print(f"  records: {len(records)}")
        leaks = [r for r in records if TOOL_OUTPUT_MARKER in r["completion"]]
        print(f"  completions containing tool output: {len(leaks)} (must be 0)")
        if leaks:
            problems.append("some completions contain tool output; re-collect traces")
        if not records:
            problems.append(
                "no SFT records; run `python -m protactrain.traces collect`"
            )
    except FileNotFoundError as exc:
        print(f"  {exc}")
        problems.append("SFT traces have not been collected")

    print()
    if problems:
        for problem in problems:
            print(f"! {problem}")
        return 1
    print("ready to train")
    return 0


def train(cfg: dict[str, Any], *, dry_run: bool = False) -> int:
    # Even --dry-run builds the trainer, which loads the model.
    require_cuda("SFT training")

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from .traces import read_records

    records = read_records(cfg)
    if not records:
        print("no SFT records to train on", file=sys.stderr)
        return 1
    leaks = sum(TOOL_OUTPUT_MARKER in r["completion"] for r in records)
    if leaks:
        print(
            f"refusing to train: {leaks} completions contain tool output, which "
            "would teach the model to write its own measurements",
            file=sys.stderr,
        )
        return 1

    dataset = build_dataset(records)
    wanted = training_settings(cfg)
    settings, dropped = adapt(SFTConfig, wanted)
    if dropped:
        print(f"note: this TRL does not accept {dropped}; proceeding without them")

    args = SFTConfig(**settings)
    peft_config = LoraConfig(**lora_settings(cfg, "sft"))

    print(f"model      : {cfg['model']['name']}")
    print(f"records    : {len(records)}")
    print(f"output_dir : {args.output_dir}")
    print(f"lora       : r={peft_config.r} alpha={peft_config.lora_alpha}")
    if dry_run:
        print("dry run: not starting the trainer")
        return 0

    trainer = SFTTrainer(
        model=cfg["model"]["name"],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    LOG.info("starting SFT; TRL reports its own progress from here")
    trainer.train()
    LOG.info("SFT finished; saving the adapter")
    trainer.save_model(args.output_dir)

    summary = {
        "records": len(records),
        "output_dir": args.output_dir,
        "dropped_settings": dropped,
        "model": cfg["model"]["name"],
    }
    Path(args.output_dir, "protactrain_sft.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"adapter saved to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.sft",
        description="SFT warmup on collected linker traces.",
    )
    parser.add_argument("command", choices=["check", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="build everything but do not train"
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    args.log_file = log_destination(cfg, args, f"sft-{args.command}")
    logs.configure_from_args(cfg, args, stem=f"sft-{args.command}")
    try:
        if args.command == "check":
            return check(cfg)
        return train(cfg, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
