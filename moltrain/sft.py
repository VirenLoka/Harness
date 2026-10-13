"""Stage 2: SFT warmup on the collected traces.

The point of this stage is not chemistry, it is protocol. A base model asked
to emit ATEM tool calls into a tool loop it has never seen will mostly produce
turns the parser rejects, and GRPO cannot learn from a malformed rollout -- it
scores badly for the wrong reason. A short warmup on traces that are known to
parse and known to have scored well gets the format out of the way, so GRPO's
gradient goes into decisions rather than syntax.

Records are prompt/completion pairs, one per assistant turn, so TRL's
completion-only loss masks every tool result. See `traces.py` for why that
matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config, resolve
from .trl_compat import adapt, describe


def training_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    sft = cfg["sft"]
    return {
        "output_dir": str(resolve(cfg, sft["output_dir"])),
        "learning_rate": float(sft["learning_rate"]),
        "num_train_epochs": float(sft["num_train_epochs"]),
        "per_device_train_batch_size": int(sft["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(sft["gradient_accumulation_steps"]),
        "max_seq_length": int(sft["max_seq_length"]),
        "packing": bool(sft["packing"]),
        "warmup_ratio": float(sft["warmup_ratio"]),
        "logging_steps": int(sft["logging_steps"]),
        "save_steps": int(sft["save_steps"]),
        "bf16": bool(sft["bf16"]),
        "gradient_checkpointing": bool(sft["gradient_checkpointing"]),
        "completion_only_loss": True,
        "report_to": [],
        "seed": int(cfg["run"]["seed"]),
    }


def lora_settings(cfg: dict[str, Any], section: str = "sft") -> dict[str, Any]:
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
    problems: list[str] = []
    print("imports")
    for module in ("torch", "transformers", "trl", "peft", "datasets"):
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
            memory = properties.total_memory / 1e9
            print(f"  device: {properties.name} ({memory:.0f} GB)")
        else:
            problems.append("no CUDA device; SFT on CPU is not practical")
    except ImportError:
        pass

    try:
        from trl import SFTConfig

        print()
        print(describe(SFTConfig, training_settings(cfg)))
    except ImportError:
        print("\nTRL is absent, so its settings cannot be checked")

    print("\ntraces")
    try:
        from .traces import read_records

        records = read_records(cfg)
        print(f"  records: {len(records)}")
        leaks = [r for r in records if "<tool_output" in r["completion"]]
        print(f"  completions containing tool output: {len(leaks)} (must be 0)")
        if leaks:
            problems.append("some completions contain tool output; re-collect traces")
        if not records:
            problems.append("no SFT records; run `python -m moltrain.traces collect`")
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
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from .traces import read_records

    records = read_records(cfg)
    if not records:
        print("no SFT records to train on", file=sys.stderr)
        return 1
    leaks = sum("<tool_output" in r["completion"] for r in records)
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
    trainer.train()
    trainer.save_model(args.output_dir)

    summary = {
        "records": len(records),
        "output_dir": args.output_dir,
        "dropped_settings": dropped,
        "model": cfg["model"]["name"],
    }
    Path(args.output_dir, "moltrain_sft.json").write_text(json.dumps(summary, indent=2))
    print(f"adapter saved to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m moltrain.sft", description="SFT warmup on collected traces."
    )
    parser.add_argument("command", choices=["check", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="build everything but do not train"
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return check(cfg) if args.command == "check" else train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
