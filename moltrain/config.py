"""YAML configuration for the molecular-optimization RL pipeline.

Same shape as the vLLM launcher's config: defaults live here, a YAML file
overrides them by deep merge, and unknown keys fail loudly rather than being
silently ignored.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "default.yaml"

DEFAULTS: dict[str, Any] = {
    # Where the molbench tool implementations live. The trainer imports them
    # directly, so no MCP server and no Node runtime are involved.
    "molbench": {
        "path": "../molbench",
        "receptor_dir": None,  # defaults to <path>/receptors
        "docking_dir": None,
        "viz_dir": None,
    },
    "episode": {
        "max_turns": 8,
        "max_tool_calls": 24,
        # Tools the policy may call. `submit_molecule` ends the episode and is
        # always available.
        "tools": [
            "validate_smiles",
            "get_molecule_metrics",
            "apply_smarts_reaction",
            "generate_3d_conformer",
            "run_docking_simulation",
            "get_sbdd_metrics",
            "dock_and_score",
        ],
        "tool_timeout_s": 900.0,
        # Bytes of tool output kept in context; longer results are truncated.
        "max_tool_result_chars": 4000,
        "reasoning_strength": "high",
    },
    "reward": {
        # Relative to each episode's own starting molecule, so targets of
        # different difficulty stay comparable.
        "thresholds": {
            "sa_score": {"direction": "decrease", "goal_delta": 0.5},
            "qed": {"direction": "increase", "goal_delta": 0.05},
            "binding_affinity": {
                "direction": "decrease",
                "goal_delta": 0.0,
                "tolerance": 0.5,
            },
        },
        "weights": {
            "thresholds": 1.0,  # shaped progress toward each threshold
            "all_met_bonus": 0.5,  # paid once when every threshold is satisfied
            "validity": 0.2,  # the submitted molecule parses
            "novelty": 0.1,  # it differs from the starting molecule
            "tool_success": 0.2,  # fraction of tool calls that did not error
            "turn_penalty": 0.005,  # per tool call beyond the free budget
            # Charged per broken "do not get worse" rule. Losing binding to
            # buy a little synthesisability is not a trade worth learning.
            "constraint_violation": 0.4,
        },
        # Calls up to here cost nothing: measuring is the task, not waste.
        "turn_penalty_free_calls": 10,
        # Ceiling on one constraint's charge, in tolerance units, so a single
        # ruinous rollout cannot swamp its group's advantage.
        "constraint_violation_cap": 3.0,
        "invalid_submission_reward": -1.0,
        "no_submission_reward": -1.0,
        "docking": {
            # off | final_only | every_candidate
            # Start with final_only; `moltrain.benchmark` reports the episode
            # cost of each so the choice can be made on measured numbers.
            "mode": "final_only",
            "engine": "vina",
            "exhaustiveness": 4,  # training setting; evaluate at 8 or more
            "num_modes": 4,
            "workers": 8,
            "timeout_s": 300.0,
            "cache_path": "cache/docking.jsonl",
        },
        "property_cache_path": "cache/properties.jsonl",
    },
    "dataset": {
        # Where each episode's starting molecule comes from.
        #   diffsbdd          - sample the pocket with DiffSBDD (needs a GPU)
        #   reference_ligand  - the crystal ligand already in the pocket
        #   smiles_file       - a flat SMILES list, paired with the pockets in order
        "source": "diffsbdd",
        "smiles_file": None,
        "tasks_file": "data/tasks.jsonl",
        "pockets": {
            # crossdocked | explicit
            "source": "crossdocked",
            # Directory holding `crossdocked_pocket10/` and `split_by_name.pt`,
            # as unpacked from the Pocket2Mol release. See README.
            "crossdocked_root": None,
            "split": "test",
            # Pocket PDBs are copied here under their own name before the
            # receptor is prepared; see stage_pocket_pdb.
            "staging_dir": "data/pockets",
            "limit": 25,  # null takes every pocket in the split
            "shuffle_seed": 42,  # null keeps the split's own order
            # [{name, pdb_file, ref_ligand}] - used when source is `explicit`
            "explicit": [],
        },
        "box": {
            # The docking box wraps the reference ligand plus this padding,
            # clamped per axis. A box much larger than the pocket makes Vina
            # slower and its scores noisier.
            "padding": 5.0,
            "min_size": 18.0,
            "max_size": 30.0,
        },
        "diffsbdd": {
            "repo_path": "../DiffSBDD",
            "checkpoint": "checkpoints/crossdocked_fullatom_cond.ckpt",
            # Interpreter for DiffSBDD's own conda env; null uses this one.
            "python": None,
            "n_samples": 20,
            "batch_size": None,  # null means one batch of n_samples
            "timesteps": None,  # null uses the checkpoint's full schedule
            "num_nodes_lig": None,
            "sanitize": True,
            "all_frags": False,
            "relax": False,
            "timeout_s": 3600.0,
            "samples_dir": "data/diffsbdd_samples",
            # Sampling a diffusion model on CPU is unusably slow, so a missing
            # GPU is treated as a misconfiguration rather than something to
            # grind through. Flip this only for a deliberate smoke test.
            "allow_cpu": False,
        },
        "filters": {
            "starts_per_pocket": 4,
            "min_heavy_atoms": 12,
            "max_heavy_atoms": 50,
            "require_ring": True,
            "max_sa_score": None,
            "dedupe": True,
        },
        "baseline": {
            # Dock every starting molecule once while building the dataset, so
            # the relative binding-affinity threshold has a reference value and
            # the first training step does not pay for it.
            "dock": True,
        },
    },
    "model": {
        "name": "meta-models/Muse-Glimmer-30B",
        "max_prompt_tokens": 24576,
        "max_completion_tokens": 2048,
    },
    "sft": {
        "output_dir": "runs/sft",
        "traces_file": "data/sft_traces.jsonl",
        # Episodes below this reward are not worth imitating. Measured
        # against the expert: a diligent episode that found no improvement
        # scores about 0.38, one that moved a threshold scores 0.59 or more.
        "min_reward": 0.45,
        "require_submission": True,
        # The reward alone would keep an episode that measured everything and
        # then submitted its input, which teaches exactly the wrong lesson.
        "require_improvement": True,
        "max_seq_length": 8192,
        "packing": False,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_steps": 100,
        "bf16": True,
        "gradient_checkpointing": True,
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
    },
    "grpo": {
        "output_dir": "runs/grpo",
        "learning_rate": 1e-6,
        "num_generations": 8,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 500,
        "beta": 0.02,  # KL coefficient against the reference policy
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": 0.35,
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
        "resume_from_sft": True,
    },
    "evaluate": {
        "output_dir": "runs/eval",
        # Training docks at exhaustiveness 4 for speed; a reported number
        # should not.
        "docking_mode": "final_only",
        "exhaustiveness": 16,
        "num_modes": 9,
        # A separate cache, so an evaluation never reads a score that was
        # measured at the cheaper training setting.
        "separate_cache": True,
        "docking_cache_path": "cache/docking_eval.jsonl",
    },
    "run": {
        "seed": 42,
        "log_dir": "runs/logs",
    },
}


class ConfigError(Exception):
    """A configuration problem worth reporting verbatim."""


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in merged:
            raise ConfigError(f"unknown config key: {where}")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            # Threshold names are an open-ended map, not a fixed schema.
            if where.startswith("reward.thresholds"):
                merged[key] = value
            else:
                merged[key] = _deep_merge(merged[key], value, where)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config file, falling back to the packaged defaults."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {config_path}")
    merged = _deep_merge(DEFAULTS, raw)
    merged["_config_path"] = str(config_path.resolve())
    merged["_root"] = str(config_path.resolve().parent.parent)
    return merged


def resolve(cfg: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the package directory."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (PACKAGE_DIR / candidate).resolve()
