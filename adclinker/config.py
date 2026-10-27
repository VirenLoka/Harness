"""YAML configuration, same shape as protacbench's: defaults here, a file
overrides them by deep merge, unknown keys fail loudly.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "default.yaml"

DEFAULTS: dict[str, Any] = {
    # Where the curated ADC linker reagents come from. Every threshold in
    # score.py is measured from this file rather than chosen.
    "reference": {
        "dataset": "../Linker-GPT/data/linkers.csv",
        "smiles_column": "smiles",
        "cache": "runs/reference_profile.json",
        "fingerprint_radius": 2,
        "fingerprint_bits": 2048,
    },
    # The design: which site chemistry, which release trigger, which drug.
    "design": {
        "handle": "maleimidocaproyl",
        "cassette": "val_cit_pab",
        "payload": "mmae",
        # spacer_before_cassette puts the generated spacer between the handle
        # and the trigger, which is where the PEG sits in clinical ADCs.
        "architecture": "spacer_before_cassette",
        # Enumerate every cassette rather than only the one named above.
        "all_cassettes": False,
    },
    "reinvent": {
        "executable": "reinvent",
        "repo_path": "../REINVENT4",
        "prior_file": "../REINVENT4/priors/linkinvent.prior",
        # null uses prior_file; point this at a fine-tuned model to sample
        # from that instead.
        "model_file": None,
        "device": "cpu",
        # Run REINVENT through a shim that neutralises its post-run
        # `resource.getrusage` call. A restricted container refuses it and
        # REINVENT dies on the way out, losing a completed sampling run.
        # Falls back to the plain console script if the environment's
        # interpreter cannot be located.
        "sandbox_safe": True,
        "working_dir": "runs",
        # Per warhead pair, not per run: the sampler multiplies this by the
        # number of lines in the warheads file.
        "num_smiles": 200,
        "unique_molecules": True,
        "randomize_smiles": True,
        "temperature": 1.0,
        "sampling_config": "runs/sampling.toml",
        "warheads_file": "runs/warheads.smi",
        "samples_csv": "runs/samples.csv",
        "timeout_s": 7200.0,
    },
    # Transfer learning: teach the prior what ADC linkers look like.
    "finetune": {
        "config_file": "runs/transfer_learning.toml",
        "training_file": "runs/tl_train.smi",
        "validation_file": "runs/tl_valid.smi",
        "output_model": "runs/linkinvent_adc.model",
        "num_epochs": 20,
        "batch_size": 64,
        "sample_batch_size": 100,
        "save_every_n_epochs": 5,
        "learning_rate": 0.0001,
        "lr_step": 5,
        "lr_gamma": 0.9,
        # How the reference reagents are cut into (warheads, linker) triples.
        "validation_fraction": 0.1,
        "min_linker_atoms": 3,
        "max_linker_atoms": 30,
        "min_warhead_atoms": 3,
        "max_pairs_per_molecule": 6,
        "seed": 42,
    },
    "scoring_plugin": {
        # Added to PYTHONPATH so REINVENT can walk the namespace package.
        "plugin_root": "plugins",
    },
    "score": {
        # A gate: reject anything with no lysosomal release motif. Turn off
        # only when deliberately designing a non-cleavable linker.
        "require_cleavable": True,
        "min_heavy_atoms": 12,
    },
    "report": {
        "output_dir": "runs/report",
        "candidates_csv": "candidates.csv",
        "summary_json": "summary.json",
        "top_n": 200,
    },
    # Where `prompts` writes, and which repair classes are worth a prompt.
    # `clean` is excluded by default: a candidate that passed every gate and
    # sits under the cLogP ceiling has nothing stated to repair, and handing
    # one to an agent invites it to change a molecule that was already fine.
    "promptout": {
        "output_dir": "runs/prompts",
        "system_file": "system_prompt.txt",
        "tasks_jsonl": "repair_tasks.jsonl",
        "manifest_csv": "manifest.csv",
        "classes": ["payload", "release", "handle", "structural", "chemistry", "greasy"],
    },
    "run": {
        "log_dir": "runs/logs",
        "log_level": "INFO",
        "log_to_file": True,
        "progress": "auto",
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
            merged[key] = _deep_merge(merged[key], value, where)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {config_path}")
    merged = _deep_merge(DEFAULTS, raw)
    merged["_config_path"] = str(config_path.resolve())
    return merged


def resolve(cfg: dict[str, Any], value: str | Path) -> Path:
    """Resolve a config path relative to the package directory."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (PACKAGE_DIR / candidate).resolve()
