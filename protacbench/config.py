"""YAML configuration, same shape as moltrain's: defaults here, a file
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
    "progat": {
        "repo_path": "../Pro-GAT",
        # Interpreter holding Pro-GAT's deps (torch, openbabel bindings).
        "python": None,
        "model_path": "checkpoints/protacs_best.ckpt",
        "repair_model": "repair_model/ckpt/best_repair_model.pt",
        "data_path": "datasets",
        "test_data_prefix": "protacs_test",
        # 0 uses each entry's own reference linker length, which is what the
        # test set was built around. A positive value forces every PROTAC to
        # the same length regardless of its geometry.
        "linker_size": 0,
        "n_samples": 10,
        "batch_size": 64,
        "output_dir": "runs/samples",
        "timeout_s": 21600.0,
        # Sampling 500 diffusion steps on CPU is not practical, so a missing
        # GPU is treated as a misconfiguration rather than something to wait
        # out.
        "allow_cpu": False,
    },
    "assess": {
        # Which Pro-GAT stages to assess. `generated` is the raw diffusion
        # output; `repaired` is what Pro-GAT's GAT produced from the failures.
        # Both, so the repair delta is visible.
        "stages": ["generated", "repaired"],
        "repaired_dir": "../Pro-GAT/repaired",
        "sdf_dir": "runs/sdf",
        "workers": 8,
    },
    "thresholds": {
        # Soft gates. A sample failing one is `poor`, not `broken`.
        # PROTACs are deliberately outside Lipinski space, so these are wide.
        "max_sa_score": 6.0,
        "min_qed": 0.05,
        "min_mw": 400.0,
        "max_mw": 1600.0,
        "max_clogp": 9.0,
        "max_rotatable_bonds": 30,
        "reject_pains": True,
        # Reject a linker carrying an aldehyde, acyl halide, epoxide,
        # Michael acceptor or other group that would react with protein
        # nucleophiles or not survive as drawn. Nothing else measured here
        # can see these: SA rewards them for being small, QED is
        # indifferent, and the motif matcher reads them as ordinary chains.
        "reject_reactive_linker": True,
        "reject_bad_linker_rings": True,
        "reject_macrocycles": True,
        "min_linker_path_bonds": 3,
        "max_linker_path_bonds": 40,
    },
    "report": {
        "output_dir": "runs/report",
        "samples_csv": "samples.csv",
        "rejects_csv": "rejects.csv",
        "summary_json": "summary.json",
        "repair_tasks": "repair_tasks.jsonl",
        # Rejected samples to carry a prompt. `broken` only, or both.
        "repair_verdicts": ["broken", "poor"],
    },
    # Driving the DeepSeek harness headlessly to repair rejected samples.
    # Everything here is a path or an endpoint, nothing is a secret: the API
    # key is read from the environment variable named in `model.api_key_env`.
    "harness": {
        "node_bin": "npx",
        "dsh_package": "@deepseek-ai/dsh@0.2.0-rc.2",
        "profile": "headless",
        # null leaves DSH_HOME as the environment has it (~/.dsh by default).
        # Point it somewhere private to keep repair sessions out of the
        # interactive harness's history.
        "dsh_home": None,
        "cwd": ".",
        "molbench_path": "../molbench",
        # null uses molbench/.venv/bin/python when it exists, which is the
        # interpreter that definitely has RDKit and FastMCP.
        "molbench_python": None,
        "model": {
            # The route name inside the harness config. Arbitrary, but it is
            # what `agent-default-model.provider` points at.
            "provider": "local",
            "name": "muse-glimmer-30b",
            # The local gateway from serve.py. On the GPU box this is right
            # as it stands; override it to use a tunnel or another host.
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_env": "VLLM_API_KEY",
            "api": "openai-completions",
            "context_window": 131072,
            "max_tokens": 8192,
        },
        # read-only | workspace-write | danger-full-access. The harness reads
        # this into both its sandbox mode and its approval policy. Repair
        # needs no file access at all, so the default is the restrictive one;
        # it only matters if a tool call ever asks to escalate.
        "permission_mode": "workspace-write",
        "require_api_key": True,
        # The harness imposes no run timeout of its own, so this one is the
        # only thing that stops a wedged run.
        "timeout_s": 900.0,
        "tool_timeout_s": 120.0,
        "patch_file": "runs/harness/molbench.patch.yml",
    },
    "repair": {
        "tasks_file": "runs/report/repair_tasks.jsonl",
        "output_dir": "runs/repair",
        "results_jsonl": "results.jsonl",
        "repaired_csv": "repaired.csv",
        "summary_json": "summary.json",
        # Concurrent harness processes. Each one is a Node process plus a
        # Python MCP server, so this is not free.
        "workers": 2,
        # A second attempt on a run that failed to produce a parseable answer.
        "retries": 1,
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
