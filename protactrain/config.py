"""YAML configuration for the PROTAC linker-optimisation RL pipeline.

Same shape as moltrain's: defaults live here, a YAML file overrides them by
deep merge, and unknown keys fail loudly rather than being silently ignored.

DEFAULTS mirrors `configs/default.yaml` key for key. The YAML is the document
a reader is meant to open -- it carries the provenance of every number -- so
when the two drift it is the YAML that is right.
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
    },
    # protacbench supplies the measurement layer (grade, structures, score,
    # repair, assess) and the repair corpus. Imported as a library, same way.
    "protacbench": {
        "path": "../protacbench",
    },
    "episode": {
        "max_turns": 8,
        "max_tool_calls": 24,
        # The six tools OPTIMISATION.md names as useful for a linker task.
        # check_protac_linker is the one that decides acceptance, and the
        # policy gets it deliberately: the repair corpus this package is built
        # from was produced with it in the prompt, so withholding it would make
        # the RL episodes incomparable to their own baseline. Grinding against
        # it is priced, not forbidden -- see turn_penalty_free_calls.
        #
        # The docking tools are absent on purpose. A PROTAC binds two proteins
        # at once and is decided by the ternary complex; docking scores one
        # ligand into one rigid receptor. At ~26 rotatable bonds the
        # conformational search is past where it stays reliable, and the score
        # would be dominated by the warhead -- the part the policy is forbidden
        # to touch and whose affinity is already known.
        "tools": [
            "check_protac_linker",
            "benchmark_transformation",
            "validate_smiles",
            "get_molecule_metrics",
            "render_molecule_2d",
            "generate_3d_conformer",
        ],
        "tool_timeout_s": 300.0,
        # Bytes of tool output kept in context; longer results are truncated.
        "max_tool_result_chars": 4000,
        "reasoning_strength": "high",
    },
    "reward": {
        # Every threshold is a band: `low`, `high`, or both. A term scores by
        # how far outside its band the value sits, and that distance is what
        # the episode is paid to close. One-sided targets leave the other null.
        #
        # Which band a value started in decides what the term IS. Outside at
        # the start -> a goal, paying shaped progress for the distance closed.
        # Inside at the start -> a hold, earning nothing and charged only if
        # the episode pushes it out. Nothing is ever paid for leaving a
        # satisfied target alone.
        #
        # The numbers are the two regimes already in this repo, not new ones.
        # `low`/`high` are OPTIMISATION.md's p25-p75 bands, measured from the
        # clean passing samples of a real 1024-sample run and sanity-checked
        # against 128 published PROTACs. `tolerance` is the slack before a hold
        # is charged, and the hard limits it defends are grade.py's grader
        # contract (SPAN_TARGET 8-17 with SPAN_FLOOR 7, MAX_SA 6.0,
        # MAX_CLOGP 7.5, MIN_TPSA 120, MAX_ROTATABLE 30).
        #
        # QED is deliberately absent, not forgotten. The 128 published PROTACs
        # this corpus is built from have a median QED of 0.064 and 34% fall
        # below 0.05; 0.10 would reject 73% of them. QED scores
        # oral-small-molecule likeness, so on a ~940 Da bifunctional degrader
        # it measures how unlike a PROTAC the molecule is. Rewarding it would
        # rank the worst candidates highest, which is also why protacbench's
        # own score.py excludes it.
        "thresholds": {
            # Span is first because it is the only one that can make the
            # molecule not a PROTAC. Below 7 bonds the two warheads cannot
            # occupy their binding sites at once, whatever else says.
            "span": {
                "low": 8,
                "high": 13,
                "tolerance": 4,  # out to grade.py's 17; the floor of 7 is a hard gate
            },
            "linker_heavy_atoms": {
                "low": 8,
                "high": 14,
                "tolerance": 11,  # out to the 5-25 hard range
            },
            "sa_score": {
                "low": None,
                "high": 4.9,
                "tolerance": 1.1,  # out to MAX_SA 6.0
            },
            "clogp": {
                "low": 3.5,
                "high": 5.6,
                "tolerance": 1.9,  # out to MAX_CLOGP 7.5
            },
            "tpsa": {
                "low": 170.0,
                "high": 217.0,
                "tolerance": 50.0,  # down to MIN_TPSA 120
            },
            "rotatable_bonds": {
                "low": 13,
                "high": 18,
                "tolerance": 12,  # out to MAX_ROTATABLE 30
            },
            "mw": {
                "low": 780.0,
                "high": 940.0,
                "tolerance": 660.0,  # out to the 400-1600 hard range
            },
        },
        # Descriptor drift charged regardless of band, from
        # grade.REGRESSION_TOLERANCE. A molecule can sit inside every band and
        # still have been walked in the wrong direction the whole way.
        "regression_tolerance": {
            "clogp": 0.15,
            "tpsa": 2.0,
            "sa_score": 0.15,
            "rotatable_bonds": 2,
        },
        # Linker length is set by ternary-complex geometry, not preference.
        # grade.LENGTH_TOLERANCE: two heavy atoms is about one bond of span.
        "length_tolerance": 2,
        "weights": {
            # --- chemistry: what the molecule became
            "thresholds": 1.0,
            "all_met_bonus": 0.5,
            "hold_violation": 0.4,
            "regression": 0.4,
            # protacbench.score.score_sample, as a dense 0-100 signal
            "synthesis_score": 0.3,
            # --- protocol: how the policy worked
            #
            # These exist because an outcome-only reward teaches tool use only
            # by accident. If a molecule can be improved by guessing, guessing
            # scores as well as measuring, and GRPO will find that out.
            # protacbench's grade scorecard already splits chemistry from
            # protocol and fails them independently; these are that second
            # half, as reward.
            #
            # `grounding` is the load-bearing one and the only term that can go
            # negative on its own: it is +1 when every number the reply stated
            # came back from a tool and -1 when none did. A model stating
            # measurements it never made is the worst habit this training could
            # instil, and an outcome reward is completely blind to it -- the
            # molecule is unaffected by what was said about it. Weighted at 0.5
            # it can overturn a mediocre chemistry result, which is the intent.
            "grounding": 0.5,
            # measure, change one thing, re-measure. Three parts, scored
            # separately.
            "discipline": 0.3,
            # called check_protac_linker before submitting
            "verification": 0.15,
            # --- cost of working
            "tool_success": 0.2,
            "turn_penalty": 0.005,
        },
        # Measuring is the task, not waste. moltrain found that a flat
        # 0.02/call penalty made its expert's sixteen honest calls cost 0.32
        # and erased the 0.116 of progress they bought, so the reward's optimum
        # was to submit blind. The free allowance is what stops that, and it
        # matters more here: the protocol terms above actively ask for more
        # calls.
        "turn_penalty_free_calls": 10,
        "violation_cap": 3.0,
        # Inadmissible, not merely poor. A deleted warhead improves SA, cLogP,
        # TPSA, rotatable bonds and MW at once, so this has to be a floor
        # rather than a weighted term: a term can be outweighed, a floor
        # cannot.
        "inadmissible_reward": -1.0,
        "no_submission_reward": -1.0,
        "property_cache_path": "cache/linker_properties.jsonl",
    },
    "dataset": {
        # protacbench's repair_tasks.jsonl, which already carries per-task
        # fragment_smiles, linker{} and properties{} -- exactly the measured
        # baseline a relative reward needs.
        "source": "repair_tasks",
        "repair_tasks": "../repair_tasks.jsonl",
        "tasks_file": "data/tasks.jsonl",
        # Pro-GAT repair is strictly upstream and never a tool: the rollout
        # path stays pure-CPU RDKit, with no second GPU model inside the
        # training step. Only the `generated` stage is used -- the RL policy IS
        # the repair step.
        "stages": ["generated"],
        "verdicts": ["poor", "broken"],
        # task_id in repair_tasks.jsonl is '{target}-{sample}-{stage}'. The
        # trailing '-generated' is stripped, because GRPO rewrites ids to
        # '{id}@{generation}' and traces to '{id}#{attempt}', and a third
        # suffix makes group membership ambiguous -- which silently destroys
        # advantages rather than failing.
        "strip_stage_suffix": True,
        "offset": 0,
        "limit": None,
        "shuffle_seed": 42,
        # Keep the qed-only tasks as controls rather than dropping them.
        "include_controls": True,
    },
    "model": {
        "name": "meta-models/Muse-Glimmer-30B",
        "max_prompt_tokens": 24576,
        "max_completion_tokens": 2048,
    },
    "sft": {
        "output_dir": "runs/sft",
        "traces_file": "data/sft_traces.jsonl",
        # Calibrate against the scripted linker expert before trusting this
        # number; moltrain's 0.45 was set against its own expert's measured
        # spread and does not transfer.
        "min_reward": None,
        "require_submission": True,
        "require_improvement": True,
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        # null means "use the epochs". Set a step count instead when the
        # traces are formulaic enough that the format is learned long before
        # the epoch ends -- see sft.training_settings.
        "max_steps": None,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 8192,
        "bf16": True,
        "gradient_checkpointing": True,
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
        # TWO H200s, ONE ENGINE EACH.
        #
        # moltrain sets TRL's use_vllm True *and* starts its own vllm.LLM
        # inside rollout_func (grpo.py:144, grpo.py:206), so one process holds
        # two engines. Under DDP that is four engines on two GPUs and neither
        # fits.
        #
        # TRL's rollout_func is the documented place to do your own generation,
        # so TRL's internal engine is dead weight the moment one is supplied.
        # Turning it off leaves exactly one engine per rank.
        "use_trl_vllm": False,
        "vllm_gpu_memory_utilization": 0.35,
        # Rank-gate the engine and the asyncio loop: both must be per-process,
        # and moltrain's `asyncio.run()` inside the training step is not
        # rank-safe as written.
        "engine_per_rank": True,
        "ddp": {
            "enabled": True,
            "world_size": 2,
            "launcher": "torchrun",  # torchrun | accelerate
            "find_unused_parameters": False,
        },
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
        "resume_from_sft": True,
    },
    "evaluate": {
        "output_dir": "runs/eval",
        # OPTIMISATION.md's four arms. `unchanged` is the floor any arm must
        # clear, and it is a real floor here: with the fragments frozen, most
        # descriptors barely move.
        "arms": ["unchanged", "one_shot", "policy", "policy_best_of_3"],
        "attempts": 3,
        # Stratify, because the three classes fail in different ways and will
        # not have the same yield: reactive (one group swap, local) is expected
        # high, structural moderate, chemistry low. A near-zero chemistry yield
        # is a finding -- those molecules need resampling, not harder
        # optimisation.
        "stratify_by": "optimisation_class",
        "min_per_class": 50,
        "separate_cache": True,
        "property_cache_path": "cache/linker_properties_eval.jsonl",
    },
    "run": {
        "seed": 42,
        "log_dir": "runs/logs",
        # Console level; the run's log file always records DEBUG.
        "log_level": "INFO",
        "log_to_file": True,
        # auto | on | off -- auto means off when stderr is not a terminal, so
        # bars stay out of redirected logs.
        "progress": "auto",
    },
}

# Subtrees whose keys are an open-ended map rather than a fixed schema, so an
# override REPLACES them instead of merging into the defaults. Without this a
# user who declares their own set of threshold names silently inherits every
# default name alongside, and trains against targets they thought they had
# removed. The same holds for regression_tolerance: it names which descriptors
# are policed at all, so dropping one has to be expressible.
_REPLACE_PREFIXES = (
    "reward.thresholds",
    "reward.regression_tolerance",
)


class ConfigError(Exception):
    """A configuration problem worth reporting verbatim."""


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in merged:
            raise ConfigError(f"unknown config key: {where}")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            if where.startswith(_REPLACE_PREFIXES):
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


def resolve_sibling(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path to a sibling repository imported as a library.

    `molbench` supplies the tools and `protacbench` the measurement layer, and
    both are put on sys.path rather than installed, so a relative default in
    the YAML has to become an absolute path before anything appends it.
    """
    if key not in ("molbench", "protacbench"):
        raise ConfigError(f"unknown sibling: {key}")
    section = cfg.get(key)
    if not isinstance(section, dict):
        raise ConfigError(f"config section {key} is missing or not a mapping")
    path = section.get("path")
    if not path:
        raise ConfigError(f"{key}.path is not set")
    return resolve(cfg, path)


def log_destination(
    cfg: dict[str, Any], args: Any, stem: str, *, suffix: str = ""
) -> str | None:
    """Where one CLI's log file goes.

    `moltrain.logs.configure_from_args` resolves `run.log_dir` with moltrain's
    own `resolve`, which is relative to the moltrain package directory -- so
    left alone every protactrain command writes its log into moltrain's run
    directory, where nobody looks for it and two packages' runs interleave.
    Every protactrain CLI therefore computes the path here first and hands it
    over as `--log-file`.

    `suffix` is for the ranks: under DDP both processes would otherwise open
    the same file and interleave their lines.
    """
    log_file = getattr(args, "log_file", None)
    if log_file:
        return str(log_file)
    if not cfg["run"].get("log_to_file", True):
        return None

    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = resolve(cfg, cfg["run"].get("log_dir", "runs/logs"))
    return str(directory / f"{stem}{suffix}-{stamp}.log")
