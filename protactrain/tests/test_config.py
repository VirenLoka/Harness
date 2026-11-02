"""`config.DEFAULTS` against `configs/default.yaml`, and the merge rules.

The two are maintained by hand in parallel, which is a standing invitation to
drift. A drift here is silent: `load_config` merges a user file onto DEFAULTS,
so a key that exists in the YAML but not in DEFAULTS reads fine from the
packaged default and vanishes the moment anyone passes `--config`. The first
test is the one that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protactrain.config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULTS,
    ConfigError,
    load_config,
    resolve,
    resolve_sibling,
)

# Injected by load_config, so they are not expected in either source.
_INJECTED = {"_config_path", "_root"}


def _flatten(node, prefix=""):
    """Every leaf of a nested mapping, as dotted-path -> value."""
    out = {}
    for key, value in node.items():
        if not prefix and key in _INJECTED:
            continue
        where = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, where))
            if not value:  # an empty mapping is itself a leaf worth comparing
                out[where] = {}
        else:
            out[where] = value
    return out


# ------------------------------------------------------- DEFAULTS mirrors YAML


def test_defaults_is_a_faithful_mirror_of_the_packaged_yaml():
    """Every key and every value, in both directions.

    Asserted as a set difference rather than `==` so a failure names the key
    that drifted instead of printing two 150-line dicts side by side.
    """
    on_disk = _flatten(yaml.safe_load(DEFAULT_CONFIG.read_text()))
    in_code = _flatten(DEFAULTS)

    missing_from_code = sorted(set(on_disk) - set(in_code))
    missing_from_yaml = sorted(set(in_code) - set(on_disk))
    assert not missing_from_code, f"in default.yaml but not in DEFAULTS: {missing_from_code}"
    assert not missing_from_yaml, f"in DEFAULTS but not in default.yaml: {missing_from_yaml}"

    differing = {
        key: (on_disk[key], in_code[key])
        for key in on_disk
        if on_disk[key] != in_code[key]
    }
    assert not differing, f"value drift (yaml, DEFAULTS): {differing}"


def test_loading_the_packaged_default_changes_nothing():
    cfg = load_config()
    assert _flatten(cfg) == _flatten(DEFAULTS)


def test_qed_is_absent_from_the_reward_entirely():
    """Not forgotten: scoring QED would rank the worst PROTACs highest.

    The 128 published degraders behind this corpus have a median QED of
    0.064. A term that rewards it measures how unlike a PROTAC the molecule
    is, which is why 952 qed-only tasks are controls rather than goals.
    """
    cfg = load_config()
    assert "qed" not in cfg["reward"]["thresholds"]
    assert "qed" not in cfg["reward"]["regression_tolerance"]
    assert "qed" not in cfg["reward"]["weights"]


def test_no_docking_key_survives_anywhere():
    """A PROTAC binds two proteins; one rigid receptor does not model that."""
    flat = _flatten(DEFAULTS)
    assert not [k for k in flat if "dock" in k.lower()], [
        k for k in flat if "dock" in k.lower()
    ]
    assert "vina" not in str(flat).lower()


# --------------------------------------------------------------- the deep merge


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "custom.yaml"
    path.write_text(body)
    return path


def test_a_scalar_override_merges_and_leaves_its_siblings_alone(tmp_path):
    cfg = load_config(_write(tmp_path, "grpo:\n  max_steps: 7\n"))
    assert cfg["grpo"]["max_steps"] == 7
    assert cfg["grpo"]["num_generations"] == DEFAULTS["grpo"]["num_generations"]
    assert cfg["model"]["name"] == DEFAULTS["model"]["name"]


def test_reward_thresholds_are_replaced_not_merged(tmp_path):
    """A user listing one band gets exactly that band.

    Merging would be the wrong default here and quietly so: someone narrowing
    the run to span alone would still be scored on six other bands they never
    wrote, and would read the resulting reward as span progress.
    """
    cfg = load_config(
        _write(tmp_path, "reward:\n  thresholds:\n    span:\n      low: 9\n      high: 12\n      tolerance: 2\n")
    )
    assert list(cfg["reward"]["thresholds"]) == ["span"]
    assert cfg["reward"]["thresholds"]["span"] == {"low": 9, "high": 12, "tolerance": 2}
    # Its siblings outside the replaced subtree are untouched.
    assert cfg["reward"]["weights"] == DEFAULTS["reward"]["weights"]


def test_regression_tolerance_is_replaced_not_merged(tmp_path):
    cfg = load_config(_write(tmp_path, "reward:\n  regression_tolerance:\n    clogp: 0.3\n"))
    assert cfg["reward"]["regression_tolerance"] == {"clogp": 0.3}


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path):
    """A typo in a config is a silent misconfiguration otherwise."""
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, "reward:\n  nope: 1\n"))
    assert "reward.nope" in str(exc.value)


def test_a_moltrain_shaped_key_is_refused(tmp_path):
    """There is no docking section here, and a ported config must say so."""
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "molbench:\n  receptor_dir: ../receptors\n"))


# ------------------------------------------------------------------- the paths


def test_relative_paths_resolve_against_the_package_not_the_cwd():
    """Package-relative, like moltrain's, so `../repair_tasks.jsonl` works
    from any directory a command is launched from."""
    cfg = load_config()
    corpus = resolve(cfg, cfg["dataset"]["repair_tasks"])
    assert corpus.is_absolute()
    assert corpus.name == "repair_tasks.jsonl"
    assert corpus.parent == Path(__file__).resolve().parents[2]


def test_siblings_resolve_to_the_repositories_next_door():
    cfg = load_config()
    assert resolve_sibling(cfg, "protacbench").name == "protacbench"
    assert resolve_sibling(cfg, "molbench").name == "molbench"
    with pytest.raises(ConfigError):
        resolve_sibling(cfg, "diffsbdd")


def test_log_destination_stays_under_protactrain():
    """moltrain's own resolve would put these logs in moltrain/runs/logs."""
    import argparse

    from protactrain.config import log_destination

    cfg = load_config()
    args = argparse.Namespace(log_file=None)
    path = Path(log_destination(cfg, args, "tasks-build"))
    assert Path(__file__).resolve().parents[2] in path.parents
    assert path.name.startswith("tasks-build-")


def test_a_rank_suffix_separates_the_two_ddp_log_files():
    import argparse

    from protactrain.config import log_destination

    cfg = load_config()
    args = argparse.Namespace(log_file=None)
    zero = Path(log_destination(cfg, args, "grpo-train", suffix="-rank0")).name
    one = Path(log_destination(cfg, args, "grpo-train", suffix="-rank1")).name
    assert zero.startswith("grpo-train-rank0-")
    assert one.startswith("grpo-train-rank1-")
