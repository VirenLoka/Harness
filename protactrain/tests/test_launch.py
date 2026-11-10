"""How many ranks the launcher starts, and who decides.

`--nproc_per_node` came from a hardcoded `${WORLD_SIZE:-2}`, which made the
`grpo.ddp` block documentation rather than configuration: setting
`world_size: 1` changed nothing, and the second rank started anyway on
device 1 -- `local_rank` is the device, so rank 1 is device 1 whether or not
device 1 is free. That is the card a single-card run exists to stay off.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protactrain.config import load_config  # noqa: E402

LAUNCHER = ROOT / "protactrain" / "scripts" / "launch_ddp.sh"


def _launch(tmp_path: Path, body: dict | None, env: dict | None = None) -> str:
    """The torchrun line the launcher would run, without running it."""
    argv = [str(LAUNCHER), "grpo"]
    if body is not None:
        config = tmp_path / "ranks.yaml"
        config.write_text(yaml.safe_dump(body))
        argv.append(str(config))
    argv.append("--dry-run")

    # WORLD_SIZE dropped deliberately: inherited from the shell it would
    # override the config and every assertion below would be about the
    # environment instead.
    environment = {k: v for k, v in os.environ.items() if k != "WORLD_SIZE"}
    environment.update(env or {})
    done = subprocess.run(
        argv, capture_output=True, text=True, env=environment, cwd=str(ROOT)
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_packaged_config_plans_one_rank():
    """The box's second H200 is in use, and rank 1 is device 1 regardless."""
    ddp = load_config(None)["grpo"]["ddp"]
    assert ddp["enabled"] is False
    assert ddp["world_size"] == 1


def test_the_launcher_starts_one_process_for_the_packaged_config(tmp_path):
    assert "--nproc_per_node=1" in _launch(tmp_path, None)


def test_the_launcher_reads_the_rank_count_out_of_the_config(tmp_path):
    """The one that catches a hardcoded number, whichever number it is.

    Pinning only the packaged config would be satisfied just as well by
    hardcoding 1, so this asks the launcher for a different answer.
    """
    output = _launch(tmp_path, {"grpo": {"ddp": {"enabled": True, "world_size": 2}}})
    assert "--nproc_per_node=2" in output
    assert "on 2 process(es)" in output


def test_disabled_ddp_means_one_rank_whatever_the_world_size_says(tmp_path):
    """`world_size if enabled else 1`, the rule `grpo check` plans with."""
    output = _launch(tmp_path, {"grpo": {"ddp": {"enabled": False, "world_size": 4}}})
    assert "--nproc_per_node=1" in output


def test_the_environment_still_overrides_the_config(tmp_path):
    """A one-off on both cards should not need a config edit."""
    assert "--nproc_per_node=2" in _launch(tmp_path, None, env={"WORLD_SIZE": "2"})


def test_the_thread_budget_follows_the_rank_count(tmp_path):
    """A rank is a full copy of the stack with its own OpenMP pool.

    Two ranks at the whole budget ask the container for twice what one rank
    does, against the same cap -- which arrives as a segfault with no usable
    traceback, not an error message.
    """
    assert "budget 8 across 1" in _launch(tmp_path, None)
    both_cards = {"grpo": {"ddp": {"enabled": True, "world_size": 2}}}
    assert "budget 8 across 2" in _launch(tmp_path, both_cards)
