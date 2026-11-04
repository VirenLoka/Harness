"""The GPU gate must not be what breaks the GPU run.

vLLM starts its EngineCore in a forked child process. A forked child may not
touch CUDA if the parent already holds a context -- torch registers a
`pthread_atfork` handler the moment one exists, and the child dies with

    RuntimeError: Cannot re-initialize CUDA in forked subprocess.

Measured on the H200 box: `python -m protactrain.grpo rollout` died exactly
that way, and the context came from our own `require_cuda`. `is_available()`
falls through to `torch._C._cuda_getDeviceCount()`, which torch's own source
comments describe as initialising "the CUDA Driver API via cuInit". So the
check that ran immediately before every engine start was the reason the engine
could not start -- and nothing in the traceback pointed back at it.

`device_count()` asks NVML first and only falls back to the poisoning call if
NVML is absent entirely, so the gate keeps its meaning without the side effect.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _stub_torch(*, devices: int) -> types.ModuleType:
    """A torch whose `is_available()` is a tripwire rather than a function."""
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")

    def is_available() -> bool:
        raise AssertionError(
            "require_cuda called torch.cuda.is_available(), which initialises a "
            "CUDA context and makes vLLM's forked EngineCore unstartable"
        )

    cuda.is_available = is_available
    cuda.device_count = lambda: devices
    cuda.is_initialized = lambda: False
    torch.cuda = cuda
    return torch


@pytest.mark.parametrize("module", ["protactrain.grpo", "protactrain.policy"])
def test_require_cuda_never_calls_is_available(module, monkeypatch):
    mod = importlib.import_module(module)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch(devices=2))
    # Two visible devices: the gate must pass, and must do so without the
    # tripwire firing.
    mod.require_cuda("a test")


@pytest.mark.parametrize("module", ["protactrain.grpo", "protactrain.policy"])
def test_require_cuda_still_refuses_a_host_with_no_device(module, monkeypatch):
    mod = importlib.import_module(module)
    monkeypatch.setitem(sys.modules, "torch", _stub_torch(devices=0))
    with pytest.raises(RuntimeError, match="CUDA device"):
        mod.require_cuda("a test")


def test_importing_the_package_sets_the_fork_safe_defaults():
    """A clean interpreter, because this one imported the package long ago.

    Both variables are stripped from the child's environment first: the point
    is what the package sets by itself, not what happens to be exported in the
    shell that is running the tests.
    """
    import json

    code = (
        "import protactrain, os, json;"
        "print(json.dumps({k: os.environ.get(k) for k in "
        "('VLLM_WORKER_MULTIPROC_METHOD', 'PYTORCH_NVML_BASED_CUDA_CHECK')}))"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VLLM_WORKER_MULTIPROC_METHOD", "PYTORCH_NVML_BASED_CUDA_CHECK"}
    }
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    env = json.loads(out.stdout.strip().splitlines()[-1])
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


def test_an_explicit_setting_still_wins():
    """`setdefault`, so a deliberate choice on the host is not overwritten."""
    code = (
        "import protactrain, os;"
        "print(os.environ['VLLM_WORKER_MULTIPROC_METHOD'])"
    )
    env = dict(os.environ)
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip().splitlines()[-1] == "fork"


# --------------------------------------------------------- the SFT warm start


def test_resume_needs_an_adapter_not_just_a_directory(tmp_path):
    """`sft.output_dir` exists as soon as a log lands in it."""
    from protactrain.grpo import should_resume_from_sft

    cfg = {
        "sft": {"output_dir": str(tmp_path / "sft")},
        "grpo": {"resume_from_sft": True},
    }
    (tmp_path / "sft").mkdir()
    adapter, resume = should_resume_from_sft(cfg)
    assert adapter == tmp_path / "sft"
    assert resume is False, "an empty directory is not a warm start"

    (tmp_path / "sft" / "adapter_config.json").write_text("{}")
    _, resume = should_resume_from_sft(cfg)
    assert resume is True


def test_resume_is_off_when_the_config_says_so(tmp_path):
    from protactrain.grpo import should_resume_from_sft

    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "adapter_config.json").write_text("{}")
    cfg = {
        "sft": {"output_dir": str(tmp_path / "sft")},
        "grpo": {"resume_from_sft": False},
    }
    assert should_resume_from_sft(cfg)[1] is False
