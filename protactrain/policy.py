"""The model side of a rollout, for the stages that are not GRPO.

Almost nothing here is new. `moltrain.policy` already owns the pieces that
matter -- the stop strings generation has to halt on, the transcript trimmer,
and `BatchingGenerator`, which collects the concurrent one-turn requests a
group of episodes makes and issues them to vLLM as a single batch. Those are
engine mechanics, not task knowledge, and copying them would be two codebases
to keep in step for no benefit.

What this module adds is the two things that are local to protactrain.

**A CUDA gate.** Everything below starts a 30B model. On the laptop this
package was written on there is nothing to start it in, and vLLM's own failure
there is a long wait followed by an allocator trace. `grpo.py` already refuses
on its own path; `traces collect --source rejection` and the three model
evaluation arms reach vLLM through here instead, so the same refusal has to
live on this side too.

**One engine for the whole run.** `grpo.py` deliberately builds its own
rank-bound engine, because under DDP the engine must adopt the rank's device.
Nothing that reaches this module is distributed -- trace collection and
evaluation are single-process -- so the simpler shared-engine shape is the
right one, and it is `moltrain.policy.vllm_factory`'s shape exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from moltrain.logs import get_logger
from moltrain.policy import (
    STOP_STRINGS,
    BatchingGenerator,
    Policy,
    SamplingSettings,
    VLLMEngine,
    build_policy,
    build_sampling_params,
    default_token_counter,
    trim_transcript,
)

from .episode import Task

LOG = get_logger("policy")

__all__ = [
    "STOP_STRINGS",
    "BatchingGenerator",
    "Policy",
    "SamplingSettings",
    "VLLMEngine",
    "build_policy",
    "build_sampling_params",
    "default_token_counter",
    "load_engine",
    "require_cuda",
    "trim_transcript",
    "vllm_factory",
]


def require_cuda(what: str) -> None:
    """Refuse, visibly and immediately, rather than hang on a host with no GPU.

    Deliberately duplicated from `grpo.require_cuda` rather than imported:
    importing it would pull `grpo` -- and through it `sft`, `trl_compat` and
    the TRL probe -- into every evaluation run that uses the GPU-free
    `unchanged` arm. The check is four lines; the import is the whole training
    stack.

    `device_count()` and not `is_available()`, for the reason spelled out in
    `grpo.require_cuda`: `is_available()` initialises a CUDA context, and a
    context in this process is what stops vLLM's forked EngineCore starting.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            f"{what} needs torch with CUDA, which is not installed here. "
            "This stage runs on the two-H200 box, not on a laptop."
        ) from exc

    if torch.cuda.device_count() == 0:
        raise RuntimeError(
            f"{what} needs a visible CUDA device and torch can see none. "
            "Run `python -m protactrain.grpo check` on the training host."
        )


def load_engine(cfg: dict[str, Any]) -> VLLMEngine:
    """Start one vLLM engine in this process.

    No rank handling, unlike `grpo.load_engine`: the callers here are
    single-process. A distributed run that reached this function would put two
    engines on device 0, which is the exact trap `grpo.py` exists to avoid, so
    it is better that this one stays obviously non-distributed.
    """
    require_cuda("starting a vLLM engine")

    from vllm import LLM

    grpo = cfg["grpo"]
    LOG.info(
        "starting vLLM on %s at %.2f memory utilisation",
        cfg["model"]["name"],
        float(grpo["vllm_gpu_memory_utilization"]),
    )
    return VLLMEngine(
        LLM(
            model=cfg["model"]["name"],
            gpu_memory_utilization=float(grpo["vllm_gpu_memory_utilization"]),
            max_model_len=int(cfg["model"]["max_prompt_tokens"])
            + int(cfg["model"]["max_completion_tokens"]),
            enable_lora=True,
            trust_remote_code=True,
        )
    )


def vllm_factory(cfg: dict[str, Any]) -> Callable[[Task], Policy]:
    """One shared engine and batcher across every task in a run.

    The batcher is shared on purpose. Each episode asks for one turn at a
    time, so without it eight concurrent episodes decode eight sequences
    serially; with it the group costs roughly one rollout's worth of decode.
    A per-task batcher would never see more than one request and would be
    the serial case with extra bookkeeping.
    """
    engine = load_engine(cfg)
    params = build_sampling_params(cfg)
    policy = build_policy(cfg, engine, params)

    def make(_task: Task) -> Policy:
        return policy

    return make
