"""GRPO training for tool-using PROTAC linker optimisation.

Sibling of moltrain, which does the same for DiffSBDD pocket tasks. The
generic machinery is imported from moltrain and the measurement layer from
protacbench; what lives here is the task, the reward and the training shape.

Thread pools are capped here, before anything else in the package can run.
A container routinely exposes every core on the host while capping how many
threads one process may create, and OpenBLAS, OMP and MKL each size their pool
from the visible core count -- so the first numpy or torch import dies with
`blas_thread_init: pthread_create failed for thread 63 of 64`. By the time any
submodule is imported it is too late to set these, because the pool size is
read once at load. An explicitly set variable always wins.
"""

from __future__ import annotations

import os

NUM_THREADS = os.environ.get("PROTACTRAIN_NUM_THREADS", "8")

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, NUM_THREADS)

# -------------------------------------------------------------- fork poisoning
#
# vLLM starts its EngineCore in a child process, and by default that child is
# *forked*. A forked child may not touch CUDA if the parent has already
# initialised it: torch registers a `pthread_atfork` handler the moment a CUDA
# context exists, and the child then dies with
#
#     RuntimeError: Cannot re-initialize CUDA in forked subprocess.
#
# Two settings, because one alone is not enough.
#
# `PYTORCH_NVML_BASED_CUDA_CHECK` stops *us* poisoning the fork. With it unset,
# `torch.cuda.is_available()` falls through to `torch._C._cuda_getDeviceCount()`
# -- which, in torch's own words, "initializes the CUDA Driver API via cuInit"
# -- and that call registers the fork handler. Our GPU gate ran before every
# engine start, so the gate itself was what made the engine unstartable. The
# gate no longer calls `is_available()` (see `grpo.require_cuda`), but this is
# set as well so that any other caller reaching for it is safe too.
#
# `VLLM_WORKER_MULTIPROC_METHOD` is the backstop for everything we do not
# control. transformers, datasets, a future moltrain import or a notebook that
# touched a tensor before importing us can all poison the fork just as easily,
# and under `spawn` the child starts clean and none of it matters.
#
# Both are `setdefault`, so an explicit value from the environment still wins.
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Allocate in segments that can grow rather than in fixed blocks. A 30B model
# plus checkpointed activations fragments the caching allocator badly, and the
# OOM this package hit on a shared H200 reported 472 MiB "reserved but
# unallocated" while failing to find 508. torch's own OOM message recommends
# this setting by name. It is not a substitute for free memory on the card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
