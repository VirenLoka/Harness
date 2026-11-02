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

__version__ = "0.1.0"
