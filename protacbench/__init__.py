"""Assessment and triage of DiffPROTAC -> Pro-GAT linker samples.

Thread pools are capped before anything else can import numpy, for the same
reason `moltrain` does it: a container exposes every core on the host while
capping how many threads one process may create, and OpenBLAS sizes its pool
from the visible core count.
"""

from __future__ import annotations

import os

NUM_THREADS = os.environ.get("MOLTRAIN_NUM_THREADS", "8")

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, NUM_THREADS)
