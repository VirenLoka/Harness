"""Host diagnostics that survive a broken dependency.

Importing candidate dependencies in this process means one bad install takes
the whole report down with it. That is not hypothetical: on the first real run
`peft`'s import chain reached scipy's BLAS loader, OpenBLAS could not spawn its
threads inside the container, and the check hung until it was interrupted --
so the report never reached the CUDA check or the TRL introspection that were
the actual reasons for running it.

So each module is imported in its own short-lived subprocess with a timeout. A
dependency that is missing, broken or hanging becomes one line in the report
instead of the end of it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

# Every library that sizes a thread pool from the visible core count.
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

BLAS_MARKER = "blas_thread_init"

BLAS_HINT = (
    "OpenBLAS could not create its threads. The host exposes many cores but "
    "caps how many threads a process may start, which is normal in a "
    "container, and every BLAS-backed import dies or hangs. moltrain sets "
    f"{THREAD_VARS[0]} and friends to MOLTRAIN_NUM_THREADS (default 8) on "
    "import, but anything started outside it needs them too:\n"
    "    export MOLTRAIN_NUM_THREADS=8\n"
    "    export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
)


def thread_limited_env(limit: str | int | None = None) -> dict[str, str]:
    """A copy of the environment with the thread pools pinned small.

    Used for every diagnostic subprocess: a probe that only prints a version
    string has no business allocating one thread per core.
    """
    environment = dict(os.environ)
    value = str(limit or environment.get("MOLTRAIN_NUM_THREADS") or 8)
    for name in THREAD_VARS:
        environment[name] = value
    environment.setdefault("PYTHONUNBUFFERED", "1")

    # Ensure CUDA visibility is preserved for GPU-accelerated subprocesses
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

    return environment


def looks_like_blas_thread_failure(text: str) -> bool:
    return BLAS_MARKER in (text or "")


@dataclass
class ModuleReport:
    name: str
    ok: bool
    detail: str  # a version, or why it could not be imported
    blas_failure: bool = False
    timed_out: bool = False


def probe_module(
    name: str, *, python: str | None = None, timeout_s: float = 90.0
) -> ModuleReport:
    """Import one module in a subprocess and report what happened."""
    script = (
        "import importlib\n"
        f"m = importlib.import_module({name!r})\n"
        "print(getattr(m, '__version__', 'present'))\n"
    )
    try:
        completed = subprocess.run(
            [python or sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=thread_limited_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ModuleReport(
            name,
            False,
            f"import did not finish within {timeout_s:.0f}s",
            timed_out=True,
        )

    stderr = completed.stderr or ""
    blas = looks_like_blas_thread_failure(stderr)
    if completed.returncode == 0:
        return ModuleReport(name, True, (completed.stdout or "").strip() or "present")

    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    interesting = [line for line in lines if not line.startswith("OpenBLAS ")]
    detail = (interesting or lines or ["no output"])[-1][:200]
    return ModuleReport(name, False, detail, blas_failure=blas)


def report_imports(
    modules: tuple[str, ...], *, python: str | None = None, timeout_s: float = 90.0
) -> tuple[list[ModuleReport], list[str]]:
    """Import each module independently; return the reports and the problems."""
    reports = [
        probe_module(name, python=python, timeout_s=timeout_s) for name in modules
    ]
    problems: list[str] = []
    for report in reports:
        if report.ok:
            continue
        if report.blas_failure or report.timed_out:
            problems.append(f"{report.name} could not be imported: {report.detail}")
        else:
            problems.append(f"{report.name} is not usable: {report.detail}")
    if any(r.blas_failure for r in reports):
        problems.append(BLAS_HINT)
    return reports, problems


CUDA_SCRIPT = """
import json
import torch
devices = []
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    devices.append({
        "name": properties.name,
        "total_memory_gb": round(properties.total_memory / 1e9, 1),
    })
print("MOLTRAIN_CUDA " + json.dumps({
    "available": bool(torch.cuda.is_available()),
    "torch": torch.__version__,
    "devices": devices,
}))
"""


def report_cuda(
    *, python: str | None = None, timeout_s: float = 120.0
) -> tuple[list[str], list[str]]:
    """What torch can see, asked in a subprocess so a crash is just a line."""
    import json

    try:
        completed = subprocess.run(
            [python or sys.executable, "-c", CUDA_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=thread_limited_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            [f"  torch did not answer within {timeout_s:.0f}s"],
            ["torch could not be queried about CUDA"],
        )

    payload = {}
    for line in (completed.stdout or "").splitlines():
        if line.startswith("MOLTRAIN_CUDA "):
            payload = json.loads(line[len("MOLTRAIN_CUDA ") :])
    if not payload:
        stderr = completed.stderr or ""
        problems = ["torch could not report on CUDA"]
        if looks_like_blas_thread_failure(stderr):
            problems.append(BLAS_HINT)
        tail = [
            line
            for line in stderr.strip().splitlines()
            if line.strip() and not line.startswith("OpenBLAS ")
        ]
        return ([f"  {(tail or ['no output'])[-1][:200]}"], problems)

    lines = [f"  torch {payload['torch']}, cuda available: {payload['available']}"]
    for device in payload["devices"]:
        lines.append(f"  device: {device['name']} ({device['total_memory_gb']} GB)")
    problems = [] if payload["available"] else ["no CUDA device is visible to torch"]
    return lines, problems


def report_binaries(names: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Which external engines are on PATH, and how to get the missing ones."""
    import shutil

    lines: list[str] = []
    missing: list[str] = []
    for name in names:
        override = os.environ.get(f"MOLBENCH_{name.upper()}_BIN")
        found = override or shutil.which(name)
        lines.append(f"  {name}: {found or 'NOT FOUND'}")
        if not found:
            missing.append(name)
    problems: list[str] = []
    if missing:
        packages = " ".join(
            "openbabel" if name == "obabel" else name for name in missing
        )
        problems.append(
            f"missing docking binaries: {', '.join(missing)}. Install with "
            f"`conda install -c conda-forge {packages}`, or point "
            f"MOLBENCH_<NAME>_BIN at each one."
        )
    return lines, problems
