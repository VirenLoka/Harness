"""How many threads this host will actually let a process create.

A JupyterLab container is the worst case for this. It exposes every core on
the machine -- `nproc` says 128 -- while capping how many tasks the container
may hold through the cgroup `pids` controller or `RLIMIT_NPROC`. Every thread
counts against that cap, not just processes.

The failure is ugly and does not name the cause:

    libgomp: Thread creation failed: Resource temporarily unavailable
    error: can't start new thread
    ... exitcode: -11 (SIGSEGV)

`Resource temporarily unavailable` is EAGAIN from `pthread_create`, which is
what the kernel returns when the cap is hit. libgomp has no way to recover, so
the segfault is the consequence rather than a separate bug -- chasing the
SIGSEGV leads nowhere.

Two ranks make it twice as likely: `torchrun --nproc_per_node=2` runs two full
copies of the stack, each with its own OpenMP pool, its own torch intra-op
pool and its own loader threads, against one shared cap.

Nothing here raises. It reports, so that `check` can say the run will not fit
before the run proves it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "SFT_GIB_PER_RANK",
    "compute_apps",
    "gpu_memory",
    "pid_ceiling",
    "report_gpus",
    "report_threads",
    "threads_in_use",
]

# Rough, and deliberately so. A 30B model in bf16 is ~60 GiB of weights before
# anything else; LoRA's own parameters and optimiser state are small, but
# activations are not, even with gradient checkpointing on. Measured on the
# H200 box: an SFT rank reached 67.5 GiB allocated, of which torch accounted
# for 66.1 GiB. 75 is that plus room, and the number exists to catch "this was
# never going to fit", not to predict the peak.
SFT_GIB_PER_RANK = 75.0


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    if text == "max":  # cgroup v2 spells "no limit" this way
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pid_ceiling() -> tuple[int | None, int | None, str]:
    """(limit, in use, where the limit came from).

    cgroup first, because in a container that is the binding one and
    `RLIMIT_NPROC` is usually untouched and enormous.
    """
    for maximum, current, source in (
        ("/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids.current", "cgroup v2"),
        (
            "/sys/fs/cgroup/pids/pids.max",
            "/sys/fs/cgroup/pids/pids.current",
            "cgroup v1",
        ),
    ):
        limit = _read_int(maximum)
        if limit is not None:
            return limit, _read_int(current), source

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (ImportError, OSError, ValueError):  # pragma: no cover - platform
        return None, None, "unknown"
    if soft in (-1, 2**63 - 1):
        return None, None, "RLIMIT_NPROC (unlimited)"
    return soft, None, "RLIMIT_NPROC"


def threads_in_use() -> int | None:
    """Threads in *this* process, from /proc. None off Linux."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def report_threads(world_size: int) -> tuple[list[str], list[str]]:
    """Lines to print, and problems to add to the caller's list."""
    lines: list[str] = []
    problems: list[str] = []

    per_rank = os.environ.get("PROTACTRAIN_NUM_THREADS", "<unset>")
    omp = os.environ.get("OMP_NUM_THREADS", "<unset>")
    lines.append(f"  PROTACTRAIN_NUM_THREADS={per_rank}  OMP_NUM_THREADS={omp}")

    limit, current, source = pid_ceiling()
    if limit is None:
        lines.append(f"  task ceiling: none found ({source})")
        return lines, problems

    threads = threads_in_use()
    lines.append(
        f"  task ceiling: {limit} from {source}"
        + (f", {current} already in use" if current is not None else "")
        + (f"; {threads} in this process" if threads is not None else "")
    )

    # A rank is not just its OpenMP pool: torch's intra-op pool, the loader,
    # the allocator and CUDA's own threads all land on the same budget. 64 per
    # rank is a deliberately rough floor from watching a 30B load; the point
    # is to catch "nowhere near enough", not to predict the true number.
    headroom = limit - (current or 0)
    needed = 64 * max(1, world_size)
    lines.append(f"  headroom {headroom}; a {world_size}-rank run wants ~{needed}")
    if headroom < needed:
        problems.append(
            f"only {headroom} task slot(s) are free under a ceiling of {limit} "
            f"({source}), and {world_size} rank(s) want roughly {needed}. A run "
            "started here dies with `libgomp: Thread creation failed: Resource "
            "temporarily unavailable` and then SIGSEGV. Lower "
            "PROTACTRAIN_NUM_THREADS, run one rank with WORLD_SIZE=1, or close "
            "the notebook kernels holding the rest of the budget."
        )
    return lines, problems


# ---------------------------------------------------------------- the devices
#
# Asked through `nvidia-smi` rather than torch, for two reasons. It reports
# what the *card* holds, including every other process on it, where
# `torch.cuda.mem_get_info` sees only free and total and tells you nothing
# about who took the rest. And it costs no CUDA context in this process --
# which matters, because creating one here is what breaks vLLM's forked
# EngineCore further down the run. See protactrain/__init__.py.


def _nvidia_smi(query: str, entity: str) -> list[list[str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{entity}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - platform
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in (out.stdout or "").strip().splitlines():
        if line.strip():
            rows.append([cell.strip() for cell in line.split(",")])
    return rows


def gpu_memory() -> list[dict[str, float]]:
    """Per device: index, total, used and free, in GiB."""
    devices = []
    for row in _nvidia_smi("index,memory.total,memory.used,memory.free", "gpu"):
        if len(row) < 4:
            continue
        try:
            devices.append(
                {
                    "index": int(row[0]),
                    # nounits gives MiB
                    "total": float(row[1]) / 1024,
                    "used": float(row[2]) / 1024,
                    "free": float(row[3]) / 1024,
                }
            )
        except ValueError:
            continue
    return devices


def compute_apps() -> list[dict[str, Any]]:
    """Every process holding memory on a device, as nvidia-smi sees it."""
    apps = []
    for row in _nvidia_smi("pid,used_gpu_memory,process_name", "compute-apps"):
        if len(row) < 2:
            continue
        try:
            apps.append(
                {
                    "pid": int(row[0]),
                    "gib": float(row[1]) / 1024,
                    "name": row[2] if len(row) > 2 else "?",
                }
            )
        except ValueError:
            continue
    return apps


def report_gpus(
    world_size: int, need_per_rank: float = SFT_GIB_PER_RANK
) -> tuple[list[str], list[str]]:
    """Whether the ranks asked for will actually fit on what is free now.

    `report_cuda` in moltrain prints each card's TOTAL memory, which is why a
    card with 200 MiB free still read as healthy right up to the OOM. Free is
    the number that decides whether a run starts.
    """
    lines: list[str] = []
    problems: list[str] = []

    devices = gpu_memory()
    if not devices:
        lines.append("  nvidia-smi is not on PATH, so free memory is unknown")
        return lines, problems

    apps = compute_apps()
    foreign = sum(app["gib"] for app in apps)
    for device in devices:
        lines.append(
            f"  device {device['index']}: {device['free']:.1f} GiB free "
            f"of {device['total']:.1f} ({device['used']:.1f} in use)"
        )
    if apps:
        lines.append(
            f"  {len(apps)} process(es) already hold {foreign:.1f} GiB across "
            "the visible devices:"
        )
        for app in sorted(apps, key=lambda a: -a["gib"])[:8]:
            lines.append(f"    pid {app['pid']}: {app['gib']:.1f} GiB  {app['name']}")

    usable = sorted((d for d in devices), key=lambda d: -d["free"])[:world_size]
    if len(usable) < world_size:
        problems.append(
            f"{world_size} rank(s) were asked for but nvidia-smi reports "
            f"{len(devices)} device(s)"
        )
        return lines, problems

    short = [d for d in usable if d["free"] < need_per_rank]
    if short:
        where = ", ".join(f"{d['index']} ({d['free']:.1f} GiB free)" for d in short)
        problems.append(
            f"a rank needs roughly {need_per_rank:.0f} GiB and these device(s) "
            f"do not have it: {where}. The run will reach "
            "`torch.OutOfMemoryError` partway through the first step, after the "
            "dataset has been tokenised. Free the devices (`nvidia-smi` lists "
            "who holds what; a crashed run of this package often leaves a "
            "process behind), or train on one card with "
            "`CUDA_VISIBLE_DEVICES=<n> WORLD_SIZE=1`."
        )
    return lines, problems
