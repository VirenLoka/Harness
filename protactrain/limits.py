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
from pathlib import Path

__all__ = ["pid_ceiling", "report_threads", "threads_in_use"]


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
