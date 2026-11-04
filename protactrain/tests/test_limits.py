"""The task ceiling has to be reported before a run proves it.

A two-rank SFT run on the H200 box died with

    libgomp: Thread creation failed: Resource temporarily unavailable
    error: can't start new thread
    ... exitcode: -11 (SIGSEGV)

EAGAIN from `pthread_create`: the container's cgroup `pids` cap, which counts
threads and not just processes. Nothing in that output names a cap, and the
SIGSEGV is libgomp failing to recover rather than a fault of its own, so the
only way to not lose an afternoon to it is to measure the ceiling up front.
"""

from __future__ import annotations

from protactrain import limits


def test_a_cgroup_v2_ceiling_is_read_and_preferred(tmp_path, monkeypatch):
    """cgroup first: in a container RLIMIT_NPROC is usually untouched."""
    (tmp_path / "pids.max").write_text("200\n")
    (tmp_path / "pids.current").write_text("150\n")

    real = limits._read_int

    def fake(path: str):
        if path == "/sys/fs/cgroup/pids.max":
            return real(str(tmp_path / "pids.max"))
        if path == "/sys/fs/cgroup/pids.current":
            return real(str(tmp_path / "pids.current"))
        return None

    monkeypatch.setattr(limits, "_read_int", fake)
    limit, current, source = limits.pid_ceiling()
    assert (limit, current) == (200, 150)
    assert source == "cgroup v2"


def test_max_means_no_limit_not_a_crash(tmp_path):
    """cgroup v2 writes the literal string `max` for an uncapped controller.

    Read as an int that is a ValueError, and a diagnostic that raises while
    diagnosing is the worst outcome available.
    """
    uncapped = tmp_path / "pids.max"
    uncapped.write_text("max\n")
    assert limits._read_int(str(uncapped)) is None

    capped = tmp_path / "capped"
    capped.write_text("4096\n")
    assert limits._read_int(str(capped)) == 4096

    assert limits._read_int(str(tmp_path / "does-not-exist")) is None


def test_a_run_that_will_not_fit_is_called_a_problem(monkeypatch):
    monkeypatch.setattr(
        limits, "pid_ceiling", lambda: (200, 180, "cgroup v2")
    )
    monkeypatch.setattr(limits, "threads_in_use", lambda: 12)
    lines, problems = limits.report_threads(2)

    assert any("task ceiling: 200" in line for line in lines)
    assert len(problems) == 1
    # The message has to name the symptom, because the symptom is what gets
    # pasted into a search box.
    assert "libgomp" in problems[0]
    assert "WORLD_SIZE=1" in problems[0]


def test_ample_headroom_is_not_a_problem(monkeypatch):
    monkeypatch.setattr(
        limits, "pid_ceiling", lambda: (100_000, 500, "cgroup v2")
    )
    monkeypatch.setattr(limits, "threads_in_use", lambda: 12)
    _lines, problems = limits.report_threads(2)
    assert problems == []


def test_no_ceiling_found_reports_rather_than_guesses(monkeypatch):
    monkeypatch.setattr(limits, "pid_ceiling", lambda: (None, None, "unknown"))
    lines, problems = limits.report_threads(2)
    assert problems == []
    assert any("none found" in line for line in lines)


def test_it_runs_on_this_host_whatever_this_host_is():
    """Never raise. A diagnostic that crashes is worse than no diagnostic."""
    lines, problems = limits.report_threads(2)
    assert lines and all(isinstance(line, str) for line in lines)
    assert all(isinstance(p, str) for p in problems)
