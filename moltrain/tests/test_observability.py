"""Progress and logging: the parts that make a long run legible.

These matter more than they look. `tasks build` spends about a minute per
pocket protonating a receptor, so twenty-five pockets is twenty-five minutes
of apparent silence unless something says otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import logs, progress
from moltrain.config import load_config


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["run"]["log_dir"] = str(tmp_path / "logs")
    return config


@pytest.fixture(autouse=True)
def quiet_root():
    """Leave the package logger as it was found."""
    logger = logging.getLogger("moltrain")
    saved = list(logger.handlers), logger.level, logger.propagate
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.handlers, logger.level, logger.propagate = saved


# -------------------------------------------------------------- progress bars


def test_wrapping_an_iterable_yields_everything_and_counts_it():
    bar = progress.track([1, 2, 3], desc="x", disable=True)
    assert list(bar) == [1, 2, 3]
    assert bar._count == 3


def test_a_bar_without_an_iterable_refuses_to_be_iterated():
    with pytest.raises(TypeError, match="not given an iterable"):
        list(progress.track(total=3, disable=True))


def test_manual_updates_and_phases_work_without_tqdm():
    with progress.track(total=4, desc="building", unit="pocket", disable=True) as bar:
        bar.postfix("3ks3-002 preparing receptor")
        bar.update(2)
        line = bar.render()
    assert "building" in line
    assert "2/4 (50%)" in line
    assert "3ks3-002 preparing receptor" in line


def test_the_status_line_carries_an_eta():
    bar = progress.track(total=10, desc="x", disable=True)
    bar.update(1)
    assert "eta" in bar.render()
    bar.close()


def test_a_finished_bar_does_not_claim_an_eta():
    bar = progress.track(total=2, desc="x", disable=True)
    bar.update(2)
    assert "eta" not in bar.render()
    bar.close()


def test_a_total_free_bar_counts_units():
    bar = progress.track(desc="docking", unit="dock", disable=True)
    bar.update(3)
    assert "3 dock" in bar.render()
    bar.close()


def test_seconds_are_formatted_for_reading():
    assert progress.format_seconds(9) == "9s"
    assert progress.format_seconds(95) == "1m35s"
    assert progress.format_seconds(3725) == "1h02m"
    assert progress.format_seconds(-5) == "0s"


def test_progress_config_maps_onto_tqdms_disable():
    assert progress.resolve_disable({"run": {"progress": "off"}}) is True
    assert progress.resolve_disable({"run": {"progress": "on"}}) is False
    # auto means None, which is tqdm's own "off when stderr is not a terminal".
    assert progress.resolve_disable({"run": {"progress": "auto"}}) is None
    assert progress.resolve_disable(None) is None
    assert progress.resolve_disable({"run": {"progress": "on"}}, True) is True


def test_the_fallback_logs_progress_rather_than_staying_silent(caplog):
    """Without tqdm the run must still say where it is."""
    logging.getLogger("moltrain.progress").propagate = True
    with caplog.at_level(logging.INFO, logger="moltrain.progress"):
        bar = progress.track(total=4, desc="building", disable=False)
        if progress.HAVE_TQDM:
            pytest.skip("tqdm is installed, so the bar replaces the log lines")
        for _ in range(4):
            bar.update(1)
        bar.close()
    messages = [r.getMessage() for r in caplog.records]
    assert any("4/4 (100%)" in m for m in messages)
    # The closing line must not repeat a count the last update already logged.
    assert sum("4/4 (100%)" in m for m in messages) == 1


def test_a_disabled_bar_logs_nothing(caplog):
    logging.getLogger("moltrain.progress").propagate = True
    with (
        caplog.at_level(logging.INFO, logger="moltrain.progress"),
        progress.track(total=2, desc="x", disable=True) as bar,
    ):
        bar.update(2)
    assert caplog.records == []


# ------------------------------------------------------------------- logging


def test_configure_is_idempotent(cfg):
    logs.configure(cfg)
    first = len(logging.getLogger("moltrain").handlers)
    logs.configure(cfg)
    assert len(logging.getLogger("moltrain").handlers) == first


def test_the_log_file_records_debug_even_when_the_console_does_not(cfg):
    path = logs.configure(cfg)
    assert path is not None
    logs.get_logger("demo").debug("a quiet detail")
    for handler in logging.getLogger("moltrain").handlers:
        handler.flush()
    assert "a quiet detail" in path.read_text()


def test_verbose_lowers_the_console_level(cfg):
    logs.configure(cfg, verbose=True)
    console = [
        h
        for h in logging.getLogger("moltrain").handlers
        if isinstance(h, logs.TqdmHandler)
    ]
    assert console and console[0].level == logging.DEBUG


def test_quiet_raises_the_console_level(cfg):
    logs.configure(cfg, quiet=True)
    console = [
        h
        for h in logging.getLogger("moltrain").handlers
        if isinstance(h, logs.TqdmHandler)
    ]
    assert console and console[0].level == logging.WARNING


def test_main_is_not_used_as_a_logger_name():
    """`python -m moltrain.tasks` sees its own name as __main__."""
    assert logs.get_logger("__main__").name == "moltrain"
    assert logs.get_logger("tasks").name == "moltrain.tasks"
    assert logs.get_logger("moltrain.oracle").name == "moltrain.oracle"


def test_an_unwritable_log_directory_does_not_stop_the_run(cfg, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    cfg["run"]["log_dir"] = str(blocker / "logs")
    assert logs.configure(cfg) is None
    logs.get_logger("demo").info("still running")  # must not raise


def test_log_to_file_can_be_turned_off(cfg):
    cfg["run"]["log_to_file"] = False
    assert logs.configure(cfg) is None


def test_cli_flags_round_trip(cfg):
    parser = argparse.ArgumentParser()
    logs.add_cli_flags(parser)
    args = parser.parse_args(["--verbose", "--no-progress"])
    assert logs.progress_disabled(cfg, args) is True
    args = parser.parse_args([])
    assert logs.progress_disabled(cfg, args) is None
    assert logs.progress_disabled(cfg, parser.parse_args(["--quiet"])) is True
