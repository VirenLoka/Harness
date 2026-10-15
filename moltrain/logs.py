"""Logging for runs that outlast your attention.

The split this package keeps: library modules *log*, CLIs *print*. A summary
table is the program's output and belongs on stdout; "docked 3ks3-002 in 4.1s"
is a diagnostic and belongs on stderr with a timestamp. Mixing them makes both
harder to use.

The console shows INFO, or DEBUG with `--verbose`. The run's log file always
records DEBUG regardless, so an overnight job leaves a full trace even though
the console stayed readable -- which is the whole point of having one.

Console records go out through `progress.write`, because printing on top of a
live tqdm bar corrupts it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import progress

ROOT_NAME = "moltrain"

CONSOLE_FORMAT = "%(levelname)-7s %(name)-20s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"


class TqdmHandler(logging.Handler):
    """Writes through tqdm so a live bar is not torn in half."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            progress.write(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """A logger under the package root, e.g. `get_logger("tasks")`.

    `__main__` is folded away: a module run as `python -m moltrain.tasks` sees
    its own name as `__main__`, which would label every line `moltrain.__main__`.
    """
    if name in ("__main__", f"{ROOT_NAME}.__main__"):
        return logging.getLogger(ROOT_NAME)
    if name.startswith(f"{ROOT_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_NAME}.{name}")


def _root() -> logging.Logger:
    return logging.getLogger(ROOT_NAME)


def configure(
    cfg: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
    quiet: bool = False,
    stem: str = "moltrain",
    log_file: str | Path | None = None,
) -> Path | None:
    """Set up console and file logging once. Returns the log file, if any."""
    logger = _root()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    run = (cfg or {}).get("run", {})
    configured = str(run.get("log_level", "INFO")).upper()
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = getattr(logging, configured, logging.INFO)

    # The file handler wants DEBUG even when the console does not, so the
    # logger itself has to pass everything down to the handlers.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = TqdmHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    logger.addHandler(console)

    path: Path | None = None
    if log_file is not None:
        path = Path(log_file).expanduser()
    elif cfg is not None and run.get("log_to_file", True):
        from .config import resolve

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = resolve(cfg, run.get("log_dir", "runs/logs")) / f"{stem}-{stamp}.log"

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(FILE_FORMAT))
            logger.addHandler(handler)
        except OSError as exc:
            # An unwritable log directory is not a reason to abandon the run.
            path = None
            logger.warning(
                "could not open a log file (%s); logging to console only", exc
            )

    logger.debug(
        "logging configured: console=%s file=%s pid=%s threads=%s",
        logging.getLevelName(level),
        path or "none",
        os.getpid(),
        os.environ.get("OPENBLAS_NUM_THREADS", "?"),
    )
    return path


def add_cli_flags(parser: argparse.ArgumentParser) -> None:
    """The observability flags every moltrain command accepts."""
    parser.add_argument(
        "--verbose", action="store_true", help="log every tool call and dock"
    )
    parser.add_argument("--quiet", action="store_true", help="warnings and errors only")
    parser.add_argument(
        "--no-progress", action="store_true", help="suppress progress bars"
    )
    parser.add_argument(
        "--log-file", default=None, help="write the run log here instead"
    )


def configure_from_args(
    cfg: dict[str, Any] | None, args: argparse.Namespace, *, stem: str
) -> Path | None:
    """Apply the CLI flags, and say on stderr where the log went."""
    path = configure(
        cfg,
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
        stem=stem,
        log_file=getattr(args, "log_file", None),
    )
    if path is not None and not getattr(args, "quiet", False):
        print(f"logging to {path}", file=sys.stderr)
    return path


def progress_disabled(
    cfg: dict[str, Any] | None, args: argparse.Namespace
) -> bool | None:
    override = True if getattr(args, "no_progress", False) else None
    if getattr(args, "quiet", False):
        override = True
    return progress.resolve_disable(cfg, override)
