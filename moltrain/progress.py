"""Progress reporting for the parts of a run that take minutes.

tqdm is optional. It is present in the training container but not in every
environment this package runs in, and the test suite must not depend on it, so
everything goes through one wrapper that falls back to periodic log lines.
A long run is never silent either way -- which is the actual requirement; the
bar is just the nicer form of it.

Bars and log lines both go to stderr, so stdout stays parseable when a command
is piped. `write` routes through tqdm when a bar is live, because printing
directly on top of one corrupts it.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any, Self

try:  # pragma: no cover - exercised by whichever branch the host provides
    from tqdm.auto import tqdm as _tqdm

    HAVE_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None
    HAVE_TQDM = False

_LOG = logging.getLogger("moltrain.progress")

# How often the fallback reports, when there is no bar to watch.
FALLBACK_INTERVAL_S = 30.0
FALLBACK_FRACTION = 0.1


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class Progress:
    """One unit of progress reporting, bar-backed or log-backed."""

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        *,
        total: int | None = None,
        desc: str = "work",
        unit: str = "it",
        disable: bool | None = None,
        leave: bool = True,
    ):
        self.desc = desc
        self.unit = unit
        self._iterable = iterable
        self._postfix = ""
        self._count = 0
        self._started = time.monotonic()
        self._last_report = self._started
        self._last_fraction = 0.0
        self._last_logged_count = -1
        self._closed = False

        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.total = total

        # `disable=None` is tqdm's own "off when stderr is not a terminal",
        # which keeps bars out of redirected logs.
        self._disabled = bool(disable)
        self._bar = None
        if HAVE_TQDM and not self._disabled:
            self._bar = _tqdm(
                total=total,
                desc=desc,
                unit=unit,
                leave=leave,
                disable=disable,
                file=sys.stderr,
                dynamic_ncols=True,
            )

    # ----------------------------------------------------------------- state

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started

    def _eta_s(self) -> float | None:
        if not self.total or self._count <= 0:
            return None
        per_item = self.elapsed_s / self._count
        return per_item * (self.total - self._count)

    # ---------------------------------------------------------------- output

    def describe(self, text: str) -> None:
        """Change the leading label."""
        self.desc = text
        if self._bar is not None:
            self._bar.set_description(text, refresh=False)

    def postfix(self, text: str) -> None:
        """Say what is happening right now, e.g. which pocket and which phase.

        Phases matter more than counts here: receptor preparation takes about a
        minute per pocket, so a bar that only counts pockets still looks stuck.
        """
        self._postfix = text
        if self._bar is not None:
            self._bar.set_postfix_str(text, refresh=True)
        elif not self._disabled:
            self._maybe_log(force=False)

    def update(self, n: int = 1) -> None:
        self._count += n
        if self._bar is not None:
            self._bar.update(n)
        else:
            self._maybe_log(force=False)

    def _maybe_log(self, *, force: bool) -> None:
        if self._disabled:
            return
        now = time.monotonic()
        fraction = (self._count / self.total) if self.total else 0.0
        due = (
            force
            or (now - self._last_report) >= FALLBACK_INTERVAL_S
            or (fraction - self._last_fraction) >= FALLBACK_FRACTION
        )
        if not due:
            return
        self._last_report = now
        self._last_fraction = fraction
        self._last_logged_count = self._count
        _LOG.info("%s", self.render())

    def render(self) -> str:
        """A one-line status, used by the fallback and worth logging at the end."""
        parts = [self.desc]
        if self.total:
            parts.append(f"{self._count}/{self.total} ({self._count / self.total:.0%})")
        else:
            parts.append(f"{self._count} {self.unit}")
        parts.append(f"elapsed {format_seconds(self.elapsed_s)}")
        eta = self._eta_s()
        if eta is not None and self._count < (self.total or 0):
            parts.append(f"eta {format_seconds(eta)}")
        if self._postfix:
            parts.append(f"- {self._postfix}")
        return " ".join(parts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.close()
        elif not self._disabled and self._count != self._last_logged_count:
            # Only if the last update did not already report this count.
            _LOG.info("%s", self.render())

    # ------------------------------------------------------------- protocols

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            raise TypeError("this Progress was not given an iterable to wrap")
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def track(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str = "work",
    unit: str = "it",
    disable: bool | None = None,
    leave: bool = True,
) -> Progress:
    """Wrap an iterable, or make a bar to `update()` by hand."""
    return Progress(
        iterable, total=total, desc=desc, unit=unit, disable=disable, leave=leave
    )


def write(text: str) -> None:
    """Emit a line without tearing a live bar in half."""
    if HAVE_TQDM:
        try:
            _tqdm.write(text, file=sys.stderr)
        except Exception:  # noqa: BLE001 - a torn bar beats a lost log line
            print(text, file=sys.stderr)
        return
    print(text, file=sys.stderr)


def resolve_disable(
    cfg: dict[str, Any] | None, override: bool | None = None
) -> bool | None:
    """Turn `run.progress` into tqdm's `disable` argument.

    `auto` becomes None, which is tqdm's own "off when stderr is not a
    terminal" and keeps bars out of redirected logs.
    """
    if override is not None:
        return override
    setting = str((cfg or {}).get("run", {}).get("progress", "auto")).lower()
    if setting in ("off", "false", "no"):
        return True
    if setting in ("on", "true", "yes"):
        return False
    return None
