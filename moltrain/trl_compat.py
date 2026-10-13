"""Passing config to TRL without betting on one TRL version's parameter names.

TRL renames training arguments between releases (`max_seq_length` became
`max_length`, for one). Rather than pin a version and fail on the training box,
each config is filtered against what the installed class actually accepts, and
whatever was dropped is printed. A silently ignored argument is the worst
outcome here -- it would mean training at a setting nobody chose.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

# Settings this package names one way and some TRL versions name another.
ALIASES: dict[str, tuple[str, ...]] = {
    "max_seq_length": ("max_length", "max_seq_length"),
    "max_length": ("max_length", "max_seq_length"),
    "num_generations": ("num_generations",),
    "beta": ("beta",),
}


def accepted_names(target: Any) -> set[str]:
    """Every keyword `target` will take, whether dataclass or plain callable."""
    if dataclasses.is_dataclass(target):
        return {field.name for field in dataclasses.fields(target)}
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return set()
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return set()  # takes **kwargs, so nothing can be filtered out
    return set(signature.parameters)


def adapt(target: Any, settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Rename what can be renamed, drop what cannot, and report the drops."""
    accepted = accepted_names(target)
    if not accepted:
        return dict(settings), []

    adapted: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in settings.items():
        if key in accepted:
            adapted[key] = value
            continue
        alternative = next(
            (name for name in ALIASES.get(key, ()) if name in accepted), None
        )
        if alternative:
            adapted[alternative] = value
        else:
            dropped.append(key)
    return adapted, dropped


def describe(target: Any, wanted: dict[str, Any]) -> str:
    """A per-setting report of what the installed TRL will and will not take."""
    accepted = accepted_names(target)
    name = getattr(target, "__name__", str(target))
    if not accepted:
        return f"{name}: takes **kwargs, nothing can be checked ahead of time"

    lines = [f"{name}:"]
    for key in sorted(wanted):
        if key in accepted:
            lines.append(f"  ok       {key}")
        else:
            alternative = next(
                (alias for alias in ALIASES.get(key, ()) if alias in accepted), None
            )
            if alternative:
                lines.append(f"  renamed  {key} -> {alternative}")
            else:
                lines.append(f"  DROPPED  {key}  (this version has no such setting)")
    return "\n".join(lines)
