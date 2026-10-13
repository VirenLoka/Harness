"""The multi-turn optimization episode the policy is trained on.

The loop is policy-agnostic: it takes a `generate` callable that maps a
transcript to the next assistant turn. A scripted function stands in for the
model during tests, vLLM drives it during GRPO rollouts, and nothing else
changes. That is what keeps the environment verifiable without a GPU.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from . import atem
from .oracle import Oracle, Properties
from .rewards import RewardBreakdown, compute_reward
from .toolkit import CallLog, Toolkit

# Either a plain or an async callable: (transcript) -> next assistant turn.
GenerateFn = Callable[[str], str | Awaitable[str]]

SYSTEM_PROMPT = """You are a medicinal chemist optimizing a molecule produced by a \
structure-based generative model.

Improve the molecule's properties while keeping it bound to the same target. You \
have tools for 2D properties, SMARTS transformations, 3D conformers and docking. \
Measure before you claim: never state a number you did not obtain from a tool call.

Work in this order:
1. Measure the starting molecule.
2. Propose and test modifications, one change at a time.
3. Call submit_molecule exactly once with your best molecule. The task ends there.

Goals for this task (all are relative to the starting molecule):
{goals}

You have at most {max_turns} turns and {max_tool_calls} tool calls."""

USER_PROMPT = """Target: {target}
Receptor file: {receptor_file}
Starting molecule (from {origin}): {smiles}

Measured starting properties:
{baseline}

Optimize it."""


@dataclass
class Task:
    """One optimization problem: a starting molecule and its target."""

    task_id: str
    smiles: str
    target: str = "unknown"
    receptor_file: str | None = None
    origin: str = "DiffSBDD"
    baseline: Properties | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Task:
        baseline = record.get("baseline")
        return cls(
            task_id=record["task_id"],
            smiles=record["smiles"],
            target=record.get("target", "unknown"),
            receptor_file=record.get("receptor_file"),
            origin=record.get("origin", "DiffSBDD"),
            baseline=Properties(**baseline) if baseline else None,
            metadata=record.get("metadata", {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "smiles": self.smiles,
            "target": self.target,
            "receptor_file": self.receptor_file,
            "origin": self.origin,
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "metadata": self.metadata,
        }


@dataclass
class EpisodeResult:
    """Everything one rollout produced, including its cost."""

    task_id: str
    transcript: str
    completion: str
    reward: float
    breakdown: dict[str, Any]
    submitted_smiles: str | None
    final_properties: dict[str, Any] | None
    turns: int
    tool_calls: int
    tool_failures: int
    malformed_calls: int
    wall_time_s: float
    tool_time_s: float
    tool_seconds_by_name: dict[str, float]
    docks_run: int
    cache_hits: int
    stop_reason: str
    candidates_evaluated: int = 0
    best_candidate: dict[str, Any] | None = None


def describe_goals(cfg: dict[str, Any]) -> str:
    lines = []
    for name, rule in cfg["reward"]["thresholds"].items():
        direction = rule.get("direction", "increase")
        goal = float(rule.get("goal_delta", 0.0))
        if goal <= 0:
            tolerance = float(rule.get("tolerance", 0.0))
            lines.append(f"- {name}: do not get worse by more than {tolerance}")
        else:
            verb = "lower" if direction == "decrease" else "raise"
            lines.append(f"- {name}: {verb} it by at least {goal}")
    return "\n".join(lines)


def build_prompt(cfg: dict[str, Any], task: Task) -> str:
    """The system and user turns, before any model output."""
    baseline = task.baseline.as_dict() if task.baseline else {}
    shown = {
        key: baseline.get(key)
        for key in ("sa_score", "qed", "exact_mw", "clogp", "tpsa", "binding_affinity")
        if baseline.get(key) is not None
    }
    baseline_text = (
        "\n".join(f"  {k}: {v}" for k, v in shown.items()) or "  (not measured yet)"
    )
    system = SYSTEM_PROMPT.format(
        goals=describe_goals(cfg),
        max_turns=cfg["episode"]["max_turns"],
        max_tool_calls=cfg["episode"]["max_tool_calls"],
    )
    user = USER_PROMPT.format(
        target=task.target,
        receptor_file=task.receptor_file or "(none; 2D optimization only)",
        smiles=task.smiles,
        origin=task.origin,
        baseline=baseline_text,
    )
    return (
        f"{atem.START}system{atem.MESSAGE}{system}{atem.END_TURN}"
        f"{atem.START}user{atem.MESSAGE}{user}{atem.END_TURN}"
        f"{atem.START}assistant"
    )


async def run_episode(
    cfg: dict[str, Any],
    task: Task,
    generate: GenerateFn,
    toolkit: Toolkit,
    oracle: Oracle,
) -> EpisodeResult:
    """Drive one optimization episode to termination and score it."""
    started = time.monotonic()
    episode_cfg = cfg["episode"]
    max_turns = int(episode_cfg["max_turns"])
    max_tool_calls = int(episode_cfg["max_tool_calls"])
    max_chars = int(episode_cfg["max_tool_result_chars"])

    baseline = task.baseline or await oracle.evaluate(
        task.smiles, task.receptor_file, is_final=False, label=f"{task.task_id}_start"
    )
    task.baseline = baseline

    prompt = build_prompt(cfg, task)
    transcript = prompt
    completion_parts: list[str] = []
    call_log = CallLog()

    turns = tool_calls = tool_failures = malformed = 0
    submitted_smiles: str | None = None
    seen_candidates: set[str] = set()
    evaluated_candidates: dict[str, Properties] = {}
    stop_reason = "max_turns"
    docks_before = oracle.stats["dock_misses"]
    hits_before = oracle.stats["dock_hits"] + oracle.stats["property_hits"]

    while turns < max_turns:
        turns += 1
        result = generate(transcript)
        text = await result if hasattr(result, "__await__") else result
        completion_parts.append(text)
        transcript += text

        parsed = atem.parse_assistant_text(text)
        malformed += len(parsed.malformed)

        if not parsed.tool_calls:
            stop_reason = (
                "final_answer" if parsed.final_text is not None else "no_tool_call"
            )
            break

        terminate = False
        for call in parsed.tool_calls:
            if tool_calls >= max_tool_calls:
                stop_reason = "max_tool_calls"
                terminate = True
                break
            tool_calls += 1
            outcome = await toolkit.call(call.name, call.arguments)
            call_log.record(outcome)
            if not outcome.ok:
                tool_failures += 1

            reply = atem.render_tool_result(
                call.name, outcome.as_text(), max_chars=max_chars
            )
            transcript += reply
            completion_parts.append(reply)

            candidate = call.arguments.get("smiles")
            if isinstance(candidate, str) and candidate.strip():
                seen_candidates.add(candidate.strip())

            spec = toolkit.specs.get(call.name)
            if spec is not None and spec.terminal and outcome.ok:
                submitted_smiles = str(
                    outcome.content.get("submitted_smiles", "")
                ).strip()
                stop_reason = "submitted"
                terminate = True
                break

        # Under `every_candidate`, every molecule the policy touched is scored,
        # which both densifies the signal and warms the cache for the final one.
        if oracle.mode == "every_candidate":
            fresh = [s for s in seen_candidates if s not in evaluated_candidates]
            if fresh:
                evaluated = await oracle.evaluate_many(
                    fresh, task.receptor_file, is_final=True
                )
                for smiles, properties in zip(fresh, evaluated, strict=True):
                    evaluated_candidates[smiles] = properties

        if terminate:
            break
        transcript += f"{atem.START}assistant"

    final_properties: Properties | None = None
    if submitted_smiles:
        final_properties = await oracle.evaluate(
            submitted_smiles,
            task.receptor_file,
            is_final=True,
            label=f"{task.task_id}_final",
        )
        # A threshold on binding affinity cannot be scored unless the starting
        # molecule was docked too; backfill it once, which the cache makes cheap.
        if (
            final_properties.binding_affinity is not None
            and baseline.binding_affinity is None
            and task.receptor_file
        ):
            affinity = await oracle.dock(
                baseline.smiles, task.receptor_file, label=f"{task.task_id}_start"
            )
            if affinity is not None:
                baseline.binding_affinity = affinity
                if baseline.heavy_atoms:
                    baseline.ligand_efficiency = round(
                        affinity / baseline.heavy_atoms, 4
                    )

    # The strongest molecule the policy produced, by whichever thresholds are
    # measurable. Logged for analysis; the reward still scores what was submitted.
    best_candidate: dict[str, Any] | None = None
    scored_candidates = [p for p in evaluated_candidates.values() if p.valid]
    if scored_candidates:

        def _rank(properties: Properties) -> tuple[float, float]:
            affinity = properties.binding_affinity
            return (
                affinity if affinity is not None else 0.0,
                properties.sa_score or 99.0,
            )

        best_candidate = min(scored_candidates, key=_rank).as_dict()

    breakdown: RewardBreakdown = compute_reward(
        cfg,
        start=baseline,
        final=final_properties,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        malformed_calls=malformed,
        submitted=submitted_smiles is not None,
        # submit_molecule is the only terminal tool, so everything else the
        # policy called was measurement.
        measurement_calls=max(0, tool_calls - (1 if submitted_smiles else 0)),
    )

    return EpisodeResult(
        task_id=task.task_id,
        transcript=transcript,
        completion="".join(completion_parts),
        reward=breakdown.total,
        breakdown=breakdown.as_dict(),
        submitted_smiles=submitted_smiles,
        final_properties=final_properties.as_dict() if final_properties else None,
        turns=turns,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        malformed_calls=malformed,
        wall_time_s=round(time.monotonic() - started, 3),
        tool_time_s=round(call_log.total_seconds, 3),
        tool_seconds_by_name={
            k: round(v, 3) for k, v in call_log.seconds_by_tool().items()
        },
        docks_run=oracle.stats["dock_misses"] - docks_before,
        cache_hits=(oracle.stats["dock_hits"] + oracle.stats["property_hits"])
        - hits_before,
        stop_reason=stop_reason,
        candidates_evaluated=len(evaluated_candidates),
        best_candidate=best_candidate,
    )
