"""Reward for one optimization episode.

Thresholds are relative to the episode's own starting molecule, so a hard
target and an easy one are scored on the same footing: what is rewarded is the
improvement the policy achieved, not the absolute quality of the pocket it was
handed.

Thresholds come in two kinds, and conflating them breaks the reward.

A **goal** (`goal_delta > 0`) asks for improvement, and contributes shaped
progress in [0, 1] rather than a step function, so a policy that moves SA from
4.9 to 4.6 against a 0.5 goal still sees gradient.

A **constraint** (`goal_delta <= 0`, "do not get worse") is a gate, not an
objective. It earns nothing when satisfied and costs real reward when
violated. Scoring it as progress was a bug worth remembering: an unchanged
molecule satisfies every constraint perfectly, so submitting the input
collected full marks on "maintain binding affinity" and outscored an episode
that actually improved something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .oracle import Properties


@dataclass
class ThresholdOutcome:
    name: str
    start: float | None
    final: float | None
    delta: float | None
    goal_delta: float
    progress: float  # 0..1
    met: bool
    # For a violated constraint, how far past its allowance it went, measured
    # in tolerance units. Zero whenever the rule is satisfied.
    severity: float = 0.0


@dataclass
class RewardBreakdown:
    """Every term that produced the scalar, kept for logging and debugging."""

    total: float
    terms: dict[str, float] = field(default_factory=dict)
    thresholds: list[ThresholdOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "terms": {k: round(v, 4) for k, v in self.terms.items()},
            "thresholds": [
                {
                    "name": t.name,
                    "start": t.start,
                    "final": t.final,
                    "delta": None if t.delta is None else round(t.delta, 4),
                    "goal_delta": t.goal_delta,
                    "progress": round(t.progress, 4),
                    "met": t.met,
                    "severity": round(t.severity, 4),
                }
                for t in self.thresholds
            ],
            "notes": self.notes,
        }


def _value(properties: Properties, name: str) -> float | None:
    return getattr(properties, name, None)


def score_thresholds(
    start: Properties, final: Properties, thresholds: dict[str, dict[str, Any]]
) -> list[ThresholdOutcome]:
    """Shaped progress toward each relative goal."""
    outcomes: list[ThresholdOutcome] = []
    for name, rule in thresholds.items():
        start_value = _value(start, name)
        final_value = _value(final, name)
        goal_delta = float(rule.get("goal_delta", 0.0))
        direction = rule.get("direction", "increase")
        tolerance = float(rule.get("tolerance", 0.0))

        if start_value is None or final_value is None:
            # A property nobody measured cannot be scored. Docking is the usual
            # case, when the policy is set to dock only final molecules.
            outcomes.append(
                ThresholdOutcome(
                    name, start_value, final_value, None, goal_delta, 0.0, False
                )
            )
            continue

        raw_delta = final_value - start_value
        # Normalize so "improvement" is always positive.
        improvement = -raw_delta if direction == "decrease" else raw_delta

        severity = 0.0
        if goal_delta <= 0:
            # A "do not get worse" rule. Satisfied inside the tolerance; past
            # it, record how far past, in tolerance units, so the charge can
            # tell a slip from a collapse.
            met = improvement >= -tolerance
            progress = (
                1.0 if met else max(0.0, 1.0 + improvement / max(tolerance, 1e-6))
            )
            if not met:
                severity = (-improvement - tolerance) / max(tolerance, 1e-6)
        else:
            progress = max(0.0, min(1.0, improvement / goal_delta))
            met = improvement >= goal_delta

        outcomes.append(
            ThresholdOutcome(
                name,
                start_value,
                final_value,
                raw_delta,
                goal_delta,
                progress,
                met,
                severity,
            )
        )
    return outcomes


def compute_reward(
    cfg: dict[str, Any],
    *,
    start: Properties,
    final: Properties | None,
    tool_calls: int,
    tool_failures: int,
    malformed_calls: int,
    submitted: bool,
    measurement_calls: int | None = None,
) -> RewardBreakdown:
    """Turn one finished episode into a scalar reward."""
    reward_cfg = cfg["reward"]
    weights = reward_cfg["weights"]

    if not submitted:
        return RewardBreakdown(
            total=float(reward_cfg["no_submission_reward"]),
            terms={"no_submission": float(reward_cfg["no_submission_reward"])},
            notes=["episode ended without calling submit_molecule"],
        )

    if final is None or not final.valid:
        return RewardBreakdown(
            total=float(reward_cfg["invalid_submission_reward"]),
            terms={
                "invalid_submission": float(reward_cfg["invalid_submission_reward"])
            },
            notes=["submitted SMILES did not parse"],
        )

    outcomes = score_thresholds(start, final, reward_cfg["thresholds"])
    scorable = [o for o in outcomes if o.delta is not None]
    goals = [o for o in scorable if o.goal_delta > 0]
    constraints = [o for o in scorable if o.goal_delta <= 0]

    # Only goals pay progress. Constraints are satisfied by inaction, so
    # paying them would make doing nothing competitive.
    progress = sum(o.progress for o in goals) / len(goals) if goals else 0.0
    violated = [o for o in constraints if not o.met]
    all_met = bool(goals) and all(o.met for o in goals) and not violated

    # Charged by how far past the allowance each constraint went, capped so a
    # single ruinous rollout cannot swamp its group's advantage.
    cap = float(reward_cfg["constraint_violation_cap"])
    severity = sum(min(outcome.severity, cap) for outcome in violated)

    total_calls = max(tool_calls, 1)
    tool_success = 1.0 - (tool_failures + malformed_calls) / total_calls

    # Calls that were not the terminating submission. A policy that submits
    # without measuring anything earns no tool-success credit, because paying
    # it would make "submit blind" the cheapest way to collect the flat terms.
    if measurement_calls is None:
        measurement_calls = max(0, tool_calls - (1 if submitted else 0))

    # Only calls beyond a free budget are penalised. Measuring, transforming
    # and verifying is the work; a flat per-call charge made it irrational,
    # because the penalty on a dozen honest calls outweighed the threshold
    # progress they bought.
    free_calls = int(reward_cfg["turn_penalty_free_calls"])
    billable = max(0, tool_calls - free_calls)

    terms = {
        "thresholds": weights["thresholds"] * progress,
        "all_met_bonus": weights["all_met_bonus"] if all_met else 0.0,
        "validity": weights["validity"],
        "novelty": weights["novelty"] if final.smiles != start.smiles else 0.0,
        "tool_success": (
            weights["tool_success"] * max(0.0, tool_success)
            if measurement_calls > 0
            else 0.0
        ),
        "turn_penalty": -weights["turn_penalty"] * billable,
        "constraint_violation": -weights["constraint_violation"] * severity,
    }

    notes: list[str] = []
    if measurement_calls == 0:
        notes.append("submitted without calling a measurement tool")
    if final.smiles == start.smiles:
        notes.append("submitted the starting molecule unchanged")
    if not scorable:
        notes.append(
            "no threshold could be scored; check that docking produced a value"
        )
    elif not goals:
        notes.append(
            "every scorable threshold is a constraint, so there was nothing to "
            "earn progress on"
        )
    for outcome in violated:
        notes.append(
            f"{outcome.name} broke its constraint: {outcome.start} -> "
            f"{outcome.final}, {outcome.severity:.2f} tolerances past its allowance"
        )

    return RewardBreakdown(
        total=sum(terms.values()), terms=terms, thresholds=outcomes, notes=notes
    )
