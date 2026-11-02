"""Scoring a linker optimisation episode.

This keeps `moltrain.rewards`' central distinction -- a goal pays shaped
progress, a constraint pays nothing when satisfied and is charged when broken
-- and adds the two things a linker task needs that a pocket task did not.

**Bands, not directions.** Half the linker targets are two-sided. cLogP wants
3.5 to 5.6, span wants 8 to 13 bonds, TPSA wants 170 to 217. `moltrain` only
knows `increase` and `decrease`, and a one-sided term here is actively
harmful: "lower cLogP" rewards stripping carbon until the linker is a polar
worm, and "shorten the span" rewards breaking the molecule, because a PROTAC
whose linker cannot reach between two binding sites is not a PROTAC whatever
its descriptors say. So a threshold here measures *distance outside a band*
and drives that to zero.

**A target already met is not an achievement.** `moltrain`'s README records
the bug where an untouched caffeine molecule scored 0.708 because satisfied
constraints paid full progress. The same trap is wider here: a band-shaped
target is satisfied at the start far more often than a "do better than you
started" one, and with the two fragments frozen most descriptors barely move.
So a target that was *already inside its band at the start* is excluded from
the progress numerator entirely and converted to a hold: it earns nothing and
is charged only if the episode pushes it out. Credit is paid for fixing what
was wrong, never for leaving alone what was right.

**Admissibility is not quality.** Deleting a warhead improves SA, cLogP, TPSA,
rotatable bonds and molecular weight simultaneously. No descriptor catches it;
only matching the fragments back does. `protacbench/OPTIMISATION.md` calls this
"the failure that looks like success", and it is the reason the identity
checks are a floor on the reward rather than another weighted term -- a term
can be outweighed, a floor cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .properties import LinkerProperties


@dataclass
class ThresholdOutcome:
    """One threshold, and why it scored what it did."""

    name: str
    kind: str  # "goal" | "hold" | "constraint"
    start: float | None
    final: float | None
    start_distance: float
    final_distance: float
    progress: float  # 0-1, and only meaningful for kind == "goal"
    met: bool
    charge: float = 0.0  # tolerance units past the allowance, for hold/constraint
    note: str = ""


@dataclass
class RewardBreakdown:
    """The number, and every term that produced it."""

    reward: float
    thresholds: list[ThresholdOutcome] = field(default_factory=list)
    terms: dict[str, float] = field(default_factory=dict)
    floor_reason: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def floored(self) -> bool:
        return bool(self.floor_reason)


def _distance(value: float, spec: dict[str, Any]) -> float:
    """How far outside its target a value sits. Zero means satisfied.

    Every threshold shape reduces to this one number, which is what lets a
    band and a one-sided bound share the same progress arithmetic.
    """
    low = spec.get("low")
    high = spec.get("high")
    if low is not None and value < low:
        return float(low) - float(value)
    if high is not None and value > high:
        return float(value) - float(high)
    return 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_thresholds(
    start: LinkerProperties,
    final: LinkerProperties,
    specs: dict[str, dict[str, Any]],
) -> list[ThresholdOutcome]:
    """Judge each threshold, splitting goals from holds by the START value.

    A threshold is a *goal* when the starting molecule was outside its band --
    that is the thing the episode was set to fix. It is a *hold* when the
    starting molecule was already inside: nothing to earn, something to lose.
    """
    outcomes: list[ThresholdOutcome] = []

    for name, spec in specs.items():
        if not spec.get("enabled", True):
            continue

        start_value = getattr(start, name, None)
        final_value = getattr(final, name, None)

        if start_value is None or final_value is None:
            outcomes.append(
                ThresholdOutcome(
                    name=name, kind="goal", start=start_value, final=final_value,
                    start_distance=0.0, final_distance=0.0, progress=0.0, met=False,
                    note="not measured on one side; skipped",
                )
            )
            continue

        start_value = float(start_value)
        final_value = float(final_value)
        start_d = _distance(start_value, spec)
        final_d = _distance(final_value, spec)
        tolerance = float(spec.get("tolerance", 0.0) or 0.0)

        if start_d > 0.0:
            # Live goal: pay for the distance actually closed.
            progress = _clamp01((start_d - final_d) / start_d)
            outcomes.append(
                ThresholdOutcome(
                    name=name, kind="goal", start=start_value, final=final_value,
                    start_distance=start_d, final_distance=final_d,
                    progress=progress, met=final_d <= 0.0,
                    note=f"was {start_d:.3g} outside, now {final_d:.3g}",
                )
            )
            continue

        # Already satisfied at the start. Earns nothing; charged if broken.
        charge = 0.0
        if final_d > tolerance:
            charge = (final_d - tolerance) / tolerance if tolerance > 0 else final_d
        outcomes.append(
            ThresholdOutcome(
                name=name, kind="hold", start=start_value, final=final_value,
                start_distance=0.0, final_distance=final_d,
                progress=0.0, met=final_d <= tolerance, charge=charge,
                note="already satisfied at the start" if final_d <= tolerance
                else f"pushed {final_d:.3g} outside a band it started inside",
            )
        )

    return outcomes


def score_regressions(
    start: LinkerProperties,
    final: LinkerProperties,
    tolerances: dict[str, float],
) -> list[ThresholdOutcome]:
    """Charge a descriptor that got meaningfully worse, band or no band.

    Separate from the thresholds because this is drift, not distance: a
    molecule can stay inside every band and still have had its cLogP walked
    1.1 units in the wrong direction, which is what a repair that deletes a
    heteroatom to make a reactive group disappear looks like. The directions
    are the ones `protacbench.grade` already treats as regressions.
    """
    worse_when_higher = {"clogp", "sa_score", "rotatable_bonds"}
    outcomes: list[ThresholdOutcome] = []

    for name, tol in tolerances.items():
        start_value = getattr(start, name, None)
        final_value = getattr(final, name, None)
        if start_value is None or final_value is None:
            continue
        start_value, final_value, tol = float(start_value), float(final_value), float(tol)

        drift = final_value - start_value
        if name not in worse_when_higher:
            drift = -drift  # tpsa and friends regress by falling

        charge = 0.0
        if drift > tol:
            charge = (drift - tol) / tol if tol > 0 else drift
        outcomes.append(
            ThresholdOutcome(
                name=f"{name}_regression", kind="constraint",
                start=start_value, final=final_value,
                start_distance=0.0, final_distance=max(0.0, drift),
                progress=0.0, met=charge == 0.0, charge=charge,
                note=f"drifted {drift:+.3g} against a {tol:g} allowance",
            )
        )
    return outcomes


def admissibility_floor(
    start: LinkerProperties,
    final: LinkerProperties,
    reference_smiles: str | None = None,
) -> tuple[str, list[str]]:
    """The checks that make an answer inadmissible rather than merely poor.

    Returns the floor reason (empty when admissible) and any flags that are
    worth recording but do not themselves sink the episode.

    These are `protacbench/OPTIMISATION.md`'s acceptance tests 1-4 and 8, in
    its own order of importance. Test 2 -- both fragments present and disjoint
    -- is the one that matters: it is an identity check, not a quality
    gradient, and everything else in this module assumes it has passed.
    """
    flags: list[str] = []

    if not final.measurable:
        return (final.error or "the molecule could not be measured"), flags
    if not final.fragments_intact:
        return "a fragment is missing or altered: this is a different compound", flags
    if final.n_attachments != 2:
        return (
            f"the linker attaches at {final.n_attachments} atoms, not 2"
        ), flags
    if final.reactive_groups:
        return (
            "the linker carries a reactive group: "
            + ", ".join(sorted(set(final.reactive_groups)))
        ), flags

    # Acceptance test 8. `moltrain`'s novelty weight (0.1) is far too weak to
    # be this gate: an episode that submits its input untouched would still
    # collect every "already satisfied" hold and come out ahead of one that
    # tried something and half-succeeded.
    if final.canonical and start.canonical and final.canonical == start.canonical:
        return "the submission is the input unchanged", flags

    # Acceptance test 9. Not a floor: recalling the published molecule is a
    # real answer to the chemistry and a worthless answer to the task, and the
    # two readings are worth keeping apart in the logs rather than collapsing.
    # 94 of 1024 generated samples already recover their reference exactly, so
    # a 30B policy has plausibly memorised some of them.
    if reference_smiles and final.canonical:
        if final.canonical == _canonical(reference_smiles):
            flags.append("recalled_reference")

    if final.liability_groups:
        flags.append("liability:" + ",".join(sorted(set(final.liability_groups))))

    return "", flags


def _canonical(smiles: str) -> str | None:
    try:
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:  # noqa: BLE001
        return None


def compute_reward(
    cfg: dict[str, Any],
    start: LinkerProperties,
    final: LinkerProperties | None,
    *,
    protocol: Any = None,
    calls: Any = None,
    reference_smiles: str | None = None,
    synthesis_score: float | None = None,
) -> RewardBreakdown:
    """The episode's number, and every term behind it.

    The ordering matters. Admissibility is settled first and short-circuits:
    an inadmissible answer does not get its descriptors scored, because the
    descriptors of a molecule with a deleted warhead are excellent and
    entirely beside the point.

    Protocol terms survive the floor, deliberately. An episode that worked
    honestly and still failed should score above one that lied and failed, or
    the only lesson available from a hard task is that effort is not worth
    spending. The floor caps the total, so protocol can soften a failure but
    never turn it into a success.
    """
    reward_cfg = cfg["reward"]
    weights = reward_cfg["weights"]
    cap = float(reward_cfg.get("violation_cap", 3.0))
    breakdown = RewardBreakdown(reward=0.0)

    protocol_total = 0.0
    if protocol is not None:
        protocol_total = (
            float(weights.get("grounding", 0.0)) * float(protocol.grounding)
            + float(weights.get("discipline", 0.0)) * float(protocol.discipline)
            + float(weights.get("verification", 0.0))
            * (1.0 if protocol.verified_before_submit else 0.0)
        )
        breakdown.terms["grounding"] = round(
            float(weights.get("grounding", 0.0)) * float(protocol.grounding), 4
        )
        breakdown.terms["discipline"] = round(
            float(weights.get("discipline", 0.0)) * float(protocol.discipline), 4
        )
        breakdown.terms["verification"] = round(
            float(weights.get("verification", 0.0))
            * (1.0 if protocol.verified_before_submit else 0.0),
            4,
        )

    # ---- no answer at all
    if final is None:
        floor = float(reward_cfg["no_submission_reward"])
        breakdown.floor_reason = "the episode ended without submitting a molecule"
        breakdown.reward = min(floor + protocol_total, floor + abs(floor))
        breakdown.terms["floor"] = floor
        return breakdown

    # ---- admissible?
    reason, flags = admissibility_floor(start, final, reference_smiles)
    breakdown.flags.extend(flags)
    if reason:
        floor = float(reward_cfg["inadmissible_reward"])
        breakdown.floor_reason = reason
        # Protocol can lift an honest failure off the floor, but never past 0:
        # an inadmissible answer is still not an answer.
        breakdown.reward = min(0.0, floor + max(0.0, protocol_total))
        breakdown.terms["floor"] = floor
        return breakdown

    # ---- chemistry
    specs = reward_cfg["thresholds"]
    outcomes = score_thresholds(start, final, specs)
    outcomes += score_regressions(
        start, final, reward_cfg.get("regression_tolerance", {})
    )
    breakdown.thresholds = outcomes

    goals = [o for o in outcomes if o.kind == "goal"]
    if goals:
        progress = sum(o.progress for o in goals) / len(goals)
    else:
        # Nothing was wrong at the start. These are the qed-only control
        # episodes: there is no progress to make, so the chemistry term is
        # zero and the episode is scored on not breaking anything and on how
        # it conducted itself. Paying full marks here is the inaction bug.
        progress = 0.0
    breakdown.terms["thresholds"] = round(
        float(weights["thresholds"]) * progress, 4
    )

    if goals and all(o.met for o in goals):
        breakdown.terms["all_met_bonus"] = float(weights["all_met_bonus"])

    hold_charge = sum(min(o.charge, cap) for o in outcomes if o.kind == "hold")
    if hold_charge:
        breakdown.terms["hold_violation"] = round(
            -float(weights["hold_violation"]) * hold_charge, 4
        )
    regression_charge = sum(
        min(o.charge, cap) for o in outcomes if o.kind == "constraint"
    )
    if regression_charge:
        breakdown.terms["regression"] = round(
            -float(weights["regression"]) * regression_charge, 4
        )

    if synthesis_score is not None and weights.get("synthesis_score"):
        breakdown.terms["synthesis_score"] = round(
            float(weights["synthesis_score"]) * (float(synthesis_score) / 100.0), 4
        )

    # ---- cost of working
    if calls is not None:
        n_calls = len(calls)
        free = int(reward_cfg.get("turn_penalty_free_calls", 0))
        charged = max(0, n_calls - free)
        if charged:
            breakdown.terms["turn_penalty"] = round(
                -float(weights["turn_penalty"]) * charged, 4
            )
        n_ok = sum(1 for c in calls if getattr(c, "ok", False))
        measured_anything = any(
            getattr(c, "name", "") not in {"submit_molecule"} for c in calls
        )
        # Withheld unless the policy actually used a tool: a lone
        # submit_molecule has a 100% success rate and must not collect this.
        if n_calls and measured_anything:
            breakdown.terms["tool_success"] = round(
                float(weights["tool_success"]) * (n_ok / n_calls), 4
            )

    breakdown.terms.update(
        {k: v for k, v in breakdown.terms.items()}
    )
    breakdown.reward = round(sum(breakdown.terms.values()), 4)
    return breakdown
