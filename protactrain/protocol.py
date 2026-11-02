"""Scoring *how* the policy worked, separately from what it produced.

An outcome-only reward teaches tool use only by accident. If a molecule can
be improved by guessing, guessing scores exactly as well as measuring, and
GRPO will find that out -- so the model that comes out the other side is good
at proposing SMILES and no better at knowing when to call a tool. That is the
opposite of the objective here.

`protacbench` already built the instrument for this and already argues the
point: its `grade` scorecard splits **chemistry** (is the molecule good) from
**protocol** (did the reply say so truthfully, in the form it was asked for),
and the two fail independently on purpose. This module turns that second half
into reward terms.

Three things are scored, and the first is the one that matters:

**Grounding.** `grade.audit_claims` extracts every number a reply asserts and
labels it `ok` (matches the measurement), `echo` (matches the input, which is
quoting the task back, not lying), `false` (matches neither) or
`unmeasurable`. A `false` claim is a tool-using model stating a measurement it
never made, which is the single worst habit this training could instil and
the one an outcome reward is blind to -- the molecule is unaffected by what
the model said about it.

**Measurement discipline.** OPTIMISATION.md's standing instruction is: measure
the input, make one change, re-measure, verify before submitting. Each of
those is checkable from the call log, and each is rewarded on its own rather
than bundled, so the gradient can distinguish "never measured" from "measured
but never re-checked".

**Verification before submitting.** `check_protac_linker` is the tool that
decides acceptance. Calling it before `submit_molecule` is the difference
between an answer and a guess, and it is cheap, so it is rewarded directly.

Every term here is bounded and small relative to the chemistry terms. Protocol
is how the work is done; it is not the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

# Tools that return a measurement, as opposed to editing or submitting. A call
# to one of these is the policy finding something out.
MEASUREMENT_TOOLS = frozenset(
    {
        "validate_smiles",
        "get_molecule_metrics",
        "check_protac_linker",
        "generate_3d_conformer",
        "render_molecule_2d",
        "benchmark_transformation",
    }
)

# Tools that propose or commit a change.
EDIT_TOOLS = frozenset({"apply_smarts_reaction", "benchmark_transformation"})

ACCEPTANCE_TOOL = "check_protac_linker"
SUBMIT_TOOL = "submit_molecule"


@dataclass
class ProtocolReport:
    """How the episode was conducted, and the score that follows from it."""

    grounding: float = 0.0  # -1 .. 1
    discipline: float = 0.0  # 0 .. 1
    verified_before_submit: bool = False

    n_claims: int = 0
    n_false_claims: int = 0
    n_ok_claims: int = 0
    n_echo_claims: int = 0
    n_hedged_false: int = 0

    measured_before_first_edit: bool = False
    remeasured_after_edit: bool = False
    one_change_at_a_time: bool = False

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounding": self.grounding,
            "discipline": self.discipline,
            "verified_before_submit": self.verified_before_submit,
            "n_claims": self.n_claims,
            "n_false_claims": self.n_false_claims,
            "n_ok_claims": self.n_ok_claims,
            "n_echo_claims": self.n_echo_claims,
            "n_hedged_false": self.n_hedged_false,
            "measured_before_first_edit": self.measured_before_first_edit,
            "remeasured_after_edit": self.remeasured_after_edit,
            "one_change_at_a_time": self.one_change_at_a_time,
            "notes": self.notes,
        }


def score_grounding(
    assistant_text: str,
    measured: dict[str, Any],
    baseline: dict[str, Any],
    hedge_relief: float = 0.5,
) -> tuple[float, dict[str, int], list[str]]:
    """Reward numbers the tools actually returned; punish the ones they did not.

    Scored as a rate rather than a count, so a terse correct reply and a long
    correct one score the same and the policy is not pushed into padding its
    answer with numbers to farm the term.

    An `echo` is neither rewarded nor punished. Quoting the task back is
    normal and `grade.audit_claims` goes out of its way to tell it apart from
    a false claim; treating it as either would be wrong.

    A hedged false claim ("cLogP is roughly 5") is charged less than a bare
    one, because stating an estimate as an estimate is a different and far
    less damaging act than asserting an unmeasured number as fact.
    """
    from protacbench.grade import audit_claims

    try:
        claims = audit_claims(assistant_text, measured, baseline)
    except Exception as exc:  # noqa: BLE001 - never let an audit sink an episode
        return 0.0, {"claims": 0}, [f"claim audit failed: {type(exc).__name__}: {exc}"]

    counts = {"ok": 0, "echo": 0, "false": 0, "unmeasurable": 0, "hedged_false": 0}
    charge = 0.0
    credit = 0.0

    for claim in claims:
        # `grade.Claim` names this field `verdict`, not `status`. Reading the
        # wrong name does not raise -- getattr falls through to the default,
        # every claim counts as unmeasurable and grounding is silently zero
        # for every episode, which deletes the one term this package exists
        # for. Keep the name in step with grade.py:150.
        status = getattr(claim, "verdict", "unmeasurable")
        hedged = bool(getattr(claim, "hedged", False))
        counts[status] = counts.get(status, 0) + 1
        if status == "ok":
            credit += 1.0
        elif status == "false":
            if hedged:
                counts["hedged_false"] += 1
                charge += hedge_relief
            else:
                charge += 1.0

    judged = credit + charge
    if judged <= 0:
        # No auditable claim either way. Not virtue and not vice: a reply that
        # states no numbers has not lied, but it has not demonstrated
        # measurement either. Zero is the honest score.
        return 0.0, counts, ["no auditable numeric claims"]

    score = (credit - charge) / judged
    notes = []
    if counts["false"]:
        notes.append(f"{counts['false']} stated number(s) no tool returned")
    return max(-1.0, min(1.0, score)), counts, notes


def score_discipline(calls: Sequence[Any]) -> tuple[float, dict[str, bool], list[str]]:
    """Did the episode measure, change one thing, and re-measure?

    `calls` is the episode's call log in order; each entry needs a `.name` and
    a truthy `.ok`. Only successful calls count, because a failed call is an
    attempt to measure, not a measurement.

    The three parts are scored equally and independently. Bundling them into
    one pass/fail would make "measured once at the start and then edited
    blind" indistinguishable from "never measured at all", and those deserve
    different gradients.
    """
    names = [getattr(c, "name", "") for c in calls if getattr(c, "ok", False)]

    first_edit = next(
        (i for i, n in enumerate(names) if n in EDIT_TOOLS), None
    )
    measured_first = bool(
        first_edit is not None
        and any(n in MEASUREMENT_TOOLS for n in names[:first_edit])
    )
    # An episode with no edit at all has not demonstrated the habit; it has
    # merely avoided the chance to break it.
    if first_edit is None:
        measured_first = False

    remeasured = bool(
        first_edit is not None
        and any(n in MEASUREMENT_TOOLS for n in names[first_edit + 1:])
    )

    n_edits = sum(1 for n in names if n in EDIT_TOOLS)
    n_measures = sum(1 for n in names if n in MEASUREMENT_TOOLS)
    # "One change at a time" is not "only one change": it is that every change
    # was followed by a look. Measurements at least matching edits is the
    # cheapest faithful reading of that from a call log alone.
    one_at_a_time = bool(n_edits > 0 and n_measures >= n_edits)

    parts = {
        "measured_before_first_edit": measured_first,
        "remeasured_after_edit": remeasured,
        "one_change_at_a_time": one_at_a_time,
    }
    score = sum(1.0 for v in parts.values() if v) / len(parts)

    notes = []
    if first_edit is None and n_measures == 0:
        notes.append("no tool was called at all")
    elif not measured_first and first_edit is not None:
        notes.append("edited before measuring")
    return score, parts, notes


def verified_before_submit(calls: Sequence[Any]) -> bool:
    """Was the acceptance tool called before the answer was committed?"""
    seen_check = False
    for call in calls:
        name = getattr(call, "name", "")
        if name == SUBMIT_TOOL:
            return seen_check
        if name == ACCEPTANCE_TOOL and getattr(call, "ok", False):
            seen_check = True
    return False


def score_protocol(
    assistant_text: str,
    calls: Sequence[Any],
    measured: dict[str, Any],
    baseline: dict[str, Any],
) -> ProtocolReport:
    """The whole protocol half of the scorecard, in one call."""
    report = ProtocolReport()

    grounding, counts, notes = score_grounding(assistant_text, measured, baseline)
    report.grounding = grounding
    report.n_ok_claims = counts.get("ok", 0)
    report.n_echo_claims = counts.get("echo", 0)
    report.n_false_claims = counts.get("false", 0)
    report.n_hedged_false = counts.get("hedged_false", 0)
    report.n_claims = sum(
        counts.get(k, 0) for k in ("ok", "echo", "false", "unmeasurable")
    )
    report.notes.extend(notes)

    discipline, parts, disc_notes = score_discipline(calls)
    report.discipline = discipline
    report.measured_before_first_edit = parts["measured_before_first_edit"]
    report.remeasured_after_edit = parts["remeasured_after_edit"]
    report.one_change_at_a_time = parts["one_change_at_a_time"]
    report.notes.extend(disc_notes)

    report.verified_before_submit = verified_before_submit(calls)
    return report
