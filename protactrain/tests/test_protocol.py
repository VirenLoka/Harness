"""The protocol half of the scorecard: grounding, discipline, verification.

This is the half an outcome-only reward is blind to. The molecule is
unaffected by what the model said about it, so if these terms do not work the
training teaches tool use only by accident -- and the first test below is the
regression test for exactly that: `score_grounding` read `claim.status` while
`grade.Claim` names the field `verdict`, which made every claim unmeasurable
and the term identically zero for every episode without raising anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protactrain.properties import measure  # noqa: E402
from protactrain.protocol import (  # noqa: E402
    score_discipline,
    score_grounding,
    score_protocol,
    verified_before_submit,
)
from protactrain.tests import fixtures  # noqa: E402


class _Call:
    """What protocol.py reads off a call-log entry, and nothing else."""

    def __init__(self, name: str, ok: bool = True):
        self.name, self.ok = name, ok


@pytest.fixture(scope="module")
def measured():
    props = measure(fixtures.PROTAC, fixtures.FRAGMENTS)
    assert props.admissible, props.error
    return props.as_dict()


@pytest.fixture(scope="module")
def baseline():
    # A different molecule over the same fragments, so "matches the baseline"
    # and "matches the measurement" are genuinely distinguishable.
    return measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS).as_dict()


# -------------------------------------------------------------- grounding


def test_the_claim_field_is_verdict_not_status(measured, baseline):
    """A regression test with a name, because the bug was invisible.

    `getattr(claim, "status", "unmeasurable")` does not raise on a Claim --
    it falls through to the default, every claim counts as unmeasurable, and
    `score_grounding` returns exactly 0.0 forever. Grounding is weighted 0.5
    and is the only term that can go negative on its own, so the failure
    silently turns the whole reward back into an outcome-only one.
    """
    from protacbench.grade import Claim

    assert hasattr(Claim("span", "10", "ok"), "verdict")
    assert not hasattr(Claim("span", "10", "ok"), "status")

    # And the function must actually judge something on a real reply.
    score, counts, _notes = score_grounding(
        f"I measured a span of {measured['span']} bonds.", measured, baseline
    )
    assert counts["ok"] >= 1
    assert score > 0.0


def test_a_number_no_tool_returned_scores_below_zero(measured, baseline):
    """A tool-using model asserting an unmeasured number is the single worst
    habit this training could instil."""
    text = "The cLogP is 2.0 and SA is 1.1, so the molecule is clean."
    score, counts, notes = score_grounding(text, measured, baseline)
    assert counts["false"] >= 1
    assert score < 0.0
    assert any("no tool returned" in n for n in notes)


def test_quoting_the_input_is_an_echo_and_scores_neither_way(measured, baseline):
    """Quoting the task back is normal. `grade.audit_claims` goes out of its
    way to tell it from a false claim, and collapsing the two would punish
    the policy for restating the problem it was set."""
    text = f"The starting molecule had a span of {baseline['span']} bonds."
    score, counts, _notes = score_grounding(text, measured, baseline)
    assert counts["echo"] >= 1
    assert counts["false"] == 0
    assert score == 0.0  # no credit and no charge: nothing was judged


def test_a_reply_stating_no_numbers_scores_zero_not_a_bonus(measured, baseline):
    """Saying nothing has not lied, but it has not demonstrated measurement."""
    score, _counts, notes = score_grounding(
        "I replaced the alkyl chain with an ether.", measured, baseline
    )
    assert score == 0.0
    assert "no auditable numeric claims" in notes


def test_grounding_is_a_rate_so_padding_with_numbers_does_not_farm_it(
    measured, baseline
):
    one = score_grounding(f"span {measured['span']}", measured, baseline)[0]
    many = score_grounding(
        f"span {measured['span']}, cLogP {measured['clogp']:.2f}, "
        f"TPSA {measured['tpsa']:.1f}",
        measured,
        baseline,
    )[0]
    assert one == pytest.approx(many)


def test_a_hedged_false_claim_is_charged_less_than_a_bare_one(measured, baseline):
    """The relief is real, but only once something true is also on the table.

    `score_grounding` is a rate: `(credit - charge) / (credit + charge)`. The
    hedge halves the charge, and in a reply that got something right that
    shows up -- a bare false claim alongside a true one scores 0.0, a hedged
    one scores +0.33.

    In a reply where EVERY claim is false the halving cancels out of the
    ratio exactly and both score -1.0, which is the one case the relief was
    described for. That is pinned here rather than asserted away: it is a
    real limitation of scoring as a rate, and it is recorded in the package
    notes rather than silently patched, because changing it would move every
    reward number the expert calibration was set against.
    """
    span = measured["span"]
    bare_only = score_grounding("The cLogP is 2.0.", measured, baseline)[0]
    hedged_only = score_grounding("The cLogP is roughly 2.0.", measured, baseline)[0]
    assert bare_only == pytest.approx(-1.0)
    assert hedged_only == pytest.approx(-1.0)

    bare_mixed = score_grounding(
        f"The span is {span} bonds and the cLogP is 2.0.", measured, baseline
    )[0]
    hedged_mixed = score_grounding(
        f"The span is {span} bonds and the cLogP is roughly 2.0.",
        measured,
        baseline,
    )[0]
    assert hedged_mixed > bare_mixed


def test_the_hedge_is_detected_even_where_it_does_not_move_the_rate(
    measured, baseline
):
    """Counted either way, so the logs can show what the score cannot."""
    _score, bare, _ = score_grounding("The cLogP is 2.0.", measured, baseline)
    _score, hedged, _ = score_grounding(
        "The cLogP is roughly 2.0.", measured, baseline
    )
    assert bare["hedged_false"] == 0
    assert hedged["hedged_false"] == 1


def test_an_audit_failure_never_sinks_the_episode():
    """Measurement paths return an error; they do not raise."""
    score, _counts, notes = score_grounding("span 10", {"span": object()}, {})
    assert isinstance(score, float)
    assert isinstance(notes, list)


# ------------------------------------------------------------- discipline


def test_discipline_distinguishes_measure_then_edit_from_edit_then_measure():
    """Bundling these into one pass/fail would make 'measured once and then
    edited blind' indistinguishable from 'never measured'."""
    measured_first = score_discipline(
        [
            _Call("check_protac_linker"),
            _Call("benchmark_transformation"),
            _Call("get_molecule_metrics"),
        ]
    )
    edited_first = score_discipline(
        [
            _Call("benchmark_transformation"),
            _Call("check_protac_linker"),
        ]
    )
    assert measured_first[1]["measured_before_first_edit"] is True
    assert edited_first[1]["measured_before_first_edit"] is False
    assert measured_first[0] > edited_first[0]
    assert "edited before measuring" in edited_first[2]


def test_editing_without_looking_afterwards_loses_the_remeasure_part():
    score, parts, _notes = score_discipline(
        [_Call("check_protac_linker"), _Call("benchmark_transformation")]
    )
    assert parts["measured_before_first_edit"] is True
    assert parts["remeasured_after_edit"] is False
    assert score == pytest.approx(2 / 3)  # the one_change_at_a_time part holds


def test_an_episode_that_never_edited_has_not_demonstrated_the_habit():
    """Avoiding the chance to break a habit is not keeping it."""
    score, parts, _notes = score_discipline(
        [_Call("check_protac_linker"), _Call("get_molecule_metrics")]
    )
    assert score == 0.0
    assert not any(parts.values())


def test_no_calls_at_all_scores_zero_and_says_so():
    score, _parts, notes = score_discipline([])
    assert score == 0.0
    assert "no tool was called at all" in notes


def test_a_failed_call_is_an_attempt_to_measure_not_a_measurement():
    _score, parts, _notes = score_discipline(
        [
            _Call("check_protac_linker", ok=False),
            _Call("benchmark_transformation"),
            _Call("get_molecule_metrics"),
        ]
    )
    assert parts["measured_before_first_edit"] is False


# ----------------------------------------------------------- verification


def test_verification_requires_the_check_before_the_submit_not_after():
    assert verified_before_submit(
        [_Call("check_protac_linker"), _Call("submit_molecule")]
    )
    assert not verified_before_submit(
        [_Call("submit_molecule"), _Call("check_protac_linker")]
    )
    assert not verified_before_submit([_Call("submit_molecule")])
    assert not verified_before_submit(
        [_Call("check_protac_linker", ok=False), _Call("submit_molecule")]
    )


def test_score_protocol_reports_all_three_halves_together(measured, baseline):
    report = score_protocol(
        f"Measured the repair: span {measured['span']} bonds.",
        [
            _Call("check_protac_linker"),
            _Call("benchmark_transformation"),
            _Call("check_protac_linker"),
            _Call("submit_molecule"),
        ],
        measured,
        baseline,
    )
    assert report.grounding > 0
    assert report.discipline == pytest.approx(1.0)
    assert report.verified_before_submit is True
    assert report.n_claims >= 1
    assert report.as_dict()["grounding"] == report.grounding
