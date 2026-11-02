"""The reward: band arithmetic, the goal/hold split, and the floors.

Every molecule is a real one from the corpus (see `fixtures.py`), measured by
`protacbench.grade.measure` through `properties.measure` -- the same call the
training loop makes. Hand-built `LinkerProperties` appear only where a test is
about the arithmetic itself and the chemistry is beside the point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protactrain.config import load_config  # noqa: E402
from protactrain.properties import LinkerProperties, measure  # noqa: E402
from protactrain.rewards import (  # noqa: E402
    _distance,
    admissibility_floor,
    compute_reward,
    score_regressions,
    score_thresholds,
)
from protactrain.tests import fixtures  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def start(cfg):
    """102-0 as measured. Admissible, with cLogP outside its band."""
    props = measure(fixtures.PROTAC, fixtures.FRAGMENTS)
    assert props.admissible, props.error
    return props


# ---------------------------------------------------------- band arithmetic


def test_distance_is_zero_inside_the_band():
    band = {"low": 3.5, "high": 5.6, "tolerance": 1.9}
    assert _distance(4.5, band) == 0.0
    assert _distance(3.5, band) == 0.0  # the edges are inside
    assert _distance(5.6, band) == 0.0


def test_distance_below_the_band_is_the_gap_to_its_floor():
    assert _distance(2.5, {"low": 3.5, "high": 5.6}) == pytest.approx(1.0)


def test_distance_above_the_band_is_the_gap_to_its_ceiling():
    assert _distance(6.6, {"low": 3.5, "high": 5.6}) == pytest.approx(1.0)


def test_a_one_sided_band_ignores_its_missing_side():
    """sa_score ships with `low: null` -- being unusually easy to make is not
    a fault, so there is nothing to charge below."""
    band = {"low": None, "high": 4.9, "tolerance": 1.1}
    assert _distance(1.0, band) == 0.0
    assert _distance(6.0, band) == pytest.approx(1.1)


# ------------------------------------------- the goal / hold split: inaction


def _props(**values) -> LinkerProperties:
    base = dict(
        smiles="X", measurable=True, fragments_intact=True, n_attachments=2
    )
    base.update(values)
    return LinkerProperties(**base)


SPAN_BAND = {"span": {"low": 8, "high": 13, "tolerance": 4}}


def test_a_target_already_inside_its_band_earns_nothing_the_inaction_bug():
    """THE INACTION BUG.

    moltrain's README records an untouched caffeine scoring 0.708 because
    satisfied constraints paid full progress. The trap is wider here: with the
    two fragments frozen most descriptors barely move, so a band-shaped target
    is satisfied at the start far more often than a beat-your-baseline one. If
    a satisfied target paid, the reward-maximising move would be to submit the
    input untouched -- and it would outscore an honest half-repair.
    """
    held = score_thresholds(_props(span=10), _props(span=10), SPAN_BAND)[0]
    assert held.kind == "hold"
    assert held.progress == 0.0
    assert held.charge == 0.0
    assert held.met is True


def test_a_hold_is_charged_only_once_it_is_pushed_past_its_tolerance():
    """Inside at the start, outside at the end, beyond the slack: charged."""
    inside_tolerance = score_thresholds(
        _props(span=10), _props(span=16), SPAN_BAND
    )[0]
    assert inside_tolerance.kind == "hold"
    assert inside_tolerance.charge == 0.0  # 3 past the band, 4 allowed
    assert inside_tolerance.met is True

    pushed_out = score_thresholds(_props(span=10), _props(span=20), SPAN_BAND)[0]
    assert pushed_out.kind == "hold"
    assert pushed_out.charge == pytest.approx((7 - 4) / 4)
    assert pushed_out.met is False
    assert "started inside" in pushed_out.note


def test_a_target_outside_its_band_at_the_start_is_a_goal_paying_progress():
    partly = score_thresholds(_props(span=4), _props(span=6), SPAN_BAND)[0]
    assert partly.kind == "goal"
    assert partly.progress == pytest.approx(0.5)  # 4 of the 4-bond gap closed
    assert partly.met is False

    fixed = score_thresholds(_props(span=4), _props(span=9), SPAN_BAND)[0]
    assert fixed.kind == "goal"
    assert fixed.progress == pytest.approx(1.0)
    assert fixed.met is True


def test_a_goal_driven_further_out_pays_zero_rather_than_going_negative():
    """The charge for making things worse is `score_regressions`' job; a goal
    that got worse simply earns nothing, so the two are not double-counted."""
    worse = score_thresholds(_props(span=6), _props(span=3), SPAN_BAND)[0]
    assert worse.kind == "goal"
    assert worse.progress == 0.0


def test_a_target_measured_on_only_one_side_is_skipped_not_scored_as_zero():
    """198 corpus rows have no measured baseline. Scoring a missing start as
    zero would pay the policy for inventing progress from nothing."""
    outcome = score_thresholds(_props(span=None), _props(span=10), SPAN_BAND)[0]
    assert outcome.progress == 0.0
    assert "skipped" in outcome.note


def test_real_controls_produce_holds_and_the_chemistry_term_stays_zero(cfg, start):
    """102-0 is one of the 952 qed-only controls.

    Its span and linker_heavy_atoms are inside their bands, so leaving them
    alone must earn nothing. cLogP at 6.67 is genuinely outside, so the task
    is not entirely empty -- which is what makes it a usable control rather
    than a no-op.
    """
    outcomes = score_thresholds(start, start, cfg["reward"]["thresholds"])
    by_name = {o.name: o for o in outcomes}
    assert by_name["span"].kind == "hold"
    assert by_name["linker_heavy_atoms"].kind == "hold"
    assert by_name["clogp"].kind == "goal"
    assert all(o.progress == 0.0 for o in outcomes)


# ------------------------------------------------------------- regressions


def test_a_descriptor_walked_the_wrong_way_inside_its_band_is_still_charged():
    """Staying inside every band is not the same as not getting worse."""
    charged = score_regressions(
        _props(clogp=4.0), _props(clogp=5.0), {"clogp": 0.15}
    )[0]
    assert charged.kind == "constraint"
    assert charged.charge == pytest.approx((1.0 - 0.15) / 0.15)


def test_tpsa_regresses_by_falling_not_by_rising():
    assert score_regressions(
        _props(tpsa=200.0), _props(tpsa=150.0), {"tpsa": 2.0}
    )[0].charge > 0
    assert score_regressions(
        _props(tpsa=200.0), _props(tpsa=250.0), {"tpsa": 2.0}
    )[0].charge == 0.0


# ----------------------------------------------------- the admissibility floors


def test_floor_a_deleted_fragment_is_a_different_compound(cfg, start):
    """The failure that looks like success: deleting the warhead improves SA,
    cLogP, TPSA, rotatable bonds and MW at once. No descriptor catches it."""
    final = measure(fixtures.PROTAC_FRAGMENT_DELETED, fixtures.FRAGMENTS)
    assert not final.fragments_intact

    reason, _flags = admissibility_floor(start, final)
    assert reason
    breakdown = compute_reward(cfg, start, final)
    assert breakdown.floored
    assert breakdown.reward <= float(cfg["reward"]["inadmissible_reward"])
    # And it is a floor, not a term: no chemistry was scored at all.
    assert breakdown.thresholds == []


def test_floor_three_attachments_is_not_a_linker(cfg):
    """1112-6, a real `linker_not_bridging` row. It parses, it is one piece
    and both fragments match -- only counting the attachments rejects it."""
    props = measure(
        fixtures.PROTAC_THREE_ATTACHMENTS, fixtures.FRAGMENTS_1112
    )
    assert props.measurable and props.fragments_intact
    assert props.n_attachments == 3
    assert not props.admissible

    reason, _flags = admissibility_floor(props, props)
    assert "3 atoms, not 2" in reason


def test_floor_a_reactive_group_in_the_linker(cfg, start):
    """An aldehyde spliced into 102-0's linker, fragments and span untouched."""
    final = measure(fixtures.PROTAC_REACTIVE, fixtures.FRAGMENTS)
    assert final.measurable and final.fragments_intact
    assert final.n_attachments == 2
    assert "aldehyde" in final.reactive_groups
    assert not final.admissible

    reason, _flags = admissibility_floor(start, final)
    assert "reactive group" in reason and "aldehyde" in reason
    assert compute_reward(cfg, start, final).floored


def test_floor_submitting_the_input_unchanged(cfg, start):
    """Acceptance test 8, and the floor the whole band design depends on.

    Without it, an episode that changed nothing would collect every satisfied
    hold and finish above one that tried something and half-succeeded.
    """
    reason, _flags = admissibility_floor(start, start)
    assert reason == "the submission is the input unchanged"
    breakdown = compute_reward(cfg, start, start)
    assert breakdown.floored
    assert breakdown.reward < 0


def test_a_genuine_repair_of_the_same_molecule_is_not_floored(cfg, start):
    """The control case for the four floors above: swapping 102-0's diamide
    linker for 102-1's PEG3 keeps both fragments and clears every floor."""
    final = measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS)
    reason, _flags = admissibility_floor(start, final)
    assert reason == ""
    breakdown = compute_reward(cfg, start, final)
    assert not breakdown.floored
    assert breakdown.thresholds


def test_recalling_the_reference_is_a_flag_not_a_floor(cfg, start):
    """A memorised published molecule answers the chemistry and not the task.
    Collapsing the two readings into one number loses that distinction."""
    final = measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS)
    reason, flags = admissibility_floor(
        start, final, reference_smiles=fixtures.PROTAC_PEG
    )
    assert reason == ""
    assert "recalled_reference" in flags


def test_no_submission_floors_the_episode(cfg, start):
    breakdown = compute_reward(cfg, start, None)
    assert breakdown.floored
    assert "without submitting" in breakdown.floor_reason


# ---------------------------------------------------- protocol against the floor


class _Call:
    def __init__(self, name: str, ok: bool = True):
        self.name, self.ok = name, ok


class _Protocol:
    def __init__(self, grounding: float, discipline: float, verified: bool):
        self.grounding = grounding
        self.discipline = discipline
        self.verified_before_submit = verified


def test_an_honest_failure_outranks_a_dishonest_one_but_neither_passes(cfg, start):
    """Protocol survives the floor on purpose.

    If an inadmissible answer scored the same however it was reached, the only
    lesson available from a hard task would be that effort is not worth
    spending. The cap is 0: it can soften a failure, never pass one.
    """
    final = measure(fixtures.PROTAC_FRAGMENT_DELETED, fixtures.FRAGMENTS)
    honest = compute_reward(
        cfg, start, final, protocol=_Protocol(1.0, 1.0, True)
    )
    dishonest = compute_reward(
        cfg, start, final, protocol=_Protocol(-1.0, 0.0, False)
    )
    assert honest.reward > dishonest.reward
    assert honest.reward <= 0.0


def test_turn_penalty_is_free_up_to_the_allowance(cfg, start):
    """Measuring is the task, not waste. A flat per-call charge made
    moltrain's expert's honest calls cost more than the progress they bought,
    so the reward's optimum there was to submit blind."""
    final = measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS)
    free = int(cfg["reward"]["turn_penalty_free_calls"])
    calls = [_Call("get_molecule_metrics") for _ in range(free)]
    assert "turn_penalty" not in compute_reward(cfg, start, final, calls=calls).terms

    charged = compute_reward(cfg, start, final, calls=calls + [_Call("validate_smiles")])
    assert charged.terms["turn_penalty"] < 0


def test_a_lone_submit_call_collects_no_tool_success_credit(cfg, start):
    """A single submit_molecule has a 100% success rate and has measured
    nothing; paying it the tool-success term would reward guessing."""
    final = measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS)
    lone = compute_reward(cfg, start, final, calls=[_Call("submit_molecule")])
    assert "tool_success" not in lone.terms

    worked = compute_reward(
        cfg, start, final, calls=[_Call("check_protac_linker"), _Call("submit_molecule")]
    )
    assert worked.terms["tool_success"] > 0
