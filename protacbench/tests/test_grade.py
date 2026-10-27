"""Grader tests. No GPU, no model endpoint, no network.

Every case here is built from the two real replies the grader was written
against, because the failures worth catching are the ones a model actually
produces: a sound molecule in an unparseable reply, and a repair that fixed
the stated defect by deleting a heteroatom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protacbench import grade

pytest.importorskip("rdkit.Chem")


# PROTAC 2276 sample 8: the linker carries an N-alkyl aldimine.
TASK_2276 = {
    "task_id": "2276-8",
    "fragments": [
        "CN1CCN(c2ccc(-c3ccccc3)cc2NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)CC1",
        "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O",
    ],
    "protac_smiles": (
        "CN1CCN(c2ccc(-c3ccc(OCCNC(=O)C/C=N/CCCCCNc4cccc5c4C(=O)N(C4CCC(=O)"
        "NC4=O)C5=O)cc3)cc2NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)CC1"
    ),
    "mw": 925.97, "exact_mw": 925.3734, "sa_score": 4.3147, "clogp": 4.7043,
    "tpsa": 214.71, "qed": 0.0518, "rotatable_bonds": 18, "heavy_atoms": 67,
    "linker_heavy_atoms": 14, "span": 14,
}

# The correct repair: the imine is reduced, so the nitrogen stays.
REDUCED = (
    "CN1CCN(c2ccc(-c3ccc(OCCNC(=O)CCNCCCCCNc4cccc5c4C(=O)N(C4CCC(=O)NC4=O)"
    "C5=O)cc3)cc2NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)CC1"
)
# What the model did: the imine nitrogen became a carbon. Same heavy-atom
# count, one fewer hydrogen-bond acceptor, an octamethylene run in its place.
TRANSMUTED = (
    "CN1CCN(c2ccc(-c3ccc(OCCNC(=O)CCCCCCCCNc4cccc5c4C(=O)N(C4CCC(=O)NC4=O)"
    "C5=O)cc3)cc2NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)CC1"
)

# PROTAC 2992 sample 0: Michael acceptor, aminal ring, and span 20.
TASK_2992 = {
    "task_id": "2992-0",
    "fragments": [
        "CC1(N)CCN(c2cnc(-c3cccc(Cl)c3Cl)c(N)n2)CC1",
        "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
    ],
    "mw": 929.95, "exact_mw": 928.3917, "sa_score": 4.4179, "clogp": 5.3286,
    "tpsa": 203.27, "qed": 0.0689, "rotatable_bonds": 19, "heavy_atoms": 65,
    "linker_heavy_atoms": 23, "span": 20,
}

# All three defects fixed: saturated, piperazine, trimmed to span 17.
FIXED_2992 = (
    "CC1(N)CCN(c2cnc(-c3cccc(Cl)c3Cl)c(NCCCNC(=O)CCN3CCN(C(=O)CCCCc4ccc5c(c4)"
    "C(=O)N(C4CCC(=O)NC4=O)C5=O)CC3)n2)CC1"
)
# Michael acceptor and span fixed, aminal left in place -- the ring is still
# the 5-membered imidazolidine. Nothing on the server tests for it.
AMINAL_LEFT = (
    "CC1(N)CCN(c2cnc(-c3cccc(Cl)c3Cl)c(NCCCNC(=O)CCN3CCN(C(=O)CCCCc4ccc5c(c4)"
    "C(=O)N(C4CCC(=O)NC4=O)C5=O)C3)n2)CC1"
)


def status(report: grade.Report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


# ----------------------------------------------------------- finding the answer


def test_the_contract_line_is_preferred_over_the_reply_body():
    reply = f"Here is a draft:\n\n```\n{REDUCED}\n```\n\nREPAIRED: {TRANSMUTED}"
    answer, source = grade.extract_answer(reply)
    assert source == "contract_line"
    assert answer == TRANSMUTED


def test_a_molecule_is_recovered_when_the_contract_line_is_missing():
    """The failure that matters: good chemistry the harness cannot parse.

    Without the fallback this scores as a refusal, which is the same result
    as a model that gave up -- and the two call for opposite fixes.
    """
    answer, source = grade.extract_answer(f"Repaired:\n\n```\n{REDUCED}\n```")
    assert source == "fenced_fallback"
    assert answer == REDUCED


def test_prose_fragments_are_not_mistaken_for_an_answer():
    answer, source = grade.extract_answer(
        "I replaced the `C(=O)` with `CCO` and checked `c1ccccc1`."
    )
    assert (answer, source) == (None, "none")


def test_a_reply_with_no_molecule_fails_without_raising():
    report = grade.grade(TASK_2276, "I could not repair this one.")
    assert report.verdict == "fail"
    assert status(report, "answer_line") == grade.FAIL
    assert status(report, "parses") == grade.FAIL


# -------------------------------------------------------------- the regressions


def test_deleting_a_heteroatom_is_caught_as_a_regression():
    """The transmuted answer passes every absolute limit and is still wrong.

    cLogP rises 1.1 and TPSA falls 12 because a hydrogen-bond acceptor was
    removed. `repair.verify` cannot see this; it is the whole reason the
    grader measures against the baseline rather than against the thresholds.
    """
    report = grade.grade(TASK_2276, f"REPAIRED: {TRANSMUTED}")
    assert status(report, "within_limits") == grade.PASS
    assert status(report, "no_regression") == grade.FAIL
    assert status(report, "no_reactive_group") == grade.PASS
    assert not report.chemistry_ok


def test_the_reduced_imine_passes_every_chemistry_check():
    report = grade.grade(TASK_2276, f"REPAIRED: {REDUCED}")
    assert report.chemistry_ok, [
        (c.name, c.detail) for c in report.checks if c.status == grade.FAIL
    ]
    assert status(report, "length_held") == grade.PASS


def test_an_improvement_is_not_a_regression():
    report = grade.grade(TASK_2992, f"REPAIRED: {FIXED_2992}")
    assert status(report, "no_regression") == grade.PASS
    assert "improved" in next(
        c.detail for c in report.checks if c.name == "no_regression"
    )


# ------------------------------------------------------------- the liabilities


def test_a_liability_no_tool_tests_for_is_still_failed():
    """check_protac_linker calls this molecule ok. The grader must not."""
    report = grade.grade(TASK_2992, f"REPAIRED: {AMINAL_LEFT}")
    assert status(report, "no_reactive_group") == grade.PASS
    assert status(report, "no_unflagged_liability") == grade.FAIL
    assert "aminal" in next(
        c.detail for c in report.checks if c.name == "no_unflagged_liability"
    )


def test_a_greasy_alkyl_run_is_noted_but_does_not_fail():
    report = grade.grade(TASK_2276, f"REPAIRED: {TRANSMUTED}")
    assert any("CH2 run" in note for note in report.notes)


def test_losing_a_fragment_fails_before_anything_is_measured():
    report = grade.grade(TASK_2276, "REPAIRED: O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1")
    assert status(report, "parses") == grade.FAIL
    assert "fragment" in next(
        c.detail for c in report.checks if c.name == "parses"
    )


# --------------------------------------------------------------- the geometry


def test_a_span_at_the_limit_passes_but_says_so():
    report = grade.grade(TASK_2992, f"REPAIRED: {FIXED_2992}")
    detail = next(c.detail for c in report.checks if c.name == "span")
    assert status(report, "span") == grade.PASS
    assert "no margin" in detail


def test_length_is_not_held_against_a_task_that_asked_for_a_cut():
    """2992 starts at span 20. Holding its 23 heavy atoms would be the bug."""
    report = grade.grade(TASK_2992, f"REPAIRED: {FIXED_2992}")
    assert status(report, "length_held") == grade.SKIP


# ------------------------------------------------------------------ the claims


def test_a_number_that_matches_the_input_is_an_echo_not_a_lie():
    """Replies quote the task back. Scoring the quote as a false claim about
    the output would make the audit useless on every real reply."""
    report = grade.grade(TASK_2992, f"starting cLogP 5.33\n\nREPAIRED: {FIXED_2992}")
    echoes = [c for c in report.claims if c.verdict == "echo"]
    assert any(c.key == "clogp" for c in echoes)
    assert not [c for c in report.claims if c.verdict == "false"]


def test_a_wrong_span_claim_is_caught():
    reply = f"shortened to ~15-16 bonds, inside the window\n\nREPAIRED: {FIXED_2992}"
    report = grade.grade(TASK_2992, reply)
    false = [c for c in report.claims if c.verdict == "false"]
    assert [c.key for c in false] == ["span"]
    assert status(report, "claims_true") == grade.FAIL


def test_exact_mw_is_not_double_counted_as_mw():
    """"Exact MW 888.36" is one claim, and it is true. Matching it twice
    turns a correct reply into a liar."""
    report = grade.grade(TASK_2992, f"Exact MW 888.36\n\nREPAIRED: {FIXED_2992}")
    assert [c.key for c in report.claims] == ["exact_mw"]
    assert report.claims[0].verdict == "ok"


def test_linker_heavy_atoms_is_not_read_as_the_whole_molecule():
    report = grade.grade(TASK_2276, f"linker heavy atoms 14\n\nREPAIRED: {REDUCED}")
    assert [c.key for c in report.claims] == ["linker_heavy_atoms"]
    assert report.claims[0].verdict == "ok"


def test_a_hedged_number_fails_even_when_it_is_right():
    report = grade.grade(TASK_2276, f"span ~ 14 bonds\n\nREPAIRED: {REDUCED}")
    assert all(c.verdict == "ok" for c in report.claims)
    assert status(report, "claims_true") == grade.FAIL


def test_a_quoted_range_beside_a_hedge_is_not_itself_hedged():
    reply = f"span ~ 14 bonds, 8-17 bond window preserved\n\nREPAIRED: {REDUCED}"
    report = grade.grade(TASK_2276, reply)
    ranges = [c for c in report.claims if c.stated == "8-17"]
    assert ranges and not ranges[0].hedged


def test_reporting_only_what_improved_is_a_protocol_failure():
    reply = f"SA 4.11, exact MW 926.39.\n\nREPAIRED: {TRANSMUTED}"
    report = grade.grade(TASK_2276, reply)
    assert status(report, "reported_what_matters") == grade.FAIL
    assert "clogp" in report.omitted and "tpsa" in report.omitted


def test_tool_markers_in_the_reply_are_recognised():
    reply = (
        f"mcp__molbench__get_molecule_metrics said cLogP 4.22, TPSA 214.4, "
        f"SA 4.15, rotatable bonds 19, span 14.\n\nREPAIRED: {REDUCED}"
    )
    report = grade.grade(TASK_2276, reply)
    assert status(report, "tool_use") == grade.PASS
    assert status(report, "reported_what_matters") == grade.PASS


# ------------------------------------------------------------------ the verdict


def test_chemistry_and_protocol_are_scored_separately():
    """A sound molecule in an unparseable reply is not the same result as a
    bad molecule, and the verdict has to say which happened."""
    report = grade.grade(TASK_2992, f"```\n{FIXED_2992}\n```")
    assert report.chemistry_ok
    assert not report.protocol_ok
    assert report.verdict == "chemistry only"


def test_a_clean_run_passes_both():
    reply = (
        "mcp__molbench__check_protac_linker returned ok.\n"
        "cLogP 4.22, TPSA 214.4, SA 4.15, rotatable bonds 19, span 14 bonds.\n"
        f"REPAIRED: {REDUCED}"
    )
    report = grade.grade(TASK_2276, reply)
    assert report.verdict == "pass", [
        (c.name, c.detail) for c in report.checks if c.status == grade.FAIL
    ]


def test_the_report_survives_a_round_trip_to_json():
    import json

    report = grade.grade(TASK_2276, f"REPAIRED: {TRANSMUTED}")
    assert json.loads(json.dumps(report.as_dict()))["verdict"] == "fail"
    assert grade.format_report(report, verbose=True).startswith("2276-8: FAIL")
