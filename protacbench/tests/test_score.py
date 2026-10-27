"""Synthesis-score tests. No GPU, no Open Babel, no model pass.

These run on SMILES directly rather than through the XYZ path, because the
score is a property of a parsed molecule and nothing about it depends on how
that molecule reached RDKit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protacbench import score as S
from protacbench.assess import molecule_properties
from protacbench.tests import fixtures

Chem = pytest.importorskip("rdkit.Chem")


def scored(smiles: str, linker_atoms: int, span: int | None, codes=()) -> S.SynthesisScore:
    """Score a SMILES, treating its first `linker_atoms` atoms as the linker."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    indices = list(range(min(linker_atoms, mol.GetNumHeavyAtoms())))
    return S.score_sample(
        mol,
        molecule_properties(mol),
        {"path_length_bonds": span},
        indices,
        list(codes),
    )


# ------------------------------------------------------------------ the motifs


def test_every_familiar_motif_is_valid_smarts():
    """A broken pattern here silently stops scoring that chemistry, and the
    score would drop for molecules that are actually fine."""
    broken = [
        name for name, smarts in S.FAMILIAR_MOTIFS if Chem.MolFromSmarts(smarts) is None
    ]
    assert broken == []


def test_peg_linker_is_fully_covered():
    """The PEG3 bridge in the fixture is exactly the chemistry the motif list
    exists to recognise."""
    smiles, linker = fixtures.protac_like()
    mol = Chem.MolFromSmiles(smiles)
    coverage, matched = S.motif_coverage(mol, linker)
    assert coverage == 1.0
    assert "peg_ether" in matched


def test_coverage_is_fractional_not_a_match_count():
    """One amide in an otherwise exotic linker must not read as familiar."""
    # A cyclopropane-rich linker with a single amide at one end.
    mol = Chem.MolFromSmiles("O=C(NC1CC1C1CC1C1CC1)c1ccccc1")
    linker = list(range(2, 12))
    coverage, matched = S.motif_coverage(mol, linker)
    assert "amide" in matched
    assert coverage < 0.6


def test_empty_linker_scores_zero_coverage():
    mol = Chem.MolFromSmiles("CCO")
    assert S.motif_coverage(mol, []) == (0.0, [])


# ----------------------------------------------------------------- the scoring


def test_a_broken_molecule_is_not_scored():
    """None, not zero. Zero is a position in the ranking and a molecule that
    did not parse does not have one."""
    result = S.score_sample(None, {}, {}, [], [])
    assert result.total is None
    assert "could not be read" in result.reason


def test_clean_peg_protac_scores_well():
    smiles, linker = fixtures.protac_like()
    result = scored(smiles, len(linker), span=11)
    assert result.total is not None
    assert result.total >= 80.0


def test_liabilities_deduct():
    """The same molecule scores lower once the triage reports an alert."""
    smiles, linker = fixtures.protac_like()
    clean = scored(smiles, len(linker), span=11)
    flagged = scored(smiles, len(linker), span=11, codes=["pains_alert"])
    assert flagged.total < clean.total


def test_a_short_span_costs_the_exit_vector_term():
    """Three bonds cannot bridge two binding sites, whatever else is true."""
    smiles, linker = fixtures.protac_like()
    result = scored(smiles, len(linker), span=3)
    term = next(t for t in result.terms if t.name == "exit_vector_span")
    assert term.score < 0.5


def test_an_unmeasurable_span_is_not_treated_as_good():
    smiles, linker = fixtures.protac_like()
    result = scored(smiles, len(linker), span=None)
    term = next(t for t in result.terms if t.name == "exit_vector_span")
    assert term.score < 0.5
    assert "could not be measured" in term.note


def test_missing_descriptors_score_neutral_rather_than_failing():
    """A descriptor that could not be computed must not read as a zero."""
    mol = Chem.MolFromSmiles("CCOCC")
    result = S.score_sample(mol, {}, {"path_length_bonds": 9}, [0, 1, 2, 3, 4], [])
    for name in ("synthesisability", "polarity", "flexibility"):
        term = next(t for t in result.terms if t.name == name)
        assert term.score == 0.5
        assert "neutral" in term.note


def test_terms_cover_every_weight():
    smiles, linker = fixtures.protac_like()
    result = scored(smiles, len(linker), span=11)
    assert {t.name for t in result.terms} == set(S.WEIGHTS)


def test_score_is_bounded():
    for smiles, n, span in (
        ("CCCCCCCCCCCCCCCCCCCCNC(=O)c1ccccc1", 20, 21),
        ("O=C(NC1CC1C1CC1CNC(=O)c1ccccc1)c1ccccc1", 12, 8),
        ("CO", 2, 1),
    ):
        result = scored(smiles, n, span)
        assert 0.0 <= result.total <= 100.0


# ----------------------------------------------------------------- the ranking


def test_rank_puts_verdict_before_score():
    """A `poor` molecule never outranks an `ok` one, however easy it is to
    make. The score says how hard it is to build, not whether it is a PROTAC."""
    rows = [
        {"verdict": "poor", "score_total": 95.0},
        {"verdict": "ok", "score_total": 60.0},
    ]
    ordered = S.rank(rows)
    assert ordered[0]["verdict"] == "ok"
    assert ordered[0]["rank"] == 1
    assert ordered[1]["rank"] == 2


def test_rank_orders_by_score_inside_a_verdict():
    rows = [
        {"verdict": "ok", "score_total": 60.0},
        {"verdict": "ok", "score_total": 90.0},
        {"verdict": "ok", "score_total": 75.0},
    ]
    ordered = S.rank(rows)
    assert [r["score_total"] for r in ordered] == [90.0, 75.0, 60.0]


def test_rank_keeps_unscored_rows_without_ranking_them():
    rows = [
        {"verdict": "broken", "score_total": None},
        {"verdict": "ok", "score_total": 70.0},
    ]
    ordered = S.rank(rows)
    assert len(ordered) == 2
    assert ordered[0]["rank"] == 1
    assert ordered[-1]["rank"] is None


def test_rank_is_stable_for_an_empty_list():
    assert S.rank([]) == []


# ------------------------------------------------------------------ the reason


def test_reason_does_not_call_a_high_score_held_back():
    """Wording is tiered: "held back by" on a 97 reads as a problem where
    there is none."""
    result = S.SynthesisScore(
        total=97.0,
        terms=[S.Term("polarity", 0.7, 10.0, "cLogP 0.3")],
    )
    assert "held back" not in S.build_reason(result)


def test_reason_names_the_weak_term_on_a_low_score():
    result = S.SynthesisScore(
        total=40.0,
        terms=[S.Term("motif_familiarity", 0.1, 25.0, "10% routine")],
    )
    assert "motif_familiarity" in S.build_reason(result)


# ------------------------------------------------------- reactive linkers


def test_a_free_aldehyde_in_the_linker_zeroes_liabilities():
    """From a real run: `NCCCCCCCC=O` was the top-scoring molecule of 1024,
    at 95/100. It is a free aldehyde that would acylate protein lysines.
    Nothing else measured can see it -- SA rewards it for being small, QED is
    indifferent, and the motif matcher reads it as an alkyl chain."""
    result = scored("O=C(NCCCCCCCC=O)c1ccccc1", 10, span=9,
                    codes=["reactive_linker_group"])
    term = next(t for t in result.terms if t.name == "liabilities")
    assert term.score == 0.0
    assert "reactive" in term.note


def test_liable_groups_cost_without_disqualifying():
    """A click handle or a solubilising amine is a real liability and can
    also be deliberate, so it is charged for rather than rejected."""
    mol = Chem.MolFromSmiles("O=C(NCCC#CCCN)c1ccccc1")
    props = molecule_properties(mol)
    plain = S.score_sample(mol, props, {"path_length_bonds": 9}, list(range(8)), [])
    flagged = S.score_sample(
        mol, {**props, "liable_groups": "alkyne;primary_amine"},
        {"path_length_bonds": 9}, list(range(8)), [],
    )
    assert flagged.total < plain.total


def test_groups_are_only_flagged_inside_the_linker():
    """A warhead is allowed to contain whatever it contains -- its structure
    is the reason the molecule exists, and flagging it would reject samples
    for something the generator never did."""
    from protacbench.assess import DISQUALIFYING_GROUPS, groups_in_linker

    mol = Chem.MolFromSmiles("O=Cc1ccccc1CCOCC")   # aldehyde on the aryl end
    aryl = list(range(0, 8))
    chain = list(range(8, mol.GetNumHeavyAtoms()))
    assert "aldehyde" in groups_in_linker(mol, aryl, DISQUALIFYING_GROUPS)
    assert groups_in_linker(mol, chain, DISQUALIFYING_GROUPS) == []


def test_ordinary_protac_linkers_are_not_flagged():
    """PEG, amide, piperazine and plain alkyl must all survive, or the gate
    rejects the chemistry it exists to favour."""
    from protacbench.assess import DISQUALIFYING_GROUPS, LIABLE_GROUPS, groups_in_linker

    for smiles in ("CCOCCOCCOCC", "CC(=O)NCCOCC", "C1CN(CCN1)CCOCC", "CCCCCC"):
        mol = Chem.MolFromSmiles(smiles)
        every = list(range(mol.GetNumHeavyAtoms()))
        assert groups_in_linker(mol, every, DISQUALIFYING_GROUPS) == [], smiles
        assert groups_in_linker(mol, every, LIABLE_GROUPS) == [], smiles


def test_qed_is_not_gated_by_default():
    """34% of the 128 real published PROTACs this test set is built from fall
    below the old 0.05 threshold. A gate there rejects a third of the ground
    truth, because QED scores oral-small-molecule likeness and a PROTAC is
    ~940 Da and bifunctional by construction."""
    from protacbench.config import load_config

    assert load_config()["thresholds"]["min_qed"] == 0.0


def test_the_prompt_limits_are_met_by_real_protacs():
    """The earlier prompt set SA <= 4.9, cLogP 3.5-5.6 and TPSA >= 170 as
    simultaneous targets. Only 22% of the 128 published PROTACs it cited as
    its source satisfy all three at once, so it was asking for molecules
    unlike the ground truth. Each bound here was chosen so the conjunction
    holds for most real PROTACs."""
    from protacbench.prompts import SYSTEM_PROMPT

    for text in ("SA score at or below 6.0", "cLogP at or below 7.5",
                 "TPSA at or above 120", "rotatable bonds at or below 30"):
        assert text in SYSTEM_PROMPT, text
    # and the old, unsatisfiable conjunction must not come back
    assert "cLogP between 3.5 and 5.6" not in SYSTEM_PROMPT
    assert "SA score at or below 4.9" not in SYSTEM_PROMPT


def test_the_prompt_only_names_tools_that_are_mounted():
    """`benchmark_transformation` lives in the TypeScript plugin, and
    harness.build_patch mounts only the Python MCP server. Telling the agent
    to prefer a tool that is not there wastes a turn on every molecule."""
    from protacbench.prompts import SYSTEM_PROMPT

    assert "benchmark_transformation" not in SYSTEM_PROMPT


def test_the_prompt_separates_linker_from_whole_molecule_limits():
    """TPSA >= 170 is unreachable for a linker and ordinary for a PROTAC.
    Printed under one heading it reads as an impossible instruction."""
    from protacbench.prompts import SYSTEM_PROMPT

    assert "LIMITS ON THE WHOLE MOLECULE" in SYSTEM_PROMPT
    assert "not\nof the linker" in SYSTEM_PROMPT


def test_the_headless_path_suppresses_qed_too():
    """build_prompt filtered nothing, so the headless repair run handed the
    agent "improve drug-likeness" while the system prompt forbade it."""
    from protacbench.assess import Assessment
    from protacbench.prompts import build_prompt

    assessment = Assessment(target_id="T", sample_id="0", stage="generated",
                            linker_size=8)
    assessment.add_failure("qed_too_low", "QED 0.03 below 0.05")
    assessment.add_failure("disconnected")
    text = build_prompt(assessment)
    assert "qed_too_low" not in text
    assert "disconnected" in text
