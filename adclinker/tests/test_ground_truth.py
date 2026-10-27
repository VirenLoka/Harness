"""Every threshold is checked against the ADCs regulators actually approved.

This file exists because of a mistake made twice in this repository, and it
is here to stop it being made a third time.

The first time, `protacbench` gated on QED at 0.05 -- a reasonable-looking
drug-likeness floor that turned out to reject 34% of the real published
PROTACs it was meant to be finding. The second time, `adclinker` compared
whole conjugates against windows measured on payload-free reagents, which
put every candidate outside every window and left 60% of the scoring weight
doing nothing; it ranked a greasy C18 chain above the spacer in brentuximab
vedotin.

Both failures look identical from inside the code. The threshold is
defensible, the arithmetic is right, and nothing crashes. The only thing that
catches them is running the ground truth through and checking it survives.

So: `data/approved_adcs.json` holds the drug-linker of every FDA- or
EMA-approved antibody-drug conjugate that could be structurally confirmed,
each with its payload excised and its conjugation handle named. If a change
to a threshold, a gate or a SMARTS pattern rejects one of them, these tests
fail, and the threshold is wrong rather than the drug.

The structures were assembled by independent searches and cross-checked: ten
of the sixteen were reported twice from separate sources and agreed on
canonical SMILES in every case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adclinker import chemistry, reference, score

Chem = pytest.importorskip("rdkit.Chem")

DATA = Path(__file__).resolve().parent / "data" / "approved_adcs.json"


@pytest.fixture(scope="module")
def approved() -> list[dict]:
    drugs = json.loads(DATA.read_text())
    assert len(drugs) >= 15, "the approved-ADC set has shrunk; was it truncated?"
    return drugs


@pytest.fixture(scope="module")
def profile(tmp_path_factory) -> reference.Profile:
    """The real reference profile, or skip. These tests are about calibration
    against real data, and there is nothing to calibrate without it."""
    dataset = Path(__file__).resolve().parents[2] / "Linker-GPT/data/linkers.csv"
    if not dataset.is_file():
        pytest.skip(f"reference dataset not present at {dataset}")
    return reference.load_profile(
        dataset, cache=tmp_path_factory.mktemp("profile") / "p.json"
    )


@pytest.fixture(scope="module")
def fingerprints() -> list:
    """The real similarity reference, not an empty list.

    Scoring with no fingerprints zeroes `adc_likeness` and maxes `novelty`
    for every molecule, which shifts the whole scale by nine to fifteen
    points. A test asserting anything about absolute scores against that is
    measuring its own fixture rather than the scorer.
    """
    dataset = Path(__file__).resolve().parents[2] / "Linker-GPT/data/linkers.csv"
    if not dataset.is_file():
        pytest.skip(f"reference dataset not present at {dataset}")
    return reference.fingerprint_reference(dataset)


def _context(
    drug: dict, profile: reference.Profile, fingerprints: list | None = None
) -> score.Context:
    return score.Context(
        profile=profile,
        reference_fingerprints=fingerprints or [],
        payload_smiles=drug["payload_smiles"],
        handle_smarts=chemistry.ADC_HANDLES[drug["handle"]][0],
        require_cleavable=drug["cleavable"],
    )


def test_every_approved_drug_linker_parses(approved):
    for drug in approved:
        assert Chem.MolFromSmiles(drug["drug_linker_smiles"]) is not None, drug["brand"]
        assert Chem.MolFromSmiles(drug["payload_smiles"]) is not None, drug["brand"]


def test_no_gate_rejects_an_approved_adc(approved, profile):
    """The headline guard. A gate that rejects an approved drug is wrong."""
    rejected = {}
    for drug in approved:
        result = score.score_smiles(drug["drug_linker_smiles"], _context(drug, profile))
        if result.gates_failed:
            rejected[drug["brand"]] = result.gates_failed
    assert not rejected, (
        "these approved ADCs were rejected by a gate, which means the gate is "
        f"wrong rather than the drug: {rejected}"
    )


def test_approved_adcs_score_above_a_greasy_chain(approved, profile, fingerprints):
    """Calibration, not just survival.

    Surviving the gates is not enough: the score has to rank real chemistry
    above bad chemistry. The failure this catches is the one that actually
    happened -- every term pinned at the same value, so the ranking was
    noise and a C18 chain came out level with the clinical spacer.
    """
    from adclinker import warheads as wh

    totals = [
        score.score_smiles(
            d["drug_linker_smiles"], _context(d, profile, fingerprints)
        ).total
        for d in approved
    ]
    assert all(t is not None for t in totals)
    assert min(totals) > 50.0, (
        f"an approved ADC scored {min(totals):.1f}; the scale has drifted"
    )
    assert max(totals) - min(totals) < 50.0, (
        "approved ADCs span too wide a range to be ranked meaningfully "
        f"({min(totals):.1f} to {max(totals):.1f})"
    )

    # The comparison the name promises: a greasy alkyl spacer must land
    # below every one of them, and by a margin that is not noise.
    pair = wh.build_pair(
        "maleimidocaproyl", "val_cit_pab", "mmae", "spacer_before_cassette"
    )
    greasy = score.score_smiles(
        wh.assemble("*NCCCCCCCCCCCC*", pair),
        score.Context(
            profile=profile,
            reference_fingerprints=fingerprints,
            payload_smiles=wh.PAYLOADS["mmae"].smiles,
            handle_smarts=chemistry.ADC_HANDLES["maleimide"][0],
            require_cleavable=True,
        ),
    )
    assert "clogp_too_high" in greasy.gates_failed, (
        f"a C12 greasy spacer was not rejected; it scored {greasy.total} "
        f"with gates {greasy.gates_failed}. Its linker cLogP is above the "
        "ceiling, so the scorer and the molbench tool must agree that it is "
        "not a candidate"
    )


def test_peg_length_is_not_ranked_by_an_artefact(profile, fingerprints):
    """Choosing between PEG2 and PEG8 is a drug-to-antibody-ratio question,
    and nothing here measures it -- so they should tie.

    They did not. Every other term ties exactly across a PEG series, so the
    synthesisability term alone decided the ranking, and RDKit's SA score
    drifts upward with molecule size. The scorer was ranking PEG length by a
    size artefact, and preferring the shortest spacer -- the opposite of the
    usual remedy for a hydrophobic drug-linker.
    """
    from adclinker import warheads as wh

    pair = wh.build_pair(
        "maleimidocaproyl", "val_cit_pab", "mmae", "spacer_before_cassette"
    )
    context = score.Context(
        profile=profile,
        reference_fingerprints=fingerprints,
        payload_smiles=wh.PAYLOADS["mmae"].smiles,
        handle_smarts=chemistry.ADC_HANDLES["maleimide"][0],
        require_cleavable=True,
    )
    totals = []
    for length in (2, 4, 6, 8):
        spacer = "*N" + "CCO" * length + "CC*"
        result = score.score_smiles(wh.assemble(spacer, pair), context)
        assert result.total is not None, (length, result.gates_failed)
        totals.append(result.total)
    assert max(totals) - min(totals) < 1.0, (
        f"PEG lengths 2 to 8 score {min(totals):.1f} to {max(totals):.1f}; "
        "something is ranking them that should not be"
    )


def test_the_clogp_ceiling_rejects_none_of_them(approved):
    """The one hard threshold, checked directly against the clinical set."""
    from rdkit.Chem import Crippen

    over = {}
    for drug in approved:
        linker = Chem.MolFromSmiles(drug["linker_only_smiles"].replace("*", "[H]"))
        if linker is None:
            continue
        value = Crippen.MolLogP(linker)
        if value > chemistry.LINKER_CLOGP_CEILING:
            over[drug["brand"]] = round(value, 2)
    assert not over, (
        f"the cLogP ceiling of {chemistry.LINKER_CLOGP_CEILING} rejects "
        f"approved ADCs: {over}"
    )


@pytest.mark.parametrize("prop", sorted(chemistry.JUDGED_PROPERTIES))
def test_a_judged_property_does_not_reject_most_of_the_clinical_set(
    prop, approved
):
    """Why fsp3 and hbd are measured but never judged.

    Their reference bands reject 10 and 9 of these 16 respectively, because
    the reference set is bare PEG and alkyl reagents while a finished linker
    carries an aromatic spacer and a dipeptide. Any property promoted into
    JUDGED_PROPERTIES has to clear the same bar.
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    read = {
        "clogp": Crippen.MolLogP,
        "tpsa": rdMolDescriptors.CalcTPSA,
        "mw": Descriptors.MolWt,
        "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
        "fsp3": rdMolDescriptors.CalcFractionCSP3,
        "hbd": rdMolDescriptors.CalcNumHBD,
        "hba": rdMolDescriptors.CalcNumHBA,
    }[prop]
    low, _median, high = chemistry.ADC_LINKER_WINDOWS[prop]
    outside = 0
    total = 0
    for drug in approved:
        linker = Chem.MolFromSmiles(drug["linker_only_smiles"].replace("*", "[H]"))
        if linker is None:
            continue
        total += 1
        value = float(read(linker))
        # cLogP is judged as a ceiling, so being below the band is not a miss.
        if prop == "clogp":
            if value > chemistry.LINKER_CLOGP_CEILING:
                outside += 1
        elif not low <= value <= high:
            outside += 1
    assert outside <= total // 4, (
        f"{prop} is in JUDGED_PROPERTIES but its band is outside for "
        f"{outside} of {total} approved ADCs"
    )


def test_the_shared_tables_are_not_empty():
    """A table emptied by a bad edit would silently pass everything."""
    assert len(chemistry.ADC_TRIGGERS) >= 8
    assert len(chemistry.RELEASE_CHEMISTRIES) >= 8
    assert len(chemistry.BAD_JUNCTIONS) >= 5
    assert len(chemistry.ADC_HANDLES) >= 7
    for name, smarts in chemistry.ADC_TRIGGERS:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, smarts, _frees, _note in chemistry.RELEASE_CHEMISTRIES:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, smarts, _why in chemistry.BAD_JUNCTIONS:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, (smarts, _target) in chemistry.ADC_HANDLES.items():
        assert Chem.MolFromSmarts(smarts) is not None, name


def test_windows_match_the_live_profile(profile):
    """The literals in `chemistry.py` are a cache of a measurement.

    They are written out so a prompt is reproducible without reading the
    dataset. If the dataset or the filtering changes, the literals are stale
    and every threshold derived from them describes a set that no longer
    exists.
    """
    for name, (p10, median, p90) in chemistry.ADC_LINKER_WINDOWS.items():
        stats = profile.properties[name]
        assert stats.p10 == pytest.approx(p10, abs=0.05), name
        assert stats.median == pytest.approx(median, abs=0.05), name
        assert stats.p90 == pytest.approx(p90, abs=0.05), name


def test_prompt_bands_match_the_shared_tables():
    """The prompt quotes numbers at the agent; they must be the real ones."""
    from adclinker import prompts

    assert prompts.CLOGP_CEILING == chemistry.LINKER_CLOGP_CEILING
    for name, band in prompts.LINKER_BANDS.items():
        assert band == chemistry.ADC_LINKER_WINDOWS[name], name
