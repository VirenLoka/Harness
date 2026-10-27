"""The ADC tools, and the guard that keeps them in step with adclinker.

Two kinds of test here.

The **conformance** test holds `molbench/server.py` and
`adclinker/chemistry.py` to identical chemistry tables. The two packages
cannot import each other -- molbench is a standalone MCP server with its own
dependency set, adclinker is a pipeline -- so the tables are written twice,
and this is what stops them drifting. They have to agree: adclinker's scorer
decides what becomes a repair task, the molbench tool decides whether a
repair is accepted, and an agent caught between two versions is optimising
against a target that moves between the task and the grade.

The **behaviour** tests are built from cases that were wrong at some point
during development, each kept as the thing that must not come back.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import server

Chem = pytest.importorskip("rdkit.Chem")

APPROVED = (
    Path(__file__).resolve().parents[2]
    / "adclinker"
    / "tests"
    / "data"
    / "approved_adcs.json"
)

# MMAE, taken from the library rather than written out here.
#
# An earlier version of this file hand-wrote the SMILES and it was subtly
# wrong -- different stereochemistry from the one the pipeline builds with --
# so the payload gate correctly reported every conjugate as epimerised and
# the tests "failed" on correct code. A payload with ten stereocentres is not
# something to transcribe from memory.
def _mmae() -> str:
    from adclinker import warheads as wh

    return wh.strip_atom_maps(wh.PAYLOADS["mmae"].smiles).replace("*", "")


MMAE = _mmae()


def call(tool: str, arguments: dict[str, Any]):
    async def run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(run())


# ------------------------------------------------------------- conformance


def test_chemistry_tables_match_adclinker():
    """The one test that keeps the duplicated tables honest."""
    from adclinker import chemistry

    assert chemistry.ADC_LINKER_WINDOWS == server.ADC_LINKER_WINDOWS
    assert chemistry.LINKER_CLOGP_CEILING == server.LINKER_CLOGP_CEILING
    assert chemistry.ADC_HANDLES == server.ADC_HANDLES
    assert chemistry.ADC_TRIGGERS == server.ADC_TRIGGERS
    assert chemistry.RELEASE_CHEMISTRIES == server.RELEASE_CHEMISTRIES
    assert chemistry.BAD_JUNCTIONS == server.BAD_JUNCTIONS
    assert chemistry.JUDGED_PROPERTIES == server.JUDGED_PROPERTIES
    assert chemistry.ADVISORY_PROPERTIES == server.ADVISORY_PROPERTIES


def test_every_smarts_compiles():
    """An uncompilable pattern raises from inside a tool call otherwise, and
    blames the molecule for the server's own typo. One was shipped with a
    space inside a recursive SMARTS and this is what caught it."""
    for name, smarts in server.ADC_TRIGGERS:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, smarts, _frees, _note in server.RELEASE_CHEMISTRIES:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, smarts, _why in server.BAD_JUNCTIONS:
        assert Chem.MolFromSmarts(smarts) is not None, name
    for name, (smarts, _t) in server.ADC_HANDLES.items():
        assert Chem.MolFromSmarts(smarts) is not None, name


# ------------------------------------------------ the approved-ADC guard


@pytest.mark.skipif(not APPROVED.is_file(), reason="approved-ADC data not present")
def test_no_approved_adc_is_rejected():
    """Sixteen drugs regulators approved, through both tools.

    A tool that rejects one of these is wrong about chemistry, not finding a
    flaw in a marketed medicine. Several thresholds were loosened because of
    this test: the Fsp3 and HBD bands rejected ten and nine of them, a
    trigger search that included the payload read DM1's own ansa-ring
    carbamate as a cleavage trigger, and a reactive-group check that exempted
    only the conjugation handle blamed DM1's epoxide and calicheamicin's
    enediyne on the linker.
    """
    drugs = json.loads(APPROVED.read_text())
    rejected = {}
    for drug in drugs:
        result = call(
            "check_adc_linker",
            {
                "smiles": drug["drug_linker_smiles"],
                "payload": drug["payload_smiles"],
                "handle": drug["handle"],
                "require_cleavable": drug["cleavable"],
            },
        )
        assert not result.is_error, (drug["brand"], result)
        if not result.data["ok"]:
            rejected[drug["brand"]] = result.data["issues"]
    assert not rejected, f"approved ADCs rejected: {rejected}"


@pytest.mark.skipif(not APPROVED.is_file(), reason="approved-ADC data not present")
def test_cleavable_approved_adcs_release_their_payload():
    drugs = json.loads(APPROVED.read_text())
    wrong = {}
    for drug in drugs:
        if not drug["cleavable"]:
            continue
        result = call(
            "check_payload_release",
            {"smiles": drug["drug_linker_smiles"], "payload": drug["payload_smiles"]},
        )
        assert not result.is_error, (drug["brand"], result)
        if not result.data["releases_free_drug"]:
            wrong[drug["brand"]] = result.data["release_chemistries"]
    assert not wrong, f"cleavable approved ADCs reported as not releasing: {wrong}"


@pytest.mark.skipif(not APPROVED.is_file(), reason="approved-ADC data not present")
def test_non_cleavable_approved_adcs_are_reported_as_not_releasing():
    """Kadcyla and Blenrep release an adduct, not the free drug. That is
    deliberate and approved, and the tool must say which it is rather than
    quietly calling everything a release."""
    drugs = json.loads(APPROVED.read_text())
    for drug in drugs:
        if drug["cleavable"]:
            continue
        result = call(
            "check_payload_release",
            {"smiles": drug["drug_linker_smiles"], "payload": drug["payload_smiles"]},
        )
        assert not result.is_error
        assert result.data["releases_free_drug"] is False, drug["brand"]
        assert "linker remnant" in result.data["released_species"] or "still carrying" \
            in result.data["released_species"]


# ------------------------------------------------------------- behaviour


def _conjugate(spacer: str) -> str:
    """maleimidocaproyl + spacer + Val-Cit-PAB + MMAE, as a SMILES."""
    from adclinker import warheads as wh

    pair = wh.build_pair(
        "maleimidocaproyl", "val_cit_pab", "mmae", "spacer_before_cassette"
    )
    return wh.assemble(spacer, pair)


def test_a_carbon_starting_spacer_is_caught_as_a_ketone():
    """The defect that prompted the junction check.

    maleimidocaproyl ends in an acyl group meant to become an amide against
    the cassette's valine. A spacer starting with a carbon makes it a dialkyl
    ketone instead -- a reactive carbonyl mid-linker. RDKit sanitises it
    happily and every descriptor looks fine.
    """
    bad = call(
        "check_adc_linker", {"smiles": _conjugate("*CCOCCOCC*"), "payload": MMAE}
    )
    assert not bad.is_error
    assert bad.data["ok"] is False
    assert "dialkyl_ketone" in bad.data["bad_junctions"]

    good = call(
        "check_adc_linker", {"smiles": _conjugate("*NCCOCCOCC*"), "payload": MMAE}
    )
    assert good.data["ok"] is True, good.data["issues"]
    assert good.data["bad_junctions"] == []


def test_a_greasy_spacer_breaches_the_clogp_ceiling():
    result = call(
        "check_adc_linker",
        {"smiles": _conjugate("*NCCCCCCCCCCCC*"), "payload": MMAE},
    )
    assert result.data["ok"] is False
    assert any("cLogP" in issue for issue in result.data["issues"])
    assert result.data["linker_properties"]["clogp"] > server.LINKER_CLOGP_CEILING


def test_properties_are_measured_on_the_linker_not_the_conjugate():
    """The calibration error this whole module exists to avoid.

    The conjugate's cLogP is dominated by MMAE and sits far outside the
    reference band whatever the spacer is. Measuring it instead of the linker
    pinned four scored terms at the same value for every candidate.
    """
    from rdkit.Chem import Crippen

    smiles = _conjugate("*NCCOCCOCC*")
    result = call("check_adc_linker", {"smiles": smiles, "payload": MMAE})
    conjugate_clogp = Crippen.MolLogP(Chem.MolFromSmiles(smiles))
    assert result.data["linker_properties"]["clogp"] < conjugate_clogp - 2.0
    assert result.data["linker_smiles"] is not None
    assert MMAE not in result.data["linker_smiles"]


def test_an_epimerised_payload_is_present_but_altered():
    """An inverted stereocentre is a different drug, and saying it is
    "missing" sends a repair agent to rebuild something already there."""
    epimer = MMAE.replace("[C@@H](O)c1ccccc1", "[C@H](O)c1ccccc1")
    assert epimer != MMAE
    smiles = _conjugate("*NCCOCCOCC*").replace(
        Chem.MolToSmiles(Chem.MolFromSmiles(MMAE)), ""
    )
    # Build the epimerised conjugate the same way the good one is built.
    from adclinker import warheads as wh

    pair = wh.build_pair(
        "maleimidocaproyl", "val_cit_pab", "mmae", "spacer_before_cassette"
    )
    bad_pair = wh.WarheadPair(
        first=pair.first,
        second=wh.join(
            wh.CASSETTES["val_cit_pab"].smiles,
            wh.PAYLOADS["mmae"].smiles.replace(
                "[C@@H](O)c1ccccc1", "[C@H](O)c1ccccc1"
            ),
            left_map=2,
            right_map=1,
        ),
        handle="maleimidocaproyl",
        cassette="val_cit_pab",
        payload="mmae",
        architecture="spacer_before_cassette",
    )
    result = call(
        "check_adc_linker",
        {"smiles": wh.assemble("*NCCOCCOCC*", bad_pair), "payload": MMAE},
    )
    assert result.data["ok"] is False
    assert result.data["payload_problem"] == "payload_altered"
    assert any("stereochemistry" in issue for issue in result.data["issues"])


def test_a_trigger_without_release_chemistry_is_reported():
    """Val-Cit bonded to the payload through a plain amide with no
    self-immolative spacer: cut, and the payload keeps an amino acid. Nothing
    measurable shows this."""
    from adclinker import warheads as wh

    val_cit = (
        "[*:1]C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=O)C(=O)[*:2]"
    )
    joined = wh.join(
        "O=C1C=CC(=O)N1CCCCCC(=O)[*:1]", val_cit, left_map=1, right_map=1
    )
    smiles = wh.join(joined, wh.PAYLOADS["mmae"].smiles, left_map=2, right_map=1)
    result = call("check_payload_release", {"smiles": smiles, "payload": MMAE})
    assert not result.is_error
    # It does release -- the protease cuts that amide itself -- but the tool
    # must say the spacer is missing, because a bulky payload bonded straight
    # to a dipeptide is cleaved slowly.
    assert result.data["releases_free_drug"] is True
    assert any("self-immolative" in a for a in result.data["advisories"])


def test_two_handles_are_rejected():
    """A second maleimide crosslinks two antibodies into an aggregate.

    The spacer keeps its two attachment points and hangs a second maleimide
    off a branch, which is exactly the shape a generator produces when it has
    seen maleimides in its training data and no reason not to add one.
    """
    doubled = _conjugate("*NCC(CN1C(=O)C=CC1=O)COCC*")
    result = call("check_adc_linker", {"smiles": doubled, "payload": MMAE})
    assert not result.is_error, result
    assert result.data["handle_count"] == 2
    assert result.data["ok"] is False
    assert any("crosslink" in issue for issue in result.data["issues"])


def test_a_mistyped_handle_name_is_an_error_not_a_silent_smarts():
    """Plain words parse as SMARTS -- every letter of "banana" is a valid
    aromatic atom symbol -- so a typo compiled into a nonsense query and came
    back as "no handle is present", which reads like a fact about the
    molecule rather than a mistake in the call."""
    smiles = _conjugate("*NCCOCCOCC*")
    for typo in ("banana", "maleimde", "maleimid"):
        result = call(
            "check_adc_linker", {"smiles": smiles, "payload": MMAE, "handle": typo}
        )
        assert result.is_error, typo
    # A real SMARTS still works, and so does a real name.
    assert not call(
        "check_adc_linker",
        {"smiles": smiles, "payload": MMAE, "handle": "O=C1C=CC(=O)N1"},
    ).is_error
    assert not call(
        "check_adc_linker",
        {"smiles": smiles, "payload": MMAE, "handle": "maleimide"},
    ).is_error


def test_a_benzylamine_payload_linkage_is_caught():
    """A payload hung on the p-aminobenzyl spacer as an amine rather than
    through a carbamate will not leave: the carbamate is there because it
    fragments to carbon dioxide plus the free amine."""
    from adclinker import warheads as wh

    cassette = (
        "[*:1]C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=O)C(=O)Nc1ccc(C[*:2])cc1"
    )
    joined = wh.join(
        "O=C1C=CC(=O)N1CCCCCC(=O)[*:1]", cassette, left_map=1, right_map=1
    )
    smiles = wh.join(joined, wh.PAYLOADS["mmae"].smiles, left_map=2, right_map=1)
    result = call("check_payload_release", {"smiles": smiles, "payload": MMAE})
    assert result.data["ok"] is False
    assert any("benzylamine" in issue for issue in result.data["issues"])


def test_the_real_dxd_is_a_camptothecin():
    """PAYLOADS['dxd'] was a placeholder with no camptothecin pharmacophore:
    MW 330 against 493, four rings against six, and no
    alpha-hydroxy-delta-lactone, which is the thing that traps topoisomerase
    I. Anything designed around it was built on a molecule that cannot kill
    a cell."""
    from rdkit.Chem import Descriptors

    from adclinker import warheads as wh

    lactone = Chem.MolFromSmarts(
        "[CX4;R]([OX2])([CH2][CH3])[CX3;R](=[OX1])[OX2;R][CH2;R]"
    )
    for name in ("dxd", "exatecan"):
        mol = Chem.MolFromSmiles(wh.PAYLOADS[name].smiles.replace("[*:1]", "[H]"))
        assert mol is not None, name
        assert mol.HasSubstructMatch(lactone), f"{name} lost its lactone"
        assert mol.GetRingInfo().NumRings() >= 6, name
    assert Descriptors.MolWt(
        Chem.MolFromSmiles(wh.PAYLOADS["dxd"].smiles.replace("[*:1]", "[H]"))
    ) == pytest.approx(493.5, abs=1.0)


@pytest.mark.parametrize("name", sorted(server.ADC_HANDLES))
def test_each_handle_pattern_matches_its_own_handle_exactly_once(name):
    """A handle SMARTS matching its own handle twice fails every design.

    `dbco` was written as `C#Cc1ccccc1`, which describes one half of a
    dibenzoazacyclooctyne. The alkyne is flanked by a benzo ring on BOTH
    sides, so that pattern matches a single DBCO group twice and
    handle_duplicated -- "a second handle crosslinks two antibodies into an
    aggregate" -- fired on every correct DBCO candidate.
    """
    from adclinker import warheads as wh

    if name not in wh.HANDLES:
        pytest.skip(f"{name} has no reference structure in warheads.HANDLES")
    smarts, _target = server.ADC_HANDLES[name]
    mol = Chem.MolFromSmiles(wh.strip_atom_maps(wh.HANDLES[name].smiles))
    assert mol is not None, name
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
    assert len(matches) == 1, (
        f"the {name} pattern matches one {name} group {len(matches)} times, "
        "so handle_duplicated rejects every design that uses it"
    )


def test_the_repair_prompts_name_every_mounted_tool_and_invent_none():
    """A prompt is a contract with the agent about what it can call.

    Both failure modes are silent. A tool the prompt never mentions is one
    the agent will not use -- that is how check_protac_linker went unused
    after it was added. A tool the prompt names that is not mounted sends
    the agent to a dead end mid-task, which is how an earlier version of
    this prompt asked for `submit_molecule`, a tool that belongs to a
    different package and was never on this server.

    So the prompt must account for all of them, including the ones it tells
    the agent NOT to call.
    """
    import re

    from protacbench import prompts as protac_prompts

    mounted = set(server.mcp.tools)
    assert len(mounted) >= 11, "tool list shrank unexpectedly"

    text = protac_prompts.SYSTEM_PROMPT
    unmentioned = {name for name in mounted if name not in text}
    assert not unmentioned, (
        "these tools are mounted but the PROTAC system prompt never mentions "
        f"them, so the agent will not know they exist: {sorted(unmentioned)}"
    )

    invented = set(re.findall(r"mcp__molbench__(\w+)", text)) - mounted
    assert not invented, (
        "the PROTAC system prompt names tools that are not mounted, which "
        f"dead-ends the agent mid-task: {sorted(invented)}"
    )


def test_an_unusable_payload_query_raises_rather_than_blaming_the_molecule():
    result = call(
        "check_adc_linker", {"smiles": _conjugate("*NCCOCC*"), "payload": "not a smiles"}
    )
    assert result.is_error


def test_a_missing_payload_is_reported_as_lost_not_altered():
    result = call(
        "check_adc_linker",
        {"smiles": "O=C1C=CC(=O)N1CCCCCC(=O)NCCOCCOCC(=O)O", "payload": MMAE},
    )
    assert result.data["payload_problem"] == "payload_lost"
    assert result.data["ok"] is False
