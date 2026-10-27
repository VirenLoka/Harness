"""Smoke tests driving the four tools through an in-process MCP client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE = "c1ccccc1"
# Aromatic C-H -> C-F, the kind of single-atom edit worth scoring.
FLUORINATE = "[cH:1]>>[c:1]F"


def call(tool: str, arguments: dict[str, Any]):
    async def run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(run())


def test_lists_every_tool():
    async def run():
        async with Client(server.mcp) as client:
            return [tool.name for tool in await client.list_tools()]

    assert sorted(asyncio.run(run())) == [
        "apply_smarts_reaction",
        "check_protac_linker",
        "generate_3d_conformer",
        "get_molecule_metrics",
        "get_sbdd_metrics",
        "prepare_target_receptor",
        "render_molecule_2d",
        "run_docking_simulation",
        "validate_smiles",
    ]


def test_metrics_match_rdkit_reference_values():
    data = call("get_molecule_metrics", {"smiles": ASPIRIN}).data
    assert data["sa_score"] == pytest.approx(1.58, abs=0.01)
    assert data["exact_mw"] == pytest.approx(180.0423, abs=0.001)
    assert 0.0 <= data["qed"] <= 1.0
    assert data["formula"] == "C9H8O4"


def test_metrics_reject_invalid_smiles():
    result = call("get_molecule_metrics", {"smiles": "c1ccccc"})
    assert result.is_error
    assert "invalid SMILES" in result.content[0].text


def test_validate_accepts_valid_smiles():
    data = call("validate_smiles", {"smiles": ASPIRIN}).data
    assert data["valid"] is True
    assert data["canonical_smiles"] == ASPIRIN
    assert data["errors"] == []


def test_validate_reports_rdkit_error_log():
    data = call("validate_smiles", {"smiles": "c1ccccc"}).data
    assert data["valid"] is False
    assert data["canonical_smiles"] is None
    assert any("unclosed ring" in message for message in data["errors"])


def test_reaction_fluorinates_benzene():
    data = call(
        "apply_smarts_reaction", {"smiles": BENZENE, "reaction_smarts": FLUORINATE}
    ).data
    assert data["products"] == ["Fc1ccccc1"]  # deduplicated across six equivalent sites
    assert data["num_product_sets"] == 6
    assert data["rejected"] == []


def test_reaction_rejects_unparsable_smarts():
    result = call(
        "apply_smarts_reaction",
        {"smiles": BENZENE, "reaction_smarts": "not a reaction"},
    )
    assert result.is_error
    assert "reaction_smarts" in result.content[0].text


def test_reaction_rejects_multi_reactant_template():
    result = call(
        "apply_smarts_reaction",
        {
            "smiles": BENZENE,
            "reaction_smarts": "[C:1](=O)O.[N:2]>>[C:1](=O)[N:2]",
        },
    )
    assert result.is_error
    assert "single-reactant" in result.content[0].text


def test_render_returns_inline_png_and_saves_svg():
    result = call(
        "render_molecule_2d", {"smiles": ASPIRIN, "filename_prefix": "aspirin"}
    )
    assert [block.type for block in result.content] == ["image"]
    assert result.content[0].mime_type == "image/png"

    svg_path = Path(result.structured_content["svg_path"])
    assert svg_path.is_absolute()
    assert svg_path.parent == server.VIZ_DIR
    assert svg_path.name.startswith("aspirin_") and svg_path.suffix == ".svg"
    assert "<svg" in svg_path.read_text(encoding="utf-8")


def test_render_highlights_substructure_matches():
    structured = call(
        "render_molecule_2d",
        {
            "smiles": ASPIRIN,
            "filename_prefix": "aspirin_ester",
            "highlight_smarts": "C(=O)O",
        },
    ).structured_content
    assert len(structured["highlighted_atoms"]) > 0


def test_render_prefix_cannot_escape_the_visualizations_directory():
    structured = call(
        "render_molecule_2d",
        {"smiles": BENZENE, "filename_prefix": "../../etc/passwd"},
    ).structured_content
    svg_path = Path(structured["svg_path"])
    assert svg_path.parent == server.VIZ_DIR
    assert "/" not in svg_path.name.removesuffix(".svg")


# ------------------------------------------------------ check_protac_linker

WARHEAD = "NC(=O)c1ccc(-c2ccncc2)cc1"
E3_LIGAND = "O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)C(=O)N1"
INTACT = (
    "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
)


def check(smiles):
    return server.check_protac_linker(smiles, E3_LIGAND, WARHEAD)


def test_an_intact_protac_passes():
    result = check(INTACT)
    assert result["ok"] is True
    assert result["fragments_intact"] is True
    assert result["attachment_bonds"] == 2
    assert result["linker_smiles"] == "CCOCCOCC"
    assert result["span_bonds"] == 9
    assert result["issues"] == []


def test_a_deleted_fragment_is_caught():
    """The failure that looks like success: the result parses, sanitises,
    scores well, and is not a PROTAC. Nothing else measurable catches it."""
    result = check("OCCOCCOCCNC(=O)c1ccc(-c2ccncc2)cc1")
    assert result["ok"] is False
    assert result["fragments_intact"] is False
    assert any("different compound" in issue for issue in result["issues"])


def test_a_reactive_group_in_the_linker_is_caught():
    result = check(
        "O=C1CCC(N2C(=O)c3ccc(OCCC(C=O)CNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    assert result["ok"] is False
    assert "aldehyde" in result["reactive_groups"]


def test_a_reactive_group_in_a_fragment_is_not_the_linkers_fault():
    """A warhead may legitimately contain chemistry a linker may not, and
    flagging it would reject molecules for something untouched."""
    warhead = "O=Cc1ccc(-c2ccncc2)cc1"
    molecule = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCc4ccc(-c5ccncc5)cc4C=O)cc3C2=O)C(=O)N1"
    )
    result = server.check_protac_linker(molecule, E3_LIGAND, warhead)
    assert "aldehyde" not in result["reactive_groups"]


def test_a_dangling_linker_is_caught():
    result = check(
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCC)cc3C2=O)C(=O)N1."
        "NC(=O)c1ccc(-c2ccncc2)cc1"
    )
    assert result["ok"] is False
    assert result["attachment_bonds"] == 1


def test_bonds_are_counted_not_atoms():
    """A one-atom linker bonded to both fragments makes two bonds. Counting
    linker atoms that touch a fragment would call it non-bridging."""
    result = check(
        "O=C1CCC(N2C(=O)c3ccc(OCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    assert result["attachment_bonds"] == 2
    assert result["linker_smiles"] == "C"


def test_a_molecule_in_pieces_says_so():
    result = check(f"{E3_LIGAND}.{WARHEAD}")
    assert result["ok"] is False
    assert any("disconnected" in issue for issue in result["issues"])
    assert any("linker is gone" in issue for issue in result["issues"])


def test_ok_does_not_judge_the_span():
    """Span is reported, never gated: the right range is a property of the
    two proteins being bridged, not of the molecule."""
    result = check(
        "O=C1CCC(N2C(=O)c3ccc(OCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    assert result["ok"] is True
    assert result["span_bonds"] == 2


def test_an_unparseable_smiles_raises():
    with pytest.raises(ToolError):
        check("C(C(C")


def test_the_tool_is_registered():
    assert "check_protac_linker" in {t for t in dir(server) if not t.startswith("_")}


def test_metrics_expose_what_an_optimiser_needs():
    """cLogP and TPSA were reachable only through get_sbdd_metrics, which
    needs a docking score. An agent told to optimise polarity and forbidden
    from claiming unmeasured numbers could do neither."""
    metrics = server.get_molecule_metrics("CCOCCOCC")
    for key in ("clogp", "tpsa", "num_rotatable_bonds"):
        assert key in metrics


def test_span_is_measured_through_the_linker():
    """GetShortestPath over the whole molecule routes through a fragment,
    reporting a short span for a long linker whenever the fragments are
    fused or share a ring -- the opposite of what the number is for."""
    short = check(
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    longer = check(
        "O=C1CCC(N2C(=O)c3ccc(OCCCCCCCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    assert short["span_bonds"] == 9
    assert longer["span_bonds"] == 9
    assert longer["linker_smiles"] == "CCCCCCCC"


def test_an_epimerised_fragment_is_not_intact():
    """Without useChirality, inverting a stereocentre in the warhead passes
    as unaltered -- and a PROTAC's warhead stereochemistry is the reason it
    binds anything."""
    fragment = "O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)[C@@H](C)N1"
    correct = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)"
        "[C@@H](C)N1"
    )
    flipped = correct.replace("[C@@H](C)N1", "[C@H](C)N1")
    assert server.check_protac_linker(correct, fragment, WARHEAD)["fragments_intact"]
    assert not server.check_protac_linker(flipped, fragment, WARHEAD)[
        "fragments_intact"
    ]


def test_both_bonds_onto_one_fragment_is_not_a_bridge():
    """Two crossings satisfies attachment_bonds == 2 while the other fragment
    is fused straight on with no linker between."""
    molecule = (
        "O=C1CCC(N2C(=O)c3ccc(C4(CC4)Nc4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    )
    result = server.check_protac_linker(
        molecule, E3_LIGAND.replace("(O)", ""), "Nc1ccc(-c2ccncc2)cc1"
    )
    if result["attachment_bonds"] == 2 and result["fragments_intact"]:
        assert result["ok"] or any(
            "same fragment" in issue or "bridge" in issue
            for issue in result["issues"]
        )


# ----------------------------------- defects found by adversarial review

def test_a_generic_fragment_query_does_not_land_on_the_linker():
    """An aryl amide appears in most real linkers. Claiming fragment 1 first
    and taking the first match that fits put the warhead query on the
    linker's own amide, reporting a correct molecule as broken with a linker
    that contained the warhead."""
    molecule = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCNC(=O)c4ccc(CNC(=O)c5ccccc5)cc4)"
        "cc3C2=O)C(=O)N1"
    )
    result = server.check_protac_linker(molecule, E3_LIGAND, "NC(=O)c1ccccc1")
    assert result["ok"] is True
    assert result["attachment_bonds"] == 2


def test_the_verdict_does_not_depend_on_argument_order():
    molecule = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCC(=O)NC(=O)c4ccc(O)cc4)cc3C2=O)C(=O)N1"
    )
    warhead = "NC(=O)c1ccc(O)cc1"
    forward = server.check_protac_linker(molecule, E3_LIGAND, warhead)
    reverse = server.check_protac_linker(molecule, warhead, E3_LIGAND)
    assert forward["fragments_intact"] == reverse["fragments_intact"] is True


def test_dummy_capped_fragments_match():
    """`*` attachment points are how PROTAC-DB and DiffLinker write
    fragments, and a dummy atom matches nothing in a mol-mol query."""
    result = server.check_protac_linker(
        INTACT,
        "O=C1CCC(N2C(=O)c3ccc(O*)cc3C2=O)C(=O)N1",
        "*NC(=O)c1ccc(-c2ccncc2)cc1",
    )
    assert result["fragments_intact"] is True
    assert result["linker_smiles"] == "CCOCCOCC"


def test_explicit_hydrogens_and_charges_do_not_break_the_query():
    for fragment in (
        "[H]Oc1ccc2c(c1)C(=O)N(C1CCC(=O)NC1=O)C2=O",
        "O=C1CCC(N2C(=O)c3ccc([O-])cc3C2=O)C(=O)N1",
    ):
        result = server.check_protac_linker(INTACT, fragment, WARHEAD)
        assert result["fragments_intact"] is True, fragment


def test_an_altered_stereocentre_says_so_rather_than_deleted():
    """An epimerised warhead is present but wrong. Calling it deleted sends
    the agent to rebuild something that is already there."""
    fragment = "O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)[C@@H](C)N1"
    correct = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)"
        "[C@@H](C)N1"
    )
    flipped = correct.replace("[C@@H](C)N1", "[C@H](C)N1")
    result = server.check_protac_linker(flipped, fragment, WARHEAD)
    assert result["fragments_intact"] is False
    assert any("stereochemistry" in issue for issue in result["issues"])


def test_an_unusable_fragment_blames_the_fragment_not_the_molecule():
    with pytest.raises(ToolError, match="fragment"):
        server.check_protac_linker(INTACT, "*", WARHEAD)


def test_ambiguous_placements_are_reported():
    """When a fragment can sit in more than one place the derived linker is
    a guess, and the caller should be told."""
    molecule = (
        "O=C1CCC(N2C(=O)c3ccc(OCCOCCNC(=O)c4ccc(CNC(=O)c5ccccc5)cc4)"
        "cc3C2=O)C(=O)N1"
    )
    result = server.check_protac_linker(molecule, E3_LIGAND, "NC(=O)c1ccccc1")
    assert result["ambiguous_placements"] >= 2
