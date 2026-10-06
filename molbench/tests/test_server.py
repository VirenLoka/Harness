"""Smoke tests driving the four tools through an in-process MCP client."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

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
