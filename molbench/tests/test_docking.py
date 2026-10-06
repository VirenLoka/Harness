"""Tests for the structure-based tools.

Unit tests run anywhere. The docking tests need the engines, so they skip when
no binary is resolvable, and the receptor test needs 3PTB either cached or
downloadable.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docking
import server

BENZAMIDINE = "c1ccc(cc1)C(=N)N"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PRISMANE = "C12C3C1C4C2C34"

# Two protein atoms, one water, one ligand atom, one metal.
MINI_PDB = """\
ATOM      1  N   ILE A  16      26.000  10.000  20.000  1.00 20.00           N
ATOM      2  CA  ILE A  16      27.000  10.000  20.000  1.00 20.00           C
HETATM 1000  O   HOH A 300       5.000   5.000   5.000  1.00 30.00           O
HETATM 1001  C1  BEN A 711      -2.000  14.000  17.000  1.00 25.00           C
HETATM 1002  C2  BEN A 711       0.000  16.000  19.000  1.00 25.00           C
HETATM 1003 ZN    ZN A 400      12.000  12.000  12.000  1.00 25.00          ZN
"""

VINA_POSES = """MODEL 1
REMARK VINA RESULT:    -6.086      0.000      0.000
ATOM      1  C   UNL     1       1.000   2.000   3.000  0.00  0.00    +0.000 C
ENDMDL
MODEL 2
REMARK VINA RESULT:    -5.421      1.234      2.345
ATOM      1  C   UNL     1       1.500   2.500   3.500  0.00  0.00    +0.000 C
ENDMDL
"""


def call(tool: str, arguments: dict):
    async def run():
        async with Client(server.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(run())


def engines_available() -> bool:
    return (
        docking.resolve_binary("vina") is not None
        and docking.resolve_binary("obabel") is not None
    )


needs_engines = pytest.mark.skipif(
    not engines_available(), reason="vina/obabel not installed (see README: conda env)"
)


# ------------------------------------------------------------------ unit tests


def test_strip_structure_removes_water_and_ligands():
    text, counts = docking.strip_structure(MINI_PDB)
    assert counts == {
        "atoms": 2,
        "waters_removed": 1,
        "hetatms_removed": 3,
        "hetatms_kept": 0,
    }
    assert "HOH" not in text and "BEN" not in text and "ZN" not in text
    assert text.count("ATOM") == 2


def test_strip_structure_keeps_requested_cofactors():
    text, counts = docking.strip_structure(MINI_PDB, keep_hetatms=["ZN"])
    assert counts["hetatms_kept"] == 1
    assert "ZN" in text


def test_ligand_centroid_averages_the_named_residue():
    center, atoms = docking.ligand_centroid(MINI_PDB, "BEN")
    assert atoms == 2
    assert center == pytest.approx((-1.0, 15.0, 18.0))


def test_ligand_centroid_rejects_a_missing_residue():
    with pytest.raises(docking.DockingError, match="no HETATM residue"):
        docking.ligand_centroid(MINI_PDB, "XYZ")


def test_parse_vina_pdbqt_reads_every_pose_in_order():
    poses = docking.parse_vina_pdbqt(VINA_POSES)
    assert [p.rank for p in poses] == [1, 2]
    assert [p.affinity for p in poses] == [-6.086, -5.421]
    assert "ENDMDL" in poses[0].block


def test_pocket_residues_selects_only_nearby_residues():
    receptor = """\
ATOM      1  CA  ASP A 189       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  SER A 195       2.000   0.000   0.000  1.00 20.00           C
ATOM      3  CA  GLY A   1      50.000  50.000  50.000  1.00 20.00           C
"""
    ligand_sdf = "lig\n\n\n  1  0  0  0  0  0            999 V2000\n    0.5000    0.0000    0.0000 C   0  0\n"
    pocket = docking.pocket_residues(receptor, ligand_sdf, cutoff=5.0)
    assert pocket == [{"chain": "A", "resi": [189, 195]}]


def test_sbdd_metrics_match_their_definitions():
    metrics = docking.sbdd_metrics(BENZAMIDINE, -6.081)
    heavy = 9  # C7N2
    assert metrics["heavy_atoms"] == heavy
    assert metrics["Binding_Affinity"] == -6.081
    assert metrics["Ligand_Efficiency"] == pytest.approx(-6.081 / heavy, abs=1e-4)
    assert metrics["estimated_pIC50"] == pytest.approx(6.081 / 1.364, abs=1e-4)
    assert metrics["Lipophilic_Ligand_Efficiency"] == pytest.approx(
        metrics["estimated_pIC50"] - metrics["cLogP"], abs=1e-4
    )
    assert metrics["meets_le_target"] is True  # -0.68 is below the -0.3 target
    assert metrics["TPSA"] > 0


def test_sbdd_metrics_rejects_bad_input():
    with pytest.raises(docking.DockingError, match="invalid SMILES"):
        docking.sbdd_metrics("c1ccccc", -7.0)


def test_embed_conformers_reports_strain_for_a_normal_molecule():
    result = docking.embed_conformers(ASPIRIN, num_conformers=5)
    assert result["force_field"] == "MMFF94"
    assert result["num_conformers"] == 5
    assert result["local_strain"] > 0  # the raw embedding sits above its minimum
    assert result["ensemble_strain"] >= 0
    assert result["minimized_energy"] >= result["global_min_energy"] - 1e-6
    assert "V2000" in result["sdf"] or "V3000" in result["sdf"]


def test_embed_conformers_fails_clearly_on_a_strained_cage():
    # RDKit raises a RuntimeError invariant violation here, not a ValueError.
    with pytest.raises(docking.DockingError, match="too strained"):
        docking.embed_conformers(PRISMANE, num_conformers=5)


def test_strained_molecule_surfaces_as_a_tool_error():
    result = call("generate_3d_conformer", {"smiles": PRISMANE})
    assert result.is_error
    assert "too strained" in result.content[0].text


def test_resolve_binary_prefers_an_explicit_override(monkeypatch, tmp_path):
    fake = tmp_path / "vina"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("MOLBENCH_VINA_BIN", str(fake))
    assert docking.resolve_binary("vina") == fake
    monkeypatch.setenv("MOLBENCH_VINA_BIN", str(tmp_path / "missing"))
    assert docking.resolve_binary("vina") is None


def test_unknown_engine_is_rejected():
    result = call(
        "run_docking_simulation",
        {
            "smiles": BENZAMIDINE,
            "receptor_file": str(__file__),
            "engine": "autodock9000",
        },
    )
    assert result.is_error
    assert "unknown engine" in result.content[0].text


def test_haddock3_reports_that_it_is_not_wired_up():
    with pytest.raises(docking.DockingError, match="HADDOCK3 is not wired up"):
        asyncio.run(docking.dock_haddock3())


# ----------------------------------------------------------- integration tests


@pytest.fixture(scope="module")
def prepared_receptor():
    if not engines_available():
        pytest.skip("docking engines not installed")
    result = call(
        "prepare_target_receptor", {"pdb_id": "3PTB", "ligand_resname": "BEN"}
    )
    if result.is_error:
        pytest.skip(f"could not prepare 3PTB: {result.content[0].text[:120]}")
    return result.data


@needs_engines
def test_receptor_preparation_strips_and_centres(prepared_receptor):
    data = prepared_receptor
    assert Path(data["receptor_file"]).is_file()
    assert data["waters_removed"] == 62
    assert data["kept_protein_atoms"] == 1629
    # Benzamidine's centroid in 3PTB.
    assert data["binding_site_center"] == pytest.approx(
        [-1.759, 14.461, 16.916], abs=0.01
    )
    assert data["box_centred_on"] == "BEN"


@needs_engines
def test_docking_reproduces_the_known_benzamidine_pose(prepared_receptor):
    result = call(
        "run_docking_simulation",
        {
            "smiles": BENZAMIDINE,
            "receptor_file": prepared_receptor["receptor_file"],
            "engine": "vina",
            "label": "pytest_benzamidine",
        },
    )
    assert not result.is_error, result.content[0].text
    data = result.data

    # Benzamidine/trypsin is a well characterised complex; Vina lands near -6.
    assert -8.0 < data["binding_affinity"] < -4.5
    assert data["poses"][0]["affinity"] == data["binding_affinity"]
    assert data["pose_strain"]["pose_strain"] >= 0

    for key in ("pose_file", "complex_file", "viewer_file"):
        assert Path(data[key]).is_file()

    # The viewer must highlight the real S1 pocket, in deposited numbering.
    html = Path(data["viewer_file"]).read_text()
    pocket = json.loads(html.split("const pocket = ")[1].split(";\n")[0])
    resi = pocket[0]["resi"]
    assert 189 in resi, "Asp189 lines trypsin's S1 pocket and must be contacted"
    assert 195 in resi, "Ser195 is the catalytic serine"


@needs_engines
@pytest.mark.skipif(
    docking.resolve_binary("smina") is None, reason="smina not installed"
)
def test_smina_agrees_with_vina(prepared_receptor):
    result = call(
        "run_docking_simulation",
        {
            "smiles": BENZAMIDINE,
            "receptor_file": prepared_receptor["receptor_file"],
            "engine": "smina",
            "label": "pytest_smina",
        },
    )
    assert not result.is_error, result.content[0].text
    assert -8.0 < result.data["binding_affinity"] < -4.5
