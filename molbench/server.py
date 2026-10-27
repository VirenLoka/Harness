#!/usr/bin/env python3
"""MolBench: an MCP server exposing RDKit and docking tools for molecule design.

2D ligand tools measure how a chemical transformation changes the heuristic
Synthetic Accessibility (SA) score, and show the structural change:

    validate_smiles        parse check plus RDKit's own error log
    get_molecule_metrics   SA score, exact molecular weight, QED
    apply_smarts_reaction  run a reaction SMARTS and return valid products
    render_molecule_2d     2D depiction: inline PNG plus a saved SVG file

3D structure-based tools dock a ligand into a target and score the result:

    prepare_target_receptor  fetch a PDB, strip it, write a receptor PDBQT
    generate_3d_conformer    ETKDGv3 embedding, MMFF94 optimization, strain
    run_docking_simulation   Vina or smina, with an interactive pose viewer
    get_sbdd_metrics         LE, LLE, cLogP and TPSA around a docking score

Run it over stdio (how MCP clients normally launch it):

    python server.py

or as a long-lived HTTP service:

    python server.py --transport http --port 8077
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import sys
from contextlib import contextmanager, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, QED, AllChem, Descriptors, RDConfig, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

import docking
from docking import DockingError

# The SA scorer ships in RDKit's Contrib tree, which is not an importable
# package, so its directory has to join sys.path before the import.
sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))

# Imported after the sys.path line above, which is what makes it importable.
import sascorer

SERVER_DIR = Path(__file__).resolve().parent
VIZ_DIR = Path(os.environ.get("MOLBENCH_VIZ_DIR") or SERVER_DIR / "visualizations")

# Prefixes come from a model, so keep them to a leaf filename: no separators,
# no traversal, no surprises.
UNSAFE_PREFIX_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
MAX_PREFIX_LENGTH = 64
MAX_PRODUCTS = 32

mcp = FastMCP(
    name="molbench",
    instructions=(
        "RDKit metrics for benchmarking SMARTS transformations. Typical loop: "
        "validate_smiles, get_molecule_metrics on the starting material, "
        "apply_smarts_reaction, then get_molecule_metrics on each product to "
        "compare SA scores. render_molecule_2d returns an inline depiction. "
        "SA score runs 1 (easy to synthesize) to 10 (hard); QED runs 0 to 1, "
        "higher being more drug-like.\n\n"
        "For structure-based work: prepare_target_receptor once per target "
        "(it caches), then run_docking_simulation per ligand, then "
        "get_sbdd_metrics with the returned affinity. Docking scores are in "
        "kcal/mol and more negative is better."
    ),
)


class _CapturedLog:
    """What `_capture_rdkit_log` yields: RDKit's complaints, as text."""

    def __init__(self) -> None:
        self.messages = ""


@contextmanager
def _capture_rdkit_log():
    """RDKit's error log for the enclosed block, on any RDKit build.

    `rdBase.CaptureErrorLog` is not present in every RDKit that satisfies
    this package's own `rdkit>=2025.3` pin -- it is absent from 2025.03.2,
    for one -- and calling it there raises AttributeError from inside
    `_parse_smiles`, which every tool on this server goes through. That turns
    a missing convenience into a server where nothing works at all.

    The fallback routes the C++ logs through Python's stderr and redirects
    that, which captures the same text and restores the stream afterwards.
    """
    capture = _CapturedLog()
    native = getattr(rdBase, "CaptureErrorLog", None)
    if native is not None:
        with native() as captured:
            yield capture
            capture.messages = captured.messages
        return

    buffer = io.StringIO()
    rdBase.LogToPythonStderr()
    try:
        with redirect_stderr(buffer):
            yield capture
    finally:
        rdBase.LogToCppStreams()
        capture.messages = buffer.getvalue()


def _parse_smiles(smiles: str) -> tuple[Chem.Mol | None, list[str]]:
    """Parse SMILES, capturing whatever RDKit wrote to its error log."""
    if not smiles or not smiles.strip():
        raise ToolError("smiles must be a non-empty string")
    with _capture_rdkit_log() as capture:
        mol = Chem.MolFromSmiles(smiles)
    messages = [line.strip() for line in capture.messages.splitlines() if line.strip()]
    return mol, messages


def _require_mol(smiles: str) -> Chem.Mol:
    mol, messages = _parse_smiles(smiles)
    if mol is None:
        detail = "; ".join(messages) or "RDKit returned no molecule"
        raise ToolError(f"invalid SMILES {smiles!r}: {detail}")
    return mol


def _metrics(mol: Chem.Mol) -> dict[str, Any]:
    return {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "sa_score": round(float(sascorer.calculateScore(mol)), 4),
        "exact_mw": round(float(Descriptors.ExactMolWt(mol)), 4),
        "qed": round(float(QED.qed(mol)), 4),
        "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        # cLogP and TPSA were reachable only through get_sbdd_metrics, which
        # needs a docking score. An agent told to optimise polarity and
        # forbidden from claiming unmeasured numbers could do neither.
        "clogp": round(float(Crippen.MolLogP(mol)), 4),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 2),
        "num_rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


def _safe_prefix(filename_prefix: str) -> str:
    cleaned = UNSAFE_PREFIX_CHARS.sub("_", (filename_prefix or "").strip()).strip("._-")
    cleaned = cleaned[:MAX_PREFIX_LENGTH]
    return cleaned or "molecule"


@mcp.tool
def get_molecule_metrics(
    smiles: Annotated[str, Field(description="SMILES string of the molecule")],
) -> dict[str, Any]:
    """Heuristic metrics for one molecule.

    Returns the RDKit Contrib Synthetic Accessibility score (1 = easy,
    10 = hard), the exact molecular weight, and the QED drug-likeness score
    (0 to 1), plus the canonical SMILES, heavy-atom count and formula.
    """
    return _metrics(_require_mol(smiles))


@mcp.tool
def validate_smiles(
    smiles: Annotated[str, Field(description="SMILES string to check")],
) -> dict[str, Any]:
    """Check whether a SMILES string parses as a valid molecule.

    Returns `valid`, the canonical SMILES when it parses, and `errors`: the
    lines RDKit itself logged while parsing, which say what is wrong.
    """
    mol, messages = _parse_smiles(smiles)
    return {
        "valid": mol is not None,
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol) if mol is not None else None,
        "errors": messages,
    }


@mcp.tool
def apply_smarts_reaction(
    smiles: Annotated[str, Field(description="SMILES of the starting material")],
    reaction_smarts: Annotated[
        str,
        Field(
            description=(
                "Reaction SMARTS, e.g. '[c:1][H]>>[c:1]F'. Single-reactant "
                "templates only, since one molecule is supplied."
            )
        ),
    ],
) -> dict[str, Any]:
    """Apply a reaction SMARTS to a molecule and return the valid products.

    Every product is sanitized; products that fail sanitization are reported
    separately in `rejected` rather than silently dropped. Products are
    deduplicated by canonical SMILES, preserving first-seen order.
    """
    mol = _require_mol(smiles)

    # RDKit raises for malformed reaction SMARTS and returns None for some
    # inputs, so both paths have to become a clean tool error.
    try:
        with _capture_rdkit_log() as capture:
            reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        parse_error = "; ".join(
            line.strip() for line in capture.messages.splitlines() if line.strip()
        )
    except ValueError as exc:
        reaction, parse_error = None, str(exc)
    if reaction is None:
        raise ToolError(
            f"could not parse reaction_smarts {reaction_smarts!r}"
            + (f": {parse_error}" if parse_error else "")
        )

    expected = reaction.GetNumReactantTemplates()
    if expected != 1:
        raise ToolError(
            f"reaction_smarts needs {expected} reactants but one molecule was "
            "supplied; use a single-reactant template"
        )

    try:
        with _capture_rdkit_log() as capture:
            product_sets = reaction.RunReactants((mol,))
    except ValueError as exc:
        raise ToolError(f"applying reaction_smarts failed: {exc}") from exc

    products: list[str] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for product_set in product_sets:
        for product in product_set:
            try:
                Chem.SanitizeMol(product)
            except (
                Chem.AtomValenceException,
                Chem.KekulizeException,
                ValueError,
            ) as exc:
                rejected.append(
                    {"smiles": Chem.MolToSmiles(product), "reason": str(exc)}
                )
                continue
            canonical = Chem.MolToSmiles(product)
            if canonical not in seen:
                seen.add(canonical)
                products.append(canonical)

    return {
        "reactant": Chem.MolToSmiles(mol),
        "reaction_smarts": reaction_smarts,
        "products": products[:MAX_PRODUCTS],
        "num_products": len(products),
        "num_product_sets": len(product_sets),
        "rejected": rejected,
        "truncated": len(products) > MAX_PRODUCTS,
        "warnings": [
            line.strip() for line in capture.messages.splitlines() if line.strip()
        ],
    }


@mcp.tool
def render_molecule_2d(
    smiles: Annotated[str, Field(description="SMILES of the molecule to draw")],
    filename_prefix: Annotated[
        str, Field(description="Leading part of the saved filename, e.g. 'aspirin'")
    ],
    highlight_smarts: Annotated[
        str | None,
        Field(
            description=(
                "Optional SMARTS; matching atoms are highlighted, which is "
                "useful for showing what a transformation changed."
            )
        ),
    ] = None,
    width: Annotated[
        int, Field(description="Image width in pixels", ge=120, le=2000)
    ] = 450,
    height: Annotated[
        int, Field(description="Image height in pixels", ge=120, le=2000)
    ] = 350,
) -> ToolResult:
    """Draw a molecule in 2D.

    Returns the depiction inline as a PNG, so it is visible in the
    conversation without opening a file, and also writes an SVG into the
    server's `visualizations/` directory, returning its absolute path.
    """
    mol = _require_mol(smiles)
    rdDepictor.Compute2DCoords(mol)

    highlight_atoms: list[int] = []
    if highlight_smarts:
        pattern = Chem.MolFromSmarts(highlight_smarts)
        if pattern is None:
            raise ToolError(f"could not parse highlight_smarts {highlight_smarts!r}")
        for match in mol.GetSubstructMatches(pattern):
            highlight_atoms.extend(match)
        highlight_atoms = sorted(set(highlight_atoms))

    def draw(drawer: Any) -> str | bytes:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=highlight_atoms or None
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    svg_text = draw(rdMolDraw2D.MolDraw2DSVG(width, height))
    png_bytes = draw(rdMolDraw2D.MolDraw2DCairo(width, height))

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]
    svg_path = VIZ_DIR / f"{_safe_prefix(filename_prefix)}_{stamp}.svg"
    svg_path.write_text(svg_text, encoding="utf-8")

    canonical = Chem.MolToSmiles(mol)
    return ToolResult(
        content=[Image(data=png_bytes, format="png").to_image_content()],
        structured_content={
            "svg_path": str(svg_path),
            "canonical_smiles": canonical,
            "highlighted_atoms": highlight_atoms,
            "width": width,
            "height": height,
            "png_bytes": len(png_bytes),
        },
    )


# ----------------------------------------------------------- 3D structure-based


@mcp.tool
async def prepare_target_receptor(
    pdb_id: Annotated[
        str | None, Field(description="RCSB PDB identifier, e.g. '3PTB'")
    ] = None,
    local_pdb_path: Annotated[
        str | None,
        Field(description="Path to a local PDB file, used instead of pdb_id"),
    ] = None,
    binding_site_center: Annotated[
        list[float] | None,
        Field(description="Docking box centre as [x, y, z] in Angstroms"),
    ] = None,
    box_size: Annotated[
        list[float] | None,
        Field(description="Box dimensions as [x, y, z] in Angstroms; default 20 cubed"),
    ] = None,
    ligand_resname: Annotated[
        str | None,
        Field(
            description=(
                "Residue name of a co-crystallized ligand, e.g. 'BEN'. Its centroid "
                "becomes the box centre, so you need no coordinates."
            )
        ),
    ] = None,
    keep_hetatms: Annotated[
        list[str] | None,
        Field(
            description="HETATM residue names to keep, e.g. ['ZN'] for a metalloenzyme"
        ),
    ] = None,
    refresh: Annotated[
        bool, Field(description="Re-prepare even when a cached receptor exists")
    ] = False,
) -> dict[str, Any]:
    """Prepare a docking receptor from a PDB entry or a local file.

    Strips water and co-crystallized ligands, adds hydrogens at pH 7.4, assigns
    Gasteiger charges, and writes a rigid `.pdbqt`. The result is cached, since
    protonating a protein takes about a minute; pass `refresh` to redo it.

    Give the box as `binding_site_center` or let `ligand_resname` place it on a
    bound ligand. The box is recorded next to the receptor, so
    `run_docking_simulation` can reuse it without being told again.
    """
    try:
        if bool(pdb_id) == bool(local_pdb_path):
            raise DockingError("supply exactly one of pdb_id or local_pdb_path")

        if pdb_id:
            pdb_text = await asyncio.to_thread(docking.fetch_pdb, pdb_id)
            source, stem = f"RCSB:{pdb_id.upper()}", pdb_id.upper()
        else:
            path = Path(local_pdb_path).expanduser()
            if not path.is_file():
                raise DockingError(f"local_pdb_path does not exist: {path}")
            pdb_text = path.read_text(encoding="utf-8", errors="replace")
            source, stem = str(path), path.stem

        ligand_atoms = 0
        if binding_site_center is not None:
            if len(binding_site_center) != 3:
                raise DockingError("binding_site_center needs exactly three numbers")
            center = tuple(float(v) for v in binding_site_center)
        elif ligand_resname:
            center, ligand_atoms = docking.ligand_centroid(pdb_text, ligand_resname)
        else:
            raise DockingError(
                "supply binding_site_center, or ligand_resname to centre the box "
                "on a co-crystallized ligand"
            )

        size = tuple(float(v) for v in (box_size or (20.0, 20.0, 20.0)))
        if len(size) != 3 or any(v <= 0 for v in size):
            raise DockingError("box_size needs three positive numbers")

        suffix = docking.safe_name(ligand_resname or "site", "site")
        receptor_path = (
            docking.RECEPTOR_DIR
            / f"{docking.safe_name(stem, 'receptor')}_{suffix}.pdbqt"
        )

        protein_pdb, counts = docking.strip_structure(pdb_text, keep_hetatms or ())
        reused = receptor_path.is_file() and not refresh
        if not reused:
            await docking.prepare_receptor_pdbqt(protein_pdb, receptor_path)
        # Keep the stripped source too: Open Babel renumbers residues in the
        # PDBQT, and viewers should show the deposited numbering.
        protein_path = receptor_path.with_name(receptor_path.stem + "_protein.pdb")
        protein_path.write_text(protein_pdb, encoding="utf-8")

        payload = {
            "receptor_file": str(receptor_path),
            "protein_pdb_file": str(protein_path),
            "source": source,
            "binding_site_center": [round(v, 3) for v in center],
            "box_size": [round(v, 3) for v in size],
            "box_centred_on": ligand_resname.upper()
            if ligand_resname and binding_site_center is None
            else None,
            "ligand_atoms_used": ligand_atoms,
            "kept_protein_atoms": counts["atoms"],
            "waters_removed": counts["waters_removed"],
            "hetatms_removed": counts["hetatms_removed"],
            "hetatms_kept": counts["hetatms_kept"],
            "reused_cached_receptor": reused,
        }
        docking.write_box_sidecar(receptor_path, payload)
        return payload
    except DockingError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def generate_3d_conformer(
    smiles: Annotated[str, Field(description="SMILES of the ligand")],
    seed: Annotated[
        int, Field(description="Random seed for reproducible embedding")
    ] = 42,
    num_conformers: Annotated[
        int,
        Field(
            description="Conformers to embed; the best becomes the ensemble minimum",
            ge=1,
            le=200,
        ),
    ] = 20,
) -> dict[str, Any]:
    """Generate a 3D conformer with ETKDGv3 and optimize it with MMFF94.

    Returns the lowest-energy conformer as an SDF block plus two strain numbers
    in kcal/mol: `local_strain`, how far the raw embedding sat above its own
    relaxed minimum, and `ensemble_strain`, how far that minimum sits above the
    best of the whole ensemble. True bound-conformation strain needs a pose, so
    `run_docking_simulation` reports it as `pose_strain`.

    Molecules that cannot be embedded, such as very strained cages, fail with a
    clear message rather than a stack trace.
    """
    try:
        result = await asyncio.to_thread(
            docking.embed_conformers, smiles, seed=seed, num_conformers=num_conformers
        )
    except DockingError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "smiles": smiles,
        "sdf": result["sdf"],
        "force_field": result["force_field"],
        "num_conformers": result["num_conformers"],
        "local_strain": result["local_strain"],
        "ensemble_strain": result["ensemble_strain"],
        "embedded_energy": result["embedded_energy"],
        "minimized_energy": result["minimized_energy"],
        "global_min_energy": result["global_min_energy"],
        "units": "kcal/mol",
    }


@mcp.tool
async def run_docking_simulation(
    smiles: Annotated[str, Field(description="SMILES of the ligand to dock")],
    receptor_file: Annotated[
        str, Field(description="Receptor .pdbqt from prepare_target_receptor")
    ],
    engine: Annotated[
        str, Field(description="Docking engine: 'vina', 'smina' or 'haddock3'")
    ] = "vina",
    binding_site_center: Annotated[
        list[float] | None,
        Field(
            description="Box centre [x, y, z]; defaults to the receptor's recorded box"
        ),
    ] = None,
    box_size: Annotated[
        list[float] | None,
        Field(
            description="Box size [x, y, z]; defaults to the receptor's recorded box"
        ),
    ] = None,
    exhaustiveness: Annotated[
        int,
        Field(
            description="Search effort; higher is slower and more thorough", ge=1, le=64
        ),
    ] = 8,
    num_modes: Annotated[int, Field(description="Poses to report", ge=1, le=20)] = 9,
    seed: Annotated[int, Field(description="Random seed for a reproducible run")] = 42,
    label: Annotated[
        str | None, Field(description="Filename prefix for saved outputs")
    ] = None,
    timeout_s: Annotated[
        float, Field(description="Give up on the engine after this long", gt=0, le=3600)
    ] = 600.0,
) -> dict[str, Any]:
    """Dock a SMILES string into a prepared receptor and score the best pose.

    Embeds and optimizes the ligand, converts it to PDBQT with Meeko, runs the
    engine as a subprocess, and keeps the results: the pose as SDF, the docked
    complex as PDB, and a self-contained HTML viewer that shows the pose in the
    pocket. Binding affinity is in kcal/mol, where more negative binds better.
    """
    try:
        engine_key = engine.strip().lower()
        runners = {
            "vina": docking.dock_vina,
            "smina": docking.dock_smina,
            "haddock3": docking.dock_haddock3,
        }
        if engine_key not in runners:
            raise DockingError(
                f"unknown engine {engine!r}; use vina, smina or haddock3"
            )

        receptor = Path(receptor_file).expanduser()
        if not receptor.is_file():
            raise DockingError(f"receptor_file does not exist: {receptor}")

        sidecar = docking.read_box_sidecar(receptor) or {}
        center = binding_site_center or sidecar.get("binding_site_center")
        size = box_size or sidecar.get("box_size") or [20.0, 20.0, 20.0]
        if not center or len(center) != 3:
            raise DockingError(
                "no docking box: pass binding_site_center, or prepare the receptor "
                "with prepare_target_receptor so the box is recorded"
            )
        center = [float(v) for v in center]
        size = [float(v) for v in size]

        conformer = await asyncio.to_thread(docking.embed_conformers, smiles, seed=seed)
        ligand_text = await asyncio.to_thread(docking.ligand_pdbqt, conformer["mol"])

        prefix = docking.safe_name(
            label or Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), "ligand"
        )
        stamp = docking.timestamp()
        with TemporaryDirectory(prefix="molbench-dock-") as scratch:
            workdir = Path(scratch)
            poses, log = await runners[engine_key](
                ligand_pdbqt_text=ligand_text,
                receptor=receptor,
                center=center,
                size=size,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                seed=seed,
                timeout=timeout_s,
                workdir=workdir,
            )
            if not poses:
                raise DockingError(
                    f"{engine_key} returned no poses; check that the box covers the "
                    "binding site and that the ligand fits inside it"
                )
            best = poses[0]
            pose_sdf = (
                best.block
                if engine_key == "smina"
                else await docking.pdbqt_pose_to_sdf(best.block, workdir)
            )

        docking.DOCKING_DIR.mkdir(parents=True, exist_ok=True)
        pose_path = docking.DOCKING_DIR / f"{prefix}_{stamp}_pose.sdf"
        pose_path.write_text(pose_sdf, encoding="utf-8")
        receptor_pdb = docking.receptor_display_pdb(receptor, sidecar)
        complex_path = docking.DOCKING_DIR / f"{prefix}_{stamp}_complex.pdb"
        complex_path.write_text(
            docking.build_complex_pdb(receptor_pdb, pose_sdf), encoding="utf-8"
        )
        viewer_path = docking.write_pose_viewer(
            destination=docking.DOCKING_DIR / f"{prefix}_{stamp}_pose.html",
            title=f"{Chem.MolToSmiles(Chem.MolFromSmiles(smiles))} in {receptor.stem}",
            subtitle=(
                f"<b>{best.affinity:.2f} kcal/mol</b> &middot; {engine_key} &middot; "
                f"exhaustiveness {exhaustiveness} &middot; {len(poses)} poses"
            ),
            receptor_pdb=receptor_pdb,
            ligand_sdf=pose_sdf,
        )

        pose_mol = Chem.MolFromMolBlock(pose_sdf, removeHs=False)
        strain = (
            await asyncio.to_thread(docking.pose_strain, pose_mol) if pose_mol else None
        )

        return {
            "smiles": smiles,
            "engine": engine_key,
            "binding_affinity": round(best.affinity, 4),
            "units": "kcal/mol",
            "poses": [
                {"rank": p.rank, "affinity": round(p.affinity, 4)} for p in poses
            ],
            "pose_file": str(pose_path),
            "complex_file": str(complex_path),
            "viewer_file": str(viewer_path),
            "receptor_file": str(receptor),
            "binding_site_center": [round(v, 3) for v in center],
            "box_size": [round(v, 3) for v in size],
            "pose_strain": strain,
            "conformer_strain": {
                "local_strain": conformer["local_strain"],
                "ensemble_strain": conformer["ensemble_strain"],
                "force_field": conformer["force_field"],
            },
            "engine_log_tail": "\n".join(log.strip().splitlines()[-8:]),
        }
    except DockingError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
def get_sbdd_metrics(
    smiles: Annotated[str, Field(description="SMILES of the docked ligand")],
    docking_score: Annotated[
        float, Field(description="Binding affinity in kcal/mol, e.g. -9.5")
    ],
) -> dict[str, Any]:
    """Combine a docking score with 2D properties into efficiency metrics.

    Reports binding affinity, Ligand Efficiency (score per heavy atom, target
    below -0.3), Lipophilic Ligand Efficiency (estimated pIC50 minus cLogP),
    cLogP and TPSA. The pIC50 estimate converts the score with 2.303RT at
    298 K; it ranks compounds, it does not predict measured potency.
    """
    try:
        return docking.sbdd_metrics(smiles, docking_score)
    except DockingError as exc:
        raise ToolError(str(exc)) from exc


# Groups that make a linker unusable however good its descriptors look. A
# warhead may legitimately contain any of these, so they are only ever
# reported from the linker region.
LINKER_REACTIVE_GROUPS: tuple[tuple[str, str], ...] = (
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]"),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
    ("isocyanate", "[NX2]=[CX2]=[OX1]"),
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
    ("epoxide", "[CX4]1O[CX4]1"),
    ("aziridine", "[CX4]1N[CX4]1"),
    ("allene", "[CX2]=[CX2]=[CX2]"),
    ("vinyl_ether", "[OX2;!$(O=*)][CX3]=[CX3]"),
    ("enol", "[OX2H][CX3]=[CX3]"),
    ("imine", "[CX3]=[NX2;H1,H0;!$(N-a);!$(N=C-N)]"),
    ("peroxide", "[OX2][OX2]"),
    ("nitroso", "[NX2]=[OX1]"),
)

_LINKER_QUERIES: list[tuple[str, Chem.Mol]] = []


def _linker_queries() -> list[tuple[str, Chem.Mol]]:
    global _LINKER_QUERIES
    if not _LINKER_QUERIES:
        _LINKER_QUERIES = [
            (name, query)
            for name, smarts in LINKER_REACTIVE_GROUPS
            if (query := Chem.MolFromSmarts(smarts)) is not None
        ]
    return _LINKER_QUERIES


def _fragment_query(smiles: str) -> Chem.Mol:
    """A fragment as a substructure query that actually matches.

    Fragment SMILES come from cutting a molecule, and every common encoding
    of a cut breaks a naive query in its own way:

    - **Attachment dummies** (`*`, `[*]`, `[*:1]`) are how PROTAC-DB and
      DiffLinker write fragments, and a dummy atom matches nothing in a
      mol-mol query. They are removed; the bond they marked is the one to
      the linker, which is not part of the fragment anyway.
    - **Open valences** become radical electrons once sanitised, and RDKit
      will not match a radical query atom against a filled-valence target.
      Zeroed.
    - **Explicit `[H]`** survives `sanitize=False` as a heavy query atom and
      then matches nothing. Removed.
    - **Formal charges** must match exactly, so a deprotonated phenol query
      fails against a neutral target. Zeroed.

    Raises rather than returning an unusable query: a fragment that cannot
    be turned into one is the caller's problem, and reporting it as "the
    molecule is a different compound" blames the wrong thing.
    """
    parsed = Chem.MolFromSmiles(smiles, sanitize=False)
    if parsed is None:
        raise ToolError(f"could not parse fragment {smiles!r}")

    query = Chem.RWMol(parsed)
    drop = [
        atom.GetIdx()
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() == 0 or atom.GetAtomicNum() == 1
    ]
    for index in sorted(drop, reverse=True):
        query.RemoveAtom(index)

    for atom in query.GetAtoms():
        atom.SetNumRadicalElectrons(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)

    result = query.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:  # noqa: BLE001 - reported as the caller's problem
        try:
            Chem.SanitizeMol(
                result,
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolError(
                f"fragment {smiles!r} cannot be used as a substructure query: "
                f"{exc}"
            ) from exc

    if not result.GetNumAtoms():
        raise ToolError(f"fragment {smiles!r} has no atoms to match")
    return result


def _span_through_linker(
    mol: Chem.Mol,
    linker: set[int],
    start_gateways: set[int],
    end_gateways: set[int],
) -> int | None:
    """Bonds from one fragment to the other, counted through the linker.

    Breadth-first over linker atoms only, plus one bond at each end for the
    two crossings. Returns None when the two sides are not connected through
    the linker, which is itself the answer for a molecule that fell apart.
    """
    if not start_gateways or not end_gateways:
        return None

    from collections import deque

    distance = {atom: 0 for atom in start_gateways}
    queue = deque(start_gateways)
    while queue:
        current = queue.popleft()
        if current in end_gateways:
            # two crossing bonds, plus the bonds walked inside the linker
            return distance[current] + 2
        for neighbour in mol.GetAtomWithIdx(current).GetNeighbors():
            index = neighbour.GetIdx()
            if index in linker and index not in distance:
                distance[index] = distance[current] + 1
                queue.append(index)
    return None


@mcp.tool
def check_protac_linker(
    smiles: Annotated[str, Field(description="SMILES of the whole PROTAC")],
    fragment_1: Annotated[
        str, Field(description="SMILES of the first fragment, which must be intact")
    ],
    fragment_2: Annotated[
        str, Field(description="SMILES of the second fragment, which must be intact")
    ],
) -> dict[str, Any]:
    """Check a PROTAC keeps both fragments and still bridges them properly.

    Answers the questions a SMILES string cannot be eyeballed for: are both
    fragments present and unaltered, is the linker bonded to them exactly
    twice, how many bonds span the two fragments, and does the linker carry
    a reactive group.

    `fragments_intact` is the one that matters. A PROTAC missing a fragment
    is a different compound: it parses, it sanitises, its descriptors look
    good, and it will not degrade anything. Nothing else measurable catches
    that.

    **A fragment query only identifies a region, not an identity.** If the
    fragment you pass also occurs inside the other fragment -- two IMiD-based
    ligands, say -- it can match there, and the tool will report both
    fragments present for a molecule missing one of them. Such a molecule
    usually fails anyway on the attachment count, but for the wrong stated
    reason. `ambiguous_placements` counts how many disjoint placements
    existed; anything above 1 means the derived linker is a choice rather
    than a deduction, and `fragment_atoms` shows which one was made.

    `ok` covers structural invariants only: both fragments present and
    disjoint, the linker bonded to them exactly twice, and no reactive group
    in the linker. It says nothing about whether the span is right for a
    ternary complex -- `span_bonds` is reported for the caller to judge
    against its own target, because that range is a property of the proteins
    being bridged, not of the molecule. `issues` says what failed in words.
    """
    mol = _require_mol(smiles)
    issues: list[str] = []

    pieces = Chem.GetMolFrags(mol)
    if len(pieces) > 1:
        issues.append(f"the molecule is in {len(pieces)} disconnected pieces")

    # Match each fragment without letting the two matches overlap, so one
    # fragment appearing twice cannot stand in for a missing other.
    # Choose BOTH placements together. Claiming fragment 1 first and taking
    # the first match that fits is wrong twice over: it is order-dependent,
    # and a generic fragment query -- an aryl amide, say, which appears in
    # most real linkers -- lands on the linker's own chemistry before it
    # finds the real warhead, reporting a correct molecule as broken.
    #
    # Among disjoint placements, prefer the one that leaves a single
    # connected linker. That is what a real PROTAC looks like, and it is the
    # signal that separates the right region from a coincidental match.
    queries = (_fragment_query(fragment_1), _fragment_query(fragment_2))
    options = [
        mol.GetSubstructMatches(query, uniquify=True, maxMatches=200)
        for query in queries
    ]

    def leftover_pieces(pair) -> int:
        taken = {atom for match in pair for atom in match}
        rest = [i for i in range(mol.GetNumAtoms()) if i not in taken]
        if not rest:
            return 0
        seen: set[int] = set()
        pieces = 0
        for start in rest:
            if start in seen:
                continue
            pieces += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for neighbour in mol.GetAtomWithIdx(current).GetNeighbors():
                    index = neighbour.GetIdx()
                    if index not in seen and index not in taken:
                        seen.add(index)
                        stack.append(index)
        return pieces

    disjoint = [
        (first, second)
        for first in options[0]
        for second in options[1]
        if not set(first).intersection(second)
    ]
    matched: list[tuple[int, ...]] = []
    if disjoint:
        # One leftover piece is a proper linker; more means the match
        # swallowed part of it. Fewest pieces wins.
        matched = list(min(disjoint, key=leftover_pieces))

    missing: list[int] = []
    if not matched:
        if not options[0]:
            missing.append(1)
        if not options[1]:
            missing.append(2)
        if not missing:
            issues.append(
                "the two fragments cannot both be found without overlapping, "
                "so at least one is not present as a distinct fragment"
            )
    claimed: set[int] = {i for match in matched for i in match}

    fragments_intact = bool(matched)

    # Stereochemistry is checked on the chosen placement, and reported as its
    # own failure. An epimerised warhead is present but wrong, and calling it
    # "deleted" sends the agent to rebuild something that is already there.
    if matched:
        stereo_wrong = []
        for index, query in enumerate(queries, start=1):
            if not mol.GetSubstructMatches(
                query, uniquify=True, useChirality=True, maxMatches=1
            ):
                stereo_wrong.append(index)
        if stereo_wrong:
            fragments_intact = False
            for index in stereo_wrong:
                issues.append(
                    f"fragment {index} is present but its stereochemistry has "
                    "been altered; a warhead's stereochemistry is the reason "
                    "it binds anything"
                )

    if len(disjoint) > 1:
        # Say so rather than pretending the placement was determined. A
        # caller seeing this should treat the derived linker with suspicion.
        result_ambiguous = len(disjoint)
    else:
        result_ambiguous = 0
    for index in missing:
        issues.append(
            f"fragment {index} is not present intact; the molecule is a "
            "different compound, not a repaired linker"
        )

    linker_indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
    result: dict[str, Any] = {
        "ok": False,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "fragments_intact": fragments_intact,
        "fragments_found": len(matched),
        "linker_heavy_atoms": len(linker_indices),
        "linker_smiles": None,
        "attachment_bonds": None,
        "span_bonds": None,
        "reactive_groups": [],
        "fragment_atoms": [list(match) for match in matched],
        "ambiguous_placements": result_ambiguous,
        "issues": issues,
    }

    if not fragments_intact:
        return result

    if not linker_indices:
        # Which diagnosis is right depends on whether the pieces are even
        # bonded: "joined directly" is misleading for a molecule that fell
        # apart, where the linker is missing rather than zero-length.
        issues.append(
            "the fragments are not connected by anything; the linker is gone"
            if len(pieces) > 1
            else "the fragments are joined directly, leaving no linker"
        )
        return result

    linker_set = set(linker_indices)
    # Count BONDS crossing the linker/fragment boundary, not linker atoms
    # that touch a fragment. The two differ exactly where it matters: a
    # one-atom linker bonded to both fragments is a single atom making two
    # bonds, and counting atoms would call it non-bridging.
    fragment_of = {}
    for which, match in enumerate(matched):
        for atom in match:
            fragment_of[atom] = which

    anchors: set[int] = set()
    crossings = 0
    touched: set[int] = set()
    # Linker atoms bonded to each fragment, which is where a span through the
    # linker has to start and end.
    gateways: dict[int, set[int]] = {0: set(), 1: set()}
    for index in linker_indices:
        for neighbour in mol.GetAtomWithIdx(index).GetNeighbors():
            other = neighbour.GetIdx()
            if other not in linker_set:
                crossings += 1
                anchors.add(other)
                which = fragment_of.get(other)
                if which is not None:
                    touched.add(which)
                    gateways[which].add(index)

    result["attachment_bonds"] = crossings
    if crossings != 2:
        issues.append(
            f"the linker is bonded to the fragments {crossings} time(s); a "
            "linker must bridge at exactly two, one per fragment"
        )
    elif len(touched) < 2:
        # Two bonds onto the SAME fragment is not a bridge: the other
        # fragment is fused straight onto the first with no linker between.
        issues.append(
            "both of the linker's bonds go to the same fragment, so it does "
            "not bridge them; the two fragments are joined directly"
        )

    with rdBase.BlockLogs():
        result["linker_smiles"] = Chem.MolFragmentToSmiles(
            mol, atomsToUse=linker_indices, canonical=True
        )

    # Walk the LINKER, not the molecule. GetShortestPath over the whole
    # graph happily routes through a fragment, which reports a short span for
    # a long linker whenever the two fragments are fused or share a ring --
    # the opposite of what the number is for.
    result["span_bonds"] = _span_through_linker(
        mol, linker_set, gateways[0], gateways[1]
    )

    reactive = []
    for name, query in _linker_queries():
        for match in mol.GetSubstructMatches(query):
            if linker_set.issuperset(match):
                reactive.append(name)
                break
    result["reactive_groups"] = reactive
    if reactive:
        issues.append(
            f"the linker contains {', '.join(reactive)}, which reacts with "
            "protein nucleophiles or will not survive as drawn"
        )

    result["ok"] = not issues
    return result


# ---------------------------------------------------------------- ADC linkers
#
# An ADC linker fails for different reasons than a PROTAC linker, so it gets
# its own checks rather than the same ones with different constants.
#
# A PROTAC linker is a GEOMETRIC problem: both ends bind proteins, and the
# span decides whether a ternary complex forms at all. An ADC linker is a
# CHEMICAL problem: the antibody holds the payload, nothing has to reach
# anything, and what decides the outcome is whether the conjugate survives
# circulation, whether the payload is cut loose inside the cell, and whether
# what comes loose is the actual drug. None of those is a distance.

# The 10th-90th percentile band of each property across 1,692 real ADC linker
# reagents: the curated Linker-GPT set, deduplicated from 3,679 catalogue rows
# to 2,764 unique structures, then filtered to the 1,692 carrying no Fmoc,
# Boc, tBu, NHS or p-nitrophenyl group, since those are activating or
# protecting groups that are gone from the installed linker.
#
# **These describe a linker WITHOUT its payload.** That is the form the
# reagents are sold in, and the numbers are meaningful only against a linker
# measured the same way. Against a whole conjugate they are worse than
# useless: MMAE alone is MW 718 with cLogP 4, so every conjugate lands outside
# every band whatever its spacer, and the comparison stops discriminating.
# `check_adc_linker` excises the payload before comparing, which is the only
# reason these numbers say anything at all.
ADC_LINKER_WINDOWS: dict[str, tuple[float, float, float]] = {
    # name: (p10, median, p90)
    "clogp": (-0.81, 0.68, 3.14),
    "tpsa": (66.4, 139.3, 284.8),
    "heavy_atoms": (16.0, 36.0, 85.0),
    "mw": (242.3, 526.1, 1236.5),
    "rotatable_bonds": (9.0, 25.0, 73.0),
    "fsp3": (0.50, 0.88, 1.00),
    "hbd": (0.0, 1.0, 4.0),
    "hba": (5.0, 10.0, 26.0),
}

# The one monotone limit rather than a band, and the only hard threshold
# here. Two independent derivations agree on it: it is the 90th percentile of
# the 1,692 reference reagents, and it sits just above 2.87, the most
# lipophilic linker among the 16 approved ADCs (Mylotarg and Besponsa's
# hydrazone-disulfide). It rejects none of them.
#
# One-sided on purpose. Hydrophobicity is monotone -- a greasier drug-linker
# aggregates more and clears faster, and there is no point at which being
# more polar starts to hurt, because the linker is discarded on release and
# never crosses a membrane. Enhertu's linker measures -1.71, below the
# reference band entirely, and is approved.
LINKER_CLOGP_CEILING = 3.14

# Which of the windows above are worth saying anything about, and which are
# measured but never judged.
#
# This split is not a matter of taste. Each band was checked against the
# linkers of 16 approved ADCs, and two of them reject most of that set:
#
#   fsp3 [0.50-1.00]  rejects 10 of 16, Adcetris, Enhertu, Polivy, Padcev
#                     and Tivdak among them
#   hbd  [0-4]        rejects 9 of 16
#
# The reason is that the reference set is not the same population as a
# finished drug-linker. It is full of bare PEG and alkyl reagents, which is
# why its median Fsp3 is 0.88 -- but an installed cleavable linker carries a
# p-aminobenzyl aromatic ring and a dipeptide, which puts a real one near
# 0.48. Judging a complete linker by a band measured on fragments flags the
# clinical standard of care as abnormal.
#
# The advisory four are wrong for 2 to 3 of 16, which is a tolerable rate for
# something that only ever prints a note. The rest are reported as raw
# numbers and compared against nothing.
JUDGED_PROPERTIES = frozenset({"clogp"})
ADVISORY_PROPERTIES = frozenset({"tpsa", "heavy_atoms", "mw", "rotatable_bonds"})

# Conjugation chemistries, keyed by what they attach to. A handle follows
# from the antibody's conjugation site and is never a design choice, so this
# is a lookup rather than something to generate.
ADC_HANDLES: dict[str, tuple[str, str]] = {
    # name: (SMARTS for the reactive core, what it conjugates to)
    "maleimide": ("O=C1C=CC(=O)N1", "a free cysteine thiol, reversibly"),
    "bromoacetyl": ("[CX4;H2](Br)C(=O)", "a free cysteine thiol, irreversibly"),
    "iodoacetyl": ("[CX4;H2](I)C(=O)", "a free cysteine thiol, irreversibly"),
    "nhs_ester": ("O=C(ON1C(=O)CCC1=O)", "a solvent-exposed lysine amine"),
    "carboxylic_acid": (
        "[CX3](=[OX1])[OX2H1]",
        "a solvent-exposed lysine amine, after activation to an NHS ester at "
        "conjugation time. This is how a lysine-targeting drug-linker is "
        "normally drawn and sold -- as the free acid -- so it counts as a "
        "handle even though it is not itself reactive",
    ),
    "dbco": ("C#Cc1ccccc1", "an engineered azide, strain-promoted"),
    "bcn": ("C1CC2CCC1C#C2", "an engineered azide, strain-promoted"),
    "tetrazine": ("c1nnc(nn1)", "a trans-cyclooctene tag"),
    "azide": ("[NX2]=[NX2+]=[NX1-]", "a strained alkyne"),
}

# What gets CUT. Distinct from how the payload then comes off: Val-Cit is a
# trigger, and the p-aminobenzyl carbamate next to it is the release
# chemistry. A cleavable design needs both, and having one without the other
# is a specific failure that nothing else here would show.
ADC_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("val_cit", "[CX4]([CX4]([CH3])[CH3])C(=O)N[CX4](CCCNC(N)=O)C(=O)"),
    ("citrulline", "[CX4](CCCNC(N)=O)C(=O)"),
    ("val_ala", "[CX4]([CX4]([CH3])[CH3])C(=O)N[CX4]([CH3])C(=O)"),
    ("ala_ala", "[NX3][CX4]([CH3])C(=O)N[CX4]([CH3])C(=O)"),
    ("phe_lys", "[CX4](Cc1ccccc1)C(=O)N[CX4](CCCC[NX3])C(=O)"),
    ("ggfg", "NCC(=O)NCC(=O)N[CX4](Cc1ccccc1)C(=O)NCC(=O)"),
    ("disulfide", "[#6][SX2][SX2][#6]"),
    ("hydrazone", "[CX3]=[NX2][NX3]"),
    ("glucuronide", "OC1OC(C(O)C(O)C1O)C(=O)O"),
    ("carbonate", "[OX2][CX3](=[OX1])[OX2]"),
)

# How the payload comes off, matched across the bond that joins it to the
# rest. Cleavable chemistries come first so a junction that offers one is
# reported as that rather than as the stable bond it also contains.
#
# `releases_free_drug` is the question no descriptor answers. A stable
# junction is not automatically wrong -- trastuzumab emtansine is approved
# and releases Lys-MCC-DM1 rather than DM1 -- but it only works for a payload
# that stays potent as that adduct, and that has to be a decision somebody
# made rather than something discovered in vivo.
RELEASE_CHEMISTRIES: tuple[tuple[str, str, bool, str], ...] = (
    (
        "pab_carbamate",
        # The payload nitrogen, a carbamate, a benzylic CH2, and a ring
        # nitrogen para to it. The para relationship is what permits the
        # 1,6-elimination, so it is in the pattern rather than assumed.
        "[#7;!$([#7][#7])]C(=O)O[CH2]c1ccc([#7])cc1",
        True,
        "self-immolative p-aminobenzyl carbamate: once the trigger is cut the "
        "spacer eliminates and the payload leaves as a free amine",
    ),
    (
        "pab_ether",
        "[OX2;!$(O=*)][CH2]c1ccc([#7])cc1",
        True,
        "self-immolative p-aminobenzyl ether, releasing a free alcohol payload",
    ),
    (
        "aminomethylene",
        # The deruxtecan-style spacer: an N-CH2-O joint that collapses once
        # the peptide in front of it is cut.
        "[NX3;$(NC=O)][CH2][OX2]",
        True,
        "self-immolative aminomethylene spacer, which collapses after the "
        "peptide is cleaved and releases unmodified payload",
    ),
    (
        "disulfide",
        "[#6][SX2][SX2][#6]",
        True,
        "reduced by intracellular glutathione, releasing a thiol payload",
    ),
    (
        "hydrazone",
        "[CX3]=[NX2][NX3]",
        True,
        "hydrolyses in the acidic endosome, releasing a ketone or aldehyde "
        "payload; known to leak in circulation",
    ),
    (
        "glucuronide",
        "OC1OC(C(O)C(O)C1O)C(=O)O",
        True,
        "cut by lysosomal beta-glucuronidase; very polar, which helps at high "
        "drug-to-antibody ratio",
    ),
    (
        "ester",
        "[CX3](=[OX1])[OX2][#6]",
        True,
        "hydrolysable, but by non-specific serum esterases rather than a "
        "lysosomal trigger, so release is not tumour-selective",
    ),
    (
        "amide",
        "[CX3](=[OX1])[NX3]",
        False,
        "a stable amide: it does not hydrolyse on its own, so the payload is "
        "freed only when the antibody itself is proteolysed, and it leaves "
        "carrying the linker remnant",
    ),
    (
        "thioether",
        "[#6][SX2;!$(SS)][#6]",
        False,
        "a stable thioether; the payload comes off only as an adduct, when "
        "the antibody is degraded",
    ),
    (
        "ether",
        "[#6][OX2;!$(OC=O)][#6]",
        False,
        "a stable ether; the payload cannot be released from it",
    ),
)

# Junctions that RDKit sanitises happily and a chemist would not make.
#
# These are checked on the ASSEMBLED molecule, and that is the gap they fill.
# A pipeline that validates its two warheads when it builds them never looks
# again, so a junction created by the GENERATED spacer is never examined. The
# one that bites: a maleimidocaproyl handle ends in an acyl group meant to
# become an amide against the cassette's valine. Hand it a spacer starting
# with a carbon and it becomes a dialkyl ketone -- a reactive carbonyl in the
# middle of the linker, and not a compound anyone would make.
#
# Checked against the LINKER only. A payload is entitled to its own ketones.
BAD_JUNCTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "dialkyl_ketone",
        "[CX4][CX3](=[OX1])[CX4]",
        "a ketone in the linker condenses with lysine amines in vivo; an acyl "
        "handle joined to a carbon-starting spacer makes one where an amide "
        "was intended. Start the spacer with N or O instead",
    ),
    (
        "triacylamine",
        "[NX3]([CX3]=[OX1])([CX3]=[OX1])[CX3]=[OX1]",
        "three acyl groups on one nitrogen is not a stable compound; this is "
        "what acylating a bare maleimide nitrogen produces",
    ),
    (
        "anhydride",
        "[CX3](=[OX1])[OX2][CX3]=[OX1]",
        "an anhydride hydrolyses in plasma within minutes",
    ),
    ("gem_diol", "[CX4]([OX2H])[OX2H]", "a gem-diol is the hydrate of a ketone, not an isolable group"),
    ("acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]", "an acyl halide does not survive water"),
    (
        "peroxide",
        "[OX2][OX2]",
        "a peroxide is not stable enough to administer",
    ),
    (
        "aldehyde",
        "[CX3H1](=O)[#6]",
        "an aldehyde cross-links protein amines",
    ),
    (
        "alpha_ketoamide",
        "[NX3][CX3](=[OX1])[CX3](=[OX1])[#6]",
        "an alpha-ketoamide is electrophilic and hydrolytically fragile. It "
        "appears when a spacer is built onto a carbamate carbonyl, which is "
        "what the spacer_after_cassette architecture hands a generator",
    ),
    (
        "free_thiol",
        "[SX2H]",
        "a free thiol in the linker caps the molecule's own maleimide, "
        "scrambles any disulfide in the cassette, and gives albumin's Cys34 "
        "somewhere to attach. Cap it as a thioether or remove it",
    ),
    (
        "unhindered_disulfide",
        "[CX4;H2,H3][SX2][SX2][CX4;H2,H3]",
        "a disulfide with no substitution on either alpha carbon is reduced "
        "by circulating glutathione long before the conjugate reaches a "
        "cell. The gem-dimethyl hindrance in every clinical disulfide linker "
        "exists for exactly this reason",
    ),
    (
        "aldehyde_derived_hydrazone",
        "[CX3;H1]=[NX2][NX3]",
        "an aldehyde-derived hydrazone hydrolyses far too readily to survive "
        "circulation. The acid-labile ADC trigger is a ketone-derived "
        "acylhydrazone, as in gemtuzumab ozogamicin's AcBut linker",
    ),
)

_ADC_QUERY_CACHE: dict[str, Chem.Mol] = {}


def _adc_query(smarts: str) -> Chem.Mol:
    """Compile once. These are matched per molecule, repeatedly."""
    cached = _ADC_QUERY_CACHE.get(smarts)
    if cached is None:
        cached = Chem.MolFromSmarts(smarts)
        if cached is None:  # our pattern, our bug
            raise ToolError(f"internal: invalid SMARTS {smarts!r}")
        _ADC_QUERY_CACHE[smarts] = cached
    return cached


def _resolve_handle(handle: str) -> tuple[str, str, str]:
    """A handle name or a raw SMARTS, as (name, smarts, what it conjugates to)."""
    key = (handle or "").strip().lower()
    if key in ADC_HANDLES:
        smarts, target = ADC_HANDLES[key]
        return key, smarts, target
    if Chem.MolFromSmarts(handle) is not None:
        return "custom", handle, "whatever you designed it for"
    raise ToolError(
        f"unknown handle {handle!r}: give one of {sorted(ADC_HANDLES)} or a "
        "valid SMARTS for the reactive core"
    )


def _payload_placement(mol: Chem.Mol, payload: str) -> tuple[list[int], str | None]:
    """Where the payload sits in the conjugate, and what is wrong if anything.

    Returns the matched atom indices and a problem code: `payload_lost` when
    it is not there at all, `payload_altered` when it is there with different
    stereochemistry. The two are kept apart deliberately -- an epimer is
    present, and telling a repair agent it was deleted sends it to rebuild
    something that is already in the molecule.
    """
    query = _fragment_query(payload)
    flat = mol.GetSubstructMatch(query)
    if not flat:
        return [], "payload_lost"
    if not mol.GetSubstructMatch(query, useChirality=True):
        return list(flat), "payload_altered"
    return list(flat), None


def _pieces_of(mol: Chem.Mol, indices: list[int]) -> int:
    """How many connected pieces `indices` falls into inside `mol`."""
    remaining = set(indices)
    pieces = 0
    while remaining:
        stack = [remaining.pop()]
        pieces += 1
        while stack:
            current = stack.pop()
            for neighbour in mol.GetAtomWithIdx(current).GetNeighbors():
                index = neighbour.GetIdx()
                if index in remaining:
                    remaining.discard(index)
                    stack.append(index)
    return pieces


def _window_note(name: str, value: float) -> tuple[bool, str]:
    """Is this value inside the reference band, and how to say so."""
    low, median, high = ADC_LINKER_WINDOWS[name]
    if value < low:
        return False, f"{value:g} is below the real-linker 10th percentile ({low:g})"
    if value > high:
        return False, f"{value:g} is above the real-linker 90th percentile ({high:g})"
    return True, f"{value:g}, inside the real-linker band {low:g}-{high:g} (median {median:g})"


@mcp.tool
def check_adc_linker(
    smiles: Annotated[str, Field(description="SMILES of the whole drug-linker: handle, spacer, cassette and payload")],
    payload: Annotated[str, Field(description="SMILES of the payload, which must survive unchanged. Attachment dummies such as * or [*:1] are fine")],
    handle: Annotated[str, Field(description="the conjugation handle: a name (maleimide, nhs_ester, dbco, bromoacetyl, iodoacetyl, bcn, tetrazine, azide) or a SMARTS for its reactive core")] = "maleimide",
    require_cleavable: Annotated[bool, Field(description="whether this design is meant to release its payload by a cleavage trigger. Set false for a deliberately non-cleavable linker such as trastuzumab emtansine's")] = True,
) -> dict[str, Any]:
    """Check an ADC drug-linker is still a conjugatable, releasable molecule.

    This is the ADC counterpart of `check_protac_linker`, and it asks
    different questions because ADC linkers fail differently. There is no
    span to satisfy: the antibody holds the payload and nothing has to reach
    anything. What matters instead is whether the thing can be attached to an
    antibody at all, whether the payload survived, and whether it can ever
    come back off.

    What it checks, in the order they matter:

    **The payload is intact, stereochemistry included.** MMAE has ten
    stereocentres and its potency depends on all of them, so an epimer is a
    different drug. Reported as `payload_altered` rather than `payload_lost`,
    because an epimerised payload is present, and telling you it was deleted
    would send you to rebuild something already there.

    **The payload is terminal.** Excising it must leave one piece. A payload
    in the middle of the chain cannot be released by cutting the linker, no
    matter how good the cassette is.

    **Exactly one conjugation handle.** None and it cannot be attached to the
    antibody; two and it crosslinks two antibodies into an aggregate.

    **A cleavage trigger is present**, when the design is meant to have one.

    **The junction chemistry is real.** Checked on the assembled molecule and
    on the linker only, which is where a generator's spacer creates things
    nobody designed -- most often a dialkyl ketone, from joining an acyl
    handle to a spacer that starts with a carbon where an amide was intended.

    **The linker, without its payload, looks like a real linker.** Every
    property is measured on the payload-free portion and compared against
    1,692 real linker reagents. Measuring the conjugate instead is a
    calibration error: the payload is a constant that puts every candidate
    outside every band.

    `ok` covers all of the above. `linker_smiles` is the payload-free portion
    that was actually measured -- read it, because if the excision went wrong
    every property below it is describing the wrong molecule.
    """
    mol = _require_mol(smiles)
    issues: list[str] = []
    advisories: list[str] = []

    pieces = Chem.GetMolFrags(mol)
    if len(pieces) > 1:
        issues.append(f"the molecule is in {len(pieces)} disconnected pieces")

    payload_atoms, payload_problem = _payload_placement(mol, payload)
    if payload_problem == "payload_lost":
        issues.append(
            "the payload is not present intact; this is a different compound, "
            "not a repaired linker"
        )
    elif payload_problem == "payload_altered":
        issues.append(
            "the payload is present but its stereochemistry has been changed; "
            "a payload's stereochemistry is the reason it is potent, so this "
            "is a different drug"
        )

    linker_indices = [i for i in range(mol.GetNumAtoms()) if i not in set(payload_atoms)]
    linker_set = set(linker_indices)
    payload_terminal = None
    if payload_atoms and linker_indices:
        payload_terminal = _pieces_of(mol, linker_indices) == 1
        if not payload_terminal:
            issues.append(
                "the payload sits between two parts of the linker rather than "
                "at its end, so cutting the linker cannot release it"
            )

    handle_name, handle_smarts, handle_target = _resolve_handle(handle)
    handle_matches = mol.GetSubstructMatches(_adc_query(handle_smarts))
    handle_atoms = {atom for match in handle_matches for atom in match}
    if not handle_matches:
        issues.append(
            f"no {handle_name} handle is present, so this cannot be "
            f"conjugated to {handle_target}"
        )
    elif len(handle_matches) > 1:
        issues.append(
            f"{len(handle_matches)} {handle_name} handles are present; a "
            "second one crosslinks two antibodies into an aggregate"
        )

    # A trigger must not lie wholly INSIDE the payload: a bond within the drug
    # does not release the drug, and searching the whole molecule otherwise
    # reports DM1's own ansa-ring carbamate as a cleavage trigger.
    #
    # Spanning the boundary is allowed, and has to be. In a disulfide-linked
    # ADC the disulfide IS the junction -- mirvetuximab soravtansine's DM4
    # contributes one of the two sulfurs -- so requiring the trigger to sit
    # wholly in the linker rejects an approved drug.
    payload_set = set(payload_atoms)
    triggers = [
        name
        for name, smarts in ADC_TRIGGERS
        if any(
            not payload_set.issuperset(match)
            for match in mol.GetSubstructMatches(_adc_query(smarts))
        )
    ]
    if require_cleavable and not triggers:
        issues.append(
            "no cleavage trigger is present, so the payload can never be "
            "released; add a protease substrate such as Val-Cit or Val-Ala, a "
            "disulfide, or a glucuronide"
        )

    # Junctions, on the linker only. The payload keeps its own chemistry.
    bad_junctions: list[str] = []
    for name, smarts, why in BAD_JUNCTIONS:
        for match in mol.GetSubstructMatches(_adc_query(smarts)):
            if not linker_set.issuperset(match):
                continue
            if handle_atoms.intersection(match):
                continue  # the handle is meant to be reactive
            bad_junctions.append(name)
            issues.append(f"the linker contains a {name}: {why}")
            break

    result: dict[str, Any] = {
        "ok": False,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "payload_intact": payload_problem is None,
        "payload_problem": payload_problem,
        "payload_terminal": payload_terminal,
        "handle": handle_name,
        "handle_conjugates_to": handle_target,
        "handle_count": len(handle_matches),
        "cleavage_triggers": triggers,
        "bad_junctions": bad_junctions,
        "linker_smiles": None,
        "linker_properties": {},
        "outside_reference": [],
        "issues": issues,
        "advisories": advisories,
    }

    if not payload_atoms or not linker_indices:
        if not issues:
            issues.append("the payload could not be located, so the linker "
                          "could not be measured")
        return result

    with rdBase.BlockLogs():
        linker_smiles = Chem.MolFragmentToSmiles(
            mol, atomsToUse=linker_indices, canonical=True
        )
        linker_mol = Chem.MolFromSmiles(linker_smiles)
    result["linker_smiles"] = linker_smiles
    if linker_mol is None:
        advisories.append(
            "the payload-free portion could not be re-parsed on its own, so "
            "its properties are not reported; the structural checks above "
            "still hold"
        )
        result["ok"] = not issues
        return result

    properties = {
        "heavy_atoms": int(linker_mol.GetNumHeavyAtoms()),
        "mw": round(float(Descriptors.MolWt(linker_mol)), 2),
        "clogp": round(float(Crippen.MolLogP(linker_mol)), 3),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(linker_mol)), 2),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(linker_mol)),
        "fsp3": round(float(rdMolDescriptors.CalcFractionCSP3(linker_mol)), 3),
        "hbd": int(rdMolDescriptors.CalcNumHBD(linker_mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(linker_mol)),
        "sa_score": round(float(sascorer.calculateScore(linker_mol)), 3),
    }
    result["linker_properties"] = properties

    # Only the properties worth judging are compared. fsp3, hbd and hba are
    # measured and returned above, and deliberately not compared against
    # anything -- their reference bands reject most of the approved ADCs,
    # because the reference set is bare reagents and a finished linker
    # carries an aromatic spacer and a peptide.
    outside = []
    for name in ADVISORY_PROPERTIES | JUDGED_PROPERTIES:
        value = float(properties[name])
        # cLogP is judged against a ceiling, not a band, so being below the
        # band is not a deviation worth reporting -- it is the good
        # direction. Enhertu's linker is at -1.71 and is approved.
        if name == "clogp":
            if value > LINKER_CLOGP_CEILING:
                outside.append(
                    {
                        "property": "clogp",
                        "note": f"{value:g} is above the ceiling of "
                        f"{LINKER_CLOGP_CEILING:g}",
                    }
                )
            continue
        inside, note = _window_note(name, value)
        if not inside:
            outside.append({"property": name, "note": note})
    result["outside_reference"] = outside

    # The one hard limit. Everything else is a band to notice, this is a
    # ceiling to respect: drug-linker hydrophobicity drives aggregation and
    # accelerated clearance, and it is the failure mode that kills ADCs in
    # animals rather than in a flask.
    if properties["clogp"] > LINKER_CLOGP_CEILING:
        issues.append(
            f"the linker's cLogP is {properties['clogp']}, above the ceiling "
            f"of {LINKER_CLOGP_CEILING} (the 90th percentile of real ADC "
            "linkers). Hydrophobic drug-linkers aggregate and are cleared "
            "fast in vivo. Add ether oxygens, a PEG stretch, or a charged "
            "group such as a sulfonate"
        )
    for entry in outside:
        if entry["property"] != "clogp":
            advisories.append(f"linker {entry['property']}: {entry['note']}")

    result["ok"] = not issues
    return result


@mcp.tool
def check_payload_release(
    smiles: Annotated[str, Field(description="SMILES of the whole drug-linker")],
    payload: Annotated[str, Field(description="SMILES of the payload. Attachment dummies such as * or [*:1] are fine")],
) -> dict[str, Any]:
    """Work out how, and whether, the payload ever comes off.

    An ADC that binds, internalises and traffics correctly still does nothing
    if the payload stays bolted to the linker. That is invisible in every
    descriptor and in the SMILES string, and it is the failure this answers.

    The bond joining the payload to the rest of the molecule is classified
    against the release chemistries that actually work in a lysosome. The
    distinction that matters is `releases_free_drug`:

    - **True** means the payload leaves as itself. A p-aminobenzyl carbamate
      next to a cut peptide eliminates and gives back the free amine; a
      disulfide is reduced by glutathione; an aminomethylene spacer collapses.
    - **False** means the payload leaves carrying a piece of linker, which
      happens only when the antibody is proteolysed. That is not automatically
      wrong -- trastuzumab emtansine is approved and releases Lys-MCC-DM1
      rather than DM1 -- but it works only for a payload that stays potent as
      that adduct, and the number of payloads that do is small.

    A trigger without a release chemistry, or a release chemistry without a
    trigger, is reported explicitly. Both are real designs that silently do
    nothing: a Val-Cit cassette bonded straight to a payload amide gets cut
    and leaves the payload still carrying the citrulline.
    """
    mol = _require_mol(smiles)
    payload_atoms, payload_problem = _payload_placement(mol, payload)
    if not payload_atoms:
        raise ToolError(
            "the payload is not present in this molecule, so there is no "
            "release chemistry to analyse"
        )

    payload_set = set(payload_atoms)
    junction_bonds = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin in payload_set) != (end in payload_set):
            junction_bonds.append((begin, end))

    if not junction_bonds:
        raise ToolError(
            "the payload is not bonded to anything else; this molecule is "
            "the bare payload, not a drug-linker"
        )

    # Classify at the junction: a match counts only if it spans the payload
    # boundary. Searching the whole molecule would find the cassette's own
    # amides and report release chemistry that has nothing to do with the
    # payload.
    junction_atoms = {atom for bond in junction_bonds for atom in bond}
    found: list[dict[str, Any]] = []
    for name, smarts, frees, note in RELEASE_CHEMISTRIES:
        for match in mol.GetSubstructMatches(_adc_query(smarts)):
            match_set = set(match)
            spans = bool(match_set & payload_set) and bool(match_set - payload_set)
            if not spans or not (match_set & junction_atoms):
                continue
            found.append(
                {"chemistry": name, "releases_free_drug": frees, "note": note}
            )
            break

    triggers = [
        name for name, smarts in ADC_TRIGGERS
        if mol.HasSubstructMatch(_adc_query(smarts))
    ]

    releasing = [f for f in found if f["releases_free_drug"]]

    # A peptide trigger that reaches the junction frees the payload by being
    # cut itself, with no spacer because none is needed -- Decnupaz's Ala-Ala
    # is bonded straight to its payload and is approved. Treating the
    # junction amide as non-releasing would reject that.
    #
    # The caveat is real but is a caveat, not a gate: cathepsin cleaves a
    # dipeptide-drug amide slowly when the payload is bulky, and that is
    # precisely why the p-aminobenzyl spacer was introduced. Said as an
    # advisory below rather than enforced.
    direct = [
        name
        for name, smarts in ADC_TRIGGERS
        if any(
            set(match) & junction_atoms
            for match in mol.GetSubstructMatches(_adc_query(smarts))
        )
    ]
    advisories: list[str] = []
    if direct and not releasing:
        releasing = [
            {
                "chemistry": f"{direct[0]}_at_the_junction",
                "releases_free_drug": True,
                "note": "the trigger is bonded straight to the payload, so "
                "cutting it releases the drug with no spacer needed",
            }
        ]
        found = found + releasing
        advisories.append(
            "the trigger is bonded directly to the payload with no "
            "self-immolative spacer. That releases free drug and is how "
            "Decnupaz works, but a protease cleaves a dipeptide-drug amide "
            "slowly when the payload is bulky -- it is the reason the "
            "p-aminobenzyl spacer exists. Consider inserting one"
        )

    issues: list[str] = []

    if not found:
        issues.append(
            "the bond holding the payload matches no release chemistry this "
            "tool recognises. Say what it is and why it would be cut, or "
            "replace it with a known one"
        )
    elif not releasing:
        stable = ", ".join(f["chemistry"] for f in found)
        article = "an" if stable[:1] in "aeiou" else "a"
        issues.append(
            f"the payload is held by {article} {stable}, which nothing in a cell "
            "cuts selectively. It will be freed only when the antibody is "
            "proteolysed, and it will leave carrying the linker remnant. "
            "That is a valid design only if this payload is potent as that "
            "adduct -- it is how trastuzumab emtansine works -- and a "
            "mistake otherwise"
        )

    # A p-aminobenzyl spacer that is wired wrong does nothing, and looks
    # exactly like one that is wired right. Three ways it fails silently,
    # each checked only when a benzylic spacer is actually present.
    benzylic = _adc_query("[CH2;$([CH2][OX2,NX3])]c1ccc([#7])cc1")
    if mol.HasSubstructMatch(benzylic):
        if mol.HasSubstructMatch(
            _adc_query(
                "[NX3;!$([NX3][CX3]=[OX1]);!$([NX3+])]"
                "c1ccc([CH2][OX2][CX3]=[OX1])cc1"
            )
        ):
            issues.append(
                "the p-aminobenzyl spacer's aniline nitrogen is not an "
                "amide, so no protease can arm it. The 1,6-elimination "
                "starts when that amide is cleaved and the nitrogen's lone "
                "pair is freed; a free amine, a sulfonamide or a urea there "
                "never triggers"
            )
        if mol.HasSubstructMatch(
            _adc_query(
                "[CX3](=[OX1])[NX3;H1]c1ccc([CH2]"
                "[NX3;!$([NX3][CX3]=[OX1]);!$([NX4+])])cc1"
            )
        ):
            issues.append(
                "the payload is joined to the p-aminobenzyl spacer as a "
                "benzylamine rather than through a carbamate. A neutral "
                "amine is a poor leaving group at a benzylic position; the "
                "carbamate exists because it fragments to carbon dioxide "
                "plus the free payload amine. Draw it Ar-CH2-O-C(=O)-N"
                "(payload)"
            )

    if triggers and not releasing:
        issues.append(
            f"there is a cleavage trigger ({', '.join(triggers)}) but no "
            "self-immolative or cleavable chemistry between it and the "
            "payload, so cutting the trigger leaves the payload still "
            "attached to part of the cassette. Insert a p-aminobenzyl "
            "carbamate or an aminomethylene spacer"
        )
    if releasing and not triggers:
        issues.append(
            "there is release chemistry at the payload but no trigger to set "
            "it off, so nothing initiates the release inside the cell"
        )

    return {
        "ok": not issues,
        "payload_problem": payload_problem,
        "attachment_bonds": len(junction_bonds),
        "release_chemistries": found,
        "releases_free_drug": bool(releasing),
        "cleavage_triggers": triggers,
        "advisories": advisories,
        "released_species": (
            "the payload itself, unmodified"
            if releasing
            else "the payload still carrying part of the linker"
        ),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MolBench MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=os.environ.get("MOLBENCH_TRANSPORT", "stdio"),
        help="stdio (default) for client-launched use, http for a long-lived service",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8077, help="HTTP port")
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdout carries the protocol, so keep the banner out of the way.
        mcp.run(show_banner=False)
    else:
        mcp.run(transport="http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
