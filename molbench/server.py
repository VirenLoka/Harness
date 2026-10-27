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
import os
import re
import sys
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


def _parse_smiles(smiles: str) -> tuple[Chem.Mol | None, list[str]]:
    """Parse SMILES, capturing whatever RDKit wrote to its error log."""
    if not smiles or not smiles.strip():
        raise ToolError("smiles must be a non-empty string")
    with rdBase.CaptureErrorLog() as capture:
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
        with rdBase.CaptureErrorLog() as capture:
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
        with rdBase.CaptureErrorLog() as capture:
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
    """A fragment as a substructure query, tolerant of open valences.

    Fragment SMILES come from cutting a molecule, so they often will not
    sanitise on their own -- an amide nitrogen that lost its acyl partner is
    not a legal molecule but is a perfectly good query.
    """
    query = Chem.MolFromSmiles(smiles, sanitize=False)
    if query is None:
        raise ToolError(f"could not parse fragment {smiles!r}")
    try:
        Chem.SanitizeMol(query)
    except Exception:  # noqa: BLE001 - an unsanitisable query still matches
        pass
    return query


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
    # Search both assignments rather than taking the first match that fits.
    # Greedy first-fit is order-dependent: a fragment with a repeated motif --
    # and a thalidomide-derived E3 ligand has several -- can claim atoms that
    # block the other fragment, reporting a perfectly good repair as a
    # deleted fragment.
    #
    # useChirality is on, so an epimerised fragment is a *different* fragment.
    # Without it, inverting a stereocentre in the warhead passes as intact.
    options_1 = mol.GetSubstructMatches(
        _fragment_query(fragment_1), uniquify=True, useChirality=True
    )
    options_2 = mol.GetSubstructMatches(
        _fragment_query(fragment_2), uniquify=True, useChirality=True
    )

    matched: list[tuple[int, ...]] = []
    for first in options_1:
        first_set = set(first)
        second = next((m for m in options_2 if not first_set.intersection(m)), None)
        if second is not None:
            matched = [first, second]
            break

    missing: list[int] = []
    if not matched:
        if not options_1:
            missing.append(1)
        if not options_2:
            missing.append(2)
        if not missing:
            # Both occur, but never disjointly: they overlap, so one of them
            # is not really there as a separate piece.
            issues.append(
                "the two fragments cannot both be found without overlapping, "
                "so at least one is not present as a distinct fragment"
            )
    claimed: set[int] = {i for match in matched for i in match}

    fragments_intact = bool(matched)
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
