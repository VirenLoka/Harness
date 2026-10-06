"""Structure-based drug design helpers: receptor prep, conformers, docking, metrics.

Everything here is engine-agnostic plumbing used by the MCP tools in
`server.py`. External programs run through `asyncio.create_subprocess_exec` so
a long dock never blocks the server loop, and RDKit's CPU-bound work is pushed
to a worker thread for the same reason.

Binaries are resolved from an environment override, then the project's
`.conda-dock/bin`, then PATH:

    MOLBENCH_VINA_BIN, MOLBENCH_SMINA_BIN, MOLBENCH_OBABEL_BIN,
    MOLBENCH_HADDOCK3_BIN
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors

SERVER_DIR = Path(__file__).resolve().parent
RECEPTOR_DIR = Path(os.environ.get("MOLBENCH_RECEPTOR_DIR") or SERVER_DIR / "receptors")
DOCKING_DIR = Path(os.environ.get("MOLBENCH_DOCKING_DIR") or SERVER_DIR / "docking")

PDB_ID = re.compile(r"^[0-9A-Za-z]{4}$")
UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
SOLVENT_RESIDUES = frozenset({"HOH", "WAT", "DOD", "TIP", "SOL"})

# ΔG = -RT ln K, so pK = -ΔG / (2.303 RT); 2.303 RT = 1.364 kcal/mol at 298 K.
KCAL_PER_LOG_UNIT = 1.364
DEFAULT_TIMEOUT_S = 600.0


class DockingError(RuntimeError):
    """A failure worth reporting to the caller verbatim."""


@dataclass(frozen=True)
class Pose:
    """One docked pose and its predicted affinity."""

    rank: int
    affinity: float
    block: str  # SDF or PDBQT text for this pose alone


# --------------------------------------------------------------------- binaries


def resolve_binary(name: str) -> Path | None:
    """Find a docking-related executable, preferring explicit configuration."""
    override = os.environ.get(f"MOLBENCH_{name.upper()}_BIN")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    bundled = SERVER_DIR / ".conda-dock" / "bin" / name
    if bundled.is_file():
        return bundled
    found = shutil.which(name)
    return Path(found) if found else None


def require_binary(name: str) -> Path:
    binary = resolve_binary(name)
    if binary is None:
        raise DockingError(
            f"{name} was not found. Install it (conda create -p .conda-dock "
            f"-c conda-forge smina vina) or set MOLBENCH_{name.upper()}_BIN to its path."
        )
    return binary


async def run_process(
    cmd: Sequence[str], *, timeout: float = DEFAULT_TIMEOUT_S, cwd: Path | None = None
) -> tuple[str, str]:
    """Run a subprocess off the event loop, raising on failure or timeout."""
    process = await asyncio.create_subprocess_exec(
        *[str(part) for part in cmd],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise DockingError(
            f"{Path(cmd[0]).name} timed out after {timeout:.0f}s"
        ) from None
    if process.returncode != 0:
        detail = (
            stderr.decode(errors="replace") or stdout.decode(errors="replace")
        ).strip()
        raise DockingError(
            f"{Path(cmd[0]).name} failed ({process.returncode}): {detail[:400]}"
        )
    return stdout.decode(errors="replace"), stderr.decode(errors="replace")


def safe_name(value: str, fallback: str) -> str:
    cleaned = UNSAFE_NAME_CHARS.sub("_", (value or "").strip()).strip("._-")[:64]
    return cleaned or fallback


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")[:-3]


# --------------------------------------------------------------------- receptor


def fetch_pdb(pdb_id: str) -> str:
    """Download a PDB entry from RCSB, caching it under the receptor directory."""
    if not PDB_ID.match(pdb_id):
        raise DockingError(
            f"pdb_id must be four alphanumeric characters, got {pdb_id!r}"
        )
    code = pdb_id.upper()
    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    cached = RECEPTOR_DIR / f"{code}.pdb"
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    url = f"https://files.rcsb.org/download/{code}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise DockingError(f"could not fetch {code} from RCSB: {exc}") from exc
    cached.write_text(text, encoding="utf-8")
    return text


def _atom_xyz(line: str) -> tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def ligand_centroid(
    pdb_text: str, resname: str
) -> tuple[tuple[float, float, float], int]:
    """Centroid of the first copy of a co-crystallized ligand.

    Picked before stripping, which is the whole point: the ligand marks the
    pocket the box should cover.
    """
    wanted = resname.strip().upper()
    chosen_key: tuple[str, str] | None = None
    coords: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM") or line[17:20].strip().upper() != wanted:
            continue
        key = (line[21:22], line[22:27])  # chain, residue sequence number
        if chosen_key is None:
            chosen_key = key
        if key == chosen_key:
            coords.append(_atom_xyz(line))
    if not coords:
        raise DockingError(f"no HETATM residue named {wanted!r} found in the structure")
    count = len(coords)
    return (
        (
            sum(c[0] for c in coords) / count,
            sum(c[1] for c in coords) / count,
            sum(c[2] for c in coords) / count,
        ),
        count,
    )


def strip_structure(
    pdb_text: str, keep_hetatms: Iterable[str] = ()
) -> tuple[str, dict[str, int]]:
    """Keep the protein (and any requested heteroatoms); drop water and ligands."""
    keep = {name.strip().upper() for name in keep_hetatms if name.strip()}
    kept: list[str] = []
    counts = {"atoms": 0, "waters_removed": 0, "hetatms_removed": 0, "hetatms_kept": 0}
    for line in pdb_text.splitlines():
        record = line[:6]
        if record.startswith("ATOM"):
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue  # one conformation is enough for a rigid receptor
            kept.append(line)
            counts["atoms"] += 1
        elif record.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            if resname in SOLVENT_RESIDUES:
                counts["waters_removed"] += 1
            elif resname in keep:
                kept.append(line)
                counts["hetatms_kept"] += 1
            else:
                counts["hetatms_removed"] += 1
        elif record.startswith("TER"):
            kept.append(line)
    if counts["atoms"] == 0:
        raise DockingError("no protein ATOM records survived stripping")
    return "\n".join(kept) + "\nEND\n", counts


async def prepare_receptor_pdbqt(
    protein_pdb: str, destination: Path, ph: float = 7.4
) -> None:
    """Add hydrogens and Gasteiger charges, then write a rigid receptor PDBQT."""
    obabel = require_binary("obabel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="molbench-receptor-") as tmp:
        source = Path(tmp) / "protein.pdb"
        source.write_text(protein_pdb, encoding="utf-8")
        # -xr writes a rigid receptor; -p adds hydrogens for the given pH and
        # Open Babel assigns Gasteiger charges when writing PDBQT.
        await run_process(
            [obabel, source, "-O", destination, "-xr", "-p", str(ph)],
            timeout=DEFAULT_TIMEOUT_S,
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise DockingError("receptor preparation produced an empty PDBQT file")


def box_sidecar_path(receptor_path: Path) -> Path:
    return receptor_path.with_suffix(".box.json")


def write_box_sidecar(receptor_path: Path, payload: dict[str, Any]) -> Path:
    sidecar = box_sidecar_path(receptor_path)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def read_box_sidecar(receptor_path: Path) -> dict[str, Any] | None:
    sidecar = box_sidecar_path(receptor_path)
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# -------------------------------------------------------------------- conformer


def _force_field(mol: Chem.Mol, conf_id: int):
    """MMFF94 where the molecule is parameterized, UFF otherwise."""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        if field is not None:
            return field, "MMFF94"
    field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if field is None:
        raise DockingError("no force field could be assigned to this molecule")
    return field, "UFF"


def embed_conformers(
    smiles: str, *, seed: int = 42, num_conformers: int = 20, max_iters: int = 500
) -> dict[str, Any]:
    """Embed with ETKDGv3, optimize, and measure how strained the result is.

    `local_strain` is the embedded geometry's energy above its own relaxed
    minimum. `ensemble_strain` is how far that relaxed structure sits above the
    best minimum found across the whole conformer ensemble, which is the
    practical stand-in for a global minimum.
    """
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise DockingError(f"invalid SMILES {smiles!r}")
    mol = Chem.AddHs(parsed)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol, numConfs=max(1, num_conformers), params=params
            )
        )
    except (ValueError, RuntimeError) as exc:
        # Geometries RDKit cannot satisfy, such as prismane, surface as a
        # ValueError or as a RuntimeError invariant violation from the
        # optimizer. Both mean the same thing to a caller.
        raise DockingError(
            f"Molecule too strained to generate 3D conformer: {exc}"
        ) from exc
    if not conf_ids:
        raise DockingError(
            "Molecule too strained to generate 3D conformer: ETKDGv3 embedding "
            "produced no conformers. Try a larger num_conformers or a less "
            "constrained structure."
        )

    pre_energies: dict[int, float] = {}
    post_energies: dict[int, float] = {}
    field_name = "MMFF94"
    try:
        for conf_id in conf_ids:
            field, field_name = _force_field(mol, conf_id)
            pre_energies[conf_id] = float(field.CalcEnergy())
        for conf_id in conf_ids:
            field, field_name = _force_field(mol, conf_id)
            field.Minimize(maxIts=max_iters)
            post_energies[conf_id] = float(field.CalcEnergy())
    except (ValueError, RuntimeError) as exc:
        raise DockingError(f"Molecule too strained to optimize in 3D: {exc}") from exc

    best_id = min(post_energies, key=lambda cid: post_energies[cid])
    first_id = conf_ids[0]
    global_min = post_energies[best_id]

    single = Chem.Mol(mol)
    single.RemoveAllConformers()
    single.AddConformer(mol.GetConformer(best_id), assignId=True)

    return {
        "mol": single,
        "sdf": Chem.MolToMolBlock(single),
        "force_field": field_name,
        "num_conformers": len(conf_ids),
        "local_strain": round(pre_energies[first_id] - post_energies[first_id], 4),
        "ensemble_strain": round(post_energies[first_id] - global_min, 4),
        "embedded_energy": round(pre_energies[first_id], 4),
        "minimized_energy": round(post_energies[first_id], 4),
        "global_min_energy": round(global_min, 4),
    }


def pose_strain(pose_mol: Chem.Mol, *, max_iters: int = 500) -> dict[str, Any] | None:
    """Energy of a docked pose above its own relaxed minimum.

    This is the strain your spec describes: the bound conformation measured
    against a minimum, available only once a pose exists.
    """
    try:
        mol = Chem.AddHs(pose_mol, addCoords=True)
        field, field_name = _force_field(mol, 0)
        bound = float(field.CalcEnergy())
        field.Minimize(maxIts=max_iters)
        relaxed = float(field.CalcEnergy())
    except (DockingError, ValueError, RuntimeError):
        return None
    return {
        "force_field": field_name,
        "bound_energy": round(bound, 4),
        "relaxed_energy": round(relaxed, 4),
        "pose_strain": round(bound - relaxed, 4),
    }


def ligand_pdbqt(mol: Chem.Mol) -> str:
    """Meeko's AutoDock-native ligand preparation."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise DockingError("Meeko could not prepare this ligand")
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise DockingError(f"Meeko could not write a PDBQT: {error}")
    return text


# ---------------------------------------------------------------------- docking


def parse_vina_pdbqt(text: str) -> list[Pose]:
    """Split Vina's multi-model PDBQT into poses with their reported affinity."""
    poses: list[Pose] = []
    current: list[str] = []
    affinity: float | None = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current, affinity = [], None
        if "VINA RESULT" in line:
            parts = line.split()
            if len(parts) >= 4:
                affinity = float(parts[3])
        current.append(line)
        if line.startswith("ENDMDL"):
            if affinity is not None:
                poses.append(
                    Pose(
                        rank=len(poses) + 1, affinity=affinity, block="\n".join(current)
                    )
                )
            current, affinity = [], None
    return poses


def parse_smina_sdf(path: Path) -> list[Pose]:
    """smina writes each pose with a `minimizedAffinity` property."""
    poses: list[Pose] = []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    for index, mol in enumerate(supplier):
        if mol is None or not mol.HasProp("minimizedAffinity"):
            continue
        poses.append(
            Pose(
                rank=index + 1,
                affinity=float(mol.GetProp("minimizedAffinity")),
                block=Chem.MolToMolBlock(mol, kekulize=False),
            )
        )
    return poses


def _box_args(center: Sequence[float], size: Sequence[float]) -> list[str]:
    return [
        "--center_x",
        f"{center[0]:.3f}",
        "--center_y",
        f"{center[1]:.3f}",
        "--center_z",
        f"{center[2]:.3f}",
        "--size_x",
        f"{size[0]:.3f}",
        "--size_y",
        f"{size[1]:.3f}",
        "--size_z",
        f"{size[2]:.3f}",
    ]


async def dock_vina(
    *,
    ligand_pdbqt_text: str,
    receptor: Path,
    center: Sequence[float],
    size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    timeout: float,
    workdir: Path,
) -> tuple[list[Pose], str]:
    binary = require_binary("vina")
    ligand = workdir / "ligand.pdbqt"
    ligand.write_text(ligand_pdbqt_text, encoding="utf-8")
    out = workdir / "poses.pdbqt"
    stdout, _ = await run_process(
        [
            binary,
            "--receptor",
            receptor,
            "--ligand",
            ligand,
            "--out",
            out,
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            *_box_args(center, size),
        ],
        timeout=timeout,
    )
    if not out.is_file():
        raise DockingError("Vina produced no output poses")
    return parse_vina_pdbqt(out.read_text(encoding="utf-8")), stdout


async def dock_smina(
    *,
    ligand_pdbqt_text: str,
    receptor: Path,
    center: Sequence[float],
    size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    timeout: float,
    workdir: Path,
) -> tuple[list[Pose], str]:
    binary = require_binary("smina")
    ligand = workdir / "ligand.pdbqt"
    ligand.write_text(ligand_pdbqt_text, encoding="utf-8")
    out = workdir / "poses.sdf"
    stdout, _ = await run_process(
        [
            binary,
            "-r",
            receptor,
            "-l",
            ligand,
            "-o",
            out,
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            *_box_args(center, size),
        ],
        timeout=timeout,
    )
    if not out.is_file():
        raise DockingError("smina produced no output poses")
    return parse_smina_sdf(out), stdout


async def dock_haddock3(**_: Any) -> tuple[list[Pose], str]:
    """Optional hook; HADDOCK3 needs CNS, which is licensed separately."""
    binary = resolve_binary("haddock3")
    raise DockingError(
        "HADDOCK3 is not wired up as a runner. "
        + (
            f"A binary was found at {binary}, but driving it needs a workflow "
            "configuration and a CNS installation, so this hook intentionally "
            "stops here rather than pretending to run."
            if binary
            else "No haddock3 binary was found; set MOLBENCH_HADDOCK3_BIN if you "
            "have one. Use engine='vina' or 'smina' for iterative docking."
        )
    )


async def pdbqt_pose_to_sdf(pose_block: str, workdir: Path) -> str:
    """Convert one Vina pose to SDF so RDKit and the viewer can read it."""
    obabel = require_binary("obabel")
    source = workdir / "pose.pdbqt"
    source.write_text(pose_block, encoding="utf-8")
    target = workdir / "pose.sdf"
    await run_process([obabel, source, "-O", target], timeout=120)
    return target.read_text(encoding="utf-8") if target.is_file() else ""


def receptor_display_pdb(
    receptor_pdbqt: Path, sidecar: dict[str, Any] | None = None
) -> str:
    """Protein coordinates for display, preferring the original PDB numbering.

    Open Babel renumbers residues sequentially when it writes a PDBQT, so a
    viewer built from that file would label the pocket wrongly. The stripped
    source PDB saved during preparation keeps the deposited numbering.
    """
    protein_path = (sidecar or {}).get("protein_pdb_file")
    if protein_path:
        candidate = Path(protein_path)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return "\n".join(
        line[:66]  # drop PDBQT's charge and atom-type columns
        for line in receptor_pdbqt.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    )


def build_complex_pdb(receptor_pdb: str, ligand_sdf: str) -> str:
    """Receptor plus docked ligand in one PDB, for viewers and downstream tools."""
    lines = [
        line
        for line in receptor_pdb.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    mol = Chem.MolFromMolBlock(ligand_sdf, removeHs=False, sanitize=False)
    if mol is not None:
        block = Chem.MolToPDBBlock(mol, flavor=4)
        for line in block.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                lines.append("HETATM" + line[6:66])
    return "\n".join(lines) + "\nEND\n"


# ----------------------------------------------------------------------- metrics


def sbdd_metrics(smiles: str, docking_score: float) -> dict[str, Any]:
    """Combine the docking score with 2D properties into efficiency metrics."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockingError(f"invalid SMILES {smiles!r}")
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        raise DockingError("molecule has no heavy atoms")
    if not math.isfinite(docking_score):
        raise DockingError("docking_score must be a finite number")

    clogp = float(Crippen.MolLogP(mol))
    # ΔG -> pK via 2.303RT at 298 K. A docking score is not a measured affinity,
    # so this is a rough scale conversion, not a prediction of potency.
    estimated_pic50 = -docking_score / KCAL_PER_LOG_UNIT
    ligand_efficiency = docking_score / heavy_atoms

    return {
        "Binding_Affinity": round(float(docking_score), 4),
        "Ligand_Efficiency": round(ligand_efficiency, 4),
        "Lipophilic_Ligand_Efficiency": round(estimated_pic50 - clogp, 4),
        "cLogP": round(clogp, 4),
        "TPSA": round(float(Descriptors.TPSA(mol)), 4),
        "estimated_pIC50": round(estimated_pic50, 4),
        "heavy_atoms": heavy_atoms,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "meets_le_target": ligand_efficiency < -0.3,
        "notes": (
            "Ligand_Efficiency uses the signed convention score/heavy_atoms, so "
            "more negative is better and the target is < -0.3. estimated_pIC50 "
            "converts the docking score with 2.303RT = 1.364 kcal/mol at 298 K; "
            "treat it as a ranking aid, not a measured potency."
        ),
    }


# ------------------------------------------------------------------------ viewer


VIEWER_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; --bg: #ffffff; --fg: #1a1a1a; --muted: #5c5c5c; --line: #d8d8d8; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #16181d; --fg: #e8e8ea; --muted: #a0a4ad; --line: #2e323a; }}
  }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 ui-sans-serif, -apple-system, system-ui, sans-serif; }}
  header {{ padding: 12px 16px; border-bottom: 1px solid var(--line); }}
  h1 {{ font-size: 15px; margin: 0 0 4px; font-weight: 600; }}
  .meta {{ color: var(--muted); font-size: 13px; }}
  .meta b {{ color: var(--fg); font-weight: 600; }}
  #viewer {{ position: relative; width: 100%; height: calc(100vh - 118px); min-height: 360px; }}
  .controls {{ padding: 8px 16px; border-top: 1px solid var(--line); display: flex;
               gap: 12px; flex-wrap: wrap; align-items: center; }}
  button {{ font: inherit; padding: 4px 10px; border: 1px solid var(--line);
            border-radius: 6px; background: transparent; color: var(--fg); cursor: pointer; }}
</style>
<header>
  <h1>{title}</h1>
  <div class="meta">{subtitle}</div>
</header>
<div id="viewer"></div>
<div class="controls">
  <button id="toggle-surface">Toggle pocket surface</button>
  <button id="toggle-cartoon">Toggle cartoon</button>
  <button id="reset">Reset view</button>
  <span class="meta">Drag to rotate, scroll to zoom.</span>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<script>
  const receptor = {receptor_json};
  const ligand = {ligand_json};
  // Pocket residues are resolved server side; 3Dmol's cross-model `within`
  // selector does not filter reliably, so explicit ids keep the view honest.
  const pocket = {pocket_json};
  const pocketSel = pocket.map((p) => ({{ model: 0, chain: p.chain, resi: p.resi }}));

  const viewer = $3Dmol.createViewer(document.getElementById('viewer'), {{ backgroundAlpha: 0 }});
  viewer.addModel(receptor, 'pdb');
  viewer.addModel(ligand, 'sdf');

  let cartoonOn = true;
  const paintProtein = () => {{
    viewer.setStyle({{ model: 0 }}, cartoonOn ? {{ cartoon: {{ color: 'spectrum', opacity: 0.75 }} }} : {{}});
    for (const sel of pocketSel) {{
      viewer.addStyle(sel, {{ stick: {{ radius: 0.13, colorscheme: 'whiteCarbon' }} }});
    }}
  }};
  paintProtein();
  viewer.setStyle({{ model: 1 }}, {{ stick: {{ radius: 0.25, colorscheme: 'greenCarbon' }} }});

  const frame = () => {{ viewer.zoomTo({{ model: 1 }}); viewer.zoom(0.45); }};
  frame();
  viewer.render();

  let surface = null;
  document.getElementById('toggle-surface').onclick = () => {{
    if (surface) {{ viewer.removeSurface(surface); surface = null; }}
    else if (pocketSel.length) {{
      surface = viewer.addSurface($3Dmol.SurfaceType.VDW,
                                  {{ opacity: 0.7, color: 'lightgrey' }}, pocketSel[0]);
    }}
    viewer.render();
  }};
  document.getElementById('toggle-cartoon').onclick = () => {{ cartoonOn = !cartoonOn; paintProtein(); viewer.render(); }};
  document.getElementById('reset').onclick = () => {{ frame(); viewer.render(); }};
</script>
"""


def pocket_residues(
    receptor_pdb: str, ligand_sdf: str, cutoff: float = 5.0
) -> list[dict[str, Any]]:
    """Residues lining the pose, grouped by chain.

    3Dmol's cross-model `within` selector does not filter reliably, so the
    pocket is resolved here and written into the page as explicit residue ids.
    """
    ligand: list[tuple[float, float, float]] = []
    for line in ligand_sdf.splitlines()[4:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                ligand.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                break
        else:
            break
    if not ligand:
        return []

    squared = cutoff * cutoff
    by_chain: dict[str, set[int]] = {}
    for line in receptor_pdb.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            x, y, z = _atom_xyz(line)
            resi = int(line[22:26])
        except ValueError:
            continue
        chain = line[21:22].strip() or "A"
        if resi in by_chain.get(chain, ()):  # already selected
            continue
        for lx, ly, lz in ligand:
            if (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2 <= squared:
                by_chain.setdefault(chain, set()).add(resi)
                break
    return [
        {"chain": chain, "resi": sorted(resi)}
        for chain, resi in sorted(by_chain.items())
    ]


def write_pose_viewer(
    *, destination: Path, title: str, subtitle: str, receptor_pdb: str, ligand_sdf: str
) -> Path:
    """Self-contained 3Dmol.js page showing the pose inside the pocket."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        VIEWER_TEMPLATE.format(
            title=title,
            subtitle=subtitle,
            receptor_json=json.dumps(receptor_pdb),
            ligand_json=json.dumps(ligand_sdf),
            pocket_json=json.dumps(pocket_residues(receptor_pdb, ligand_sdf)),
        ),
        encoding="utf-8",
    )
    return destination
