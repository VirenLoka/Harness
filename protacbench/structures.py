"""Reading Pro-GAT's output, and recording exactly why a sample is unusable.

Pro-GAT's own pipeline destroys this information. `run_progat.py`'s
`convert_xyz_to_sdf` deletes the SDF whenever conversion, parsing or the
connectivity check fails, and `compute_repair.compute_metrics` then records
only `"{uuid}/{sample}"` with no reason -- so four distinct causes (an
impossible valence, an Open Babel write failure, an RDKit parse failure, and a
disconnected molecule) all arrive downstream as the same thing: a missing file.

Everything here keeps the evidence instead. Each step returns a reason code
and a human-readable detail, and nothing is deleted.

Atom order is the other thing this module depends on. Pro-GAT writes
fragments first and the linker as a trailing contiguous block -- confirmed
against `protacs_test.pt`, where entry 0 has 46 atoms with the linker at
indices 32-45 -- and Open Babel preserves input order for xyz to sdf. That
makes the linker recoverable exactly by index, which is far more reliable than
matching the fragments as a substructure.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pro-GAT's one-hot alphabet (const.IDX2ATOM).
ATOM_TYPES = ("C", "O", "N", "F", "S", "Cl", "Br", "I", "P")

# Open Babel's own ceiling check in run_progat.py: a summed bond order above
# this means perception produced an impossible atom.
MAX_BOND_ORDER_SUM = 8

# Two heavy atoms closer than this are not a bond, they are a collapsed
# geometry the diffusion model failed to separate.
MIN_HEAVY_ATOM_DISTANCE = 0.9


class StructureError(Exception):
    """A structure problem worth reporting verbatim."""


@dataclass
class XyzFrame:
    """One Pro-GAT XYZ file: heavy atoms only, no bonds."""

    path: Path
    symbols: list[str]
    coords: list[tuple[float, float, float]]

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


@dataclass
class LoadResult:
    """The outcome of turning one XYZ into an RDKit molecule."""

    ok: bool
    mol: Any | None = None
    sdf_path: Path | None = None
    reason: str | None = None  # a stable code, for grouping
    detail: str | None = None  # the message a human reads
    warnings: list[str] = field(default_factory=list)


def read_xyz(path: str | Path) -> XyzFrame:
    """Parse an XYZ written by `visualizer.save_xyz_file`."""
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise StructureError(f"cannot read {path}: {exc}") from exc
    if not lines:
        raise StructureError(f"{path} is empty")

    try:
        declared = int(lines[0].strip())
    except ValueError as exc:
        raise StructureError(f"{path} has no atom count on its first line") from exc

    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in lines[2 : 2 + declared]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if len(symbols) != declared:
        raise StructureError(
            f"{path} declares {declared} atoms but holds {len(symbols)}"
        )
    return XyzFrame(path=path, symbols=symbols, coords=coords)


def obabel_binary() -> Path:
    """Open Babel, which turns a bondless XYZ into a bonded SDF."""
    import os

    override = os.environ.get("MOLBENCH_OBABEL_BIN")
    found = override or shutil.which("obabel")
    if not found:
        raise StructureError(
            "obabel was not found. Install it with "
            "`conda install -c conda-forge openbabel`, or set "
            "MOLBENCH_OBABEL_BIN to its path."
        )
    return Path(found)


def convert_xyz_to_sdf(
    xyz_path: str | Path, sdf_path: str | Path, *, timeout_s: float = 60.0
) -> tuple[bool, str | None, str | None]:
    """Run Open Babel, returning (ok, reason_code, detail).

    The CLI is used rather than the Python bindings so this works wherever
    `obabel` is installed, without the `openbabel` module that Pro-GAT's own
    conversion needs.
    """
    xyz_path, sdf_path = Path(xyz_path), Path(sdf_path)
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        binary = obabel_binary()
    except StructureError as exc:
        return False, "obabel_missing", str(exc)

    try:
        completed = subprocess.run(
            [str(binary), str(xyz_path), "-O", str(sdf_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "obabel_timeout", f"obabel did not finish in {timeout_s:.0f}s"

    if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
        tail = (completed.stderr or "").strip().splitlines()[-2:]
        return (
            False,
            "obabel_wrote_nothing",
            "obabel produced no output: " + (" / ".join(tail) or "no message"),
        )
    return True, None, None


def restore_implicit_hydrogens(mol: Any) -> Any:
    """Let open valences be filled by implicit hydrogens.

    Open Babel writes heavy-atom-only SDFs, and molblock semantics tell RDKit
    that the listed valences are complete. Sanitising then assigns radical
    electrons to every unfilled valence, so a perfectly ordinary amide comes
    back as `O=C([N][C][C]O...)` and every descriptor computed from it is
    wrong -- measured on a reference PROTAC-like molecule, QED read 0.5485
    instead of 0.6029 and the mass was 24 hydrogens short. Pro-GAT's own
    pipeline reports SMILES and metrics from exactly this state.

    Clearing the radicals and allowing implicit hydrogens recovers the
    molecule exactly. It does not paper over real valence errors: an atom with
    too many bonds still fails sanitisation afterwards, which is reported.
    """
    from rdkit import Chem

    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
    return editable.GetMol()


def _bond_order_overflow(mol: Any) -> tuple[int, float] | None:
    """The atom whose summed bond order exceeds what any element allows."""
    worst: tuple[int, float] | None = None
    for atom in mol.GetAtoms():
        total = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        if total > MAX_BOND_ORDER_SUM and (worst is None or total > worst[1]):
            worst = (atom.GetIdx(), total)
    return worst


def load_molecule(
    xyz_path: str | Path,
    sdf_path: str | Path,
    *,
    expected_symbols: list[str] | None = None,
) -> LoadResult:
    """XYZ to a sanitized RDKit molecule, keeping the reason for any failure.

    The order of the checks matters: each one is only meaningful once the
    previous has passed, and reporting the first real cause is the point of
    the exercise.
    """
    from rdkit import Chem, rdBase

    ok, reason, detail = convert_xyz_to_sdf(xyz_path, sdf_path)
    if not ok:
        return LoadResult(ok=False, reason=reason, detail=detail)

    sdf_path = Path(sdf_path)
    warnings: list[str] = []

    # Read without sanitizing first, so a valence problem can be described
    # rather than just making the molecule vanish.
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        raw = next(iter(supplier), None)

    if raw is None:
        return LoadResult(
            ok=False,
            sdf_path=sdf_path,
            reason="unreadable_sdf",
            detail="RDKit could not parse the SDF Open Babel wrote",
        )

    if expected_symbols is not None:
        got = [atom.GetSymbol() for atom in raw.GetAtoms()]
        if got != expected_symbols:
            # Index-based linker extraction depends on this, so a mismatch is
            # reported rather than silently producing the wrong linker.
            warnings.append(
                f"atom order changed in conversion: {len(expected_symbols)} "
                f"expected, {len(got)} found"
            )

    overflow = _bond_order_overflow(raw)
    if overflow is not None:
        index, total = overflow
        symbol = raw.GetAtomWithIdx(index).GetSymbol()
        return LoadResult(
            ok=False,
            sdf_path=sdf_path,
            reason="impossible_valence",
            detail=(
                f"atom {index} ({symbol}) was given a summed bond order of "
                f"{total:g}, above the maximum of {MAX_BOND_ORDER_SUM}"
            ),
            warnings=warnings,
        )

    mol = restore_implicit_hydrogens(raw)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - the message is the deliverable
        return LoadResult(
            ok=False,
            sdf_path=sdf_path,
            reason="sanitization_failed",
            detail=f"{type(exc).__name__}: {exc}",
            warnings=warnings,
        )

    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        sizes = sorted((len(f) for f in fragments), reverse=True)
        return LoadResult(
            ok=False,
            mol=mol,
            sdf_path=sdf_path,
            reason="disconnected",
            detail=(
                f"the molecule came apart into {len(fragments)} pieces of "
                f"{sizes} atoms; the linker did not bond to both fragments"
            ),
            warnings=warnings,
        )

    return LoadResult(ok=True, mol=mol, sdf_path=sdf_path, warnings=warnings)


# ------------------------------------------------------------------ the linker


@dataclass
class LinkerView:
    """The linker, and how it attaches to the two fragments."""

    smiles: str
    atom_indices: list[int]
    n_heavy_atoms: int
    attachment_atoms: list[int]  # linker atoms bonded to a fragment
    fragment_anchor_atoms: list[int]  # fragment atoms bonded to the linker
    n_attachments: int
    path_length_bonds: int | None  # shortest route between the two anchors
    exit_vector_distance: float | None  # anchor-to-anchor distance in Angstroms
    rotatable_bonds: int
    heteroatom_fraction: float
    in_ring_fraction: float


def linker_indices(n_atoms: int, linker_len: int) -> list[int]:
    """Pro-GAT writes the linker as the trailing contiguous block."""
    if linker_len <= 0:
        return []
    if linker_len > n_atoms:
        raise StructureError(
            f"a linker of {linker_len} atoms cannot fit in {n_atoms} atoms"
        )
    return list(range(n_atoms - linker_len, n_atoms))


def describe_fragments(mol: Any, linker_indices: list[int]) -> list[str]:
    """The warhead and the E3 ligand, as SMILES, by removing the linker.

    Recorded at assessment time because after a repair there is no way to
    recover them: the agent has rewritten the molecule, and the trailing-block
    trick that locates the linker in Pro-GAT's output no longer applies. These
    strings are what makes "do not change either fragment" checkable rather
    than merely asked for.

    Returned in a stable order (longest first) so two runs on the same sample
    produce the same list.
    """
    from rdkit import Chem, rdBase

    linker_set = set(linker_indices)
    keep = [i for i in range(mol.GetNumAtoms()) if i not in linker_set]
    if not keep:
        return []

    with rdBase.BlockLogs():
        try:
            smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=keep, canonical=True)
        except Exception:  # noqa: BLE001 - a partial read is a bonus, not a failure
            return []

    pieces = [p for p in smiles.split(".") if p]
    return sorted(pieces, key=len, reverse=True)


def describe_linker(mol: Any, linker_len: int) -> LinkerView:
    """Pull the linker out by index and measure how it bridges the fragments.

    Pro-GAT writes the linker as the trailing block, so the length is enough
    to locate it. After a repair it is not -- the agent has rewritten the
    molecule -- which is what `linker_view` is for.
    """
    return linker_view(mol, linker_indices(mol.GetNumAtoms(), linker_len))


def linker_view(mol: Any, indices: list[int]) -> LinkerView:
    """Measure how the given atoms bridge the rest of the molecule.

    Separate from `describe_linker` because a repaired molecule's linker is
    found by subtracting the matched fragments rather than by counting back
    from the end, but everything measured about it is the same.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    linker_set = set(indices)

    attachment_atoms: list[int] = []
    anchor_atoms: list[int] = []
    for index in indices:
        for neighbour in mol.GetAtomWithIdx(index).GetNeighbors():
            if neighbour.GetIdx() not in linker_set:
                attachment_atoms.append(index)
                anchor_atoms.append(neighbour.GetIdx())
    attachment_atoms = sorted(set(attachment_atoms))
    anchor_atoms = sorted(set(anchor_atoms))

    with rdBase.BlockLogs():
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=indices, canonical=True)
        fragment = Chem.MolFromSmiles(smiles, sanitize=False)
        rotatable = 0
        if fragment is not None:
            try:
                Chem.SanitizeMol(fragment)
                rotatable = rdMolDescriptors.CalcNumRotatableBonds(fragment)
            except Exception:  # noqa: BLE001 - a descriptor is not worth failing for
                rotatable = 0

    # The bond count between the two fragment anchors is the length that
    # governs ternary-complex geometry, and it is not the same as the atom
    # count: a linker can be long but fold through a ring.
    path_length: int | None = None
    exit_distance: float | None = None
    if len(anchor_atoms) >= 2:
        try:
            path = Chem.GetShortestPath(mol, anchor_atoms[0], anchor_atoms[1])
            path_length = max(len(path) - 1, 0) or None
        except Exception:  # noqa: BLE001
            path_length = None
        if mol.GetNumConformers():
            conformer = mol.GetConformer()
            first = conformer.GetAtomPosition(anchor_atoms[0])
            second = conformer.GetAtomPosition(anchor_atoms[1])
            exit_distance = round(first.Distance(second), 3)

    heteroatoms = sum(
        1 for index in indices if mol.GetAtomWithIdx(index).GetSymbol() != "C"
    )
    in_ring = sum(1 for index in indices if mol.GetAtomWithIdx(index).IsInRing())
    denominator = max(len(indices), 1)

    return LinkerView(
        smiles=smiles,
        atom_indices=indices,
        n_heavy_atoms=len(indices),
        attachment_atoms=attachment_atoms,
        fragment_anchor_atoms=anchor_atoms,
        n_attachments=len(attachment_atoms),
        path_length_bonds=path_length,
        exit_vector_distance=exit_distance,
        rotatable_bonds=rotatable,
        heteroatom_fraction=round(heteroatoms / denominator, 3),
        in_ring_fraction=round(in_ring / denominator, 3),
    )


# ---------------------------------------------------------------- the geometry


@dataclass
class GeometryReport:
    min_distance: float | None
    clashing_pairs: int
    centroid_spread: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_interatomic_distance": self.min_distance,
            "clashing_atom_pairs": self.clashing_pairs,
            "centroid_spread": self.centroid_spread,
        }


def check_geometry(
    frame: XyzFrame, *, minimum: float = MIN_HEAVY_ATOM_DISTANCE
) -> GeometryReport:
    """Look for collapsed coordinates before trusting any bond perception.

    Open Babel infers bonds from distance, so two atoms placed on top of each
    other produce a confidently wrong molecule rather than an error. Checking
    the raw coordinates catches that at the source.
    """
    import math

    coords = frame.coords
    if len(coords) < 2:
        return GeometryReport(None, 0, None)

    smallest = math.inf
    clashes = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            smallest = min(smallest, distance)
            if distance < minimum:
                clashes += 1

    centre = [sum(c[k] for c in coords) / len(coords) for k in range(3)]
    spread = max(math.dist(c, centre) for c in coords)
    return GeometryReport(
        min_distance=round(smallest, 3),
        clashing_pairs=clashes,
        centroid_spread=round(spread, 3),
    )
