"""Building Pro-GAT-shaped sample directories, including broken ones.

Real Pro-GAT output needs a GPU to produce, so the tests construct files with
the same layout, the same filenames, and the same trailing-linker atom order,
then break them in the specific ways the diffusion model breaks them.
"""

from __future__ import annotations

from pathlib import Path

# A thalidomide-like glutarimide on one end, an aryl-pyridine on the other,
# bridged by PEG3: 587 Da over 43 heavy atoms, which is squarely inside the
# PROTAC range the default thresholds expect. A smaller fixture gets correctly
# rejected for being too light, which makes it useless for testing anything
# else.
PROTAC_SMILES = (
    "O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
)

# The PEG bridge, located by substructure rather than hardcoded, so the
# fixture stays correct if the molecule is ever edited.
LINKER_SMARTS = "[CH2][CH2]O[CH2][CH2]O[CH2][CH2]"


def protac_like() -> tuple[str, list[int]]:
    """A PROTAC-shaped molecule, and which of its atoms form the linker."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(PROTAC_SMILES)
    match = mol.GetSubstructMatch(Chem.MolFromSmarts(LINKER_SMARTS))
    if not match:
        raise ValueError("the fixture molecule no longer contains its PEG bridge")
    return PROTAC_SMILES, list(match)


def ordered_mol(smiles: str, linker_indices: list[int]):
    """Embed the molecule and renumber so the linker is the trailing block."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable fixture SMILES: {smiles}")
    fragments = [i for i in range(mol.GetNumAtoms()) if i not in set(linker_indices)]
    mol = Chem.RenumberAtoms(mol, fragments + list(linker_indices))

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xABCDEF) != 0:
        raise ValueError("could not embed the fixture molecule")
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def write_xyz(path: Path, mol, *, break_mode: str | None = None) -> Path:
    """Write heavy atoms in order, optionally breaking the geometry.

    The break modes mirror what the diffusion model actually produces:
    a linker that never bonded to a fragment, and atoms placed on top of each
    other.
    """
    conformer = mol.GetConformer()
    rows: list[tuple[str, float, float, float]] = []
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        rows.append((atom.GetSymbol(), position.x, position.y, position.z))

    if break_mode == "disconnected":
        # Push the trailing linker block far away, so no bond is perceived.
        linker_len = len(protac_like()[1])
        for i in range(len(rows) - linker_len, len(rows)):
            symbol, x, y, z = rows[i]
            rows[i] = (symbol, x + 40.0, y, z)
    elif break_mode == "clash":
        # Put the last atom exactly on the first.
        symbol = rows[-1][0]
        rows[-1] = (symbol, rows[0][1], rows[0][2], rows[0][3])
    elif break_mode == "overcrowded":
        # Collapse several atoms onto one centre, so bond perception gives it
        # far more bonds than carbon permits.
        anchor = rows[0]
        for offset, i in enumerate(range(1, 7)):
            symbol = rows[i][0]
            rows[i] = (
                symbol,
                anchor[1] + 1.1 + 0.01 * offset,
                anchor[2] + 0.02 * offset,
                anchor[3],
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(rows)), ""]
    lines += [f"{s} {x:.9f} {y:.9f} {z:.9f}" for s, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def build_run(
    root: Path,
    *,
    targets: int = 2,
    samples: int = 2,
    break_modes: dict[tuple[int, int], str] | None = None,
) -> Path:
    """A sampling output directory: `{uuid}/{sample}_.xyz` plus `true_.xyz`."""
    smiles, linker = protac_like()
    mol = ordered_mol(smiles, linker)
    breaks = break_modes or {}

    for target in range(targets):
        target_dir = root / f"T{target:03d}"
        write_xyz(target_dir / "true_.xyz", mol)
        write_xyz(target_dir / "frag_.xyz", mol)
        for sample in range(samples):
            write_xyz(
                target_dir / f"{sample}_.xyz",
                mol,
                break_mode=breaks.get((target, sample)),
            )
    return root


def build_repaired(
    root: Path, *, target: str = "T000", sample: int = 0, linker_len: int = 8
) -> Path:
    """The repair layout: a flat `{uuid}-{linker}_{sample}_.xyz`."""
    smiles, linker = protac_like()
    mol = ordered_mol(smiles, linker)
    run_dir = root / "samples"
    write_xyz(run_dir / f"{target}-{linker_len}_{sample}_.xyz", mol)
    return run_dir
