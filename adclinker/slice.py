"""Turning real ADC reagents into LinkInvent training examples.

LinkInvent learns from `(warhead_pair, linker)` triples. The curated reference
set is whole reagents with no attachment points, so the triples have to be
made by cutting each molecule in two places and calling the middle piece the
linker.

**This is the step where the fine-tuning either does or does not make
biological sense**, because an arbitrary pair of cuts teaches an arbitrary
lesson. Three constraints make the lesson the right one.

**One warhead must contain a conjugation handle.** A real ADC reagent is
`handle - spacer - trigger`, and what we want the model to learn is "given a
handle on one side and something to reach on the other, produce the spacer
that real reagents use". Cutting `Fmoc-Val-Cit-PAB` into two random halves
teaches peptide chemistry instead. Requiring a handle in a warhead keeps the
generated piece in the position it will occupy at inference.

**Cuts are single acyclic bonds between heavy atoms, never in a ring.**
Breaking a ring produces two fragments that no chemistry reassembles, and a
model trained on them learns to propose bonds that cannot be made.

**The linker is size-bounded at both ends.** The reference set runs from PEG2
building blocks to 336-atom PEG9 reagents. A three-atom "linker" teaches
nothing and a hundred-atom one is the whole molecule.

What comes out is deliberately *not* every possible cut. Capping the triples
per molecule stops a handful of large PEG reagents, which have dozens of
equivalent cut sites, from dominating the training set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .warheads import HANDLES

# Substructures that mark a fragment as the antibody-facing end. Derived from
# the handle library rather than written twice.
HANDLE_CORES: tuple[tuple[str, str], ...] = (
    ("maleimide", "O=C1C=CC(=O)N1"),
    ("nhs_ester", "O=C(ON1C(=O)CCC1=O)"),
    ("dbco", "C#Cc1ccccc1"),
    ("haloacetyl", "[CX4]([F,Cl,Br,I])C(=O)N"),
    ("azide", "[NX2]=[NX2+]=[NX1-]"),
    ("alkyne", "C#C"),
    ("thiol", "[SX2;H1]"),
)


# How many bond pairs to try per molecule before giving up on finding more
# acceptable cuts. Bounds the cost of the largest reagents.
_MAX_PAIRS_CONSIDERED = 120


class SliceError(Exception):
    """The reference set could not be turned into training triples."""


@dataclass
class Triple:
    """One LinkInvent training example."""

    warhead_1: str
    warhead_2: str
    linker: str
    source: str
    handle: str | None = None

    def as_line(self) -> str:
        """TAB separated, warheads first.

        The two sides use different notation, which is REINVENT's convention
        rather than ours: the input vocabulary has a bare `*` token and the
        target vocabulary a bracketed `[*]`. Writing both the same way fails
        token validation on one side or the other.
        """
        return f"{self.warhead_1}|{self.warhead_2}\t{self.linker}"


@dataclass
class SliceReport:
    molecules: int = 0
    molecules_used: int = 0
    triples: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "molecules_read": self.molecules,
            "molecules_used": self.molecules_used,
            "triples": self.triples,
            "rejected": dict(sorted(self.rejected.items(), key=lambda kv: -kv[1])),
        }


def cuttable_bonds(mol: Any) -> list[int]:
    """Single, acyclic bonds between two heavy atoms.

    Ring bonds are excluded because breaking one does not give two pieces that
    any reaction puts back together, and a model trained on such a split
    learns to propose impossible bonds.
    """
    from rdkit import Chem

    indices = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        if bond.GetBeginAtom().GetAtomicNum() == 1:
            continue
        if bond.GetEndAtom().GetAtomicNum() == 1:
            continue
        indices.append(bond.GetIdx())
    return indices


_HANDLE_QUERIES: list[tuple[str, Any]] = []


def _handle_queries() -> list[tuple[str, Any]]:
    """Compiled once. Called several times per candidate cut, over tens of
    thousands of cuts, so recompiling here dominated the runtime."""
    global _HANDLE_QUERIES
    if not _HANDLE_QUERIES:
        from rdkit import Chem

        _HANDLE_QUERIES = [
            (name, q)
            for name, smarts in HANDLE_CORES
            if (q := Chem.MolFromSmarts(smarts)) is not None
        ]
    return _HANDLE_QUERIES


def _has_handle(mol: Any) -> str | None:
    for name, query in _handle_queries():
        if mol.HasSubstructMatch(query):
            return name
    return None


def slice_molecule(
    smiles: str,
    *,
    min_linker_atoms: int = 3,
    max_linker_atoms: int = 30,
    min_warhead_atoms: int = 3,
    require_handle: bool = True,
    max_triples: int = 6,
    rng: random.Random | None = None,
    report: SliceReport | None = None,
) -> list[Triple]:
    """Every acceptable two-bond cut of one reagent, capped and shuffled."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if report:
            report.reject("unparseable")
        return []

    bonds = cuttable_bonds(mol)
    if len(bonds) < 2:
        if report:
            report.reject("too_few_cuttable_bonds")
        return []

    found: list[Triple] = []
    # A long PEG reagent has hundreds of cuttable bonds and so tens of
    # thousands of bond pairs, nearly all of which are equivalent. Sampling a
    # bounded number keeps one such molecule from costing more than the rest
    # of the set put together.
    pairs = list(combinations(bonds, 2))
    if rng is not None:
        if len(pairs) > _MAX_PAIRS_CONSIDERED:
            pairs = rng.sample(pairs, _MAX_PAIRS_CONSIDERED)
        else:
            rng.shuffle(pairs)

    for first, second in pairs:
        if len(found) >= max_triples:
            break
        with rdBase.BlockLogs():
            try:
                broken = Chem.FragmentOnBonds(
                    mol, [first, second], addDummies=True
                )
                pieces = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
            except Exception:  # noqa: BLE001 - a bad cut is a result
                continue

        # Three pieces exactly: two ends and a middle. Two pieces means the
        # cuts were on the same branch and the molecule only fell in half.
        if len(pieces) != 3:
            if report:
                report.reject("cut_did_not_give_three_pieces")
            continue

        middle = [p for p in pieces if _dummy_count(p) == 2]
        ends = [p for p in pieces if _dummy_count(p) == 1]
        if len(middle) != 1 or len(ends) != 2:
            if report:
                report.reject("ambiguous_middle")
            continue

        linker_mol = middle[0]
        linker_size = _heavy_atoms(linker_mol)
        if not (min_linker_atoms <= linker_size <= max_linker_atoms):
            if report:
                report.reject("linker_size")
            continue
        if any(_heavy_atoms(p) < min_warhead_atoms for p in ends):
            if report:
                report.reject("warhead_too_small")
            continue

        handle = _has_handle(ends[0]) or _has_handle(ends[1])
        if require_handle and handle is None:
            if report:
                report.reject("no_conjugation_handle_in_either_warhead")
            continue

        # Put the handle-bearing warhead first, so the model consistently
        # sees the antibody side on the left. Inference presents it the same
        # way, and a model trained on a random order learns neither.
        first_end, second_end = ends
        if _has_handle(second_end) and not _has_handle(first_end):
            first_end, second_end = second_end, first_end

        found.append(
            Triple(
                warhead_1=_as_warhead(first_end),
                warhead_2=_as_warhead(second_end),
                linker=_as_target(linker_mol),
                source=smiles,
                handle=handle,
            )
        )

    if report:
        report.triples += len(found)
        if found:
            report.molecules_used += 1
    return found


def _dummy_count(mol: Any) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


def _heavy_atoms(mol: Any) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _as_warhead(mol: Any) -> str:
    """Bare `*`, which is what LinkInvent's input vocabulary tokenises."""
    from rdkit import Chem

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    return Chem.MolToSmiles(copy)


def _as_target(mol: Any) -> str:
    """`[*]`, which is what the target vocabulary stores."""
    from rdkit import Chem

    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(copy)
    # RDKit writes a bare `*` for an unmapped dummy; the target side wants it
    # bracketed.
    return smiles.replace("*", "[*]").replace("[[*]]", "[*]")


def build_triples(
    smiles_list: Iterable[str],
    *,
    min_linker_atoms: int = 3,
    max_linker_atoms: int = 30,
    min_warhead_atoms: int = 3,
    require_handle: bool = True,
    max_triples_per_molecule: int = 6,
    seed: int = 42,
) -> tuple[list[Triple], SliceReport]:
    """Cut a whole reference set into training triples."""
    rng = random.Random(seed)
    report = SliceReport()
    triples: list[Triple] = []

    for smiles in smiles_list:
        report.molecules += 1
        triples.extend(
            slice_molecule(
                smiles,
                min_linker_atoms=min_linker_atoms,
                max_linker_atoms=max_linker_atoms,
                min_warhead_atoms=min_warhead_atoms,
                require_handle=require_handle,
                max_triples=max_triples_per_molecule,
                rng=rng,
                report=report,
            )
        )

    if not triples:
        raise SliceError(
            "no training triples survived filtering. Loosen "
            "finetune.min_linker_atoms / max_linker_atoms, or set "
            "require_handle false if the reference set has no recognisable "
            "conjugation chemistry."
        )
    return triples, report


def write_split(
    triples: list[Triple],
    training_path: str | Path,
    validation_path: str | Path | None,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Write the TAB-separated files, splitting by source molecule.

    Split by molecule, not by triple. Several triples cut from the same
    reagent are near-duplicates, and splitting them across train and
    validation would put the same chemistry on both sides -- the exact
    leakage that makes the Linker-GPT paper's own numbers unreliable.
    """
    rng = random.Random(seed)

    # Deduplicate the examples themselves before grouping. Splitting by source
    # molecule stops cuts of ONE reagent spanning the split, but a catalogue
    # is full of near-identical reagents -- PEG4 and PEG5 variants cut at the
    # same bonds give byte-identical triples from different sources. Without
    # this, the same training example lands on both sides and the validation
    # loss flatters the model.
    seen: set[str] = set()
    unique: list[Triple] = []
    for triple in triples:
        line = triple.as_line()
        if line in seen:
            continue
        seen.add(line)
        unique.append(triple)
    triples = unique

    by_source: dict[str, list[Triple]] = {}
    for triple in triples:
        by_source.setdefault(triple.source, []).append(triple)

    sources = sorted(by_source)
    rng.shuffle(sources)
    n_validation = (
        int(len(sources) * validation_fraction) if validation_path else 0
    )
    validation_sources = set(sources[:n_validation])

    training = [t for s in sources if s not in validation_sources for t in by_source[s]]
    validation = [t for s in validation_sources for t in by_source[s]]

    train_file = Path(training_path)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    train_file.write_text("\n".join(t.as_line() for t in training) + "\n")

    written = {
        "training_file": str(train_file),
        "training_triples": len(training),
        "training_molecules": len(sources) - len(validation_sources),
        "validation_triples": 0,
    }
    if validation_path and validation:
        valid_file = Path(validation_path)
        valid_file.parent.mkdir(parents=True, exist_ok=True)
        valid_file.write_text("\n".join(t.as_line() for t in validation) + "\n")
        written["validation_file"] = str(valid_file)
        written["validation_triples"] = len(validation)
        written["validation_molecules"] = len(validation_sources)
    return written
