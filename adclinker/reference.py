"""What a real ADC linker looks like, measured rather than asserted.

Every threshold in `score.py` comes from here, and here it comes from the
curated linker set in the Linker-GPT repository -- roughly 2,700 unique
commercial ADC linker reagents with names like `Mc-Val-Cit-PABC-PNP`. That
set is the one genuinely valuable thing in that repository, and it is hard to
assemble by hand.

Three corrections are applied before anything is measured, and each one
changes the numbers:

**Deduplicate on canonical SMILES.** The file has about 25% duplicates --
`Val-cit-PAB-OH` and `Val-Cit-PAB` are the same molecule under two catalogue
names. Leaving them in weights the common reagents twice.

**Drop the protecting and activating groups.** About 39% of the file carries
Fmoc, Boc, tBu, NHS or p-nitrophenyl. Those exist to make the reagent
couplable on the bench and are *gone* from the finished conjugate. Measuring
cLogP across them describes a bottle on a shelf, not a linker on an antibody.
Molecules carrying them are excluded from the property profile, though they
stay in the similarity reference, where resemblance to a real reagent is the
point.

**Measure percentiles, not means.** The distribution is long-tailed -- PEG9
reagents run past 1,200 Da -- so a mean is pulled somewhere no real linker
sits.

The profile that comes out is worth reading before trusting any score built
on it. Median cLogP is about 0.7, against 2-3 for an oral drug: these
molecules are far more polar than drug-like chemistry. Median Fsp3 is about
0.9 with ~25 rotatable bonds: long saturated flexible chains, almost no
aromatic ring systems. That is the opposite shape to a PROTAC linker, and it
is why `adclinker` cannot reuse `protacbench`'s thresholds even though it
reuses its machinery.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Groups that exist only to make a reagent couplable and are absent from the
# installed linker. Matching any of these excludes a molecule from the
# property profile.
SYNTHESIS_ONLY_GROUPS: tuple[tuple[str, str], ...] = (
    ("fmoc", "O=C(O[CH2]C1c2ccccc2-c2ccccc21)N"),
    ("boc", "O=C(OC(C)(C)C)N"),
    ("pnp_carbonate", "O=C(Oc1ccc([N+](=O)[O-])cc1)O"),
    ("nhs_ester", "O=C(ON1C(=O)CCC1=O)"),
    ("alloc", "O=C(OCC=C)N"),
    ("tbu_ester", "O=C(OC(C)(C)C)[#6]"),
)

# Motifs that make a linker cleavable in the lysosome. Presence is a gate in
# `score.py`, not a scored term: a linker that never releases its payload is
# not a worse candidate, it is not a candidate.
CLEAVABLE_MOTIFS: tuple[tuple[str, str], ...] = (
    # Citrulline side chain -- the Val-Cit and Val-Ala cathepsin B substrates.
    ("citrulline", "[CH](CCCNC(N)=O)"),
    # Glutathione-reducible.
    ("disulfide", "[#6]SS[#6]"),
    # p-aminobenzyl self-immolative spacer.
    ("pab", "Nc1ccc(CO)cc1"),
    # Acid-labile hydrazone.
    ("hydrazone", "[#6]=[NX2][NX3]"),
    # beta-glucuronidase substrate.
    ("glucuronide", "OC1OC(C(O)C(O)C1O)C(=O)O"),
)

# Properties profiled, and which direction is better. `None` means a window
# rather than a direction.
PROFILED = (
    "heavy_atoms",
    "mw",
    "clogp",
    "tpsa",
    "rotatable_bonds",
    "hbd",
    "hba",
    "fsp3",
)


class ReferenceError(Exception):
    """The reference set could not be read or was unusable."""


@dataclass
class Percentiles:
    """The shape of one property across the reference set."""

    p10: float
    median: float
    p90: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float]:
        """Full precision on purpose: this is what the cache stores, and a
        cache that rounds does not reload the profile it was built from.
        Display formats these itself."""
        return {
            "p10": self.p10,
            "median": self.median,
            "p90": self.p90,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass
class Profile:
    """The measured shape of real ADC linkers."""

    n_total: int = 0
    n_unique: int = 0
    n_profiled: int = 0  # unique, minus those carrying synthesis-only groups
    n_cleavable: int = 0
    properties: dict[str, Percentiles] = field(default_factory=dict)
    source: str = ""

    def window(self, name: str) -> tuple[float, float]:
        """The 10th-90th percentile band, which is what the score plateaus on."""
        stats = self.properties.get(name)
        if stats is None:
            raise ReferenceError(f"no reference statistics for {name}")
        return stats.p10, stats.p90

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_total": self.n_total,
            "n_unique": self.n_unique,
            "n_profiled": self.n_profiled,
            "n_cleavable": self.n_cleavable,
            "properties": {k: v.as_dict() for k, v in self.properties.items()},
        }


def _percentiles(values: list[float]) -> Percentiles:
    ordered = sorted(values)
    n = len(ordered)
    return Percentiles(
        p10=ordered[max(0, int(n * 0.10) - 1)] if n > 1 else ordered[0],
        median=statistics.median(ordered),
        p90=ordered[min(n - 1, int(n * 0.90))],
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def measure(mol: Any) -> dict[str, float]:
    """The properties profiled, for one molecule."""
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mw": float(Descriptors.MolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


_QUERY_CACHE: dict[int, list[tuple[str, Any]]] = {}


def _compiled(pairs: tuple[tuple[str, str], ...]) -> list[tuple[str, Any]]:
    """Compile once and keep. These are matched per molecule over thousands
    of molecules, and recompiling the SMARTS each time dominated the runtime.
    """
    cached = _QUERY_CACHE.get(id(pairs))
    if cached is not None:
        return cached

    from rdkit import Chem

    out = []
    for name, smarts in pairs:
        query = Chem.MolFromSmarts(smarts)
        if query is None:  # a bad pattern here is our bug, not the molecule's
            raise ReferenceError(f"invalid SMARTS for {name}: {smarts}")
        out.append((name, query))
    _QUERY_CACHE[id(pairs)] = out
    return out


def carries_synthesis_group(mol: Any) -> bool:
    return any(mol.HasSubstructMatch(q) for _, q in _compiled(SYNTHESIS_ONLY_GROUPS))


def cleavable_motifs(mol: Any) -> list[str]:
    """Which lysosomal release motifs this molecule carries, if any."""
    return [name for name, q in _compiled(CLEAVABLE_MOTIFS) if mol.HasSubstructMatch(q)]


def read_reference_smiles(path: str | Path, column: str = "smiles") -> list[str]:
    """Canonical, deduplicated SMILES from the curated reagent file.

    The file is not valid UTF-8 -- a stray byte in one of the catalogue
    product names -- so it is read leniently. Pandas with default settings
    raises on it, which is worth knowing before debugging a crash somewhere
    further in.
    """
    from rdkit import Chem, rdBase

    source = Path(path)
    if not source.is_file():
        raise ReferenceError(
            f"reference set not found: {source}\n"
            "Clone the Linker-GPT repository next to this one, or point "
            "reference.dataset at data/linkers.csv wherever it lives."
        )

    seen: set[str] = set()
    ordered: list[str] = []
    with source.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ReferenceError(
                f"{source} has no '{column}' column; found {reader.fieldnames}"
            )
        with rdBase.BlockLogs():
            for row in reader:
                smiles = (row.get(column) or "").strip()
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol)
                if canonical in seen:
                    continue
                seen.add(canonical)
                ordered.append(canonical)
    if not ordered:
        raise ReferenceError(f"no readable SMILES in {source}")
    return ordered


def build_profile(path: str | Path, column: str = "smiles") -> Profile:
    """Measure the reference set and return the distribution behind the score."""
    from rdkit import Chem, rdBase

    unique = read_reference_smiles(path, column)
    profile = Profile(n_unique=len(unique), source=str(Path(path).resolve()))

    collected: dict[str, list[float]] = {name: [] for name in PROFILED}
    with rdBase.BlockLogs():
        for smiles in unique:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if cleavable_motifs(mol):
                profile.n_cleavable += 1
            # The property profile describes the linker as installed, so the
            # bench-only groups are excluded from it.
            if carries_synthesis_group(mol):
                continue
            profile.n_profiled += 1
            for name, value in measure(mol).items():
                collected[name].append(value)

    if profile.n_profiled < 50:
        raise ReferenceError(
            f"only {profile.n_profiled} molecules survived filtering in "
            f"{path}; too few to calibrate a score against"
        )

    profile.properties = {
        name: _percentiles(values) for name, values in collected.items() if values
    }
    profile.n_total = profile.n_unique
    return profile


# ------------------------------------------------------------------- the cache


def load_profile(
    dataset: str | Path, cache: str | Path | None = None, refresh: bool = False
) -> Profile:
    """Build the profile, or read it back from a cached JSON.

    Cached because it is a few seconds of RDKit over several thousand
    molecules, and because pinning it makes a scoring run reproducible even if
    the reference file is later edited.
    """
    cache_path = Path(cache) if cache else None
    if cache_path and cache_path.is_file() and not refresh:
        raw = json.loads(cache_path.read_text())
        profile = Profile(
            n_total=raw["n_total"],
            n_unique=raw["n_unique"],
            n_profiled=raw["n_profiled"],
            n_cleavable=raw.get("n_cleavable", 0),
            source=raw.get("source", ""),
        )
        profile.properties = {
            name: Percentiles(
                p10=stats["p10"],
                median=stats["median"],
                p90=stats["p90"],
                minimum=stats["min"],
                maximum=stats["max"],
            )
            for name, stats in raw["properties"].items()
        }
        return profile

    profile = build_profile(dataset)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(profile.as_dict(), indent=2) + "\n")
    return profile


# ------------------------------------------------------- the similarity anchor


def fingerprint_reference(
    dataset: str | Path, radius: int = 2, n_bits: int = 2048
) -> list[Any]:
    """Morgan fingerprints of every reference reagent, for the likeness term.

    Protected reagents are kept here deliberately. The question this answers
    is "does this look like something a linker chemist would recognise", and a
    Boc group does not make a molecule less recognisable.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    fingerprints = []
    with rdBase.BlockLogs():
        for smiles in read_reference_smiles(dataset):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints


def nearest_similarity(mol: Any, reference: list[Any], radius: int = 2,
                       n_bits: int = 2048) -> float:
    """Tanimoto to the closest reference reagent, 0-1.

    Used as a floor rather than something to maximise: a candidate that looks
    nothing like any known linker is probably not one, but a candidate
    identical to a known linker is not an invention. `score.py` pairs this
    with a novelty term for that reason.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    if not reference:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    probe = generator.GetFingerprint(mol)
    return float(max(DataStructs.BulkTanimotoSimilarity(probe, reference)))
