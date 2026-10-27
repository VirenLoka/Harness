"""The parameters each sample is judged on.

Three groups, and the distinction between them is what drives triage:

**Hard gates** decide whether a molecule exists at all -- it parsed, it is one
connected piece, its valences are possible, the linker actually bridges both
fragments. Failing one makes a sample `broken`, and the reason is what the
repair handoff needs.

**Soft gates** decide whether a molecule is worth keeping. These are
thresholds on measured properties, and failing one makes a sample `poor`.
A `poor` sample is chemically real, just not useful.

**Descriptors** are measured and reported but never gate anything, because
no threshold on them would be defensible.

Synthetic accessibility comes from RDKit's own contributed scorer, the same
source `molbench` uses, so numbers agree across the two packages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .score import score_sample
from .structures import (
    LinkerView,
    LoadResult,
    XyzFrame,
    check_geometry,
    describe_fragments,
    describe_linker,
    load_molecule,
    read_xyz,
)

_SASCORER: Any = None


def sa_scorer() -> Any:
    """RDKit's contributed SA scorer, imported the way molbench does."""
    global _SASCORER
    if _SASCORER is None:
        from rdkit.Chem import RDConfig

        contrib = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(contrib) not in sys.path:
            sys.path.append(str(contrib))
        import sascorer

        _SASCORER = sascorer
    return _SASCORER


# Failure codes that mean the molecule is not chemically real. Ordered by how
# early they are detected, which is also roughly how fundamental they are.
HARD_FAILURES = {
    "obabel_missing": "Open Babel is not installed, so nothing could be converted",
    "obabel_timeout": "Open Babel did not finish converting the coordinates",
    "obabel_wrote_nothing": "Open Babel could not perceive a molecule from the coordinates",
    "unreadable_sdf": "the converted file could not be parsed",
    "impossible_valence": "an atom was given more bonds than the element permits",
    "sanitization_failed": "the molecule violates a valence or aromaticity rule",
    "disconnected": "the molecule is in more than one piece",
    "atom_clash": "two atoms are closer together than any bond length",
    "linker_not_bridging": "the linker does not join exactly two fragments",
    "linker_size_mismatch": "the file holds a different number of atoms than requested",
    "missing_xyz": "the generated coordinate file is absent",
}

# Functional groups that make a linker unusable, however good its descriptors
# look. These are checked inside the linker only -- a warhead is allowed to
# contain whatever it contains, because its structure is the reason the
# molecule exists.
#
# This gate was added after a real run: of 468 samples that passed every other
# check, 197 carried one of these, and the single highest-scoring molecule in
# the whole run was `NCCCCCCCC=O` -- a free aldehyde on a short alkyl chain,
# scoring 95/100 because it is small and trivially synthesisable. Nothing else
# measured here can see that. SA rewards it, QED is indifferent, and the motif
# matcher reads it as a perfectly ordinary alkyl chain.
DISQUALIFYING_GROUPS: tuple[tuple[str, str], ...] = (
    # Electrophiles that acylate or alkylate lysines and cysteines in vivo.
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]"),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
    ("isocyanate", "[NX2]=[CX2]=[OX1]"),
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
    ("epoxide", "[CX4]1O[CX4]1"),
    ("aziridine", "[CX4]1N[CX4]1"),
    # Unstable or tautomerising, so the drawn structure is not what you get.
    ("allene", "[CX2]=[CX2]=[CX2]"),
    ("vinyl_ether", "[OX2;!$(O=*)][CX3]=[CX3]"),
    ("enol", "[OX2H][CX3]=[CX3]"),
    ("imine", "[CX3]=[NX2;H1,H0;!$(N-a);!$(N=C-N)]"),
    ("peroxide", "[OX2][OX2]"),
    ("nitroso", "[NX2]=[OX1]"),
)

# Groups that are a real liability but can be deliberate -- a click handle, a
# conjugation point, a solubilising amine. Scored down in `score.py` rather
# than gated, because rejecting them would throw away legitimate designs.
LIABLE_GROUPS: tuple[tuple[str, str], ...] = (
    ("alkyne", "[CX2]#[CX2]"),
    ("ketone", "[CX4][CX3](=[OX1])[CX4]"),
    ("hydrazine", "[NX3;!$(N=*)][NX3;!$(N=*)]"),
    ("free_thiol", "[SX2H]"),
    ("primary_amine", "[NX3;H2;!$(NC=O);!$(N=*)]"),
    ("azide", "[NX2]=[NX2+]=[NX1-]"),
    ("disulfide", "[SX2][SX2]"),
)

SOFT_FAILURES = {
    "reactive_linker_group": "the linker carries a reactive or unstable group",
    "sa_too_high": "harder to synthesise than the threshold allows",
    "qed_too_low": "drug-likeness below the threshold",
    "mw_out_of_range": "molecular weight outside the PROTAC range",
    "clogp_too_high": "too lipophilic",
    "too_flexible": "more rotatable bonds than the threshold allows",
    "pains_alert": "matches a pan-assay interference substructure",
    "bad_linker_ring": "the linker contains a ring of an implausible size",
    "macrocycle": "the linker closes a macrocycle through the fragments",
    "linker_too_short": "fewer bonds between the fragments than a ternary complex needs",
    "linker_too_long": "more bonds between the fragments than the threshold allows",
}


@dataclass
class Assessment:
    """Everything measured about one sample, flattened for a CSV row."""

    target_id: str
    sample_id: str
    stage: str  # generated | repaired
    linker_size: int

    verdict: str = "ok"  # ok | poor | broken
    failure_codes: list[str] = field(default_factory=list)
    failure_detail: str = ""
    warnings: list[str] = field(default_factory=list)

    protac_smiles: str | None = None
    linker_smiles: str | None = None
    # The two fragments, recorded here because a repaired molecule has no
    # trailing linker block to recover them from. See `repair.verify`.
    fragment_smiles: list[str] = field(default_factory=list)
    reference_smiles: str | None = None
    recovers_reference: bool | None = None

    properties: dict[str, Any] = field(default_factory=dict)
    linker: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)

    # Set only for a molecule that parsed and bridges its fragments. A broken
    # sample has no position in a ranking, so it gets no score rather than a
    # zero. See `score.py`.
    score: Any = None

    @property
    def is_broken(self) -> bool:
        return self.verdict == "broken"

    def add_failure(self, code: str, detail: str | None = None) -> None:
        if code not in self.failure_codes:
            self.failure_codes.append(code)
        text = detail or HARD_FAILURES.get(code) or SOFT_FAILURES.get(code) or code
        self.failure_detail = (
            f"{self.failure_detail}; {text}".strip("; ")
            if self.failure_detail
            else text
        )
        if code in HARD_FAILURES:
            self.verdict = "broken"
        elif self.verdict != "broken":
            self.verdict = "poor"

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target_id": self.target_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "linker_size": self.linker_size,
            "verdict": self.verdict,
            "failure_codes": ";".join(self.failure_codes),
            "failure_detail": self.failure_detail,
            "warnings": ";".join(self.warnings),
            "protac_smiles": self.protac_smiles or "",
            "linker_smiles": self.linker_smiles or "",
            "fragment_smiles": ".".join(self.fragment_smiles),
            "reference_smiles": self.reference_smiles or "",
            "recovers_reference": self.recovers_reference,
        }
        row.update({f"prop_{k}": v for k, v in self.properties.items()})
        row.update({f"linker_{k}": v for k, v in self.linker.items()})
        row.update({f"geom_{k}": v for k, v in self.geometry.items()})
        if self.score is not None:
            row.update(self.score.as_row())
        row.update(self.paths)
        return row


# ------------------------------------------------------------------ descriptors


def molecule_properties(mol: Any) -> dict[str, Any]:
    """Everything measured about the whole PROTAC."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    try:
        sa = round(sa_scorer().calculateScore(mol), 4)
    except Exception:  # noqa: BLE001 - a descriptor is not worth failing for
        sa = None
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": round(Descriptors.MolWt(mol), 2),
        "exact_mw": round(Descriptors.ExactMolWt(mol), 4),
        "sa_score": sa,
        "qed": round(Descriptors.qed(mol), 4),
        "clogp": round(Crippen.MolLogP(mol), 4),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": rings,
        "aromatic_rings": aromatic,
        "aliphatic_rings": rings - aromatic,
        "stereocentres": len(
            Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
        ),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
    }


_PAINS_CATALOG: Any = None


def pains_catalog() -> Any:
    """RDKit's built-in PAINS filter set."""
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        from rdkit.Chem import FilterCatalog

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def pains_hits(mol: Any) -> list[str]:
    try:
        entries = pains_catalog().GetMatches(mol)
    except Exception:  # noqa: BLE001
        return []
    return [entry.GetDescription() for entry in entries]


_GROUP_QUERIES: dict[int, list[tuple[str, Any]]] = {}


def _group_queries(table: tuple[tuple[str, str], ...]) -> list[tuple[str, Any]]:
    """Compile once; these run per molecule over thousands of molecules."""
    cached = _GROUP_QUERIES.get(id(table))
    if cached is not None:
        return cached
    from rdkit import Chem

    compiled = []
    for name, smarts in table:
        query = Chem.MolFromSmarts(smarts)
        if query is None:  # a bad pattern here is our bug, not the molecule's
            raise ValueError(f"invalid SMARTS for {name}: {smarts}")
        compiled.append((name, query))
    _GROUP_QUERIES[id(table)] = compiled
    return compiled


def groups_in_linker(
    mol: Any, linker_indices: list[int], table: tuple[tuple[str, str], ...]
) -> list[str]:
    """Which of `table`'s groups sit wholly inside the linker.

    Wholly inside on purpose. A match that straddles the junction belongs
    partly to a fragment, and the fragments are a warhead and an E3 ligand
    whose structures are not ours to judge -- flagging their chemistry would
    reject molecules for something the generator never did.
    """
    linker_set = set(linker_indices)
    if not linker_set:
        return []
    found: list[str] = []
    for name, query in _group_queries(table):
        for match in mol.GetSubstructMatches(query):
            if linker_set.issuperset(match):
                found.append(name)
                break
    return found


def implausible_linker_rings(mol: Any, linker_indices: list[int]) -> list[int]:
    """Ring sizes in the linker that a medicinal chemist would not accept.

    Three- and four-membered rings are strained, and anything above six in a
    generated linker is usually a perception artefact rather than a design.
    """
    linker_set = set(linker_indices)
    bad: list[int] = []
    for ring in mol.GetRingInfo().AtomRings():
        if not linker_set.intersection(ring):
            continue
        if len(ring) < 5 or len(ring) > 6:
            bad.append(len(ring))
    return sorted(bad)


def macrocycle_sizes(mol: Any, minimum: int = 9) -> list[int]:
    return sorted(len(r) for r in mol.GetRingInfo().AtomRings() if len(r) >= minimum)


# -------------------------------------------------------------------- the gates


def apply_soft_gates(
    assessment: Assessment, thresholds: dict[str, Any], mol: Any, linker: LinkerView
) -> None:
    """Judge a chemically real molecule against the configured thresholds."""
    properties = assessment.properties

    sa = properties.get("sa_score")
    max_sa = thresholds.get("max_sa_score")
    if sa is not None and max_sa is not None and sa > float(max_sa):
        assessment.add_failure("sa_too_high", f"SA {sa} above {max_sa}")

    qed = properties.get("qed")
    min_qed = thresholds.get("min_qed")
    if qed is not None and min_qed is not None and qed < float(min_qed):
        assessment.add_failure("qed_too_low", f"QED {qed} below {min_qed}")

    mw = properties.get("mw")
    low, high = thresholds.get("min_mw"), thresholds.get("max_mw")
    if (
        mw is not None
        and low is not None
        and high is not None
        and not float(low) <= mw <= float(high)
    ):
        assessment.add_failure("mw_out_of_range", f"MW {mw} outside {low}-{high}")

    clogp = properties.get("clogp")
    max_clogp = thresholds.get("max_clogp")
    if clogp is not None and max_clogp is not None and clogp > float(max_clogp):
        assessment.add_failure("clogp_too_high", f"cLogP {clogp} above {max_clogp}")

    rotatable = properties.get("rotatable_bonds")
    max_rot = thresholds.get("max_rotatable_bonds")
    if rotatable is not None and max_rot is not None and rotatable > int(max_rot):
        assessment.add_failure(
            "too_flexible", f"{rotatable} rotatable bonds, above {max_rot}"
        )

    if thresholds.get("reject_pains", True):
        hits = pains_hits(mol)
        if hits:
            assessment.properties["pains_hits"] = ";".join(hits[:3])
            assessment.add_failure("pains_alert", f"PAINS: {hits[0]}")

    if thresholds.get("reject_reactive_linker", True):
        reactive = groups_in_linker(mol, linker.atom_indices, DISQUALIFYING_GROUPS)
        if reactive:
            assessment.properties["reactive_groups"] = ";".join(reactive)
            assessment.add_failure(
                "reactive_linker_group",
                f"the linker contains {', '.join(reactive)}, which would react "
                "with protein nucleophiles or will not survive as drawn",
            )

    # Recorded whether or not it gates, because `score.py` charges for these
    # and the repair prompt needs to name them.
    liable = groups_in_linker(mol, linker.atom_indices, LIABLE_GROUPS)
    if liable:
        assessment.properties["liable_groups"] = ";".join(liable)

    if thresholds.get("reject_bad_linker_rings", True):
        bad = implausible_linker_rings(mol, linker.atom_indices)
        if bad:
            assessment.add_failure(
                "bad_linker_ring", f"linker ring sizes {bad} are implausible"
            )

    if thresholds.get("reject_macrocycles", True):
        macro = macrocycle_sizes(mol)
        if macro:
            assessment.add_failure(
                "macrocycle", f"ring sizes {macro} close a macrocycle"
            )

    path = linker.path_length_bonds
    if path is not None:
        shortest = thresholds.get("min_linker_path_bonds")
        longest = thresholds.get("max_linker_path_bonds")
        if shortest is not None and path < int(shortest):
            assessment.add_failure(
                "linker_too_short", f"{path} bonds between fragments, below {shortest}"
            )
        if longest is not None and path > int(longest):
            assessment.add_failure(
                "linker_too_long", f"{path} bonds between fragments, above {longest}"
            )


# ------------------------------------------------------------------ the pipeline


def assess_sample(
    target_id: str,
    sample_id: str,
    xyz_path: str | Path,
    sdf_path: str | Path,
    *,
    linker_size: int,
    thresholds: dict[str, Any],
    stage: str = "generated",
    reference_smiles: str | None = None,
) -> Assessment:
    """Measure one sample and give it a verdict.

    Every step records why it stopped. Nothing returns a bare failure.
    """
    from rdkit import Chem

    assessment = Assessment(
        target_id=target_id,
        sample_id=sample_id,
        stage=stage,
        linker_size=linker_size,
        reference_smiles=reference_smiles,
    )
    xyz_path = Path(xyz_path)
    assessment.paths = {"xyz_path": str(xyz_path), "sdf_path": str(sdf_path)}

    if not xyz_path.is_file():
        assessment.add_failure("missing_xyz", f"{xyz_path} does not exist")
        return assessment

    frame: XyzFrame | None = None
    try:
        frame = read_xyz(xyz_path)
    except Exception as exc:  # noqa: BLE001 - a malformed file is a result
        assessment.add_failure("missing_xyz", f"{type(exc).__name__}: {exc}")
        return assessment

    geometry = check_geometry(frame)
    assessment.geometry = geometry.as_dict()
    if geometry.clashing_pairs:
        # Checked before bond perception: Open Babel infers bonds from
        # distance, so overlapping atoms yield a confidently wrong molecule
        # rather than an error.
        assessment.add_failure(
            "atom_clash",
            f"{geometry.clashing_pairs} atom pair(s) closer than "
            f"{geometry.min_distance} A",
        )
        return assessment

    if linker_size > frame.n_atoms:
        assessment.add_failure(
            "linker_size_mismatch",
            f"{frame.n_atoms} atoms in the file cannot hold a "
            f"{linker_size}-atom linker",
        )
        return assessment

    result: LoadResult = load_molecule(
        xyz_path, sdf_path, expected_symbols=frame.symbols
    )
    assessment.warnings.extend(result.warnings)
    if not result.ok:
        assessment.add_failure(result.reason or "unreadable_sdf", result.detail)
        # Salvage whatever is readable. A disconnected molecule still has its
        # linker as the trailing block, and that SMILES is the single most
        # useful thing the repair agent can be handed -- it says what to
        # reattach rather than leaving the agent to guess.
        if result.mol is not None:
            try:
                assessment.protac_smiles = Chem.MolToSmiles(result.mol)
            except Exception as exc:  # noqa: BLE001 - a partial read is a bonus
                assessment.warnings.append(f"could not write SMILES: {exc}")
            try:
                salvaged = describe_linker(result.mol, linker_size)
                assessment.linker_smiles = salvaged.smiles
                assessment.fragment_smiles = describe_fragments(
                    result.mol, salvaged.atom_indices
                )
                assessment.linker = {
                    "heavy_atoms": salvaged.n_heavy_atoms,
                    "attachments": salvaged.n_attachments,
                    "attachment_atoms": ";".join(
                        str(i) for i in salvaged.attachment_atoms
                    ),
                    "fragment_anchor_atoms": ";".join(
                        str(i) for i in salvaged.fragment_anchor_atoms
                    ),
                    "path_length_bonds": salvaged.path_length_bonds,
                    "exit_vector_distance": salvaged.exit_vector_distance,
                    "rotatable_bonds": salvaged.rotatable_bonds,
                    "heteroatom_fraction": salvaged.heteroatom_fraction,
                    "in_ring_fraction": salvaged.in_ring_fraction,
                }
            except Exception as exc:  # noqa: BLE001
                assessment.warnings.append(f"could not isolate the linker: {exc}")
        return assessment

    mol = result.mol
    assessment.protac_smiles = Chem.MolToSmiles(mol)
    assessment.properties = molecule_properties(mol)

    linker = describe_linker(mol, linker_size)
    assessment.linker_smiles = linker.smiles
    assessment.fragment_smiles = describe_fragments(mol, linker.atom_indices)
    assessment.linker = {
        "heavy_atoms": linker.n_heavy_atoms,
        "attachments": linker.n_attachments,
        "attachment_atoms": ";".join(str(i) for i in linker.attachment_atoms),
        "fragment_anchor_atoms": ";".join(str(i) for i in linker.fragment_anchor_atoms),
        "path_length_bonds": linker.path_length_bonds,
        "exit_vector_distance": linker.exit_vector_distance,
        "rotatable_bonds": linker.rotatable_bonds,
        "heteroatom_fraction": linker.heteroatom_fraction,
        "in_ring_fraction": linker.in_ring_fraction,
    }

    # A linker is a bridge. One attachment means it dangles off a single
    # fragment; three or more means it branched into something else.
    if linker.n_attachments != 2:
        assessment.add_failure(
            "linker_not_bridging",
            f"the linker touches a fragment at {linker.n_attachments} atom(s), "
            "so it does not bridge the two fragments",
        )
        return assessment

    if reference_smiles:
        assessment.recovers_reference = assessment.protac_smiles == reference_smiles

    apply_soft_gates(assessment, thresholds, mol, linker)

    # Scored last, because the liabilities term reads the soft-gate failures.
    # Only molecules that got this far are scored: everything above returned
    # early, and those are `broken`.
    assessment.score = score_sample(
        mol,
        assessment.properties,
        assessment.linker,
        linker.atom_indices,
        assessment.failure_codes,
    )
    return assessment
