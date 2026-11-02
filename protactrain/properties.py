"""What a proposed PROTAC is worth, measured.

`moltrain.oracle.Properties` is built around a pocket: its fields are
`binding_affinity` and `ligand_efficiency`, and its cache key carries a
receptor, an engine and an exhaustiveness. None of that applies to a linker
task, where there is no single pocket to dock into -- a PROTAC binds two
proteins at once and the thing that decides whether it works is the ternary
complex, which docking does not model. `protacbench/OPTIMISATION.md` says so
explicitly and declines to put docking on the linker tool list.

So this is the linker analogue. The measurement itself is not reimplemented:
`protacbench.grade.measure` already returns every whole-molecule descriptor,
every linker metric, the fragments-intact verdict and the reactive-group scan
in one call, and it is the same function the hand-run grading loop scores
replies with. Reimplementing it here would let the trained objective and the
reported result drift apart, which is the one failure this package cannot
detect from the inside.

`rewards._value` reaches fields by `getattr`, so every name a threshold can be
written against has to be an attribute on this dataclass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Fields `protacbench.grade.measure` returns under a different name than the
# one a threshold is written against. Keeping the mapping explicit means a
# rename upstream fails loudly here rather than silently measuring nothing:
# a `getattr` miss returns None, and a None threshold is skipped, so a typo
# would quietly delete a reward term.
_FROM_MEASURE: dict[str, str] = {
    "span": "span",
    "linker_heavy_atoms": "linker_heavy_atoms",
    "linker_rotatable_bonds": "linker_rotatable_bonds",
    "n_attachments": "attachments",
    "linker_smiles": "linker_smiles",
    "canonical": "canonical",
    "mw": "mw",
    "clogp": "clogp",
    "tpsa": "tpsa",
    "sa_score": "sa_score",
    "qed": "qed",
    "rotatable_bonds": "rotatable_bonds",
    "hbd": "hbd",
    "hba": "hba",
    "heavy_atoms": "heavy_atoms",
    "rings": "rings",
    "aromatic_rings": "aromatic_rings",
    "aliphatic_rings": "aliphatic_rings",
    "fraction_csp3": "fraction_csp3",
    "stereocentres": "stereocentres",
    "formal_charge": "formal_charge",
    "exact_mw": "exact_mw",
}


@dataclass
class LinkerProperties:
    """Everything a linker reward term may be written against.

    `measurable` is False when the molecule could not be measured at all --
    it did not parse, it came back in pieces, or a fragment went missing. The
    reward turns that into a floor rather than trying to score the numbers,
    because a molecule with a deleted warhead scores *better* on every
    descriptor here and is not the compound.
    """

    smiles: str
    measurable: bool = True
    error: str = ""

    # --- the identity checks. These are not quality, they are admissibility.
    fragments_intact: bool = False
    n_attachments: int | None = None
    reactive_groups: list[str] = field(default_factory=list)
    liability_groups: list[str] = field(default_factory=list)
    longest_alkyl_run: int | None = None

    # --- linker geometry and composition
    linker_smiles: str | None = None
    span: int | None = None
    linker_heavy_atoms: int | None = None
    linker_rotatable_bonds: int | None = None

    # --- whole-molecule descriptors
    canonical: str | None = None
    mw: float | None = None
    exact_mw: float | None = None
    clogp: float | None = None
    tpsa: float | None = None
    sa_score: float | None = None
    qed: float | None = None
    rotatable_bonds: int | None = None
    hbd: int | None = None
    hba: int | None = None
    heavy_atoms: int | None = None
    rings: int | None = None
    aromatic_rings: int | None = None
    aliphatic_rings: int | None = None
    fraction_csp3: float | None = None
    stereocentres: int | None = None
    formal_charge: int | None = None

    @property
    def admissible(self) -> bool:
        """Does this even count as an answer to the task it was set?

        Deliberately stricter than `measurable`: a molecule can be perfectly
        measurable and still not be a repair of the PROTAC it was handed.
        """
        return (
            self.measurable
            and self.fragments_intact
            and self.n_attachments == 2
            and not self.reactive_groups
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_measurement(smiles: str, measured: dict[str, Any]) -> LinkerProperties:
    """Turn `protacbench.grade.measure`'s dict into the dataclass.

    `measure` reports failure by putting an `error` key in an otherwise
    populated dict, so a partial result still carries whatever descriptors it
    managed to compute. That is deliberate upstream and worth preserving:
    knowing a rejected molecule's cLogP is how you tell a near-miss from a
    molecule that was never close.
    """
    error = str(measured.get("error") or "")
    props = LinkerProperties(smiles=smiles, measurable=not error, error=error)

    for attr, key in _FROM_MEASURE.items():
        if key in measured and measured[key] is not None:
            setattr(props, attr, measured[key])

    # `measure` reports these as lists of (name, smarts) pairs or plain names
    # depending on the group table; keep only the names.
    props.reactive_groups = _names(measured.get("reactive"))
    props.liability_groups = _names(measured.get("grader_only"))
    if measured.get("longest_alkyl_run") is not None:
        props.longest_alkyl_run = int(measured["longest_alkyl_run"])

    # measure() only reaches the fragment match when the molecule parsed and
    # was connected, and it returns an error the moment a fragment is missing.
    props.fragments_intact = bool(
        not error and measured.get("linker_smiles") is not None
    )
    return props


def _names(raw: Any) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and item:
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def measure(smiles: str, fragment_smiles: list[str]) -> LinkerProperties:
    """Measure one proposed PROTAC against the two fragments it must keep.

    Imported lazily so that this module can be read, and the dataclass used,
    on a machine with no RDKit -- which is how the tests run on the Mac.
    """
    from protacbench.grade import measure as _measure

    try:
        raw = _measure(smiles, list(fragment_smiles))
    except Exception as exc:  # noqa: BLE001 - a measurement must never raise
        return LinkerProperties(
            smiles=smiles, measurable=False, error=f"{type(exc).__name__}: {exc}"
        )
    return from_measurement(smiles, raw)
