"""Scoring an ADC linker-payload, calibrated against real linkers.

This is the one object the whole pipeline turns on. It is REINVENT's reward,
the filter on anything a generator emits, the ranking criterion, and the
objective the LLM stage optimises against. Written once, used four times.

Every window here comes from `reference.py`, which measures roughly 1,700
real ADC linker reagents. Nothing is a number I chose.

**Why this is not `protacbench.score` with different constants.** It reads
the same descriptors and inverts most of them. A PROTAC linker is a geometric
problem: it has to hold two warheads at a separation that lets a ternary
complex form, so span is decisive and 25 rotatable bonds is a failure. An ADC
linker is a chemical problem: nothing holds the payload anywhere, so geometry
barely matters, and the reference set's *median* is 25 rotatable bonds with
Fsp3 0.88 -- long, saturated, floppy chains. Scoring one with the other's
thresholds would reject every real ADC linker. The helpers are duplicated
rather than shared for the same reason: tuning one must not silently move the
other.

**Gates versus scores.** Three properties are not tradeable. A linker-payload
that lost its payload, lost its cleavage trigger, or carries two reactive
handles is not a worse candidate -- it is not a candidate, and averaging it
into a weighted score would let a high cLogP term hide it. Those reject.

**Why likeness is a floor and not a target.** Nearest-neighbour similarity to
the reference set answers "would a linker chemist recognise this". Maximising
it would return MC-Val-Cit-PAB every time, which is not a design. It is
scored as a floor, paired with an explicit novelty term, so the objective
reads "close enough to be plausible, far enough to be new".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .reference import Profile, cleavable_motifs, nearest_similarity

# Reasons a candidate is rejected outright rather than ranked low.
GATES: dict[str, str] = {
    "unparseable": "RDKit could not read the SMILES",
    "disconnected": "the molecule is in more than one piece",
    "payload_lost": "the payload is no longer intact; this is a different drug",
    "trigger_lost": "the cleavable motif is gone, so the payload can never be released",
    "handle_lost": "no reactive handle, so it cannot be conjugated",
    "handle_duplicated": "more than one reactive handle, which would crosslink",
    "too_small": "smaller than any linker in the reference set",
}

# name -> (weight, what it is for)
#
# Polarity carries the most weight, and that is the deliberate inversion from
# the PROTAC scorer. Linker-payload hydrophobicity drives aggregation and
# accelerated clearance, and it is the failure mode that kills ADCs in vivo.
# A model optimising anything else will drift greasy, because burying
# hydrophobic surface looks favourable on almost every other metric.
WEIGHTS: dict[str, tuple[float, str]] = {
    "polarity": (30.0, "cLogP against the reference window"),
    "saturation": (15.0, "Fsp3 -- real linkers are aliphatic chains, not ring systems"),
    "adc_likeness": (15.0, "nearest-neighbour similarity to real linkers, as a floor"),
    "novelty": (10.0, "distance from the nearest known linker"),
    "polar_surface": (10.0, "TPSA, which tracks solubility"),
    "size": (10.0, "heavy atoms against the reference window"),
    "synthesisability": (10.0, "RDKit SA score"),
}


def _ramp(value: float, good: float, bad: float) -> float:
    """1.0 at `good`, 0.0 at `bad`, linear between; either direction."""
    if good == bad:
        return 1.0
    return max(0.0, min(1.0, (value - bad) / (good - bad)))


def _window(value: float, low: float, high: float, slack: float = 0.5) -> float:
    """1.0 inside [low, high], tapering outside over `slack` of the width.

    The reference set's 10th-90th band becomes the plateau, and the taper is
    proportional to the band's own width so that a tight distribution is
    judged tightly and a loose one loosely.
    """
    if high <= low:
        return 1.0
    margin = max((high - low) * slack, 1e-6)
    if value < low:
        return _ramp(value, low, low - margin)
    if value > high:
        return _ramp(value, high, high + margin)
    return 1.0


@dataclass
class Term:
    name: str
    score: float  # 0-1
    weight: float
    note: str

    @property
    def contribution(self) -> float:
        return round(self.score * self.weight, 3)


@dataclass
class ADCScore:
    """The verdict on one linker-payload."""

    total: float | None  # 0-100, or None when a gate rejected it
    gates_failed: list[str] = field(default_factory=list)
    gate_detail: str = ""
    terms: list[Term] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None
    triggers: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return not self.gates_failed and self.total is not None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "score_total": self.total,
            "accepted": self.accepted,
            "gates_failed": ";".join(self.gates_failed),
            "gate_detail": self.gate_detail,
            "score_reason": self.reason,
            "nearest_similarity": self.similarity,
            "cleavable_triggers": ";".join(self.triggers),
        }
        row.update({f"prop_{k}": v for k, v in self.properties.items()})
        for term in self.terms:
            row[f"score_{term.name}"] = round(term.score, 4)
        return row

    def weakest(self, count: int = 2) -> list[Term]:
        return sorted(
            self.terms, key=lambda t: (1.0 - t.score) * t.weight, reverse=True
        )[:count]


@dataclass
class Context:
    """What a candidate is judged against: the pair it was designed for."""

    profile: Profile
    reference_fingerprints: list[Any] = field(default_factory=list)
    payload_smiles: str | None = None  # with its attachment dummy removed
    handle_smarts: str | None = None
    require_cleavable: bool = True
    min_heavy_atoms: int = 12


def _payload_query(payload_smiles: str) -> Any:
    """The payload as a substructure query, dummies stripped."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(payload_smiles, sanitize=False)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    for index in sorted(
        (a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0), reverse=True
    ):
        editable.RemoveAtom(index)
    query = editable.GetMol()
    try:
        Chem.SanitizeMol(query)
    except Exception:  # noqa: BLE001 - an unsanitisable query still matches
        pass
    return query


def check_gates(mol: Any, context: Context, score: ADCScore) -> None:
    """Apply the non-tradeable checks. Populates `score.gates_failed`."""
    from rdkit import Chem

    pieces = Chem.GetMolFrags(mol)
    if len(pieces) > 1:
        score.gates_failed.append("disconnected")
        score.gate_detail = f"the molecule is in {len(pieces)} pieces"
        return

    if mol.GetNumHeavyAtoms() < context.min_heavy_atoms:
        score.gates_failed.append("too_small")
        score.gate_detail = (
            f"{mol.GetNumHeavyAtoms()} heavy atoms, below the floor of "
            f"{context.min_heavy_atoms}"
        )

    # The payload must survive unchanged. A generator or an LLM that
    # "improves" the molecule by editing the drug has produced something that
    # scores well and is not the compound you wanted.
    if context.payload_smiles:
        query = _payload_query(context.payload_smiles)
        if query is not None and query.GetNumAtoms():
            if not mol.HasSubstructMatch(query):
                score.gates_failed.append("payload_lost")
                score.gate_detail = (
                    "the payload substructure is not present intact"
                )

    # The release trigger. Without it a cleavable design never releases, and
    # no descriptor would reveal that.
    score.triggers = cleavable_motifs(mol)
    if context.require_cleavable and not score.triggers:
        score.gates_failed.append("trigger_lost")
        score.gate_detail = (
            score.gate_detail or "no lysosomal cleavage motif is present"
        )

    # Exactly one reactive handle: none cannot be conjugated, two crosslinks.
    if context.handle_smarts:
        query = Chem.MolFromSmarts(context.handle_smarts)
        if query is not None:
            matches = mol.GetSubstructMatches(query)
            if not matches:
                score.gates_failed.append("handle_lost")
            elif len(matches) > 1:
                score.gates_failed.append("handle_duplicated")
                score.gate_detail = (
                    score.gate_detail or f"{len(matches)} reactive handles"
                )


def score_molecule(mol: Any, context: Context) -> ADCScore:
    """Gate, then score, one assembled linker-payload."""
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    score = ADCScore(total=None)
    if mol is None:
        score.gates_failed.append("unparseable")
        score.gate_detail = GATES["unparseable"]
        score.reason = "rejected: " + GATES["unparseable"]
        return score

    check_gates(mol, context, score)

    properties = {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }
    score.properties = properties

    if score.gates_failed:
        score.reason = "rejected: " + "; ".join(
            GATES.get(code, code) for code in score.gates_failed
        )
        return score

    profile = context.profile
    terms: list[Term] = []

    low, high = profile.window("clogp")
    terms.append(
        Term(
            "polarity",
            _window(properties["clogp"], low, high),
            WEIGHTS["polarity"][0],
            f"cLogP {properties['clogp']} against reference {low:.1f} to {high:.1f}",
        )
    )

    low, high = profile.window("fsp3")
    terms.append(
        Term(
            "saturation",
            _window(properties["fsp3"], low, high),
            WEIGHTS["saturation"][0],
            f"Fsp3 {properties['fsp3']} (reference median "
            f"{profile.properties['fsp3'].median:.2f})",
        )
    )

    similarity = nearest_similarity(mol, context.reference_fingerprints)
    score.similarity = round(similarity, 4)
    # A floor, not a target: full credit once it is recognisably a linker.
    terms.append(
        Term(
            "adc_likeness",
            _ramp(similarity, good=0.45, bad=0.15),
            WEIGHTS["adc_likeness"][0],
            f"nearest real linker Tanimoto {similarity:.3f}",
        )
    )
    # And the other half of the same coin, so the optimum is not a copy.
    terms.append(
        Term(
            "novelty",
            _ramp(1.0 - similarity, good=0.65, bad=0.25),
            WEIGHTS["novelty"][0],
            "identical to a known linker" if similarity > 0.95 else "distinct",
        )
    )

    low, high = profile.window("tpsa")
    terms.append(
        Term(
            "polar_surface",
            _window(properties["tpsa"], low, high),
            WEIGHTS["polar_surface"][0],
            f"TPSA {properties['tpsa']}",
        )
    )

    low, high = profile.window("heavy_atoms")
    terms.append(
        Term(
            "size",
            _window(float(properties["heavy_atoms"]), low, high),
            WEIGHTS["size"][0],
            f"{properties['heavy_atoms']} heavy atoms",
        )
    )

    sa = _sa_score(mol)
    terms.append(
        Term(
            "synthesisability",
            0.5 if sa is None else _ramp(sa, good=3.0, bad=7.0),
            WEIGHTS["synthesisability"][0],
            "SA could not be computed" if sa is None else f"SA {sa:.2f}",
        )
    )

    score.terms = terms
    total_weight = sum(t.weight for t in terms)
    score.total = round(
        sum(t.contribution for t in terms) / total_weight * 100, 2
    )
    score.reason = build_reason(score)
    return score


_SASCORER: Any = None


def _sa_score(mol: Any) -> float | None:
    """RDKit's contributed synthetic accessibility scorer."""
    global _SASCORER
    if _SASCORER is None:
        import sys

        from rdkit.Chem import RDConfig

        sys.path.append(f"{RDConfig.RDContribDir}/SA_Score")
        try:
            import sascorer  # type: ignore

            _SASCORER = sascorer
        except Exception:  # noqa: BLE001 - a missing contrib is not fatal
            _SASCORER = False
    if not _SASCORER:
        return None
    try:
        return float(_SASCORER.calculateScore(mol))
    except Exception:  # noqa: BLE001
        return None


def build_reason(score: ADCScore) -> str:
    if score.gates_failed:
        return "rejected: " + "; ".join(score.gates_failed)
    if score.total is None:
        return "not scored"
    weak = [t for t in score.weakest(2) if t.score < 0.85]
    if not weak:
        return f"{score.total}: nothing objectionable"
    detail = "; ".join(f"{t.name} ({t.note})" for t in weak)
    if score.total >= 85.0:
        return f"{score.total}: good, with {detail}"
    if score.total >= 60.0:
        return f"{score.total}: usable, held back by {detail}"
    return f"{score.total}: weak on {detail}"


def score_smiles(smiles: str, context: Context) -> ADCScore:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return score_molecule(mol, context)


def reward(smiles: str, context: Context) -> float:
    """The scalar REINVENT optimises: 0-1, with a rejected candidate at zero.

    Zero rather than a small number on purpose. A gate failure is not a near
    miss, and letting it score 0.3 would let RL trade a lost payload against a
    good cLogP.
    """
    result = score_smiles(smiles, context)
    if not result.accepted or result.total is None:
        return 0.0
    return result.total / 100.0


def rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accepted candidates first, best score first; rejects keep their place."""

    def key(row: dict[str, Any]) -> tuple[int, float]:
        accepted = bool(row.get("accepted"))
        total = row.get("score_total")
        return (0 if accepted else 1, -float(total) if total is not None else 0.0)

    ordered = sorted(rows, key=key)
    position = 0
    for row in ordered:
        if row.get("score_total") is None or not row.get("accepted"):
            row["rank"] = None
            continue
        position += 1
        row["rank"] = position
    return ordered
