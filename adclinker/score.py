"""Scoring an ADC linker-payload, calibrated against real linkers.

This is the one object the whole pipeline turns on. It is REINVENT's reward,
the filter on anything a generator emits, the ranking criterion, and the
objective the LLM stage optimises against. Written once, used four times.

Every threshold here comes from `reference.py`, which measures 1,692 real ADC
linker reagents, and every one is then checked against the drug-linkers of
the 16 approved ADCs in `tests/data/approved_adcs.json`. Nothing is a number
chosen by hand, and nothing survives that rejects a marketed drug.

**What is measured, and what is gated, are different molecules.** A gate asks
whether this conjugate is a candidate at all -- payload intact, one handle,
a trigger, nothing reactive loose -- and those are properties of the whole
assembled thing. A scored term asks how good the *linker* is, and the
reference set is payload-free reagents, so the payload is excised before
anything is measured. Mixing the two was the defect this module was rewritten
to fix; `linker_portion` records what it cost.

**Two thresholds are one-sided, not windows, and that is evidence not taste.**
Hydrophobicity and polar surface are monotone for an ADC linker: the linker
is discarded on release and never crosses a membrane, so there is no point at
which being more polar starts to hurt. Enhertu's linker measures cLogP -1.71
and Trodelvy's has TPSA 340, both outside the reference band on the good
side, and both are approved. A two-sided window marked them down for the
property that makes them work.

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

**Gates versus scores.** Some properties are not tradeable. A linker-payload
that lost its payload, lost its cleavage trigger, has nothing that frees the
drug once the trigger is cut, or carries two conjugation handles is not a
worse candidate -- it is not a candidate, and averaging it into a weighted
score would let a good cLogP term hide it. Those reject.

The gates match the ones `molbench`'s `check_adc_linker` enforces, and the
chemistry behind both lives in `chemistry.py` with a test holding the two
copies identical. They have to agree: this module decides what becomes a
repair task and that tool decides whether a repair is accepted, so an agent
caught between two versions is chasing a target that moves between the task
and the grade.

**Why likeness is a floor and not a target.** Nearest-neighbour similarity to
the reference set answers "would a linker chemist recognise this". Maximising
it would return MC-Val-Cit-PAB every time, which is not a design. It is
scored as a floor, paired with an explicit novelty term, so the objective
reads "close enough to be plausible, far enough to be new".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import chemistry
from .reference import Profile, nearest_similarity

# Reasons a candidate is rejected outright rather than ranked low.
# Reactive or unstable groups that have no business in an ADC spacer.
#
# The same gate exists in `protacbench.assess`, for the same reason -- nothing
# else measured can see these -- but it cannot be the same check. An ADC's
# conjugation handle is *deliberately* reactive: a maleimide is a Michael
# acceptor, an NHS ester is an activated acyl. Those are the point. So matches
# that overlap the declared handle are exempt, and only reactive chemistry
# found elsewhere in the molecule counts against it.
DISQUALIFYING_GROUPS: tuple[tuple[str, str], ...] = (
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]"),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
    ("isocyanate", "[NX2]=[CX2]=[OX1]"),
    ("epoxide", "[CX4]1O[CX4]1"),
    ("aziridine", "[CX4]1N[CX4]1"),
    ("allene", "[CX2]=[CX2]=[CX2]"),
    ("vinyl_ether", "[OX2;!$(O=*)][CX3]=[CX3]"),
    ("enol", "[OX2H][CX3]=[CX3]"),
    ("peroxide", "[OX2][OX2]"),
    ("nitroso", "[NX2]=[OX1]"),
    # Michael acceptors are listed, and the handle exemption is what keeps
    # maleimide-based designs legal.
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
)

GATES: dict[str, str] = {
    "unparseable": "RDKit could not read the SMILES",
    "reactive_group": "a reactive or unstable group outside the conjugation handle",
    "disconnected": "the molecule is in more than one piece",
    "payload_lost": "the payload is no longer intact; this is a different drug",
    "payload_altered": (
        "the payload is present but its stereochemistry has been changed, "
        "which makes it a different drug with different potency"
    ),
    "payload_internal": (
        "the payload is in the middle of the molecule rather than at the end, "
        "so cleaving the linker cannot release it"
    ),
    "trigger_lost": "the cleavable motif is gone, so the payload can never be released",
    "no_release_chemistry": (
        "there is a cleavage trigger but nothing between it and the payload "
        "that frees the drug, so cutting the trigger leaves the payload still "
        "carrying part of the cassette"
    ),
    "bad_junction": (
        "the linker contains a junction nobody would make, usually from "
        "joining an acyl handle to a spacer that starts with a carbon"
    ),
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
    # The same descriptors for the linker alone, which is what the reference
    # windows are measured on and what every scored term is judged against.
    linker_properties: dict[str, Any] = field(default_factory=dict)
    linker_smiles: str | None = None
    payload_atoms: list[int] = field(default_factory=list)
    scored_on: str = "conjugate"  # or "linker"
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
            "scored_on": self.scored_on,
            "linker_smiles": self.linker_smiles,
        }
        row.update({f"prop_{k}": v for k, v in self.properties.items()})
        row.update({f"linker_{k}": v for k, v in self.linker_properties.items()})
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
    """The payload as a substructure query, dummies stripped.

    The same three corrections the PROTAC checker needs, for the same
    reasons: a dummy atom matches nothing, removing one from an unsanitised
    mol leaves a radical that will not match a filled valence, and an
    explicit `[H]` survives as a heavy query atom that matches nothing.
    Without them a perfectly correct conjugate reports its payload missing.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(payload_smiles, sanitize=False)
    if mol is None:
        return None
    editable = Chem.RWMol(mol)
    for index in sorted(
        (
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0 or a.GetAtomicNum() == 1
        ),
        reverse=True,
    ):
        editable.RemoveAtom(index)
    for atom in editable.GetAtoms():
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(False)
    query = editable.GetMol()
    try:
        Chem.SanitizeMol(query)
    except Exception:  # noqa: BLE001 - an unsanitisable query still matches
        pass
    return query


_QUERY_CACHE: dict[str, Any] = {}


def _cached_query(smarts: str) -> Any:
    """Compile once. These run per molecule across thousands of candidates."""
    from rdkit import Chem

    query = _QUERY_CACHE.get(smarts)
    if query is None:
        query = Chem.MolFromSmarts(smarts)
        if query is None:  # our pattern, our bug
            raise ValueError(f"invalid SMARTS in chemistry.py: {smarts!r}")
        _QUERY_CACHE[smarts] = query
    return query


def _piece_count(mol: Any, indices: list[int]) -> int:
    """How many connected pieces `indices` falls into within `mol`."""
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


def linker_portion(mol: Any, payload_atoms: list[int]) -> Any | None:
    """The conjugate with its payload excised: handle + spacer + cassette.

    **This is the only thing comparable to the reference set, and scoring
    the conjugate instead was a measurement error, not a design choice.**
    `reference.py` profiles linker *reagents* -- Mc-Val-Cit-PABC-PNP and its
    kin -- which carry no payload at all. An MMAE conjugate is MW 1460 with
    cLogP 5.5; the reference cLogP window is -0.8 to 3.1. Every candidate
    therefore landed outside every window regardless of its spacer, which
    pinned the polarity, polar-surface, size and synthesisability terms --
    60% of the total weight -- at the same value for everything.

    What that cost, measured: brentuximab vedotin's own drug-linker scored
    43/100, a C12 greasy spacer scored 42.9, and PEG8 scored 37.4. The
    scorer preferred a greasy chain to PEG, which inverts the one property
    its own docstring calls the failure mode that kills ADCs in vivo.

    Excising the payload restores the discrimination: on the same three
    spacers the real PEG2 lands inside all three windows and the C12 chain
    is correctly flagged out on cLogP.

    Returns None when the payload cannot be excised cleanly, and the caller
    falls back to the whole molecule rather than guessing.
    """
    from rdkit import Chem

    if not payload_atoms:
        return None
    keep = [i for i in range(mol.GetNumAtoms()) if i not in set(payload_atoms)]
    if not keep:
        return None
    smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=keep, canonical=True)
    return Chem.MolFromSmiles(smiles)


_REACTIVE_QUERIES: list[tuple[str, Any]] = []


def reactive_groups_outside(mol: Any, exempt: set[int]) -> list[str]:
    """Disqualifying groups whose atoms lie outside `exempt`.

    `exempt` must hold BOTH the conjugation handle and the payload, and both
    for the same reason: their chemistry is given, not designed. The handle
    is deliberately reactive -- a maleimide is a Michael acceptor, which is
    the point -- and a payload is whatever the medicinal chemistry programme
    made it.

    Exempting only the handle rejected four approved ADCs. DM1 carries an
    epoxide in its maytansinoid ring, so Kadcyla and Elahere were rejected
    for `epoxide`; calicheamicin's enediyne matches the Michael-acceptor
    pattern, so Mylotarg and Besponsa were rejected for `michael_acceptor`.
    In every case the group was a feature of the approved drug, blamed on
    the linker.
    """
    global _REACTIVE_QUERIES
    from rdkit import Chem

    if not _REACTIVE_QUERIES:
        _REACTIVE_QUERIES = [
            (name, q)
            for name, smarts in DISQUALIFYING_GROUPS
            if (q := Chem.MolFromSmarts(smarts)) is not None
        ]

    found: list[str] = []
    for name, query in _REACTIVE_QUERIES:
        for match in mol.GetSubstructMatches(query):
            if not exempt.intersection(match):
                found.append(name)
                break
    return found


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
    #
    # Stereochemistry is checked separately and reported separately. MMAE has
    # ten stereocentres and its potency depends on every one of them, so an
    # epimer is a different drug -- but it is a drug that is *present*, and
    # calling it "lost" sends a repair agent to rebuild something already
    # there. Matching without `useChirality` silently accepts the epimer,
    # which is the worse of the two errors and the one this had.
    if context.payload_smiles:
        query = _payload_query(context.payload_smiles)
        if query is not None and query.GetNumAtoms():
            flat_match = mol.GetSubstructMatch(query)
            if not flat_match:
                score.gates_failed.append("payload_lost")
                score.gate_detail = (
                    "the payload substructure is not present intact"
                )
            else:
                score.payload_atoms = list(flat_match)
                if not mol.GetSubstructMatch(query, useChirality=True):
                    score.gates_failed.append("payload_altered")
                    score.gate_detail = score.gate_detail or (
                        "the payload is present but at least one stereocentre "
                        "has been inverted"
                    )
                # A payload in the middle of the chain cannot be released by
                # cutting the linker, however good the cassette is. Excising
                # it leaves two pieces, which is how that is visible at all.
                remainder = [
                    i for i in range(mol.GetNumAtoms()) if i not in set(flat_match)
                ]
                if remainder and _piece_count(mol, remainder) > 1:
                    score.gates_failed.append("payload_internal")
                    score.gate_detail = score.gate_detail or (
                        "the payload sits between two parts of the linker "
                        "rather than at its end"
                    )

    # The release trigger. Without it a cleavable design never releases, and
    # no descriptor would reveal that.
    #
    # Matched against the shared table in `chemistry.py` rather than this
    # module's own list, so the scorer and the molbench tool that grades a
    # repair agree on what a trigger is. A trigger wholly inside the payload
    # does not count -- DM1 carries a carbamate in its ansa ring, and a bond
    # within the drug does not release the drug.
    payload_set = set(score.payload_atoms)
    score.triggers = [
        name
        for name, smarts in chemistry.ADC_TRIGGERS
        if any(
            not payload_set.issuperset(match)
            for match in mol.GetSubstructMatches(_cached_query(smarts))
        )
    ]
    if context.require_cleavable and not score.triggers:
        score.gates_failed.append("trigger_lost")
        score.gate_detail = (
            score.gate_detail or "no lysosomal cleavage motif is present"
        )
    elif context.require_cleavable and score.payload_atoms:
        # A trigger is not release. A cassette bonded straight to the payload
        # by an amide is cut and leaves the payload still wearing an amino
        # acid; it looks perfect in every descriptor and does nothing.
        junction = {
            atom
            for bond in mol.GetBonds()
            for atom in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if (bond.GetBeginAtomIdx() in payload_set)
            != (bond.GetEndAtomIdx() in payload_set)
        }
        frees = False
        for _name, smarts, releases, _note in chemistry.RELEASE_CHEMISTRIES:
            if not releases:
                continue
            for match in mol.GetSubstructMatches(_cached_query(smarts)):
                found = set(match)
                if found & payload_set and found - payload_set and found & junction:
                    frees = True
                    break
            if frees:
                break

        # A peptide trigger that reaches the junction releases the payload by
        # being cut itself -- there is no spacer because none is needed. That
        # is how Decnupaz's Ala-Ala works, and rejecting it would reject an
        # approved drug. The cleavage is slow when the payload is bulky,
        # which is the reason the p-aminobenzyl spacer exists, but slow is
        # not the same as impossible and this is not the place to gate on it.
        if not frees:
            for name, smarts in chemistry.ADC_TRIGGERS:
                if any(
                    set(match) & junction
                    for match in mol.GetSubstructMatches(_cached_query(smarts))
                ):
                    frees = True
                    break

        if not frees:
            score.gates_failed.append("no_release_chemistry")
            score.gate_detail = score.gate_detail or (
                "nothing between the trigger and the payload frees the drug"
            )

    # Exactly one reactive handle: none cannot be conjugated, two crosslinks.
    handle_atoms: set[int] = set()
    if context.handle_smarts:
        query = Chem.MolFromSmarts(context.handle_smarts)
        if query is not None:
            matches = mol.GetSubstructMatches(query)
            for match in matches:
                handle_atoms.update(match)
            if not matches:
                score.gates_failed.append("handle_lost")
            elif len(matches) > 1:
                score.gates_failed.append("handle_duplicated")
                score.gate_detail = (
                    score.gate_detail or f"{len(matches)} reactive handles"
                )

    # Reactive chemistry anywhere other than the handle. The handle's own
    # atoms are excluded, which is what lets a maleimide -- a Michael acceptor
    # by construction, and the single most common ADC conjugation chemistry --
    # pass while a stray aldehyde in the spacer does not.
    # Junctions the generated spacer created, checked on the linker only --
    # a payload is entitled to its own ketones. The same table the molbench
    # tool uses, so a candidate the scorer passes is not one the tool then
    # rejects.
    linker_only = {i for i in range(mol.GetNumAtoms()) if i not in payload_set}
    junctions: list[str] = []
    for name, smarts, _why in chemistry.BAD_JUNCTIONS:
        for match in mol.GetSubstructMatches(_cached_query(smarts)):
            if linker_only.issuperset(match) and not handle_atoms.intersection(match):
                junctions.append(name)
                break
    if junctions:
        score.gates_failed.append("bad_junction")
        score.gate_detail = score.gate_detail or ", ".join(junctions)
        score.properties["bad_junctions"] = ";".join(junctions)

    stray = reactive_groups_outside(mol, handle_atoms | payload_set)
    if stray:
        score.gates_failed.append("reactive_group")
        score.gate_detail = score.gate_detail or (
            f"{', '.join(stray)} outside the conjugation handle"
        )
        score.properties["reactive_groups"] = ";".join(stray)


def _descriptors(mol: Any) -> dict[str, Any]:
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mw": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def score_molecule(mol: Any, context: Context) -> ADCScore:
    """Gate the whole conjugate, then score the linker inside it.

    The two halves judge different molecules on purpose. A gate asks whether
    this conjugate is a candidate at all -- payload intact, handle present
    once, trigger present, nothing reactive loose -- and those are properties
    of the assembled thing. A scored term asks how good the *linker* is, and
    the windows it is judged against were measured on payload-free reagents,
    so it has to be the payload-free portion that is measured. See
    `linker_portion` for what mixing the two cost.
    """
    from rdkit import Chem

    score = ADCScore(total=None)
    if mol is None:
        score.gates_failed.append("unparseable")
        score.gate_detail = GATES["unparseable"]
        score.reason = "rejected: " + GATES["unparseable"]
        return score

    check_gates(mol, context, score)

    properties = _descriptors(mol)
    score.properties = properties

    if score.gates_failed:
        score.reason = "rejected: " + "; ".join(
            GATES.get(code, code) for code in score.gates_failed
        )
        return score

    # The payload is a constant across every candidate for one design, so
    # leaving it in adds the same offset to every bulk descriptor while
    # destroying the calibration. Take it out.
    judged = mol
    linker_mol = linker_portion(mol, score.payload_atoms)
    if linker_mol is not None and linker_mol.GetNumHeavyAtoms():
        judged = linker_mol
        score.scored_on = "linker"
        score.linker_properties = _descriptors(linker_mol)
        score.linker_smiles = Chem.MolToSmiles(linker_mol)
    elif context.payload_smiles:
        # Say so rather than silently reverting to the broken comparison.
        score.scored_on = "conjugate (payload could not be excised)"

    profile = context.profile
    terms: list[Term] = []
    # Everything from here down is measured on `judged` -- the linker alone
    # wherever the payload could be excised -- because that is the molecule
    # the reference windows describe.
    measured = score.linker_properties or properties
    where = "linker" if score.scored_on == "linker" else "conjugate"

    # One-sided, not a window, for the same reason the gate is a ceiling:
    # hydrophobicity is monotone. Full credit at or below the reference
    # median, falling to zero at the ceiling.
    #
    # A two-sided window penalised exactly the wrong molecules. Enhertu's
    # linker measures cLogP -1.71, below the reference band, and scored 66.7
    # against 82.4 for Adcetris -- it was marked down for being more polar
    # than a typical reagent, which is the property that lets it carry eight
    # payloads per antibody instead of four.
    ceiling = chemistry.LINKER_CLOGP_CEILING
    median_clogp = profile.properties["clogp"].median
    terms.append(
        Term(
            "polarity",
            1.0
            if measured["clogp"] <= median_clogp
            else _ramp(measured["clogp"], good=median_clogp, bad=ceiling),
            WEIGHTS["polarity"][0],
            f"{where} cLogP {measured['clogp']} against a ceiling of "
            f"{ceiling} (reference median {median_clogp:.2f})",
        )
    )

    # Fsp3 is scored softly here and is deliberately NOT in
    # `chemistry.JUDGED_PROPERTIES`, which governs what the molbench tool
    # treats as a pass/fail. The distinction is the point: the reagent band
    # [0.50-1.00] is outside for 10 of the 16 approved ADCs, because the
    # reference set is bare PEG and alkyl while a finished linker carries an
    # aromatic spacer and a dipeptide. As a hard check that would reject the
    # standard of care. As a ranking signal with the window's own taper it
    # costs those 16 an average of 1.2 points out of 100, which is a nudge
    # away from fully aromatic linkers rather than a verdict.
    low, high = profile.window("fsp3")
    terms.append(
        Term(
            "saturation",
            _window(measured["fsp3"], low, high),
            WEIGHTS["saturation"][0],
            f"{where} Fsp3 {measured['fsp3']} (reference median "
            f"{profile.properties['fsp3'].median:.2f})",
        )
    )

    similarity = nearest_similarity(judged, context.reference_fingerprints)
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

    # A floor, not a window, and for the same reason. Polar surface helps an
    # ADC linker and there is no amount that starts to hurt: Trodelvy's
    # linker has TPSA 340, well above the reference band, and is approved.
    low, _median, _high = (
        profile.window("tpsa")[0],
        profile.properties["tpsa"].median,
        profile.window("tpsa")[1],
    )
    terms.append(
        Term(
            "polar_surface",
            1.0 if measured["tpsa"] >= _median else _ramp(
                measured["tpsa"], good=_median, bad=low * 0.5
            ),
            WEIGHTS["polar_surface"][0],
            f"{where} TPSA {measured['tpsa']} (reference median {_median:.0f})",
        )
    )

    low, high = profile.window("heavy_atoms")
    terms.append(
        Term(
            "size",
            _window(float(measured["heavy_atoms"]), low, high),
            WEIGHTS["size"][0],
            f"{measured['heavy_atoms']} {where} heavy atoms",
        )
    )

    sa = _sa_score(judged)
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
