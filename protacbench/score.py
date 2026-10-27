"""A synthesis-plausibility score, so samples can be ranked rather than only
sorted into `ok` / `poor` / `broken`.

The triage in `assess.py` answers "is this molecule real, and is it worth
keeping". That is a verdict, not an ordering, and a verdict cannot tell you
which of four hundred `ok` PROTACs a chemist should make first. This module
adds the ordering.

Six terms, each scored 0-1 and then weighted. Every term reports its own
number and a sentence, because a single composite with no breakdown is not
reviewable -- a chemist who disagrees with a ranking needs to see which term
caused it.

The weights encode one opinion worth stating plainly: for a PROTAC the
limiting factor is almost never drug-likeness, it is whether the linker is
made of chemistry a lab already runs. QED is deliberately absent. PROTACs
violate Lipinski by construction, so QED measures how unlike a PROTAC the
molecule is, and rewarding it would rank the worst candidates highest. SA and
motif familiarity carry 55% of the weight between them for the same reason.

The score is only defined for a molecule that parsed. A `broken` sample gets
`None`, not a zero: zero is a position in the ranking and these molecules do
not have one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Linker chemistry that a medicinal chemistry lab actually runs. A linker
# built from these is one somebody can order reagents for on Monday; a linker
# built from anything else is a research project before it is a compound.
#
# Each entry is matched against the linker atoms only. Coverage is the
# fraction of linker heavy atoms touched by at least one match, so a linker
# that is entirely PEG-and-amide scores 1.0 and one full of exotic ring
# systems scores near 0.
FAMILIAR_MOTIFS: tuple[tuple[str, str], ...] = (
    ("amide", "[CX3](=[OX1])[NX3]"),
    ("peg_ether", "[CX4;!R][OX2;!R][CX4;!R]"),
    ("alkyl_chain", "[CX4;!R;H2]"),
    ("piperazine", "[NX3;R][CX4;R][CX4;R][NX3;R][CX4;R][CX4;R]"),
    ("piperidine", "[NX3;R]1[CX4;R][CX4;R][CX4;R][CX4;R][CX4;R]1"),
    ("triazole", "c1cnnn1"),
    ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]"),
    ("carbamate", "[NX3][CX3](=[OX1])[OX2]"),
    ("ester", "[CX3](=[OX1])[OX2][#6]"),
    ("aryl_ether", "c[OX2][#6]"),
    ("secondary_amine", "[NX3;H1;!$([NX3][CX3]=[OX1])]"),
    ("aromatic_ring", "c1ccccc1"),
)

# name -> (weight, what the term is for)
#
# SA carries less than motif familiarity on purpose. SA is close to flat
# across generated linkers -- a long alkyl chain and a PEG both score about 2,
# because both are genuinely easy to make -- so on its own it does not
# separate a good linker from a useless one. Motif coverage and the exit
# vector do.
WEIGHTS: dict[str, tuple[float, str]] = {
    "synthesisability": (25.0, "RDKit SA score, the headline difficulty number"),
    "motif_familiarity": (25.0, "how much of the linker is routine lab chemistry"),
    "exit_vector_span": (20.0, "bond span between the fragments"),
    "polarity": (10.0, "cLogP, which drives solubility and purification"),
    "flexibility": (10.0, "rotatable bonds"),
    "liabilities": (10.0, "PAINS, strained rings, macrocycles, new stereocentres"),
}

# Ranking is by verdict first. The score says how plausible a molecule is to
# make; it does not say whether the thing you would be making is a PROTAC.
# A three-bond linker is trivially synthesisable and cannot possibly work, so
# ordering on the score alone would put it above a clean molecule.
VERDICT_ORDER: dict[str, int] = {"ok": 0, "poor": 1, "broken": 2}


def _ramp(value: float, good: float, bad: float) -> float:
    """1.0 at `good`, 0.0 at `bad`, linear between. Works in either direction."""
    if good == bad:
        return 1.0
    fraction = (value - bad) / (good - bad)
    return max(0.0, min(1.0, fraction))


def _window(value: float, low_bad: float, low_ok: float, high_ok: float, high_bad: float) -> float:
    """A plateau of 1.0 between `low_ok` and `high_ok`, tapering to 0 outside."""
    if value < low_ok:
        return _ramp(value, low_ok, low_bad)
    if value > high_ok:
        return _ramp(value, high_ok, high_bad)
    return 1.0


@dataclass
class Term:
    """One scored component, with the number that produced it."""

    name: str
    score: float  # 0-1
    weight: float
    note: str

    @property
    def contribution(self) -> float:
        return round(self.score * self.weight, 3)


@dataclass
class SynthesisScore:
    """The composite, the terms behind it, and why it came out that way."""

    total: float | None  # 0-100, or None when the molecule did not parse
    terms: list[Term] = field(default_factory=list)
    motif_coverage: float | None = None
    matched_motifs: list[str] = field(default_factory=list)
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "score_total": self.total,
            "score_reason": self.reason,
            "score_motifs": ";".join(self.matched_motifs),
            "score_motif_coverage": self.motif_coverage,
        }
        for term in self.terms:
            row[f"score_{term.name}"] = round(term.score, 4)
        return row

    def weakest_terms(self, count: int = 2) -> list[Term]:
        """The terms dragging the score down, worst first. Weighted, because a
        bad score on a 10-point term matters less than a mediocre one on a 30.
        """
        scored = [t for t in self.terms if t.weight > 0]
        return sorted(scored, key=lambda t: (1.0 - t.score) * t.weight, reverse=True)[
            :count
        ]


def motif_coverage(mol: Any, linker_indices: list[int]) -> tuple[float, list[str]]:
    """Fraction of linker heavy atoms covered by familiar chemistry, and which
    motifs matched.

    Coverage, not a match count: one amide in a twenty-atom exotic linker
    should not read as "familiar", and counting matches would say it does.
    """
    from rdkit import Chem

    if not linker_indices:
        return 0.0, []

    linker_set = set(linker_indices)
    covered: set[int] = set()
    matched: list[str] = []

    for name, smarts in FAMILIAR_MOTIFS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:  # a bad SMARTS is a bug here, not in the molecule
            continue
        hit_here = False
        for match in mol.GetSubstructMatches(pattern):
            inside = linker_set.intersection(match)
            if inside:
                covered.update(inside)
                hit_here = True
        if hit_here:
            matched.append(name)

    return round(len(covered) / len(linker_set), 3), matched


def linker_stereocentres(mol: Any, linker_indices: list[int]) -> int:
    """Stereocentres inside the linker only.

    Fragment stereocentres come with the warhead and the E3 ligand and are
    not the generator's doing. A new one in the linker is, and each one is
    another separation or another chiral synthesis.
    """
    from rdkit import Chem

    linker_set = set(linker_indices)
    try:
        centres = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    except Exception:  # noqa: BLE001 - a descriptor is not worth failing for
        return 0
    return sum(1 for index, _ in centres if index in linker_set)


def score_sample(
    mol: Any,
    properties: dict[str, Any],
    linker: dict[str, Any],
    linker_indices: list[int],
    failure_codes: list[str] | None = None,
) -> SynthesisScore:
    """Score one assessed molecule. `mol` may be None when it did not parse."""
    if mol is None:
        return SynthesisScore(
            total=None, reason="not scored: the molecule could not be read"
        )

    codes = set(failure_codes or ())
    terms: list[Term] = []

    # --- synthesisability -------------------------------------------------
    # SA runs 1 (trivial) to 10 (heroic). Generated PROTAC linkers cluster
    # between 2.5 and 5, so the useful discrimination is in that band: full
    # credit at 3.0 and below, nothing at 6.0 and above.
    sa = properties.get("sa_score")
    if sa is None:
        terms.append(Term("synthesisability", 0.5, WEIGHTS["synthesisability"][0],
                          "SA could not be computed; scored neutral"))
    else:
        value = _ramp(float(sa), good=3.0, bad=6.0)
        terms.append(Term("synthesisability", value, WEIGHTS["synthesisability"][0],
                          f"SA {sa}"))

    # --- motif familiarity ------------------------------------------------
    coverage, matched = motif_coverage(mol, linker_indices)
    terms.append(Term("motif_familiarity", coverage, WEIGHTS["motif_familiarity"][0],
                      f"{int(coverage * 100)}% of the linker is routine chemistry"
                      + (f" ({', '.join(matched)})" if matched else "")))

    # --- exit vector span -------------------------------------------------
    # Published PROTACs cluster around 8-16 bonds between the two anchors.
    # Shorter and the warheads cannot occupy both sites at once; much longer
    # and the ternary complex pays an entropic cost for no gain.
    span = linker.get("path_length_bonds")
    if span is None:
        terms.append(Term("exit_vector_span", 0.3, WEIGHTS["exit_vector_span"][0],
                          "the span between fragments could not be measured"))
    else:
        value = _window(float(span), low_bad=2.0, low_ok=7.0, high_ok=17.0, high_bad=30.0)
        terms.append(Term("exit_vector_span", value, WEIGHTS["exit_vector_span"][0],
                          f"{span} bonds between the fragments"))

    # --- polarity ---------------------------------------------------------
    # Greasy PROTACs are hard to purify and harder to dose. The window is wide
    # because the fragments themselves carry most of the logP.
    clogp = properties.get("clogp")
    if clogp is None:
        terms.append(Term("polarity", 0.5, WEIGHTS["polarity"][0],
                          "cLogP could not be computed; scored neutral"))
    else:
        value = _window(float(clogp), low_bad=-2.0, low_ok=1.0, high_ok=5.0, high_bad=9.0)
        terms.append(Term("polarity", value, WEIGHTS["polarity"][0], f"cLogP {clogp}"))

    # --- flexibility ------------------------------------------------------
    rotatable = properties.get("rotatable_bonds")
    if rotatable is None:
        terms.append(Term("flexibility", 0.5, WEIGHTS["flexibility"][0],
                          "rotatable bonds could not be counted; scored neutral"))
    else:
        value = _ramp(float(rotatable), good=12.0, bad=30.0)
        terms.append(Term("flexibility", value, WEIGHTS["flexibility"][0],
                          f"{rotatable} rotatable bonds"))

    # --- liabilities ------------------------------------------------------
    # Deductions rather than a measurement: each of these is a specific thing
    # a chemist would object to, so each costs a stated amount.
    penalties: list[str] = []
    value = 1.0

    # A reactive linker is not a quality gradient, so this zeroes the term
    # outright rather than nibbling at it. Without this a free aldehyde on a
    # short chain scored 95/100 on a real run -- small, low SA, and read by
    # the motif matcher as an ordinary alkyl chain.
    reactive = str(properties.get("reactive_groups") or "")
    if "reactive_linker_group" in codes or reactive:
        value = 0.0
        penalties.append(f"reactive group ({reactive or 'in linker'})")

    # Liabilities that can be deliberate -- a click handle, a conjugation
    # point, a solubilising amine -- cost rather than disqualify.
    liable = [g for g in str(properties.get("liable_groups") or "").split(";") if g]
    if liable:
        value -= min(0.15 * len(liable), 0.45)
        penalties.append(", ".join(liable))

    if "pains_alert" in codes:
        value -= 0.5
        penalties.append("PAINS alert")
    if "bad_linker_ring" in codes:
        value -= 0.3
        penalties.append("implausible linker ring")
    if "macrocycle" in codes:
        value -= 0.3
        penalties.append("macrocycle")
    centres = linker_stereocentres(mol, linker_indices)
    if centres:
        value -= min(0.2 * centres, 0.4)
        penalties.append(f"{centres} new stereocentre{'s' if centres > 1 else ''}")
    value = max(0.0, value)
    terms.append(Term("liabilities", value, WEIGHTS["liabilities"][0],
                      ", ".join(penalties) if penalties else "none"))

    total_weight = sum(t.weight for t in terms)
    total = round(sum(t.contribution for t in terms) / total_weight * 100, 2)

    score = SynthesisScore(
        total=total,
        terms=terms,
        motif_coverage=coverage,
        matched_motifs=matched,
    )
    score.reason = build_reason(score)
    return score


def build_reason(score: SynthesisScore) -> str:
    """One sentence naming what decided the score.

    The wording is tiered, because "held back by" on a 97 reads as a problem
    where there is none. A high scorer with one soft term is still a good
    molecule and the sentence should say so.
    """
    if score.total is None:
        return score.reason or "not scored"

    weak = [t for t in score.weakest_terms(2) if t.score < 0.85]
    if not weak:
        return f"{score.total}: nothing objectionable"

    detail = "; ".join(f"{t.name} ({t.note})" for t in weak)
    if score.total >= 85.0:
        return f"{score.total}: good, with {detail}"
    if score.total >= 60.0:
        return f"{score.total}: usable, held back by {detail}"
    return f"{score.total}: weak on {detail}"


def rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows best-first by verdict, then by score, and stamp each with its
    rank.

    Verdict leads because the score answers "how hard is this to make", not
    "is this a PROTAC". Those come apart: a three-bond linker scores well on
    synthesis and cannot bridge two binding sites. The triage already caught
    that and called it `poor`, so honouring the verdict first keeps the easy
    mistake out of the top of the list.

    Unscored rows stay in the output rather than being dropped. A broken
    molecule is still a result, and a run where most of them are broken should
    look that way.
    """

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        verdict = VERDICT_ORDER.get(str(row.get("verdict", "")), len(VERDICT_ORDER))
        total = row.get("score_total")
        # Negated so that a higher score sorts earlier under an ascending sort.
        return (verdict, -float(total) if total is not None else 0.0)

    ordered = sorted(rows, key=sort_key)
    position = 0
    for row in ordered:
        if row.get("score_total") is None:
            row["rank"] = None
            continue
        position += 1
        row["rank"] = position
    return ordered
