"""Score a model's repair reply against the rubric it was given.

`repair.verify` answers one question -- is the proposed molecule an acceptable
PROTAC -- and it answers it only for replies that carry a `REPAIRED:` line.
That is the right contract for a production run and the wrong one for judging
a model, because the two failure modes seen most often are invisible to it:

  * a chemically sound answer in a reply the harness cannot parse, which
    scores as `no_answer` and is indistinguishable from a refusal, and
  * a parseable answer that fixed the stated defect by trading away a
    property the prompt said not to touch, which scores as `repaired`
    because every absolute limit still holds.

So this module grades two things separately. CHEMISTRY asks whether the
molecule is good. PROTOCOL asks whether the reply told the truth about it in
the form it was asked to. A run can pass one and fail the other, and knowing
which is the whole point -- a protocol failure is a prompt or harness bug and
is cheap to fix, while a chemistry failure is the model.

Grading is against the numbers the agent was actually given in
`prompts.SYSTEM_PROMPT`, not against `configs/default.yaml`. The config gates
generation; the prompt is the contract the model was held to, and holding it
to a limit it was never shown measures nothing.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assess import DISQUALIFYING_GROUPS, groups_in_linker, molecule_properties
from .repair import match_fragments, parse_answer
from .structures import linker_view

# The limits quoted to the agent in `prompts.SYSTEM_PROMPT`. Kept here as
# literals rather than imported, because the prompt states them as prose and
# a grader that re-parsed that prose would fail silently the day the wording
# changed.
SPAN_TARGET = (8, 17)
SPAN_FLOOR = 7
MAX_SA = 6.0
MAX_CLOGP = 7.5
MIN_TPSA = 120.0
MAX_ROTATABLE = 30

# How far a descriptor may drift before it counts as a regression. A repair
# that moves cLogP by 0.02 has not "pushed it further out" in any sense a
# chemist would recognise; one that moves it by 1.1 has.
REGRESSION_TOLERANCE = {
    "clogp": 0.15,
    "tpsa": 2.0,
    "sa_score": 0.15,
    "rotatable_bonds": 2,
}

# Linker heavy atoms may drift this far before the length instruction counts
# as broken. Two atoms is about one bond of span either way.
LENGTH_TOLERANCE = 2

# Groups that are a genuine liability and that nothing in `assess.py` tests
# for. They live here rather than in DISQUALIFYING_GROUPS deliberately: that
# table gates generation, and widening it would retroactively re-triage runs
# already on disk. A grader has no such constraint, and a model that only
# satisfies the checker should not score the same as one that does chemistry.
GRADER_ONLY_GROUPS: tuple[tuple[str, str], ...] = (
    # N-CH2-N. Hydrolyses to the diamine and formaldehyde at physiological
    # pH, so an imidazolidine drawn in place of a piperazine is not the
    # molecule you would have in a cell.
    ("aminal", "[NX3;!$(N=*)][CX4H2][NX3;!$(N=*)]"),
    # O-CH2-N. The same hydrolysis, one heteroatom over.
    ("hemiaminal_ether", "[OX2;!$(O=*)][CX4H2][NX3;!$(N=*)]"),
    # Three or more contiguous heteroatom-free rotatable CH2 units beyond
    # what a PEG or amide linker needs is not unstable, only greasy -- so it
    # is reported, never failed. See `_long_alkyl_run`.
)

# Descriptor names as a model writes them, mapped to the measurement key.
# Ordered longest-first so "exact mw" wins over "mw".
_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("exact_mw", r"exact(?:\s+molecular)?\s*(?:mass|mw|weight)"),
    # Before the bare forms, so "linker heavy atoms 14" is not read as a
    # claim about the whole molecule's 67 and scored as a lie.
    ("linker_heavy_atoms", r"linker[\s:]*(?:of\s+)?heavy[\s-]*atoms?"),
    ("linker_heavy_atoms", r"linker[^.\n]{0,12}?heavy[\s-]*atoms?"),
    ("sa_score", r"\bsa(?:\s*score)?\b"),
    ("clogp", r"\bc?logp\b"),
    ("tpsa", r"\btpsa\b"),
    ("rotatable_bonds", r"rotatable(?:\s+bonds?)?"),
    ("heavy_atoms", r"heavy[\s-]*atoms?"),
    ("qed", r"\bqed\b"),
    ("mw", r"\b(?:mw|molecular\s+weight)\b"),
    ("span", r"span(?:s|ning|ned)?|bond\s+span"),
)

# Claims written the other way round, with the number first: "15-16 bonds",
# "spans 17 bonds". Models state linker length like this far more often than
# they write "span = 17", and it is the number most worth auditing, because
# the span decides whether the molecule is in range at all.
_CLAIM_PATTERNS_SUFFIX: tuple[tuple[str, str], ...] = (
    ("span", r"bonds?\b"),
)

# How close a stated number must be to count as the measured one.
_CLAIM_TOLERANCE = {
    "exact_mw": 0.05,
    "mw": 0.6,
    "sa_score": 0.06,
    "clogp": 0.06,
    "tpsa": 0.6,
    "qed": 0.006,
    "rotatable_bonds": 0,
    "heavy_atoms": 0,
    "linker_heavy_atoms": 0,
    "span": 0,
}

_HEDGES = ("~", "≈", "approximately", "roughly", "about", "circa", "around")

_TOOL_MARKERS = (
    "mcp__molbench__",
    "check_protac_linker",
    "get_molecule_metrics",
    "validate_smiles",
    "apply_smarts_reaction",
    "generate_3d_conformer",
)

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class Check:
    """One rubric line: what was asked, what happened, and the evidence."""

    name: str
    group: str  # "chemistry" or "protocol"
    status: str
    detail: str

    @property
    def counts(self) -> bool:
        """WARN and SKIP are reported but do not fail the run."""
        return self.status in (PASS, FAIL)


@dataclass
class Claim:
    """A number the reply asserted, and whether it survives measurement."""

    key: str
    stated: str
    verdict: str  # "ok", "false", "echo", "unmeasurable"
    measured: Any = None
    hedged: bool = False


@dataclass
class Report:
    task_id: str
    answer: str | None = None
    answer_source: str = "none"
    measured: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    omitted: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, group: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, group, status, detail))

    def tally(self, group: str) -> tuple[int, int]:
        """(passed, counted) for one group."""
        counted = [c for c in self.checks if c.group == group and c.counts]
        return sum(1 for c in counted if c.status == PASS), len(counted)

    @property
    def chemistry_ok(self) -> bool:
        passed, total = self.tally("chemistry")
        return total > 0 and passed == total

    @property
    def protocol_ok(self) -> bool:
        passed, total = self.tally("protocol")
        return total > 0 and passed == total

    @property
    def verdict(self) -> str:
        if self.chemistry_ok and self.protocol_ok:
            return "pass"
        if self.chemistry_ok:
            return "chemistry only"
        if self.protocol_ok:
            return "protocol only"
        return "fail"

    def as_dict(self) -> dict[str, Any]:
        chem_pass, chem_total = self.tally("chemistry")
        prot_pass, prot_total = self.tally("protocol")
        return {
            "task_id": self.task_id,
            "verdict": self.verdict,
            "chemistry": f"{chem_pass}/{chem_total}",
            "protocol": f"{prot_pass}/{prot_total}",
            "answer": self.answer,
            "answer_source": self.answer_source,
            "measured": self.measured,
            "checks": [
                {"name": c.name, "group": c.group, "status": c.status,
                 "detail": c.detail}
                for c in self.checks
            ],
            "claims": [
                {"key": c.key, "stated": c.stated, "verdict": c.verdict,
                 "measured": c.measured, "hedged": c.hedged}
                for c in self.claims
            ],
            "omitted": self.omitted,
            "notes": self.notes,
        }


# ------------------------------------------------------------------ the answer

# A SMILES long enough to be a PROTAC and made only of SMILES characters.
# Deliberately strict: a model's prose is full of short fragments like
# `C(=O)` and matching those would pick a sub-structure as the answer.
_SMILES_CHARS = r"[A-Za-z0-9@\+\-\[\]\(\)=#$:/\\\.%]"
_FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)


def extract_answer(reply: str, min_atoms: int = 40) -> tuple[str | None, str]:
    """The proposed SMILES, and how it had to be found.

    The contract line is tried first and is the only source a production run
    would accept. The fenced-block fallback exists so that a reply which did
    good chemistry but ignored the output format can still be graded -- the
    missing line is then recorded as a protocol failure rather than silently
    becoming a refusal. Without this split the two are the same result, and
    they call for completely different fixes.
    """
    from rdkit import Chem, rdBase

    contract = parse_answer(reply)
    if contract:
        return contract, "contract_line"

    # Largest plausible molecule in a fenced block wins. Largest, because a
    # reply usually shows the linker as well as the whole PROTAC and the
    # whole PROTAC is the answer.
    best: str | None = None
    best_atoms = 0
    candidates = [block.strip() for block in _FENCE.findall(reply)]
    candidates += [line.strip() for line in reply.splitlines()]
    for candidate in candidates:
        candidate = candidate.strip("`*_ ").strip()
        if not candidate or not re.fullmatch(f"{_SMILES_CHARS}+", candidate):
            continue
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            continue
        atoms = mol.GetNumHeavyAtoms()
        if atoms >= min_atoms and atoms > best_atoms:
            best, best_atoms = candidate, atoms
    return (best, "fenced_fallback") if best else (None, "none")


# ------------------------------------------------------------- the measurement


def measure(smiles: str, fragments: list[str]) -> dict[str, Any]:
    """Everything the rubric needs about a proposed molecule.

    Returns a dict with an `error` key when the molecule cannot be measured,
    so the caller can turn that into a failed check rather than an exception.
    """
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "RDKit could not read the SMILES"}

    pieces = Chem.GetMolFrags(mol, asMols=False)
    if len(pieces) > 1:
        return {"error": f"the molecule is in {len(pieces)} disconnected pieces"}

    out: dict[str, Any] = dict(molecule_properties(mol))
    out["canonical"] = Chem.MolToSmiles(mol)

    matches = match_fragments(mol, fragments)
    if matches is None:
        return {**out, "error": "at least one fragment is missing or altered"}

    claimed = {i for match in matches for i in match}
    linker_indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
    if not linker_indices:
        return {**out, "error": "the fragments are joined directly, with no linker"}

    view = linker_view(mol, linker_indices)
    out.update(
        {
            "linker_smiles": view.smiles,
            "linker_heavy_atoms": view.n_heavy_atoms,
            "attachments": view.n_attachments,
            "span": view.path_length_bonds,
            "linker_rotatable_bonds": view.rotatable_bonds,
            "reactive": groups_in_linker(mol, linker_indices, DISQUALIFYING_GROUPS),
            "grader_only": groups_in_linker(mol, linker_indices, GRADER_ONLY_GROUPS),
            "longest_alkyl_run": _long_alkyl_run(mol, linker_indices),
        }
    )
    return out


def _long_alkyl_run(mol: Any, linker_indices: list[int]) -> int:
    """Longest unbroken chain of sp3 CH2 carbons inside the linker.

    Not a failure on its own -- plenty of published linkers carry a C6 alkyl
    run -- but it is the signature of a repair that removed a heteroatom to
    make a problem go away, so it is worth seeing next to a cLogP rise.
    """
    inside = set(linker_indices)
    chain = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetIdx() in inside
        and a.GetSymbol() == "C"
        and a.GetTotalNumHs() == 2
        and not a.GetIsAromatic()
        and not a.IsInRing()
    }
    best = 0
    seen: set[int] = set()
    for start in chain:
        if start in seen:
            continue
        # Walk the connected component; a CH2 run is a path, so its size is
        # its length.
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            for neighbour in mol.GetAtomWithIdx(node).GetNeighbors():
                if neighbour.GetIdx() in chain:
                    stack.append(neighbour.GetIdx())
        seen |= component
        best = max(best, len(component))
    return best


# ------------------------------------------------------------------ the claims


def audit_claims(reply: str, measured: dict[str, Any],
                 baseline: dict[str, Any]) -> list[Claim]:
    """Every number the reply asserted, checked against measurement.

    A stated value that matches the BASELINE rather than the measurement is
    recorded as `echo`, not as a lie. Replies routinely quote the task back,
    and scoring the quoted input as a false claim about the output would make
    the audit useless on exactly the replies worth auditing.
    """
    claims: list[Claim] = []
    # Character ranges already attributed to a claim. The pattern table runs
    # longest-first, so an earlier, more specific reading wins: without this,
    # "Exact MW 888.36" is scored once correctly as exact_mw and then again
    # as a false claim about mw.
    taken: list[tuple[int, int]] = []

    def record(key: str, match: re.Match, number: str) -> None:
        start, end = match.span()
        if any(start < seen_end and seen_start < end for seen_start, seen_end in taken):
            return
        taken.append((start, end))
        # Only the match itself and the two characters before it. A wider
        # window catches a hedge belonging to a neighbouring number -- in
        # "span ~ 14 bonds, 8-17 bond window" the 8-17 is a quoted range,
        # not an estimate, and flagging it buries the real hedge.
        window = reply[max(0, start - 2): end]
        hedged = any(h in window.lower() for h in _HEDGES)
        claims.append(_judge_claim(key, number.strip(), hedged, measured, baseline))

    # A stated value may be "4.11", "15-16" or "888.36".
    number = r"(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)"
    # Each pattern is wrapped, because several carry a top-level alternation
    # and concatenating one unwrapped makes the trailing number part of only
    # the last branch -- group(1) is then None on a match from any other.
    for key, pattern in _CLAIM_PATTERNS:
        rx = re.compile(f"(?:{pattern})" + r"[^0-9\n]{0,24}?" + number,
                        re.IGNORECASE)
        for match in rx.finditer(reply):
            record(key, match, match.group(1))
    for key, pattern in _CLAIM_PATTERNS_SUFFIX:
        rx = re.compile(number + r"[^0-9\n]{0,4}?" + f"(?:{pattern})",
                        re.IGNORECASE)
        for match in rx.finditer(reply):
            record(key, match, match.group(1))
    return claims


def _judge_claim(key: str, stated: str, hedged: bool,
                 measured: dict[str, Any], baseline: dict[str, Any]) -> Claim:
    actual = measured.get(key)
    if actual is None:
        return Claim(key, stated, "unmeasurable", None, hedged)

    tolerance = _CLAIM_TOLERANCE.get(key, 0.05)
    if _within(stated, actual, tolerance):
        return Claim(key, stated, "ok", actual, hedged)
    if _within(stated, baseline.get(key), tolerance):
        return Claim(key, stated, "echo", actual, hedged)
    return Claim(key, stated, "false", actual, hedged)


def _within(stated: str, actual: Any, tolerance: float) -> bool:
    """Does a stated scalar or `a-b` range cover the measured value."""
    if actual is None:
        return False
    parts = [p.strip() for p in re.split(r"[-–]", stated) if p.strip()]
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return False
    if not numbers:
        return False
    low, high = min(numbers), max(numbers)
    return (low - tolerance) <= float(actual) <= (high + tolerance)


# ----------------------------------------------------------------- the grading


def grade(baseline: dict[str, Any], reply: str) -> Report:
    """Score one reply. `baseline` comes from `load_baseline`."""
    report = Report(task_id=baseline.get("task_id", "?"), baseline=baseline)

    answer, source = extract_answer(reply)
    report.answer, report.answer_source = answer, source

    # --- protocol: the output contract -------------------------------------
    if source == "contract_line":
        report.add("answer_line", "protocol", PASS,
                   "reply ends with a parseable REPAIRED: line")
    elif source == "fenced_fallback":
        report.add(
            "answer_line", "protocol", FAIL,
            "no REPAIRED: line; a molecule was recovered from the reply body "
            "for grading, but a production run would score this as no answer",
        )
    else:
        report.add("answer_line", "protocol", FAIL,
                   "no REPAIRED: line and no molecule found in the reply")

    tools = sorted({m for m in _TOOL_MARKERS if m in reply})
    if tools:
        report.add("tool_use", "protocol", PASS,
                   f"reply references {len(tools)} tool(s): {', '.join(tools)}")
    else:
        report.add(
            "tool_use", "protocol", WARN,
            "no tool call appears in the reply. If the harness strips the "
            "trace this is expected; if not, every number here is unmeasured",
        )

    if answer is None:
        report.add("parses", "chemistry", FAIL, "there is no molecule to measure")
        return report

    measured = measure(answer, baseline.get("fragments", []))
    report.measured = measured

    # --- chemistry: is it the same PROTAC ----------------------------------
    if "error" in measured:
        report.add("parses", "chemistry", FAIL, measured["error"])
        report.claims = audit_claims(reply, measured, baseline)
        return report

    report.add("parses", "chemistry", PASS,
               f"one connected molecule, {measured['heavy_atoms']} heavy atoms")
    report.add("fragments_intact", "chemistry", PASS,
               "both fragments present, unaltered and non-overlapping")

    attachments = measured.get("attachments")
    if attachments == 2:
        report.add("bridges", "chemistry", PASS,
                   "the linker is bonded to the fragments exactly twice")
    else:
        report.add("bridges", "chemistry", FAIL,
                   f"the linker touches a fragment at {attachments} atom(s)")

    # --- chemistry: geometry -----------------------------------------------
    span = measured.get("span")
    low, high = SPAN_TARGET
    if span is None:
        report.add("span", "chemistry", SKIP, "the span could not be computed")
    elif low <= span <= high:
        margin = min(span - low, high - span)
        note = " -- at the limit, no margin" if margin == 0 else ""
        report.add("span", "chemistry", PASS,
                   f"span {span} bonds, inside {low}-{high}{note}")
    elif span < SPAN_FLOOR:
        report.add("span", "chemistry", FAIL,
                   f"span {span} bonds is below the floor of {SPAN_FLOOR}")
    else:
        report.add("span", "chemistry", FAIL,
                   f"span {span} bonds is outside {low}-{high}")

    _grade_length(report, baseline, measured)

    # --- chemistry: liabilities --------------------------------------------
    reactive = measured.get("reactive") or []
    if reactive:
        report.add("no_reactive_group", "chemistry", FAIL,
                   f"the linker still carries: {', '.join(reactive)}")
    else:
        report.add("no_reactive_group", "chemistry", PASS,
                   "no reactive or unstable group inside the linker")

    hidden = measured.get("grader_only") or []
    if hidden:
        report.add(
            "no_unflagged_liability", "chemistry", FAIL,
            f"the linker carries {', '.join(hidden)}, which no tool on the "
            "server tests for -- check_protac_linker would call this ok",
        )
    else:
        report.add("no_unflagged_liability", "chemistry", PASS,
                   "clear of the liabilities the server's own checks miss")

    # --- chemistry: absolute limits and regressions ------------------------
    _grade_limits(report, measured)
    _grade_regressions(report, baseline, measured)

    run = measured.get("longest_alkyl_run") or 0
    if run >= 6:
        report.notes.append(
            f"longest unbroken CH2 run in the linker is {run} atoms; published "
            "linkers put PEG, amide or piperazine here"
        )

    # --- protocol: did it tell the truth -----------------------------------
    report.claims = audit_claims(reply, measured, baseline)
    _grade_claims(report, measured)
    return report


def _grade_length(report: Report, baseline: dict[str, Any],
                  measured: dict[str, Any]) -> None:
    """Did the linker keep the length the task told it to keep.

    Skipped when the task's own span was out of range, because such a task
    asks for the length to change and holding it would be the failure.
    """
    before = baseline.get("linker_heavy_atoms")
    after = measured.get("linker_heavy_atoms")
    if before is None or after is None:
        return report.add("length_held", "chemistry", SKIP,
                          "the baseline linker size is unknown")

    base_span = baseline.get("span")
    low, high = SPAN_TARGET
    if base_span is not None and not (low <= base_span <= high):
        return report.add(
            "length_held", "chemistry", SKIP,
            f"the task started at span {base_span}, outside {low}-{high}, so "
            "it asked for the length to change",
        )

    drift = after - before
    if abs(drift) <= LENGTH_TOLERANCE:
        report.add("length_held", "chemistry", PASS,
                   f"linker {before} -> {after} heavy atoms ({drift:+d})")
    else:
        report.add(
            "length_held", "chemistry", FAIL,
            f"linker {before} -> {after} heavy atoms ({drift:+d}); the length "
            "comes from the ternary complex, not from preference",
        )


def _grade_limits(report: Report, measured: dict[str, Any]) -> None:
    """The four absolute limits the agent was quoted."""
    rules = (
        ("sa_score", "SA", measured.get("sa_score"), lambda v: v <= MAX_SA,
         f"at or below {MAX_SA}"),
        ("clogp", "cLogP", measured.get("clogp"), lambda v: v <= MAX_CLOGP,
         f"at or below {MAX_CLOGP}"),
        ("tpsa", "TPSA", measured.get("tpsa"), lambda v: v >= MIN_TPSA,
         f"at or above {MIN_TPSA}"),
        ("rotatable_bonds", "rotatable bonds",
         measured.get("rotatable_bonds"), lambda v: v <= MAX_ROTATABLE,
         f"at or below {MAX_ROTATABLE}"),
    )
    breached = [
        f"{label} {value} (wants {want})"
        for _, label, value, ok, want in rules
        if value is not None and not ok(value)
    ]
    if breached:
        report.add("within_limits", "chemistry", FAIL,
                   "; ".join(breached))
    else:
        report.add(
            "within_limits", "chemistry", PASS,
            "SA {sa_score}, cLogP {clogp}, TPSA {tpsa}, rotatable "
            "{rotatable_bonds} -- all inside the quoted limits".format(**measured),
        )


def _grade_regressions(report: Report, baseline: dict[str, Any],
                       measured: dict[str, Any]) -> None:
    """"Fix the stated problem without pushing any of them further out."

    This is the check `repair.verify` cannot make, because every value here
    can regress badly while still satisfying its absolute limit. It is also
    the one that catches a repair which deleted a heteroatom to make a
    reactive group disappear.
    """
    worse: list[str] = []
    better: list[str] = []
    directions = {
        "clogp": -1,           # lower is further from the ceiling
        "sa_score": -1,
        "rotatable_bonds": -1,
        "tpsa": +1,            # higher is further from the floor
    }
    for key, good in directions.items():
        before, after = baseline.get(key), measured.get(key)
        if before is None or after is None:
            continue
        delta = after - before
        if abs(delta) <= REGRESSION_TOLERANCE.get(key, 0.1):
            continue
        label = {"clogp": "cLogP", "sa_score": "SA", "tpsa": "TPSA",
                 "rotatable_bonds": "rotatable bonds"}[key]
        moved = f"{label} {before} -> {after} ({delta:+.2f})".replace(".00)", ")")
        (better if (delta > 0) == (good > 0) else worse).append(moved)

    if worse:
        report.add("no_regression", "chemistry", FAIL,
                   "pushed further out: " + "; ".join(worse))
    else:
        detail = "improved: " + "; ".join(better) if better else \
            "every limit descriptor held within tolerance"
        report.add("no_regression", "chemistry", PASS, detail)


def _grade_claims(report: Report, measured: dict[str, Any]) -> None:
    """Were the reply's numbers true, and did it report the ones that matter.

    Two separate protocol checks, because a reply can be scrupulously honest
    about two descriptors while never mentioning the three it broke. Silence
    is not a lie, but the prompt asked for measurement, and a reply that
    reports only what improved is not a measured answer.
    """
    false = [c for c in report.claims if c.verdict == "false"]
    hedged = [c for c in report.claims if c.hedged and c.verdict != "echo"]

    if false:
        report.add(
            "claims_true", "protocol", FAIL,
            "; ".join(
                f"said {c.key} {c.stated}, measured {c.measured}" for c in false
            ),
        )
    elif hedged:
        report.add(
            "claims_true", "protocol", FAIL,
            "no false numbers, but "
            + "; ".join(f"{c.key} hedged as '{c.stated}'" for c in hedged)
            + " -- these are tool outputs, not estimates",
        )
    elif report.claims:
        report.add("claims_true", "protocol", PASS,
                   f"all {len(report.claims)} stated number(s) match measurement")
    else:
        report.add("claims_true", "protocol", FAIL,
                   "the reply states no measured number at all")

    # Which of the five the reply never mentioned.
    stated = {c.key for c in report.claims if c.verdict != "echo"}
    wanted = ("clogp", "tpsa", "sa_score", "rotatable_bonds", "span")
    report.omitted = [k for k in wanted if k not in stated and measured.get(k) is not None]
    if report.omitted:
        report.add(
            "reported_what_matters", "protocol", FAIL,
            "never stated: " + ", ".join(report.omitted),
        )
    else:
        report.add("reported_what_matters", "protocol", PASS,
                   "every limit descriptor and the span were stated")


# ------------------------------------------------------------- loading a task


def load_baseline(source: str | Path, task_id: str) -> dict[str, Any] | None:
    """The starting molecule and its measurements, from a samples.csv row."""
    target, _, sample = task_id.partition("-")
    with Path(source).open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target_id") == target and row.get("sample_id") == sample:
                return _baseline_from_row(row, task_id)
    return None


def _baseline_from_row(row: dict[str, str], task_id: str) -> dict[str, Any]:
    def number(key: str, cast: type = float) -> Any:
        raw = (row.get(key) or "").strip()
        try:
            return cast(float(raw))
        except (TypeError, ValueError):
            return None

    return {
        "task_id": task_id,
        "target_id": row.get("target_id", ""),
        "sample_id": row.get("sample_id", ""),
        "protac_smiles": row.get("protac_smiles", ""),
        "linker_smiles": row.get("linker_smiles", ""),
        "fragments": [f for f in (row.get("fragment_smiles") or "").split(".") if f],
        "mw": number("prop_mw"),
        "exact_mw": number("prop_exact_mw"),
        "sa_score": number("prop_sa_score"),
        "clogp": number("prop_clogp"),
        "tpsa": number("prop_tpsa"),
        "qed": number("prop_qed"),
        "rotatable_bonds": number("prop_rotatable_bonds", int),
        "heavy_atoms": number("prop_heavy_atoms", int),
        "linker_heavy_atoms": number("linker_heavy_atoms", int),
        "span": number("linker_path_length_bonds", int),
    }


# ------------------------------------------------------------------ reporting


def format_report(report: Report, verbose: bool = False) -> str:
    """The human-readable scorecard."""
    marks = {PASS: "PASS", FAIL: "FAIL", WARN: "warn", SKIP: "skip"}
    lines: list[str] = []
    chem_pass, chem_total = report.tally("chemistry")
    prot_pass, prot_total = report.tally("protocol")

    lines.append(f"{report.task_id}: {report.verdict.upper()}")
    lines.append(
        f"  chemistry {chem_pass}/{chem_total}    protocol {prot_pass}/{prot_total}"
    )
    if report.answer_source == "fenced_fallback":
        lines.append("  (answer recovered from the reply body, not a REPAIRED: line)")
    lines.append("")

    for group in ("chemistry", "protocol"):
        rows = [c for c in report.checks if c.group == group]
        if not rows:
            continue
        lines.append(f"  {group}")
        for check in rows:
            lines.append(f"    {marks[check.status]:<4}  {check.name:<24}{check.detail}")
        lines.append("")

    if report.measured and "error" not in report.measured:
        m = report.measured
        b = report.baseline
        lines.append("  measured            before      after")
        for key, label in (
            ("span", "linker span"), ("linker_heavy_atoms", "linker heavy atoms"),
            ("clogp", "cLogP"), ("tpsa", "TPSA"),
            ("sa_score", "SA"), ("rotatable_bonds", "rotatable bonds"),
            ("mw", "MW"),
        ):
            before = b.get(key)
            after = m.get(key)
            if after is None:
                continue
            lines.append(
                f"    {label:<20}{'-' if before is None else before:<12}{after}"
            )
        lines.append("")

    if report.claims and verbose:
        lines.append("  numbers the reply stated")
        for claim in report.claims:
            flag = " (hedged)" if claim.hedged else ""
            lines.append(
                f"    {claim.verdict:<13}{claim.key} = {claim.stated}"
                f"  [measured {claim.measured}]{flag}"
            )
        lines.append("")

    for note in report.notes:
        lines.append(f"  note: {note}")
    if report.notes:
        lines.append("")
    return "\n".join(lines)
