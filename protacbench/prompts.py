"""Repair prompts for the harness agent, one per rejected sample.

The agent that repairs these runs in the DeepSeek harness with molbench
mounted as an MCP server, so it has SMILES validation, metrics, SMARTS
transformations and docking. A prompt here is written to be acted on with
those tools, and it says plainly what is wrong, what must not change, and what
counts as done.

Two details about the tool surface are load-bearing.

**The tools are namespaced.** MCP tools reach the model as
`mcp__molbench__<name>`, so the prompt names them that way. Naming them bare
sends the agent looking for tools that are not mounted.

**There is no `submit_molecule`.** That tool belongs to `moltrain`'s own
toolkit, which synthesises it as a terminal action; molbench's MCP server
never had it. A prompt that asks for it dead-ends every repair. The answer
instead comes back as a `REPAIRED:` line in the final message, which
`repair.py` parses -- a text contract, because the harness gives us the final
message losslessly and a tool call would have to be invented to carry it.

Two things matter in the wording. The fragments are a warhead and an E3-ligase
ligand whose structures are the whole point of the molecule, so the prompt has
to forbid touching them -- an agent that "fixes" a disconnected PROTAC by
deleting a fragment has destroyed it. And the required linker length is a
geometric constraint from the ternary complex, not a preference.
"""

from __future__ import annotations

import json
from typing import Any

from .assess import HARD_FAILURES, SOFT_FAILURES, Assessment

# What the agent should do about each failure, specifically. Generic advice
# produces generic edits.
REMEDIES: dict[str, str] = {
    "disconnected": (
        "Join the pieces into one molecule. The linker fragment must bond to "
        "both the warhead and the E3 ligand, one bond at each end."
    ),
    "impossible_valence": (
        "Find the over-bonded atom and reduce it to a legal valence, either by "
        "removing a bond or by substituting an element that accepts the bonds "
        "present."
    ),
    "sanitization_failed": (
        "Fix the valence or aromaticity violation. Use "
        "mcp__molbench__validate_smiles after each edit to confirm the "
        "molecule parses."
    ),
    "linker_not_bridging": (
        "Rebuild the linker so it has exactly two attachment points, one to "
        "each fragment. A linker that touches one fragment twice is a "
        "substituent, not a linker."
    ),
    "atom_clash": (
        "Two atoms were generated on top of each other, so the 3D geometry is "
        "unusable. Rebuild the linker in 2D with the required atom count, then "
        "regenerate a conformer with mcp__molbench__generate_3d_conformer."
    ),
    "obabel_wrote_nothing": (
        "Bond perception failed on the raw coordinates. Propose a linker of "
        "the required length in SMILES directly rather than repairing the "
        "geometry."
    ),
    "unreadable_sdf": (
        "The converted structure could not be parsed. Propose a linker of the "
        "required length in SMILES directly."
    ),
    "reactive_linker_group": (
        "Replace the reactive group in the linker with a stable equivalent: "
        "an aldehyde or ketone becomes an amide or an alcohol, a Michael "
        "acceptor becomes the saturated chain, an epoxide becomes a diol or "
        "an ether. Do not simply delete the atoms -- keep the linker's length."
    ),
    "bad_linker_ring": (
        "Replace the strained or oversized ring in the linker with an "
        "open chain, or with a five- or six-membered ring."
    ),
    "macrocycle": (
        "Break the macrocycle. The linker should connect the two fragments "
        "without closing a ring through them."
    ),
    "sa_too_high": (
        "Simplify the linker: prefer amide, ether and short alkyl chains over "
        "unusual ring systems and dense substitution."
    ),
    "qed_too_low": "Improve drug-likeness without changing either fragment.",
    "clogp_too_high": (
        "Make the linker more polar. Ether oxygens or an amide in place of "
        "alkyl is the usual move."
    ),
    "too_flexible": (
        "Reduce rotatable bonds, for instance by replacing part of a flexible "
        "chain with a para-substituted ring or a short rigid spacer."
    ),
    "mw_out_of_range": "Adjust the linker length to bring the mass into range.",
    "linker_too_short": (
        "Lengthen the linker. The two fragments need enough separation to "
        "occupy their own binding sites simultaneously."
    ),
    "linker_too_long": "Shorten the linker while keeping both attachment points.",
    "pains_alert": (
        "Remove the interference substructure from the linker. Do not alter "
        "either fragment, even if the alert matches there."
    ),
}

TEMPLATE = """Repair a PROTAC linker that a 3D generative model produced invalid.

Target: {target_id}, sample {sample_id} (Pro-GAT stage: {stage})

What is wrong:
{problems}

The molecule as generated:
  PROTAC: {protac_smiles}
  Linker: {linker_smiles}

Measured:
{measured}

Constraints you must respect:
- Do not change either fragment. They are a target warhead and an E3-ligase
  ligand, and their structures are the reason the molecule exists.
- The linker must attach at exactly two points, one to each fragment.
- Keep the linker at roughly {linker_size} heavy atoms. This length comes from
  the ternary-complex geometry, not from preference; {path_note}
- Measure before you claim. Never state a number you did not get from a tool.

Tools available to you:
- mcp__molbench__validate_smiles(smiles)
- mcp__molbench__get_molecule_metrics(smiles)
- mcp__molbench__apply_smarts_reaction(smiles, reaction_smarts)
- mcp__molbench__generate_3d_conformer(smiles)

How to work:
1. mcp__molbench__validate_smiles on the molecule above to see how it parses.
2. mcp__molbench__get_molecule_metrics to confirm the starting values.
3. Repair the linker, one change at a time, re-measuring after each.
4. mcp__molbench__validate_smiles on your final answer, so you know it parses
   before you report it.

Done means: a single connected molecule that sanitises, both fragments intact,
exactly two linker attachments, and SA and QED no worse than the values above.

Finish your reply with exactly one line in this form, and nothing after it:

REPAIRED: <the full repaired PROTAC SMILES>

If you cannot repair it, finish with this line instead, and say why above it:

REPAIRED: none
"""


def describe_problems(assessment: Assessment) -> str:
    """The failures, each with what to do about it.

    `SUPPRESSED_CODES` is applied here too. The row-based path already
    filtered it; without this, the headless repair run still handed the agent
    "improve drug-likeness" while the system prompt forbade exactly that.
    """
    lines: list[str] = []
    for code in assessment.failure_codes:
        if code in SUPPRESSED_CODES:
            continue
        what = HARD_FAILURES.get(code) or SOFT_FAILURES.get(code) or code
        remedy = REMEDIES.get(code)
        lines.append(f"- {code}: {what}." + (f" {remedy}" if remedy else ""))
    if assessment.failure_detail:
        lines.append(f"- detail: {assessment.failure_detail}")
    return "\n".join(lines) or "- no failure was recorded"


def describe_measurements(assessment: Assessment) -> str:
    shown = (
        ("heavy_atoms", "heavy atoms"),
        ("mw", "molecular weight"),
        ("sa_score", "SA score"),
        ("qed", "QED"),
        ("clogp", "cLogP"),
        ("tpsa", "TPSA"),
        ("rotatable_bonds", "rotatable bonds"),
    )
    lines = [
        f"  {label}: {assessment.properties[key]}"
        for key, label in shown
        if assessment.properties.get(key) is not None
    ]
    linker_atoms = assessment.linker.get("heavy_atoms")
    if linker_atoms is not None:
        lines.append(f"  linker heavy atoms: {linker_atoms}")
    path = assessment.linker.get("path_length_bonds")
    if path is not None:
        lines.append(f"  bonds between fragments: {path}")
    return "\n".join(lines) or "  nothing could be measured; the molecule did not parse"


def build_prompt(assessment: Assessment) -> str:
    """A self-contained repair task for one rejected sample."""
    path = assessment.linker.get("path_length_bonds")
    path_note = (
        f"the generated linker spans {path} bonds between the fragments."
        if path is not None
        else "the span between fragments could not be measured."
    )
    return TEMPLATE.format(
        target_id=assessment.target_id,
        sample_id=assessment.sample_id,
        stage=assessment.stage,
        problems=describe_problems(assessment),
        protac_smiles=assessment.protac_smiles or "(could not be read)",
        linker_smiles=assessment.linker_smiles or "(could not be read)",
        measured=describe_measurements(assessment),
        linker_size=assessment.linker_size,
        path_note=path_note,
    )


def build_task(assessment: Assessment) -> dict[str, Any]:
    """The same repair task as structured data, for driving a batch."""
    return {
        "task_id": f"{assessment.target_id}-{assessment.sample_id}-{assessment.stage}",
        "target_id": assessment.target_id,
        "sample_id": assessment.sample_id,
        "stage": assessment.stage,
        "verdict": assessment.verdict,
        "failure_codes": assessment.failure_codes,
        "failure_detail": assessment.failure_detail,
        "protac_smiles": assessment.protac_smiles,
        "linker_smiles": assessment.linker_smiles,
        # The warhead and the E3 ligand. `repair.verify` matches these back
        # into whatever the agent returns, which is the only way to tell a
        # repaired linker from a deleted fragment.
        "fragment_smiles": assessment.fragment_smiles,
        "linker_size": assessment.linker_size,
        "score_total": (
            assessment.score.total if assessment.score is not None else None
        ),
        "linker": assessment.linker,
        "properties": assessment.properties,
        "paths": assessment.paths,
        "prompt": build_prompt(assessment),
    }


def write_tasks(path: Any, assessments: list[Assessment]) -> int:
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for assessment in assessments:
            handle.write(json.dumps(build_task(assessment)) + "\n")
    return len(assessments)


# --------------------------------------------------------- the manual handoff

# Columns that must never reach the agent. `reference_smiles` is the real
# published PROTAC each target was built from: hand it over and the agent
# reads the answer off instead of repairing anything, and the salvage rate
# measures recall rather than chemistry. The score columns are withheld for a
# softer reason -- a composite is gameable in a way raw descriptors are not,
# and the agent should be fixing the molecule rather than the number.
WITHHELD_COLUMNS = frozenset(
    {
        "reference_smiles",
        "recovers_reference",
        "score_total",
        "score_reason",
        "rank",
    }
)

# Given once, as the agent's standing instruction. The two paragraphs that
# earn their place are the fragment prohibition and the explicit refusal of
# drug-likeness: without the second, an agent does the natural thing and
# optimises QED, which for a 940 Da bifunctional degrader makes it worse at
# its job. 34% of the real PROTACs in the test set fail a QED gate of 0.05.
SYSTEM_PROMPT = """\
You are a medicinal chemist optimising the linker of a PROTAC.

A PROTAC has three parts: a warhead that binds a target protein, an
E3-ubiquitin-ligase ligand, and a linker joining them. It works by holding
both proteins together in a ternary complex, so the linker's length and
composition decide whether it functions at all.

THE TWO FRAGMENTS ARE NOT YOURS TO CHANGE. The warhead and the E3 ligand are
the reason the molecule exists; their structures came from medicinal
chemistry programmes, not from a generator. You may only edit the linker
between them. A molecule that scores better because you deleted or altered a
fragment is a different compound and will be rejected.

You are NOT making this molecule drug-like in the oral small-molecule sense.
PROTACs are around 900 Da and violate Lipinski by construction. Do not
optimise QED. Do not reduce molecular weight toward typical drug ranges.
Doing either makes the molecule worse at its job.

WHAT A GOOD LINKER LOOKS LIKE. These describe the LINKER itself, measured
from 128 published PROTACs:
  - spans 8 to 17 bonds between the two fragments, and never fewer than 7
  - built from chemistry a lab runs routinely: amide, PEG ether, short alkyl,
    piperazine, piperidine, triazole, sulfonamide, carbamate
  - contains NO reactive or unstable group: no aldehyde, in-chain ketone,
    imine, Michael acceptor, epoxide, aziridine, acyl halide, anhydride,
    vinyl ether, enol, allene, peroxide or nitroso. These react with protein
    lysines and cysteines in vivo, or do not survive as drawn.

LIMITS ON THE WHOLE MOLECULE. These are properties of the entire PROTAC, not
of the linker, and the fragments already account for most of each one. 80% of
published PROTACs satisfy all four at once, so treat them as limits not to
cross rather than targets to hit:
  - SA score at or below 6.0        (published median 4.7)
  - cLogP at or below 7.5           (published median 5.0)
  - TPSA at or above 120            (published median 213)
  - rotatable bonds at or below 30  (published median 18)

Your real objective is simpler than those numbers: fix the stated problem
without making any of the four worse than it already is. The task tells you
the starting value for each. A repair that solves the problem and leaves them
roughly where they were is a success; one that trades a fixed problem for a
new limit breach is not.

TOOLS available to you:

  mcp__molbench__check_protac_linker(smiles, fragment_1, fragment_2)
      The one that decides whether your answer is accepted. Confirms both
      fragments are still present and unaltered, that the linker is bonded
      to them exactly twice, and that it carries no reactive group. Also
      reports the bond span. Call it on every candidate before reporting.

  mcp__molbench__validate_smiles(smiles)
      Does it parse, and what did RDKit complain about.

  mcp__molbench__get_molecule_metrics(smiles)
      SA score, cLogP, TPSA, rotatable bonds, QED, exact mass, heavy atoms,
      formula. This is where every number in the limits above comes from.

  mcp__molbench__apply_smarts_reaction(smiles, reaction_smarts)
      Apply a transformation and get the products back.

  mcp__molbench__render_molecule_2d(smiles, filename_prefix)
      A picture. Worth one call to confirm you edited the linker and not a
      fragment; a mangled warhead is obvious in a depiction and invisible
      in a SMILES string.

  mcp__molbench__generate_3d_conformer(smiles)
      Returns local_strain and ensemble_strain. A linker that only reaches
      the required span in a high-energy conformation is a bad linker that
      looks fine in 2D. Worth checking on your final candidate.

Docking tools exist on this server and are NOT useful here. A PROTAC binds
two proteins at once, and docking scores one ligand into one rigid receptor,
so it does not measure what decides whether a PROTAC works. These molecules
also have around 18 rotatable bonds at the median, far beyond where that
search stays reliable.

MEASUREMENT DISCIPLINE. Never state a number you did not get from a tool. If
you believe a change lowered cLogP, call get_molecule_metrics and read it.
Make ONE change at a time and re-measure after each. An unmeasured claim is
treated as false.

HOW TO WORK:
  1. validate_smiles on the input, to see how it parses.
  2. get_molecule_metrics, to confirm the starting values.
  3. Identify the smallest edit to the linker that addresses the stated
     problem. Prefer replacing a group over rebuilding the linker.
  4. Apply it, re-measure, and check it did not break something else.
  5. Repeat until the problem is fixed, or until you have tried three
     distinct approaches.
  6. check_protac_linker on your final answer. If `ok` is false, you are not
     finished -- read `issues` and fix what it names. An answer that fails
     this will be rejected, so there is nothing to gain by reporting it.
  7. Compare `span_bonds` against the target range for this task. The tool
     reports it but does not judge it.

FINISH your reply with exactly one line, and nothing after it:

REPAIRED: <the full repaired PROTAC SMILES>

If you cannot repair it without changing a fragment, or you exhausted your
three approaches without check_protac_linker returning ok, finish with:

REPAIRED: none

If you have a molecule that fixes the stated problem but still breaches a
limit, report it with REPAIRED: and say which limit, above the line. A
partial repair is a result; a silent one is not.

and say above that line what you tried and why it failed. A truthful failure
is worth more than a molecule that scores well by cheating.
"""

ROW_TEMPLATE = """\
Repair the linker of PROTAC {target_id}, sample {sample_id}.

WHAT IS WRONG
{problems}

THE MOLECULE
  Full PROTAC: {protac_smiles}
  Linker only: {linker_smiles}
{fragments}

AS MEASURED
  MW {mw}            SA {sa}       cLogP {clogp}
  TPSA {tpsa}        rotatable bonds {rotatable}
  linker: {linker_atoms} heavy atoms spanning {span} bonds between the fragments
{reactive}
WHAT TO AIM FOR
  linker span 8-13 bonds (currently {span})
  SA at or below 4.9 (currently {sa})
  cLogP between 3.5 and 5.6 (currently {clogp})
  no reactive or unstable groups in the linker

The linker must keep roughly its current length: {linker_size} heavy atoms.
That length comes from the geometry of the ternary complex, not from
preference. Shortening it to improve a descriptor breaks the molecule.
"""


# Failure codes that no longer gate anything and must not reach the agent.
# `qed_too_low` carries the remedy "improve drug-likeness", which directly
# contradicts the system prompt's instruction not to optimise QED -- and the
# instruction is the correct one, since 34% of real published PROTACs fail a
# QED gate of 0.05. A samples.csv written before the gate was removed still
# carries the code, so it is dropped here rather than relied on being absent.
SUPPRESSED_CODES = frozenset({"qed_too_low"})


def classify_row(row: dict[str, Any], reactive: list[str] | None = None) -> str:
    """Which optimisation class a CSV row belongs to.

    `reactive` is passed in rather than recomputed so a caller that already
    has it does not pay for the substructure search twice.
    """
    codes = {c for c in (row.get("failure_codes") or "").split(";") if c}
    if (row.get("verdict") or "") == "broken":
        return "structural"
    if reactive or "reactive_linker_group" in codes:
        return "reactive"
    if codes - {"qed_too_low"}:
        return "chemistry"
    if codes == {"qed_too_low"}:
        return "qed_only"
    return "clean"


def build_prompt_from_row(
    row: dict[str, Any], reactive: list[str] | None = None
) -> str:
    """The per-task prompt for one CSV row, with the answer columns withheld."""
    codes = [
        c
        for c in (row.get("failure_codes") or "").split(";")
        if c and c not in SUPPRESSED_CODES
    ]
    lines = []
    for code in codes:
        what = HARD_FAILURES.get(code) or SOFT_FAILURES.get(code) or code
        remedy = REMEDIES.get(code)
        lines.append(f"- {code}: {what}." + (f" {remedy}" if remedy else ""))
    if reactive:
        lines.append(
            f"- reactive_linker_group: the linker contains "
            f"{', '.join(reactive)}. {REMEDIES['reactive_linker_group']}"
        )
    detail = "; ".join(
        part.strip()
        for part in (row.get("failure_detail") or "").split(";")
        if part.strip() and "QED" not in part
    )
    if detail:
        lines.append(f"- detail: {detail}")
    problems = "\n".join(lines) or "- no failure was recorded"

    pieces = [p for p in (row.get("fragment_smiles") or "").split(".") if p]
    if pieces:
        fragments = "\n".join(
            f"  Fragment {i}:  {p}    <- must survive unchanged"
            for i, p in enumerate(pieces, start=1)
        )
    else:
        fragments = (
            "  (the fragments could not be isolated; do not alter anything "
            "that looks like a warhead or an E3 ligand)"
        )

    def value(key: str, fallback: str = "not measured") -> str:
        raw = (row.get(key) or "").strip()
        return raw if raw else fallback

    reactive_line = (
        f"  reactive groups in the linker: {', '.join(reactive)}\n"
        if reactive
        else ""
    )

    return ROW_TEMPLATE.format(
        target_id=row.get("target_id", "?"),
        sample_id=row.get("sample_id", "?"),
        problems=problems,
        protac_smiles=value("protac_smiles", "(could not be read)"),
        linker_smiles=value("linker_smiles", "(could not be read)"),
        fragments=fragments,
        mw=value("prop_mw"),
        sa=value("prop_sa_score"),
        clogp=value("prop_clogp"),
        tpsa=value("prop_tpsa"),
        rotatable=value("prop_rotatable_bonds"),
        linker_atoms=value("linker_heavy_atoms"),
        span=value("linker_path_length_bonds", "unknown"),
        reactive=reactive_line,
        linker_size=value("linker_size", "its current length"),
    )
