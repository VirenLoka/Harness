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
    """The failures, each with what to do about it."""
    lines: list[str] = []
    for code in assessment.failure_codes:
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
