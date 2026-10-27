"""Repair prompts for an ADC drug-linker, one per rejected candidate.

The agent reading these runs in the DeepSeek harness with molbench mounted,
so it has `check_adc_linker`, `check_payload_release`, SMILES validation and
metrics. A prompt here is written to be acted on with those, and says what is
wrong, what must not change, and what counts as done.

**Why this is not `protacbench.prompts` with different numbers.** The two
molecules fail differently, so the instructions invert.

A PROTAC linker is a geometric problem. Both ends bind proteins, the span
decides whether a ternary complex forms, and 25 rotatable bonds is a failure.
So that prompt is mostly about length, and it tells the agent to keep the
linker at the length it has.

An ADC linker is a chemical problem. The antibody holds the payload, nothing
has to reach anything, and the reference set's own median is 25 rotatable
bonds with Fsp3 0.88 -- long, floppy, saturated chains. Length barely
matters. What matters is whether the conjugate survives in plasma, whether
the payload is cut loose inside the cell, and whether what comes loose is the
drug rather than the drug still wearing part of the linker. None of those is
a distance, and two of them are invisible in every descriptor.

**The numbers describe the payload-free linker.** Every limit below was
measured on 1,692 real ADC linker reagents, which carry no payload. Applying
them to a whole conjugate is a calibration error, not a stricter standard:
MMAE alone is MW 718 with cLogP 4, so every conjugate breaches every limit
whatever its spacer, and the comparison stops saying anything. `check_adc_
linker` excises the payload before measuring, and so must any number quoted
here.
"""

from __future__ import annotations

import json
from typing import Any

# The 10th-90th percentile band of each property across the 1,692 reference
# reagents, which is what `reference.build_profile` measures. Kept here as
# literals so a prompt is reproducible without reading the dataset, and
# checked against the live profile by the test suite so the two cannot drift.
LINKER_BANDS: dict[str, tuple[float, float, float]] = {
    "clogp": (-0.81, 0.68, 3.14),
    "tpsa": (66.4, 139.3, 284.8),
    "heavy_atoms": (16.0, 36.0, 85.0),
    "mw": (242.3, 526.1, 1236.5),
    "rotatable_bonds": (9.0, 25.0, 73.0),
    "fsp3": (0.50, 0.88, 1.00),
}

# The one hard ceiling, and the only property here with a monotone
# relationship to outcome rather than a band. A more hydrophobic drug-linker
# aggregates and is cleared faster, with no lower bound at which being polar
# starts to hurt -- the linker is discarded on release and never has to cross
# a membrane. Set at the reference set's own 90th percentile.
CLOGP_CEILING = 3.14

# What to do about each rejection, specifically. Generic advice produces
# generic edits.
REMEDIES: dict[str, str] = {
    "payload_lost": (
        "The payload is gone. Put it back exactly as supplied and edit only "
        "the spacer. A conjugate without its payload is not a worse "
        "candidate, it is a different compound."
    ),
    "payload_altered": (
        "The payload is present but a stereocentre has been inverted. Restore "
        "the payload SMILES character for character. MMAE has ten "
        "stereocentres and its potency depends on all of them."
    ),
    "payload_internal": (
        "The payload has linker on both sides of it. Move the spacer so the "
        "payload sits at the end of the chain; nothing can release a payload "
        "that is in the middle."
    ),
    "trigger_lost": (
        "There is no cleavage trigger, so the payload can never be released. "
        "Put back a protease substrate (Val-Cit, Val-Ala, or the GGFG "
        "tetrapeptide), a hindered disulfide, or a glucuronide."
    ),
    "no_release_chemistry": (
        "There is a trigger but nothing between it and the payload that "
        "frees the drug once the trigger is cut. Insert a p-aminobenzyl "
        "carbamate (the cut peptide's amide, then an aniline, then a "
        "benzylic CH2, then a carbamate to the payload's amine) or a "
        "deruxtecan-style aminomethylene spacer."
    ),
    "handle_lost": (
        "There is no conjugation handle, so this cannot be attached to an "
        "antibody at all. Restore the handle exactly as supplied; which "
        "chemistry it is follows from the antibody's conjugation site and is "
        "not a design choice."
    ),
    "handle_duplicated": (
        "There are two conjugation handles. The second one crosslinks two "
        "antibodies into an aggregate. Remove it."
    ),
    "reactive_group": (
        "There is reactive chemistry in the spacer, outside the conjugation "
        "handle. Replace it with a stable equivalent: a ketone or aldehyde "
        "becomes an amide or an alcohol, a Michael acceptor becomes the "
        "saturated chain, an epoxide becomes a diol. Keep the spacer's "
        "length."
    ),
    "dialkyl_ketone": (
        "The spacer meets the handle's acyl group carbon-to-carbon, which "
        "makes a ketone where an amide was intended. Start the spacer with a "
        "nitrogen or an oxygen instead -- `*NCCOCC*` rather than `*CCOCC*` -- "
        "and the same junction becomes an amide or an ester."
    ),
    "triacylamine": (
        "Three acyl groups on one nitrogen is not an isolable compound. This "
        "is what acylating a bare maleimide nitrogen produces; put a spacer "
        "between them, which is exactly why clinical ADCs use "
        "maleimidocaproyl rather than maleimide."
    ),
    "clogp_too_high": (
        "The linker is too greasy. Hydrophobic drug-linkers aggregate and are "
        "cleared quickly in vivo, and it is the failure mode that kills ADCs "
        "in animals rather than in a flask. Replace alkyl stretches with PEG "
        "ethers, insert an amide, or add a sulfonate or a phosphate."
    ),
    "too_small": (
        "The linker is shorter than any in the reference set. Lengthen the "
        "spacer; too little distance between the antibody surface and the "
        "cassette makes the protease unable to reach its substrate."
    ),
    "disconnected": (
        "Join the pieces into one molecule. The spacer must bond to the "
        "handle at one end and the cassette at the other."
    ),
    "unparseable": (
        "The SMILES does not parse. Use mcp__molbench__validate_smiles to see "
        "what RDKit objected to."
    ),
}


# Given once, as the agent's standing instruction.
SYSTEM_PROMPT = """\
You are a medicinal chemist optimising the linker of an antibody-drug
conjugate.

An ADC has three parts: a monoclonal antibody that finds the tumour cell, a
cytotoxic payload far too toxic to give on its own, and a linker joining
them. The antibody binds its antigen, the conjugate is internalised, and in
the lysosome the linker is cut and the payload is released inside the cell.

The linker itself has four pieces, and only ONE of them is yours:

  [conjugation handle] - [SPACER] - [cleavage cassette] - [payload]
                            ^^^^^^
                       this is the part you design

THE PAYLOAD IS NOT YOURS TO CHANGE, STEREOCHEMISTRY INCLUDED. MMAE has ten
stereocentres and its potency depends on every one of them. An inverted
centre is not a small edit: it is a different drug, with different potency,
and it will be rejected. Copy the payload SMILES through character for
character.

THE CONJUGATION HANDLE IS NOT YOURS TO CHANGE. Which chemistry it is follows
from the antibody's conjugation site -- a free cysteine takes a maleimide, a
surface lysine takes an activated ester, an engineered azide takes a strained
alkyne. These work because of decades of wet-lab validation, and reactivity
is not something to reason about from a structure. There must be exactly one:
none cannot be attached to the antibody, and two crosslink two antibodies
into an aggregate.

THE CASSETTE MAY BE SWAPPED, BUT ONLY FOR ANOTHER KNOWN ONE. Val-Cit, Val-Ala
and GGFG are cut by cathepsin B; a hindered disulfide is reduced by
glutathione; a glucuronide is cut by beta-glucuronidase. The enzyme is not in
any model you have, so you cannot invent a new substrate for it and check
whether it works. Choose from the ones that exist.

YOU ARE NOT MAKING THIS DRUG-LIKE. A drug-linker is 1,000 to 1,500 Da and the
whole conjugate is an antibody. It is never swallowed and never crosses a
membrane as drawn. Do not optimise QED. Do not reduce molecular weight toward
oral-drug ranges. Both make it worse at its job.

RELEASE IS TWO THINGS, AND YOU NEED BOTH. A TRIGGER is what gets cut --
Val-Cit, a disulfide, a glucuronide. A RELEASE CHEMISTRY is what then lets
the payload leave as itself, usually a p-aminobenzyl carbamate that
eliminates once the peptide in front of it is gone. A cassette bonded
straight to the payload by an amide has a trigger and no release chemistry:
it is cut, and the payload stays attached to the leftover amino acid. That
design looks perfect in every descriptor and does nothing, and
check_payload_release is the only thing here that sees it.

LIMITS ON THE LINKER, NOT ON THE CONJUGATE. Every number below was measured
on 1,692 real ADC linker reagents, which carry no payload. check_adc_linker
excises the payload before measuring, and so must you -- quoting the whole
conjugate's cLogP against these is comparing different things, and every
conjugate breaches every limit whatever its spacer.

  HARD CEILING, not a target:
    - linker cLogP at or below 3.14   (real-linker median 0.68)
      This is the one limit with a monotone relationship to outcome. More
      hydrophobic means more aggregation and faster clearance, and there is
      no point at which being more polar starts to hurt.

  BANDS that real linkers sit inside, worth noticing when you leave one:
    - TPSA            66 to 285      (median 139)
    - heavy atoms     16 to 85       (median 36)
    - molecular mass  242 to 1237    (median 526)
    - rotatable bonds 9 to 73        (median 25)
    - Fsp3            0.50 to 1.00   (median 0.88)

  Those bands are wide on purpose: they are the 10th to 90th percentile of
  real reagents, not a target to hit. A linker outside one is worth a look,
  not an automatic failure. The cLogP ceiling is the one to respect.

TOOLS available to you:

  mcp__molbench__check_adc_linker(smiles, payload, handle, require_cleavable)
      The one that decides whether your answer is accepted. Confirms the
      payload is intact with its stereochemistry, that it sits at the end of
      the chain, that there is exactly one handle, that a trigger is present,
      that the spacer contains no junction chemistry nobody would make, and
      that the payload-free linker is inside the reference bands. Call it on
      every candidate before reporting. Read `linker_smiles` in the result:
      that is the molecule its numbers describe.

  mcp__molbench__check_payload_release(smiles, payload)
      Whether, and how, the payload ever comes off. `releases_free_drug` is
      the question. False means the payload leaves carrying linker, which
      works only for a payload potent as that adduct -- it is how
      trastuzumab emtansine works and it is a mistake everywhere else.

  mcp__molbench__validate_smiles(smiles)
      Does it parse, and what did RDKit object to.

  mcp__molbench__get_molecule_metrics(smiles)
      SA score, cLogP, TPSA, rotatable bonds, QED, exact mass, heavy atoms.
      Call this on the LINKER (the `linker_smiles` from check_adc_linker),
      not on the conjugate, when you are comparing against the limits above.

  mcp__molbench__apply_smarts_reaction(smiles, reaction_smarts)
      Apply a transformation and get the products back.

  mcp__molbench__render_molecule_2d(smiles, filename_prefix)
      A picture. Worth one call to confirm you edited the spacer and not the
      payload; a mangled payload is obvious in a depiction and invisible in a
      SMILES string.

Docking tools exist on this server and are USELESS here. Docking scores one
small molecule into one rigid pocket. The payload is held by an antibody and
released inside a cell; nothing about this molecule's behaviour is a binding
pose, and the drug-linker has around 25 rotatable bonds, far past where that
search stays reliable.

MEASUREMENT DISCIPLINE. Never state a number you did not get from a tool. If
you believe a change lowered cLogP, call the tool and read it. Make ONE
change at a time and re-measure after each. An unmeasured claim is treated as
false.

HOW TO WORK:
  1. validate_smiles on the input, to see how it parses.
  2. check_adc_linker on the input, to confirm what is actually wrong and to
     get the payload-free linker SMILES.
  3. check_payload_release on the input, which catches the failure the
     descriptors cannot see.
  4. Identify the smallest edit to the SPACER that addresses the stated
     problem. Prefer replacing a group over rebuilding the linker.
  5. Apply it, re-measure, and check it did not break something else.
  6. Repeat until fixed, or until you have tried three distinct approaches.
  7. check_adc_linker AND check_payload_release on your final answer. If
     either returns ok false, you are not finished: read `issues` and fix
     what it names. An answer failing these is rejected, so there is nothing
     to gain by reporting it.

A NOTE ON JUNCTIONS. When you join a spacer to an acyl handle such as
maleimidocaproyl, start the spacer with a nitrogen or an oxygen. Starting it
with a carbon turns the handle's acid into a dialkyl ketone instead of an
amide -- a reactive carbonyl that condenses with lysine in vivo, and not a
compound anyone would make. `*NCCOCCOCC*` is right where `*CCOCCOCC*` is
wrong, and the difference is invisible unless you look for it.

FINISH your reply with exactly one line, and nothing after it:

REPAIRED: <the full repaired drug-linker SMILES>

If you cannot repair it without changing the payload or the handle, or you
exhausted three approaches without check_adc_linker returning ok, finish
with:

REPAIRED: none

and say above that line what you tried and why it failed. If you have a
molecule that fixes the stated problem but still breaches a band, report it
with REPAIRED: and name the band above the line. A partial repair is a
result; a silent one is not. A truthful failure is worth more than a
molecule that scores well by cheating.
"""


ROW_TEMPLATE = """\
Repair an ADC drug-linker: {handle} + {cassette} + {payload_name}.

WHAT IS WRONG
{problems}

THE MOLECULE
  Drug-linker: {smiles}
  Linker only (payload excised, and what the limits apply to):
    {linker_smiles}

  Payload:     {payload_smiles}
               ^ must survive unchanged, stereochemistry included
  Handle:      {handle} ({handle_note})
               ^ must be present exactly once

AS MEASURED ON THE LINKER, NOT THE CONJUGATE
  cLogP {clogp}        TPSA {tpsa}        heavy atoms {heavy_atoms}
  mass {mw}            rotatable bonds {rotatable_bonds}        Fsp3 {fsp3}
  cleavage triggers found: {triggers}
  payload released as free drug: {releases}

WHAT TO AIM FOR
  linker cLogP at or below {ceiling} (currently {clogp}) -- the hard one
  a cleavage trigger present, and release chemistry between it and the payload
  exactly one {handle} handle
  the payload intact and at the end of the chain
  no ketone, anhydride or other junction chemistry in the spacer

Edit the spacer only. The payload, the handle, and the cassette's identity
are fixed. If you must change the cassette, swap it for another known one
rather than inventing a substrate.
"""


# Columns withheld from the agent. The score columns go because a composite
# is gameable in a way raw descriptors are not, and the agent should be
# fixing the molecule rather than the number. `reinvent_smiles` goes because
# it is the generator's own assembly, with the payload's stereochemistry
# stripped -- handing it over invites the agent to copy a flattened payload
# back in and fail the one gate that matters.
WITHHELD_COLUMNS = frozenset(
    {
        "score_total",
        "score_reason",
        "rank",
        "reinvent_smiles",
        "nll",
    }
)


def classify_row(row: dict[str, Any]) -> str:
    """Which repair class a candidates.csv row belongs to.

    The classes are ordered by how fundamental the failure is, and a row gets
    the first that applies. `payload` first because nothing else matters if
    the drug is gone; `release` next because a conjugate that never releases
    is inert whatever its descriptors say.
    """
    gates = {g for g in (row.get("gates_failed") or "").split(";") if g}
    if gates & {"payload_lost", "payload_altered", "payload_internal"}:
        return "payload"
    if gates & {"trigger_lost", "no_release_chemistry"}:
        return "release"
    if gates & {"handle_lost", "handle_duplicated"}:
        return "handle"
    if gates & {"reactive_group", "disconnected", "unparseable"}:
        return "structural"
    if gates:
        return "chemistry"
    try:
        clogp = float(row.get("linker_clogp") or row.get("prop_clogp") or 0.0)
    except (TypeError, ValueError):
        clogp = 0.0
    if clogp > CLOGP_CEILING:
        return "greasy"
    return "clean"


def describe_problems(row: dict[str, Any]) -> str:
    """The failures, each with what to do about it."""
    lines: list[str] = []
    for gate in (row.get("gates_failed") or "").split(";"):
        if not gate:
            continue
        remedy = REMEDIES.get(gate)
        lines.append(f"- {gate}." + (f" {remedy}" if remedy else ""))
    for junction in (row.get("bad_junctions") or "").split(";"):
        if not junction:
            continue
        remedy = REMEDIES.get(junction)
        lines.append(f"- {junction}." + (f" {remedy}" if remedy else ""))
    try:
        clogp = float(row.get("linker_clogp") or 0.0)
    except (TypeError, ValueError):
        clogp = 0.0
    if clogp > CLOGP_CEILING:
        lines.append(
            f"- clogp_too_high: the linker's cLogP is {clogp}, above the "
            f"ceiling of {CLOGP_CEILING}. {REMEDIES['clogp_too_high']}"
        )
    detail = (row.get("gate_detail") or "").strip()
    if detail:
        lines.append(f"- detail: {detail}")
    return "\n".join(lines) or "- no failure was recorded"


def build_prompt_from_row(
    row: dict[str, Any], payload_smiles: str, handle_note: str = ""
) -> str:
    """The per-task prompt for one candidates.csv row."""

    def value(key: str, fallback: str = "not measured") -> str:
        raw = str(row.get(key) or "").strip()
        return raw if raw else fallback

    triggers = value("cleavable_triggers", "none found")
    releases = value("releases_free_drug", "not checked")
    return ROW_TEMPLATE.format(
        handle=value("handle", "?"),
        cassette=value("cassette", "?"),
        payload_name=value("payload", "?"),
        problems=describe_problems(row),
        smiles=value("smiles", "(could not be read)"),
        linker_smiles=value("linker_smiles", "(could not be isolated)"),
        payload_smiles=payload_smiles,
        handle_note=handle_note or "fixed by the antibody's conjugation site",
        clogp=value("linker_clogp"),
        tpsa=value("linker_tpsa"),
        heavy_atoms=value("linker_heavy_atoms"),
        mw=value("linker_mw"),
        rotatable_bonds=value("linker_rotatable_bonds"),
        fsp3=value("linker_fsp3"),
        triggers=triggers,
        releases=releases,
        ceiling=CLOGP_CEILING,
    )


def build_task(
    row: dict[str, Any], payload_smiles: str, handle_note: str = ""
) -> dict[str, Any]:
    """One repair task as structured data, for driving a batch."""
    shared = {k: v for k, v in row.items() if k not in WITHHELD_COLUMNS}
    return {
        "task_id": f"{row.get('handle','?')}-{row.get('cassette','?')}-"
        f"{row.get('rank') or row.get('smiles','')[:12]}",
        "repair_class": classify_row(row),
        "payload_smiles": payload_smiles,
        "row": shared,
        "prompt": build_prompt_from_row(row, payload_smiles, handle_note),
    }


def write_tasks(path: Any, tasks: list[dict[str, Any]]) -> int:
    from pathlib import Path as _Path

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")
    return len(tasks)
