"""A scripted linker expert, and the number it exists to measure.

`moltrain.experts` cold-starts SFT. This one does that too, but it is on the
critical path for a second reason worth stating plainly: **there is no
demonstration corpus for this task.** Of the 2,229 rows in
`repair_tasks.jsonl`, only 144 carry a before/after pair, their mean
`sa_delta` is -0.0039, and 4 of them improved. Nothing in that is a
demonstration of a repair; it is a record of a generator being resampled.

So `sft.min_reward` cannot be guessed. `moltrain`'s 0.45 was set against its
own expert's measured anchors -- 0.38 for a diligent no-op, 0.59 for a real
move -- and those numbers belong to a different reward with different terms.
The `calibrate` subcommand below runs this expert over a stratified sample and
reports the spread it actually produces, including the no-op anchor, so the
threshold is set from evidence rather than inherited.

Like `moltrain`'s expert this is an ordinary `GenerateFn`: it takes a
transcript and returns the next assistant turn, so `episode.run_episode`
executes its tool calls through the real toolkit. Every number that reaches a
trace came back from a tool. That matters more here than it did there,
because `protocol.score_grounding` pays the policy for exactly that property
and teaching the opposite during the warmup would train the two stages
against each other.

What the expert knows is the *protocol* -- measure, change one thing, verify,
submit -- plus a move library of functional-group swaps. GRPO is what is
supposed to find the chemistry; this is the floor it has to beat.

**Every move preserves the linker's length.** `protacbench/OPTIMISATION.md`
says so in the remedy table -- "Do not delete the atoms, keep the linker's
length" -- because the span comes from ternary-complex geometry, not from
preference. A repair that fixes a descriptor by shortening the bridge has
made a molecule that cannot hold two proteins together.

**Every move is confined to the linker.** The fragments are matched with
`protacbench.repair.match_fragments` before a move is proposed, and a move
whose SMARTS touches a fragment atom is never offered. After the edit,
`check_protac_linker` confirms both fragments survived. Belt and braces on
purpose: a repair that ate a warhead improves SA, cLogP, TPSA, rotatable
bonds and MW simultaneously, and it is the one failure that looks like
success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from moltrain import logs, progress
from moltrain.atem import START, render_reasoning, render_tool_call

from .properties import LinkerProperties
from .rewards import score_thresholds

LOG = logs.get_logger("protac.experts")

_TOOL_OUTPUT = re.compile(
    r'<tool_output name="(?P<name>[^"]+)">\s*(?P<body>.*?)\s*</tool_output>',
    re.DOTALL,
)

# How many distinct approaches one episode gets. OPTIMISATION.md's standing
# instruction is "repeat until the problem is fixed or you have tried three
# distinct approaches", and the turn budget agrees: with
# `episode.max_turns: 8`, one survey turn plus one submit turn leaves six, and
# a move costs two (propose, then verify).
DEFAULT_MAX_MOVES = 3

# `episode.max_turns` in configs/default.yaml. Restated rather than read
# because `traces.scripted_factory` constructs the expert with the thresholds
# alone; `calibrate` below passes the configured value in. The expert counts
# its own turns against this so that it always keeps one back for the
# submission -- an episode that runs out of turns mid-move submits nothing,
# and that scores the same -1.0 as deleting a warhead.
DEFAULT_TURN_BUDGET = 8

# How many times one move may be re-offered after it was accepted and the
# group it repairs is still there. A reaction SMARTS rewrites one matched site
# per product, so a linker carrying the same reactive group twice needs the
# same move twice. Measured over the 2,229 tasks: 397 linkers carry one copy
# of a disqualifying group, 18 carry two and 3 carry three, so two retries
# clears every molecule in the corpus.
MAX_MOVE_RETRIES = 2

# Mirrors `reward.weights.hold_violation` and `reward.violation_cap` in
# configs/default.yaml. The expert is handed `thresholds` and not the whole
# reward, so these are restated rather than read; they are here so that
# "better" means the same thing to the expert as it does to the thing paying
# it, and they are constructor arguments so a changed config can be passed in.
HOLD_PENALTY = 0.4
VIOLATION_CAP = 3.0


@dataclass(frozen=True)
class Move:
    """One linker edit, with what it measured when it was vetted.

    `addresses` are the `failure_codes` this move is a remedy for, taken from
    OPTIMISATION.md's per-failure table. They order the search; they do not
    gate it, because a molecule carrying an epoxide has an epoxide whatever
    the assessor chose to complain about.

    `measured` is prose rather than a number because these moves do not have
    one effect. `imine_to_amine` lowers cLogP on every molecule it touches and
    moves SA in either direction depending on what it landed next to, and a
    single headline delta would be a claim the probe did not support.
    """

    name: str
    smarts: str
    addresses: tuple[str, ...]
    rationale: str
    measured: str


# Every move below was run through `protacbench.grade.measure` on real rows of
# repair_tasks.jsonl -- not on a toy probe -- and the medians quoted are over
# the molecules the move actually fired on. `applicable` is how many of the
# 2,229 tasks carry the group inside the linker, which is why the order is
# what it is: the corpus has 168 imines and three enols.
#
# FOUR CANDIDATES WERE MEASURED AND DROPPED:
#
#   peroxide -> ether by deleting an oxygen ("[#6:1][OX2:2][OX2:3][#6:4]>>
#   [#6:1][O:2][#6:4]") works, and shortened the span by a bond on 12 of 12
#   probes. That is the length rule, so the surviving peroxide move replaces
#   the second oxygen with a methylene instead and holds the span at zero.
#
#   aziridine -> pendant primary amine ("[C:1]1[N:2][C:3]1>>[C:1][C:3][N:2]")
#   cleared the reactive group, but a primary amine is in
#   `assess.LIABLE_GROUPS` and `protacbench.score` charges for it, so the move
#   trades a hard failure for a scored one; it also lengthened the span on one
#   of twelve probes. Opening the ring in-chain does the same job with the
#   amine inside the bridge.
#
#   opening a seven-membered carbocycle ("...>>[C:1][C:2]...[C:7]") fired on
#   3 of 2,229 tasks and added a median of five rotatable bonds, trading a
#   `bad_linker_ring` complaint for a `too_flexible` one.
#
#   aldehyde -> amide has no move because there is no nitrogen to form it
#   from without inventing atoms; the primary-alcohol remedy is the
#   length-preserving half of OPTIMISATION.md's "amide or primary alcohol".
MOVES: tuple[Move, ...] = (
    # --- reactive-group repairs, in corpus frequency order -----------------
    Move(
        "imine_to_amine",
        "[CX3;!$(C=N-C=O):1]=[NX2;H1,H0;!$(N-a);!$(N=C-N):2]>>[C:1][N:2]",
        ("reactive_linker_group",),
        "an imine hydrolyses back to the carbonyl and the amine in water; "
        "reducing it to the secondary amine keeps every atom in place",
        # applicable to 108 tasks; 12/12 probes admissible; median span 0,
        # linker heavy atoms 0, cLogP -0.483, SA +0.135, TPSA -0.33.
        "span held, cLogP down about half a unit, SA slightly up",
    ),
    Move(
        "vinyl_ether_to_ether",
        "[OX2;!$(O=*):1][CX3:2]=[CX3:3]>>[O:1][C:2][C:3]",
        ("reactive_linker_group",),
        "a vinyl ether hydrolyses to an aldehyde under acid; saturating the "
        "alkene leaves a plain ether, which is the commonest linker motif "
        "there is",
        # applicable to 78 tasks; 12/12 admissible; median span 0, heavy atoms
        # 0, SA -0.134, cLogP -0.124, rotatable +1.
        "span held, SA down, one rotatable bond gained",
    ),
    Move(
        "michael_to_saturated",
        "[CX3:1]=[CX3:2][CX3:3]=[OX1:4]>>[C:1][C:2][C:3]=[O:4]",
        ("reactive_linker_group",),
        "an enone alkylates cysteine thiols in vivo; saturating the alkene "
        "leaves the carbonyl and the chain exactly as long as it was",
        # applicable to 71 tasks; 12/12 admissible; median span 0, heavy atoms
        # 0, SA -0.06, cLogP -0.103, rotatable +1.
        "span held, SA and cLogP both slightly down",
    ),
    Move(
        "aziridine_to_amine_chain",
        "[CX4:1]1[NX3:2][CX4:3]1>>[C:1][N:2][C:3]",
        ("reactive_linker_group", "bad_linker_ring"),
        "an aziridine alkylates protein nucleophiles and is strained as well; "
        "opening it to the in-chain secondary amine clears both complaints",
        # applicable to 51 tasks; 12/12 admissible; median span 0, heavy atoms
        # 0, SA -0.236, cLogP +0.248, rotatable +1.
        "span held, SA down, cLogP slightly up",
    ),
    Move(
        "peroxide_to_ether",
        "[#6:1][OX2:2][OX2:3][#6:4]>>[#6:1][O:2][C:3][#6:4]",
        ("reactive_linker_group",),
        "a peroxide is not a linker, it is an oxidant; swapping the second "
        "oxygen for a methylene leaves an ether of the same length",
        # applicable to 19 tasks; 12/12 admissible; median span 0, heavy atoms
        # 0, SA -0.093, cLogP +0.458, TPSA -9.23.
        "span held, SA down, cLogP up about half a unit",
    ),
    Move(
        "epoxide_to_diol",
        "[CX4:1]1[OX2:2][CX4:3]1>>[C:1]([O:2])[C:3]O",
        ("reactive_linker_group", "bad_linker_ring", "clogp_too_high"),
        "an epoxide opens on contact with a lysine; hydrating it to the diol "
        "keeps the carbons and makes the linker markedly more polar",
        # applicable to 16 tasks; 6/6 admissible; median span 0, linker heavy
        # atoms +1, cLogP -1.046, TPSA +27.93.
        "span held, cLogP down a full unit, TPSA up sharply",
    ),
    Move(
        "nitroso_to_amine",
        "[#6:1][NX2:2]=[OX1:3]>>[#6:1][N:2]",
        ("reactive_linker_group",),
        "a nitroso group is a genotoxicity flag and tautomerises; the amine "
        "is what it reduces to and sits in the same place on the chain",
        # applicable to 5 tasks; 3/3 admissible; median span 0, linker heavy
        # atoms -1 (the oxygen is pendant, not in the bridge), cLogP -0.808.
        "span held, cLogP down",
    ),
    Move(
        "enol_to_alcohol",
        "[OX2H:1][CX3:2]=[CX3:3]>>[O:1][C:2][C:3]",
        ("reactive_linker_group",),
        "an enol is drawn, not isolated: it tautomerises to the ketone. "
        "Saturating it gives the alcohol that actually sits there",
        # applicable to 3 tasks; 3/3 admissible; median span 0, heavy atoms 0,
        # cLogP -0.691, SA +0.111.
        "span held, cLogP down",
    ),
    Move(
        "aldehyde_to_alcohol",
        "[CX3H1:1](=[OX1:2])[#6:3]>>[#6:3][C:1][O:2]",
        ("reactive_linker_group",),
        "an aldehyde forms a Schiff base with any lysine it meets; reducing "
        "it to the primary alcohol keeps the carbon and the oxygen",
        # The corpus has three linker aldehydes and all three sit on
        # disconnected molecules, which cannot be measured -- a linker
        # aldehyde needs a loose end, and in a connected PROTAC the linker has
        # none. Measured instead on a connected corpus molecule (102-0) with a
        # formyl branch spliced into its linker: span 0, linker heavy atoms 0,
        # SA -0.051, cLogP -0.207, TPSA +3.16, aldehyde gone.
        "span held, cLogP slightly down, the aldehyde cleared",
    ),
    Move(
        "ketone_to_amide",
        "[CX4:1][CX3:2](=[OX1:3])[CX4:4]>>[C:1][C:2](=[O:3])[N:4]",
        ("reactive_linker_group", "clogp_too_high"),
        "an in-chain ketone is an electrophile and a metabolic soft spot; the "
        "amide is the single most familiar linker motif and the same length",
        # applicable to 101 tasks; 12/12 admissible; median span 0, heavy
        # atoms 0, cLogP -0.885, TPSA +12.03, rotatable -1.
        "span held, cLogP down almost a unit, TPSA up",
    ),
    Move(
        "ketone_to_alcohol",
        "[#6:1][CX3:2](=[OX1:3])[#6:4]>>[#6:1][C:2]([O:3])[#6:4]",
        ("reactive_linker_group",),
        "the gentler ketone repair: reduce it to the secondary alcohol, which "
        "changes polarity far less than the amide does",
        # applicable to 128 tasks; 12/12 admissible; median span 0, heavy
        # atoms 0, cLogP -0.208, TPSA +3.16, SA +0.131.
        "span held, cLogP slightly down",
    ),
    # --- polarity ----------------------------------------------------------
    Move(
        "alkyl_to_ether",
        "[CH2;!R:1][CH2;!R:2][CH2;!R:3]>>[C:1][O:2][C:3]",
        ("clogp_too_high",),
        "OPTIMISATION.md's own remedy for cLogP: an ether oxygen in place of "
        "an alkyl carbon. It is one atom for one atom, so the span cannot "
        "move, and it turns an alkyl run into the start of a PEG",
        # applicable to 1,579 tasks; 12/12 admissible; median span 0, heavy
        # atoms 0, cLogP -0.806, TPSA +9.23, rotatable 0.
        "span held, cLogP down about four fifths of a unit, TPSA up",
    ),
    Move(
        "alkyl_to_amide",
        "[CH2;!R:1][CH2;!R:2][CH2;!R:3]>>[C:1][C:2](=O)[N:3]",
        ("clogp_too_high", "too_flexible", "sa_too_high"),
        "the stronger polarity move: an amide where an alkyl run was. It is "
        "the chemistry a lab runs on Monday, it stiffens the chain, and the "
        "two chain carbons stay where they were",
        # applicable to 1,579 tasks; 12/12 admissible; median span 0, linker
        # heavy atoms +1 (the carbonyl oxygen is pendant), cLogP -1.707,
        # TPSA +29.1, rotatable -1.
        "span held, cLogP down well over a unit, TPSA up sharply",
    ),
    # --- rigidification ----------------------------------------------------
    Move(
        "chain_to_para_phenylene",
        "[#6;!R:1][CH2;!R][CH2;!R][CH2;!R][CH2;!R][#6;!R:2]>>[#6:1]c1ccc([#6:2])cc1",
        ("too_flexible",),
        "a para-substituted ring in place of a four-carbon run: the paper "
        "behind this project ties linker rigidification to better metabolic "
        "stability and less efflux, and para substitution spans the same four "
        "bonds the chain did",
        # applicable to 492 tasks; 12/12 admissible; median span 0, linker
        # heavy atoms +2, rotatable -3, cLogP -0.094.
        "span held, three rotatable bonds removed",
    ),
    Move(
        "chain_to_piperazine",
        "[#6;!R:1][CH2;!R][CH2;!R][CH2;!R][CH2;!R][#6;!R:2]>>[#6:1]N1CCN([#6:2])CC1",
        ("too_flexible", "clogp_too_high"),
        "piperazine is on OPTIMISATION.md's list of linker chemistry a lab "
        "runs routinely, and it rigidifies and polarises in one move while "
        "bridging the same distance",
        # applicable to 492 tasks; 12/12 admissible; median span 0, linker
        # heavy atoms +2, cLogP -2.028, rotatable -3.5, TPSA +6.48.
        "span held, three or four rotatable bonds removed, cLogP down two "
        "units",
    ),
    Move(
        "chain_to_piperidine",
        "[#6;!R:1][CH2;!R][CH2;!R][CH2;!R][CH2;!R][#6;!R:2]>>[#6:1]C1CCN([#6:2])CC1",
        ("too_flexible",),
        "the gentler rigidification, for a molecule whose cLogP is already at "
        "the low end of its band: one ring nitrogen instead of two, so the "
        "chain stiffens without the polarity swing",
        # applicable to 492 tasks; 12/12 admissible; median span 0, linker
        # heavy atoms +2, cLogP -1.238, rotatable -3. Kept alongside
        # piperazine, which dominates it on both numbers, because dominating
        # is not the same as right: a task whose only complaint is flexibility
        # and whose cLogP sits near 3.5 is pushed out of its band by -2.028
        # and held inside it by -1.238.
        "span held, three rotatable bonds removed, cLogP down about a unit",
    ),
    # --- ring repairs ------------------------------------------------------
    Move(
        "open_cyclopropane",
        "[CX4;R1:1]1[CX4;R1:2][CX4;R1:3]1>>[C:1][C:2][C:3]",
        ("bad_linker_ring", "sa_too_high"),
        "a cyclopropane in a linker is ring strain bought for nothing and is "
        "the most expensive feature in a route; opening it to the propyl "
        "chain keeps all three carbons",
        # applicable to 40 tasks; 12/12 admissible; median SA -0.288,
        # rotatable +2, and median span +1 -- opening the ring makes the
        # bridge take the long way round. Within the span band's tolerance of
        # four bonds, and the acceptance test below is what decides per task.
        "SA down, span a bond longer",
    ),
    Move(
        "cyclobutane_to_chain",
        "[CX4;R1:1]1[CX4;R1:2][CX4;R1:3][CX4;R1:4]1>>[C:1][C:2][C:3][C:4]",
        ("bad_linker_ring",),
        "a cyclobutane is the other strained ring OPTIMISATION.md rejects; "
        "opening it to the butyl chain holds the span exactly",
        # applicable to 7 tasks; 6/6 admissible; median span 0, heavy atoms 0,
        # SA +0.076, rotatable +2.
        "span held, two rotatable bonds gained",
    ),
)

MOVES_BY_NAME: dict[str, Move] = {move.name: move for move in MOVES}


# --------------------------------------------------------------- the chemistry


def linker_atoms(smiles: str, fragment_smiles: Sequence[str]) -> list[int] | None:
    """Which atoms are the linker: everything the two fragments do not claim.

    None when the molecule cannot be read or a fragment is not present, which
    is the same answer `grade.measure` gives and means the same thing: there
    is no linker to edit because this is not the compound.

    Never raises. A measurement path that throws turns a bad molecule into a
    crashed episode.
    """
    try:
        from rdkit import Chem, rdBase

        from protacbench.repair import match_fragments

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        matches = match_fragments(mol, list(fragment_smiles))
        if matches is None:
            return None
        claimed = {index for match in matches for index in match}
        return [i for i in range(mol.GetNumAtoms()) if i not in claimed]
    except Exception as exc:  # noqa: BLE001 - a failed match is data, not a crash
        LOG.debug("linker_atoms failed on %s: %s", smiles[:60], exc)
        return None


def moves_inside_linker(
    smiles: str, fragment_smiles: Sequence[str]
) -> list[Move] | None:
    """The moves whose every matched atom lies inside the linker.

    This is the guard the whole package is built around. A reaction SMARTS
    rewrites what it matched, so a template that matches across the boundary
    rewrites a warhead -- and a rewritten warhead is a different compound
    whose descriptors look better than the real one's.

    None means the check could not be made (no RDKit, or the fragments did not
    match). The caller then falls back to proposing by failure code and leans
    on `check_protac_linker` to catch what this would have prevented, which is
    slower and costs turns but is not less safe.
    """
    inside = linker_atoms(smiles, fragment_smiles)
    if inside is None:
        return None
    try:
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            inside_set = set(inside)
            usable: list[Move] = []
            for move in MOVES:
                query = Chem.MolFromSmarts(move.smarts.split(">>")[0])
                if query is None:
                    LOG.warning("move %s has an unreadable pattern", move.name)
                    continue
                hits = mol.GetSubstructMatches(query, uniquify=True, maxMatches=50)
                if any(set(hit).issubset(inside_set) for hit in hits):
                    usable.append(move)
        return usable
    except Exception as exc:  # noqa: BLE001
        LOG.debug("move matching failed on %s: %s", smiles[:60], exc)
        return None


def select_moves(
    smiles: str,
    fragment_smiles: Sequence[str],
    failure_codes: Sequence[str],
    *,
    limit: int = DEFAULT_MAX_MOVES,
) -> list[Move]:
    """The moves to try, best bet first.

    Ordered by whether the move is a remedy for something the assessor
    actually complained about, then by the library's own order, which is
    corpus frequency. A move that no code asked for is still offered once the
    targeted ones run out: the failure codes say what was *measured* to be
    wrong, and a molecule can carry an epoxide while being flagged only for
    cLogP.
    """
    candidates = moves_inside_linker(smiles, fragment_smiles)
    if candidates is None:
        # No containment check was possible. Offer only the moves the codes
        # ask for, because an untargeted edit with nothing checking where it
        # lands is the fragment-loss failure waiting to happen.
        codes = set(failure_codes)
        candidates = [m for m in MOVES if codes.intersection(m.addresses)]

    codes = set(failure_codes)
    targeted = [m for m in candidates if codes.intersection(m.addresses)]
    rest = [m for m in candidates if not codes.intersection(m.addresses)]
    return (targeted + rest)[: max(0, limit)]


def properties_from_tools(
    smiles: str,
    check: Any,
    metrics: Any,
) -> LinkerProperties:
    """Assemble a `LinkerProperties` out of what the tools returned.

    Deliberately not `properties.measure`. The expert must not be able to see
    anything a policy with the same tool list cannot see, or its traces teach
    a standard of evidence the model has no way to meet -- and the acceptance
    decision below would be made on numbers that never appeared in the
    transcript.

    `check_protac_linker` supplies the identity half (fragments, attachments,
    reactive groups) and the linker geometry; `get_molecule_metrics` supplies
    the whole-molecule descriptors.
    """
    props = LinkerProperties(smiles=smiles)

    if not isinstance(check, dict) or not isinstance(metrics, dict):
        props.measurable = False
        props.error = "a measurement call did not return a result"
        return props

    props.fragments_intact = bool(check.get("fragments_intact"))
    props.n_attachments = check.get("attachment_bonds")
    props.reactive_groups = [str(g) for g in (check.get("reactive_groups") or ())]
    props.linker_smiles = check.get("linker_smiles")
    props.linker_heavy_atoms = check.get("linker_heavy_atoms")
    props.span = check.get("span_bonds")
    props.canonical = check.get("canonical_smiles") or metrics.get("canonical_smiles")

    props.sa_score = metrics.get("sa_score")
    props.qed = metrics.get("qed")
    props.clogp = metrics.get("clogp")
    props.tpsa = metrics.get("tpsa")
    props.exact_mw = metrics.get("exact_mw")
    props.rotatable_bonds = metrics.get("num_rotatable_bonds")
    props.heavy_atoms = metrics.get("num_heavy_atoms")
    # `mw` is the only threshold the tool list cannot measure directly: there
    # is no average mass on get_molecule_metrics, only the exact one. Over the
    # 1,993 corpus rows carrying both, they differ by a median of 0.84 Da and
    # at most 2.84, against an mw band of 780-940 with 660 of tolerance -- so
    # standing one in for the other cannot change which side of a band this
    # molecule is on. The reward still scores the real `mw` from
    # grade.measure; this value exists only to pick between two candidates.
    props.mw = metrics.get("exact_mw")

    props.measurable = props.canonical is not None
    if not props.measurable:
        props.error = "neither tool returned a canonical SMILES"
    return props


def tool_outputs(text: str) -> list[tuple[str, Any]]:
    """Every tool result in a slice of transcript, parsed when it is JSON."""
    results: list[tuple[str, Any]] = []
    for match in _TOOL_OUTPUT.finditer(text):
        body = match.group("body")
        try:
            results.append((match.group("name"), json.loads(body)))
        except json.JSONDecodeError:
            results.append((match.group("name"), body))
    return results


def _turn(reasoning: str, *calls: tuple[str, dict[str, Any]]) -> str:
    """One assistant turn carrying a thought and any number of tool calls.

    `atem.render_continuation` renders one call per turn. The episode loop
    executes every call in a turn in order, and with `max_turns: 8` the expert
    cannot afford a turn per call: the survey alone is three. Calls are
    batched only where the second does not depend on the first's answer.

    The generation prompt already ends with `<|start|>assistant`, so the first
    channel header is emitted without it, exactly as `render_continuation`
    does.
    """
    parts = [render_reasoning(reasoning)]
    parts.extend(render_tool_call(name, arguments) for name, arguments in calls)
    return "".join(parts).removeprefix(START + "assistant")


def _format(value: Any) -> str:
    """A number as a chemist writes it, so a quoted value audits as its own.

    `grade.audit_claims` compares what the reply said against what was
    measured with a per-descriptor tolerance. Printing float noise would not
    fail that, but printing a rounded value that no tool returned would be a
    stated number the tools did not produce, which is the habit this training
    exists to suppress.
    """
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class ScriptedExpert:
    """A greedy linker repairer that measures before it accepts anything.

    One instance per episode. It reads the transcript only for the tool
    results it has not seen yet, which is what the real policy conditions on
    too.

    The accept test is the reward's own, not a single descriptor.
    `moltrain.experts` records the bug this avoids: judging on SA alone made
    its expert trade QED away for it, and SFT would then have taught exactly
    that trade. Here the trap is worse, because the descriptors pull against
    each other by construction -- every polarity move costs cLogP and buys
    TPSA, and a linker shortened to improve SA is not a PROTAC.
    """

    def __init__(
        self,
        task: Any,
        thresholds: dict[str, dict[str, Any]],
        *,
        max_moves: int = DEFAULT_MAX_MOVES,
        edit_tool: str = "benchmark_transformation",
        hold_penalty: float = HOLD_PENALTY,
        turn_budget: int = DEFAULT_TURN_BUDGET,
    ):
        self.task = task
        self.thresholds = thresholds
        self.max_moves = max_moves
        self.edit_tool = edit_tool
        self.hold_penalty = hold_penalty
        self.turn_budget = turn_budget

        self.fragments: list[str] = list(getattr(task, "fragment_smiles", []) or [])
        self.start_smiles: str = getattr(task, "protac_smiles", "")
        self.failure_codes: list[str] = list(getattr(task, "failure_codes", []) or [])

        self.stage = "survey"
        self.turns_emitted = 0
        self.cursor = 0  # how much of the transcript has been read
        self.start: LinkerProperties | None = None
        self.best: str = self.start_smiles
        self.best_props: LinkerProperties | None = None
        self.best_objective = 0.0
        self.moves: list[Move] = []
        self.move_index = 0
        self.pending: str | None = None
        self.pending_move: Move | None = None
        self.accepted: list[str] = []
        self.retries: dict[str, int] = {}
        self.rejected: list[str] = []
        self.notes: list[str] = []

    # ---------------------------------------------------------------- reading

    def _fresh(self, transcript: str) -> dict[str, Any]:
        """Tool results that arrived since the last turn, keyed by tool name.

        Last-wins within a turn: a verify turn calls `check_protac_linker` and
        `get_molecule_metrics` once each, and the survey turn calls three
        tools once each, so there is never more than one result per name to
        choose between.
        """
        slice_ = transcript[self.cursor :]
        self.cursor = len(transcript)
        latest: dict[str, Any] = {}
        for name, body in tool_outputs(slice_):
            latest[name] = body
        return latest

    # ------------------------------------------------------------- judgement

    def _objective(self, candidate: LinkerProperties) -> float:
        """How much better this is, as the reward would see it.

        Goal progress minus what the holds would be charged. A goal is a band
        the molecule started outside; a hold is one it started inside and can
        only lose. Scoring the holds here is what stops the expert from
        closing one band by walking another out of range -- which is what a
        single-descriptor objective does on this task every time.
        """
        if self.start is None:
            return 0.0
        outcomes = score_thresholds(self.start, candidate, self.thresholds)
        goals = [o for o in outcomes if o.kind == "goal"]
        progress = (
            sum(o.progress for o in goals) / len(goals) if goals else 0.0
        )
        charge = sum(
            min(o.charge, VIOLATION_CAP) for o in outcomes if o.kind == "hold"
        )
        return progress - self.hold_penalty * charge

    def _rank(self, candidate: LinkerProperties) -> tuple[int, float]:
        """How good a candidate is, in the order the reward settles things.

        Reactive groups come first and descriptors second, because
        `rewards.compute_reward` floors on the former before it scores the
        latter: a molecule with one reactive group left has no descriptors
        worth comparing.

        Counting them rather than testing for none is what makes a molecule
        with two of them repairable. Measured on the corpus: 37 tasks carry
        `bad_linker_ring` alongside `reactive_linker_group` and several
        linkers carry an aziridine and a Michael acceptor at once. Each single
        fix leaves the other, so a pass/fail test on admissibility rejects
        every intermediate and the expert can never take the second step.
        """
        return (-len(candidate.reactive_groups), self._objective(candidate))

    def _accept(self, candidate: LinkerProperties) -> bool:
        """Is this candidate worth keeping over what we already have?

        The hard gates are the identity checks, and they are absolute: a
        molecule missing a fragment or attached three times is not a worse
        answer, it is not an answer, and it is the shape every
        fragment-eating repair arrives in. Everything that clears them is
        ranked.
        """
        if not candidate.measurable or candidate.span is None:
            return False
        if not candidate.fragments_intact or candidate.n_attachments != 2:
            return False
        if self.best_props is None:
            return True
        # The starting molecule may itself have failed the gates -- it is
        # attached three times, say. Anything that passes them beats it,
        # whatever the descriptors say, because the reward floors the
        # alternative.
        if (
            not self.best_props.fragments_intact
            or self.best_props.n_attachments != 2
        ):
            return True
        best = (-len(self.best_props.reactive_groups), self.best_objective)
        current = self._rank(candidate)
        return current > (best[0], best[1] + 1e-9)

    # ----------------------------------------------------------------- turns

    def __call__(self, transcript: str) -> str:
        fresh = self._fresh(transcript)
        self.turns_emitted += 1
        if self.stage == "survey":
            return self._survey()
        if self.stage == "read_survey":
            return self._read_survey(fresh)
        if self.stage == "read_products":
            return self._read_products(fresh)
        if self.stage == "verify":
            return self._verify(fresh)
        return self._submit()

    def _survey(self) -> str:
        """Measure the input three ways before touching it.

        OPTIMISATION.md's working order starts with validate, then metrics;
        `check_protac_linker` is added because it is the tool that decides
        acceptance and the only one that reports the span. None of the three
        depends on another's answer, so they go in one turn.
        """
        self.stage = "read_survey"
        return _turn(
            "Measure the molecule I was given before changing anything: how it "
            "parses, what its descriptors are, and whether the linker bridges "
            "the two fragments as it stands. Every later comparison has to be "
            "against a number I actually hold.",
            ("validate_smiles", {"smiles": self.start_smiles}),
            ("get_molecule_metrics", {"smiles": self.start_smiles}),
            # The toolkit binds this task's fragments to the tool; passing them
            # again earns a note saying they were ignored.
            ("check_protac_linker", {"smiles": self.start_smiles}),
        )

    def _read_survey(self, fresh: dict[str, Any]) -> str:
        self.start = properties_from_tools(
            self.start_smiles,
            fresh.get("check_protac_linker"),
            fresh.get("get_molecule_metrics"),
        )
        self.best_props = self.start
        self.best = self.start.canonical or self.start_smiles
        self.moves = select_moves(
            self.start_smiles,
            self.fragments,
            self.failure_codes,
            limit=self.max_moves,
        )
        if not self.moves:
            self.notes.append(
                "no move in the library matches inside this linker"
            )
        self.stage = "propose"
        return self._propose()

    def _propose(self) -> str:
        if self.move_index >= len(self.moves):
            return self._submit()
        # A move costs the turn being emitted now plus a verify turn, and the
        # submission needs one of its own. Running out of turns mid-move
        # costs the whole episode: `rewards.no_submission_reward` is -1.0,
        # which is the same floor as eating a warhead and strictly worse than
        # submitting the best molecule measured so far.
        if self.turn_budget - self.turns_emitted < 2:
            self.notes.append("out of turns before the next move could be verified")
            return self._submit()
        move = self.moves[self.move_index]
        self.pending_move = move
        self.stage = "read_products"
        return _turn(
            f"Try {move.name.replace('_', ' ')}: {move.rationale}. It keeps "
            "the bridge the same length, which the ternary complex decides "
            "and I do not. One change at a time, so the effect is "
            "attributable.",
            (
                self.edit_tool,
                {
                    "smiles": self.best,
                    "reaction_smarts": move.smarts,
                },
            ),
        )

    def _read_products(self, fresh: dict[str, Any]) -> str:
        """Pick a product to measure, preferring one the tool says is intact.

        The composite ranks products by band distance and reports
        `fragments_intact` per product, so the choice is made from what the
        tool returned rather than from a local opinion about it. When the
        toolkit falls back to moltrain's plainer composite neither key is
        there, and the first product is taken and checked like any other.
        """
        result = fresh.get(self.edit_tool)
        products: list[dict[str, Any]] = []
        if isinstance(result, dict):
            for entry in result.get("products") or ():
                if isinstance(entry, dict) and entry.get("smiles"):
                    products.append(entry)
                elif isinstance(entry, str):
                    products.append({"smiles": entry})

        chosen = next(
            (p for p in products if p.get("fragments_intact") is not False), None
        )
        if chosen is None:
            name = self.pending_move.name if self.pending_move else "?"
            self.rejected.append(name)
            self.notes.append(f"{name}: no product kept both fragments")
            self.move_index += 1
            self.stage = "propose"
            return self._propose()

        self.pending = str(chosen["smiles"])
        self.stage = "verify"
        return _turn(
            "The transformation applied. Check both fragments survived it and "
            "the linker still bridges them, and measure the product, before "
            "deciding whether to keep it.",
            ("check_protac_linker", {"smiles": self.pending}),
            ("get_molecule_metrics", {"smiles": self.pending}),
        )

    def _verify(self, fresh: dict[str, Any]) -> str:
        name = self.pending_move.name if self.pending_move else "?"
        candidate = (
            properties_from_tools(
                self.pending,
                fresh.get("check_protac_linker"),
                fresh.get("get_molecule_metrics"),
            )
            if self.pending
            else None
        )

        if candidate is not None and self._accept(candidate):
            self.best = candidate.canonical or self.pending or self.best
            self.best_props = candidate
            # The real number, not a floored one: a move that clears a
            # reactive group and costs a little band distance is still the
            # right move, and clamping at zero would hide the cost from every
            # comparison after it.
            self.best_objective = self._objective(candidate)
            self.accepted.append(name)
            if (
                candidate.reactive_groups
                and self.pending_move is not None
                and "reactive_linker_group" in self.pending_move.addresses
                and self.retries.get(name, 0) < MAX_MOVE_RETRIES
            ):
                # The group is still there, so the template matched more than
                # one site and the product only repaired one of them. Offer
                # the same move again against the molecule it just produced
                # rather than moving on with a reactive linker, which the
                # reward floors outright.
                self.retries[name] = self.retries.get(name, 0) + 1
                self.pending = None
                self.stage = "propose"
                return self._propose()
        else:
            self.rejected.append(name)
            if candidate is not None and not candidate.admissible:
                self.notes.append(f"{name}: the product was not admissible")
            else:
                self.notes.append(f"{name}: measured no better than what I had")

        self.pending = None
        self.move_index += 1
        self.stage = "propose"
        return self._propose()

    def _submit(self) -> str:
        """Validate the answer, then commit it. Both in the last turn.

        `validate_smiles` on the final answer is step six of OPTIMISATION.md's
        working order. It goes in the same turn as the submission because
        `submit_molecule` is terminal and the episode ends the moment it
        returns, so a separate turn for the check would cost the budget a
        whole move.
        """
        self.stage = "done"
        props = self.best_props
        measured = ""
        # The only numbers this expert ever states. They describe the molecule
        # being submitted and came from the verify turn that measured it, so
        # the claim audit reads them against the same measurement. Every
        # earlier turn speaks in directions rather than digits: a number about
        # an intermediate candidate is true of nothing the audit can check,
        # and would be scored as a stated measurement the tools never returned.
        if props is not None and props.measurable:
            parts = []
            if props.span is not None:
                # The span is quoted from `check_protac_linker`, which is the
                # only span a policy can see. On 9 of 300 corpus molecules
                # that number disagrees with `grade.measure`'s -- sometimes
                # by one bond, once by eleven -- while both agree on the
                # linker's heavy-atom count, so the two differ on which route
                # between the anchors counts and not on where the linker is.
                # The audit scores a stated span against grade.measure, so on
                # those molecules an honestly quoted tool reading is recorded
                # as a false claim. Quoting it anyway is the right side of
                # that trade: suppressing the number would hide a
                # disagreement that the reward is already paying on.
                parts.append(f"span {_format(props.span)} bonds")
            if props.clogp is not None:
                parts.append(f"cLogP {_format(props.clogp)}")
            if props.sa_score is not None:
                parts.append(f"SA {_format(props.sa_score)}")
            if props.tpsa is not None:
                parts.append(f"TPSA {_format(props.tpsa)}")
            if parts:
                measured = " As measured: " + ", ".join(parts) + "."

        if self.accepted:
            summary = (
                "Applied " + ", ".join(self.accepted)
                + ". Each was measured and kept only because it moved the "
                "molecule toward its targets without pushing another one out, "
                "and none of them changed the linker's length."
            )
        else:
            reason = self.notes[0] if self.notes else "nothing measured better"
            summary = (
                "I could not improve this linker without changing a fragment "
                f"or making something else worse ({reason}), so my answer is "
                "the molecule as it stands. A truthful failure is worth more "
                "than a molecule that scores well by cheating."
            )

        return _turn(
            summary + measured,
            ("validate_smiles", {"smiles": self.best}),
            ("submit_molecule", {"smiles": self.best, "rationale": summary}),
        )


# --------------------------------------------------------------- calibration


@dataclass
class EpisodeOutcome:
    """One scored episode, flattened for the report."""

    task_id: str
    optimisation_class: str
    failure_codes: list[str] = field(default_factory=list)
    reward: float = 0.0
    floored: str = ""
    accepted: list[str] = field(default_factory=list)
    submitted: str | None = None
    changed: bool = False
    n_tool_calls: int = 0
    n_tool_failures: int = 0
    n_malformed_calls: int = 0
    grounding: float = 0.0
    discipline: float = 0.0
    verified: bool = False
    terms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "optimisation_class": self.optimisation_class,
            "failure_codes": self.failure_codes,
            "reward": round(self.reward, 4),
            "floored": self.floored,
            "accepted_moves": self.accepted,
            "submitted_smiles": self.submitted,
            "changed": self.changed,
            "n_tool_calls": self.n_tool_calls,
            "n_tool_failures": self.n_tool_failures,
            "n_malformed_calls": self.n_malformed_calls,
            "grounding": round(self.grounding, 4),
            "discipline": round(self.discipline, 4),
            "verified_before_submit": self.verified,
            "terms": self.terms,
        }


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "min": round(ordered[0], 4),
        "p25": round(ordered[max(0, (len(ordered) - 1) // 4)], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[min(len(ordered) - 1, 3 * (len(ordered) - 1) // 4)], 4),
        "max": round(ordered[-1], 4),
    }


@dataclass
class CalibrationReport:
    """The expert's measured spread, and what it implies for sft.min_reward."""

    episodes: list[EpisodeOutcome] = field(default_factory=list)
    anchors: list[EpisodeOutcome] = field(default_factory=list)

    def _rewards(self, rows: Sequence[EpisodeOutcome]) -> list[float]:
        return [row.reward for row in rows]

    def suggestion(self) -> tuple[float | None, str]:
        """A threshold between the no-op anchor and a real repair.

        Three outcomes, and which one comes back is itself the finding.

        The two populations overlap -> None. If a diligent episode that
        changed nothing scores as well as one that repaired something, no
        cutoff separates the traces and the reward is what needs fixing.

        The anchors are admissible and score below the repairs -> the
        midpoint, which is how `moltrain`'s 0.45 was set between its 0.38
        no-op and its 0.59 real move.

        Every anchor was floored -> the anchors do not constrain the cutoff
        at all, because `traces.collect` drops a floored episode before it
        ever looks at `min_reward`. Then the threshold's only job is to drop
        the weakest repairs, and the first quartile of the repairs themselves
        is the evidence available for where that is.
        """
        moved = sorted(
            row.reward for row in self.episodes if row.changed and not row.floored
        )
        if not moved:
            return None, (
                "no episode both changed the molecule and stayed admissible, "
                "so there is nothing to set a threshold above"
            )
        if not self.anchors:
            return None, "no no-op anchor was run"

        admissible = sorted(row.reward for row in self.anchors if not row.floored)
        quartile = moved[max(0, (len(moved) - 1) // 4)]
        if not admissible:
            return round(quartile, 2), (
                f"every one of the {len(self.anchors)} no-op anchors was "
                "floored as the input unchanged, so the anchor does not bind: "
                "traces.collect drops a floored episode before min_reward is "
                f"read. Set at the first quartile of real repairs ({quartile:.3f})"
            )

        ceiling = admissible[
            min(len(admissible) - 1, int(0.9 * (len(admissible) - 1)))
        ]
        floor = statistics.median(moved)
        if floor <= ceiling:
            return None, (
                f"the median real repair ({floor:.3f}) does not clear the "
                f"ninetieth-percentile admissible no-op ({ceiling:.3f}); a "
                "reward threshold cannot separate these two populations"
            )
        return round((ceiling + floor) / 2, 2), (
            f"midway between the ninetieth-percentile admissible no-op "
            f"({ceiling:.3f}) and the median real repair ({floor:.3f})"
        )

    def as_dict(self) -> dict[str, Any]:
        by_class: dict[str, dict[str, Any]] = {}
        for row in self.episodes:
            bucket = by_class.setdefault(
                row.optimisation_class,
                {"n": 0, "changed": 0, "floored": 0, "rewards": []},
            )
            bucket["n"] += 1
            bucket["changed"] += int(row.changed)
            bucket["floored"] += int(bool(row.floored))
            bucket["rewards"].append(row.reward)
        for bucket in by_class.values():
            bucket["reward"] = _summary(bucket.pop("rewards"))

        floors: dict[str, int] = {}
        for row in self.episodes:
            if row.floored:
                floors[row.floored] = floors.get(row.floored, 0) + 1

        moves: dict[str, int] = {}
        for row in self.episodes:
            for name in row.accepted:
                moves[name] = moves.get(name, 0) + 1

        suggestion, why = self.suggestion()
        return {
            "episodes": len(self.episodes),
            "changed": sum(1 for row in self.episodes if row.changed),
            "floored": sum(1 for row in self.episodes if row.floored),
            "reward_all": _summary(self._rewards(self.episodes)),
            "reward_changed": _summary(
                [r.reward for r in self.episodes if r.changed and not r.floored]
            ),
            "reward_noop_anchor": _summary(self._rewards(self.anchors)),
            "grounding": _summary([r.grounding for r in self.episodes]),
            "discipline": _summary([r.discipline for r in self.episodes]),
            "verified_before_submit": sum(
                1 for r in self.episodes if r.verified
            ),
            "tool_failures": sum(r.n_tool_failures for r in self.episodes),
            "malformed_calls": sum(r.n_malformed_calls for r in self.episodes),
            "by_class": by_class,
            "floor_reasons": floors,
            "accepted_moves": dict(
                sorted(moves.items(), key=lambda kv: -kv[1])
            ),
            "suggested_min_reward": suggestion,
            "suggestion_basis": why,
        }


def score_episode(
    cfg: dict[str, Any], task: Any, result: Any, oracle: Any = None
) -> EpisodeOutcome:
    """Score one finished episode the way training will score it.

    Assembled here rather than imported because `traces.collect` scores for
    its own purposes and the evaluation arms score the same transcript under
    more than one configuration; what both share is `compute_reward`, and that
    is what is called.
    """
    from .protocol import score_protocol
    from .rewards import compute_reward

    final = getattr(result, "final_properties", None)
    start = getattr(result, "start_properties", None)
    measured = final.as_dict() if final is not None else {}
    baseline = start.as_dict() if start is not None else {}

    protocol = score_protocol(
        getattr(result, "assistant_text", ""),
        getattr(result, "calls", []),
        measured,
        baseline,
    )

    synthesis = None
    if oracle is not None and result.submitted_smiles:
        synthesis = oracle.synthesis_score(
            result.submitted_smiles, task.fragment_smiles
        )

    breakdown = compute_reward(
        cfg,
        start if start is not None else LinkerProperties(smiles=task.protac_smiles),
        final,
        protocol=protocol,
        calls=getattr(result, "calls", []),
        reference_smiles=getattr(task, "reference_smiles", None),
        synthesis_score=synthesis,
    )

    submitted = result.submitted_smiles
    changed = bool(
        submitted
        and final is not None
        and start is not None
        and final.canonical
        and start.canonical
        and final.canonical != start.canonical
    )
    return EpisodeOutcome(
        task_id=task.task_id,
        optimisation_class=getattr(task, "optimisation_class", "chemistry"),
        failure_codes=list(getattr(task, "failure_codes", []) or []),
        reward=float(breakdown.reward),
        floored=breakdown.floor_reason,
        submitted=submitted,
        changed=changed,
        n_tool_calls=getattr(result, "n_tool_calls", 0),
        n_tool_failures=getattr(result, "n_tool_failures", 0),
        n_malformed_calls=getattr(result, "n_malformed_calls", 0),
        grounding=protocol.grounding,
        discipline=protocol.discipline,
        verified=protocol.verified_before_submit,
        terms=breakdown.terms,
    )


def sample_tasks(
    tasks: Sequence[Any], *, per_class: int, seed: int
) -> list[Any]:
    """A stratified sample, because the classes do not behave alike.

    `evaluate.stratify_by` says the same thing about the evaluation arms, and
    for the same reason: a `reactive` task is one group swap and a `chemistry`
    task may not be fixable by a linker edit at all. An unstratified sample of
    this corpus is 43% control episodes and would calibrate the threshold
    against molecules with nothing to repair.
    """
    import random

    buckets: dict[str, list[Any]] = {}
    for task in tasks:
        buckets.setdefault(
            getattr(task, "optimisation_class", "chemistry"), []
        ).append(task)

    rng = random.Random(seed)
    chosen: list[Any] = []
    for name in sorted(buckets):
        bucket = list(buckets[name])
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_class])
    return chosen


async def calibrate(
    cfg: dict[str, Any],
    tasks: Sequence[Any],
    *,
    anchors: bool = True,
    disable_progress: bool | None = None,
) -> CalibrationReport:
    """Run the expert over `tasks` and report the reward spread it produced.

    Two episodes per task when `anchors` is on. The second runs the same
    expert with its move budget set to zero, which makes it survey the
    molecule properly and then submit it untouched -- a diligent episode that
    changed nothing. That is the floor `sft.min_reward` has to sit above, and
    running it rather than deriving it means the protocol terms in it are the
    ones this expert really earns.
    """
    from .episode import run_episode
    from .oracle import LinkerOracle
    from .toolkit import build_toolkit

    thresholds = cfg["reward"]["thresholds"]
    turns = int(cfg["episode"]["max_turns"])
    base = build_toolkit(cfg)
    oracle = LinkerOracle(cfg)
    report = CalibrationReport()

    bar = progress.track(
        total=len(tasks) * (2 if anchors else 1),
        desc="calibrating expert",
        unit="episode",
        disable=disable_progress,
    )
    for task in tasks:
        toolkit = base.for_fragments(task.fragment_smiles)
        bar.postfix(task.task_id)

        expert = ScriptedExpert(task, thresholds, turn_budget=turns)
        result = await run_episode(cfg, task, expert, toolkit, oracle)
        outcome = score_episode(cfg, task, result, oracle)
        outcome.accepted = list(expert.accepted)
        report.episodes.append(outcome)
        LOG.info(
            "%s [%s] reward %.3f after %d calls%s",
            task.task_id,
            outcome.optimisation_class,
            outcome.reward,
            outcome.n_tool_calls,
            f" ({', '.join(expert.accepted)})" if expert.accepted else " (no move)",
        )
        bar.update(1)

        if anchors:
            # A fresh task object: run_episode measures the starting molecule
            # onto the result, and reusing one would let the anchor inherit
            # the first episode's state.
            twin = type(task).from_dict(task.as_dict())
            noop = ScriptedExpert(twin, thresholds, max_moves=0, turn_budget=turns)
            anchor_result = await run_episode(cfg, twin, noop, toolkit, oracle)
            report.anchors.append(
                score_episode(cfg, twin, anchor_result, oracle)
            )
            bar.update(1)
    bar.close()

    LOG.info(
        "calibrated on %d episodes in %s",
        len(report.episodes),
        progress.format_seconds(bar.elapsed_s),
    )
    return report


def _print_report(report: CalibrationReport, episodes_path: Path, json_path: Path) -> int:
    summary = report.as_dict()
    moved = summary["reward_changed"]
    anchor = summary["reward_noop_anchor"]

    print(
        f"{summary['episodes']} episodes: {summary['changed']} changed the "
        f"molecule, {summary['floored']} floored"
    )
    if moved:
        print(
            "  real repair : "
            f"median {moved['median']:+.3f}  "
            f"p25 {moved['p25']:+.3f}  p75 {moved['p75']:+.3f}  "
            f"range {moved['min']:+.3f}..{moved['max']:+.3f}  (n={moved['n']})"
        )
    else:
        print("  real repair : none")
    if anchor:
        print(
            "  no-op anchor: "
            f"median {anchor['median']:+.3f}  "
            f"p25 {anchor['p25']:+.3f}  p75 {anchor['p75']:+.3f}  "
            f"range {anchor['min']:+.3f}..{anchor['max']:+.3f}  (n={anchor['n']})"
        )
    for name, bucket in sorted(summary["by_class"].items()):
        line = (
            f"  {name:<11}: n={bucket['n']:<4} changed={bucket['changed']:<4} "
            f"floored={bucket['floored']:<4}"
        )
        stats = bucket["reward"]
        if stats:
            line += f" median {stats['median']:+.3f}"
        print(line)
    if summary["accepted_moves"]:
        print("  moves kept  : " + ", ".join(
            f"{name} x{count}" for name, count in summary["accepted_moves"].items()
        ))

    suggested = summary["suggested_min_reward"]
    if suggested is None:
        print(f"\nsft.min_reward: NOT SETTABLE -- {summary['suggestion_basis']}")
    else:
        print(f"\nsft.min_reward: {suggested}  ({summary['suggestion_basis']})")

    print(f"\nreport   -> {json_path}")
    print(f"episodes -> {episodes_path}")
    return 0 if report.episodes else 1


async def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from .config import resolve
    from .tasks import TaskBuildError, read_tasks

    try:
        tasks = read_tasks(cfg, args.tasks)
    except TaskBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not tasks:
        print("no tasks to run", file=sys.stderr)
        return 1

    sampled = sample_tasks(
        tasks, per_class=args.per_class, seed=int(cfg["run"]["seed"])
    )
    if args.limit:
        sampled = sampled[: args.limit]

    report = await calibrate(
        cfg,
        sampled,
        anchors=not args.no_anchors,
        disable_progress=logs.progress_disabled(cfg, args),
    )

    out = resolve(cfg, args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    episodes_path = out.with_name(out.stem + "_episodes.jsonl")
    with episodes_path.open("w") as handle:
        for row in report.episodes:
            handle.write(json.dumps({**row.as_dict(), "arm": "expert"}) + "\n")
        for row in report.anchors:
            handle.write(json.dumps({**row.as_dict(), "arm": "noop"}) + "\n")

    return _print_report(report, episodes_path, out)


def _cmd_moves(args: argparse.Namespace) -> int:
    """Print the library, so a move can be read without reading the source."""
    for move in MOVES:
        print(f"{move.name}")
        print(f"  for      : {', '.join(move.addresses)}")
        print(f"  smarts   : {move.smarts}")
        print(f"  measured : {move.measured}")
        print(f"  why      : {move.rationale}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.experts",
        description="The scripted linker expert, and the calibration that "
        "sets sft.min_reward from its measured spread.",
    )
    parser.add_argument("command", choices=["calibrate", "moves"])
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--tasks", default=None, help="tasks.jsonl to run")
    parser.add_argument(
        "--per-class",
        type=int,
        default=25,
        help="episodes per optimisation class (default 25)",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the sample")
    parser.add_argument(
        "--out",
        default="runs/calibration/expert.json",
        help="where to write the report",
    )
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="skip the no-op anchor episode for each task",
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    if args.command == "moves":
        return _cmd_moves(args)

    from .config import ConfigError, load_config, log_destination

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    args.log_file = log_destination(cfg, args, f"experts-{args.command}")
    logs.configure_from_args(cfg, args, stem=f"experts-{args.command}")

    try:
        return asyncio.run(_run(cfg, args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
