"""The multi-turn linker-repair episode the policy is trained on.

The loop is policy-agnostic, exactly as `moltrain.episode` is: it takes a
`generate` callable that maps a transcript to the next assistant turn. A
scripted function stands in for the model during tests, vLLM drives it during
GRPO rollouts, and nothing else changes. That is what keeps the environment
verifiable on a laptop with no CUDA.

What differs from the pocket task is the shape of the problem. There is no
receptor and no docking -- a PROTAC binds two proteins at once and is decided
by the ternary complex, which docking one rigid receptor does not model, so
`protacbench/OPTIMISATION.md` leaves the docking tools off the list. What takes
their place is an identity constraint: the warhead and the E3 ligand are frozen
and only the linker between them may be edited. That constraint is the reason
the prompts in this module are worded the way they are, and the reason
`submit_molecule` is the terminal act rather than a parsed `REPAIRED:` line --
a tool call is unambiguous where a line of prose is a parsing problem.

The reward is NOT computed here. The caller scores the result, because the
episode is also run by the evaluation arms, which score the same transcript
several different ways.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from moltrain.atem import (
    END_TURN,
    MESSAGE,
    START,
    parse_assistant_text,
    render_tool_result,
)
from moltrain.logs import get_logger

from .properties import LinkerProperties, measure
from .protocol import SUBMIT_TOOL

LOG = get_logger("episode")

# Either a plain or an async callable: (transcript) -> next assistant turn.
GenerateFn = Callable[[str], str | Awaitable[str]]


# What the agent should do about each failure, specifically. Transcribed from
# the per-failure remedy table in protacbench/OPTIMISATION.md; the wording is
# deliberately concrete because that document records that generic advice
# produces generic edits.
REMEDIES: dict[str, str] = {
    "reactive_linker_group": (
        "Replace the reactive group with a stable equivalent: aldehyde or "
        "in-chain ketone -> amide or primary alcohol; Michael acceptor -> the "
        "saturated chain; imine -> the corresponding amine; epoxide -> diol or "
        "ether. Do not delete the atoms -- keep the linker's length."
    ),
    "disconnected": (
        "Join the pieces into one molecule. The linker must bond to both "
        "fragments, one bond at each end."
    ),
    "sanitization_failed": (
        "Fix the valence or aromaticity violation. Use validate_smiles after "
        "each edit."
    ),
    "linker_not_bridging": (
        "Rebuild so the linker has exactly two attachment points, one per "
        "fragment. Touching one fragment twice makes it a substituent, not a "
        "linker."
    ),
    "bad_linker_ring": (
        "Replace the strained or oversized ring with an open chain or a five- "
        "or six-membered ring."
    ),
    "pains_alert": (
        "Remove the interference substructure from the linker. Do not alter "
        "either fragment, even if the alert matches there."
    ),
    "sa_too_high": (
        "Simplify: prefer amide, ether and short alkyl over unusual rings and "
        "dense substitution."
    ),
    "clogp_too_high": (
        "Make the linker more polar. Ether oxygens or an amide in place of "
        "alkyl is the usual move."
    ),
    "too_flexible": (
        "Replace part of a flexible chain with a para-substituted ring or a "
        "short rigid spacer."
    ),
    "macrocycle": (
        "Break the macrocycle. The linker should connect the fragments without "
        "closing a ring through them."
    ),
}


SYSTEM_PROMPT = """You are a medicinal chemist optimising the linker of a PROTAC.

A PROTAC has three parts: a warhead that binds a target protein, an
E3-ubiquitin-ligase ligand, and a linker joining them. It works by holding
both proteins together in a ternary complex, so the linker's length and
composition decide whether it functions at all.

THE TWO FRAGMENTS ARE NOT YOURS TO CHANGE. The warhead and the E3 ligand are
the reason the molecule exists; their structures came from medicinal
chemistry programmes, not from a generator. You may only edit the linker
between them. A molecule that scores better because you deleted or altered a
fragment is a different compound and will be rejected automatically.

You are NOT making this molecule drug-like in the oral small-molecule sense.
PROTACs are ~900 Da and violate Lipinski by construction. Do not optimise
QED. Do not reduce molecular weight toward typical drug ranges. Doing so
makes the molecule worse at its job.

WHAT A GOOD LINKER LOOKS LIKE, measured from 128 published PROTACs and from
the passing samples of this run:
  - spans 8 to 13 bonds between the two fragments (never below 7 or above 17)
  - 8 to 14 heavy atoms
  - built from chemistry a lab runs on Monday: amide, PEG ether, short alkyl,
    piperazine, piperidine, triazole, sulfonamide, carbamate
  - SA score at or below 4.9
  - cLogP between 3.5 and 5.6
  - TPSA at or above 170
  - contains NO reactive or unstable group: no aldehyde, ketone in-chain,
    imine, Michael acceptor, epoxide, aziridine, acyl halide, anhydride,
    vinyl ether, enol, allene, peroxide or nitroso. These react with protein
    lysines and cysteines in vivo or do not survive as drawn.

TOOLS:
{tools}

MEASUREMENT DISCIPLINE. Never state a number you did not get from a tool. If
you believe a change lowered cLogP, call get_molecule_metrics and read it.
Make ONE change at a time and re-measure after each. An unmeasured claim is
treated as false.

HOW TO WORK:
  1. validate_smiles on the input, to see how it parses.
  2. get_molecule_metrics, to confirm the starting values.
  3. Identify the smallest edit to the linker that addresses the stated
     problem. Prefer replacing a group over rebuilding the linker.
  4. Apply it, re-measure, check it did not break something else.
  5. Repeat until the problem is fixed or you have tried three distinct
     approaches.
  6. validate_smiles on your final answer before reporting it.

FINISH by calling submit_molecule exactly once, with the full repaired PROTAC
SMILES. The task ends there, so call nothing after it.

If you cannot repair it without changing a fragment, call submit_molecule with
the molecule you were given, unchanged, and say in the rationale what you tried
and why it failed. A truthful failure is worth more than a molecule that scores
well by cheating.

You have at most {max_turns} turns and {max_tool_calls} tool calls."""


# One line each, in the order protactrain/configs/default.yaml lists them. The
# text is OPTIMISATION.md's own table, because the agent's picture of what a
# tool is for should not be a second, paraphrased description that can drift.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "check_protac_linker": (
        "check_protac_linker(smiles, fragment_1, fragment_2) -- the one that "
        "decides acceptance: fragments intact and disjoint, exactly two bonds "
        "to the linker, no reactive group, and the span in bonds."
    ),
    "benchmark_transformation": (
        "benchmark_transformation(smiles, reaction_smarts) -- validate, "
        "transform and score in one call; products ranked by SA improvement."
    ),
    "validate_smiles": (
        "validate_smiles(smiles) -- does it parse, and what RDKit objected to."
    ),
    "get_molecule_metrics": (
        "get_molecule_metrics(smiles) -- SA, QED, mass, heavy atoms."
    ),
    "render_molecule_2d": (
        "render_molecule_2d(smiles) -- a mangled warhead is obvious in a "
        "picture and invisible in a SMILES string."
    ),
    "generate_3d_conformer": (
        "generate_3d_conformer(smiles) -- strain: a linker that only reaches "
        "its span in a high-energy conformation is a bad linker."
    ),
    SUBMIT_TOOL: (
        "submit_molecule(smiles, rationale) -- commit your answer. This ends "
        "the task."
    ),
}


USER_PROMPT = """Repair the linker of PROTAC {target_id}, sample {sample_id}.

WHAT IS WRONG
{problems}

THE MOLECULE
  Full PROTAC: {protac_smiles}
  Linker only: {linker_smiles}
  Fragment 1:  {fragment_1}    <- must survive unchanged
  Fragment 2:  {fragment_2}    <- must survive unchanged

AS MEASURED
{measured}

WHAT TO AIM FOR
{goals}

The linker must keep roughly its current length: {linker_size} heavy atoms.
That length comes from the geometry of the ternary complex, not from
preference. Shortening it to improve a descriptor breaks the molecule."""


@dataclass
class LinkerTask:
    """One linker optimisation problem, as the dataset hands it over."""

    task_id: str
    target_id: str
    sample_id: str
    protac_smiles: str
    fragment_smiles: list[str]  # exactly two: warhead and E3 ligand
    linker_smiles: str | None = None
    linker_size: int = 0
    verdict: str = "poor"  # poor | broken
    failure_codes: list[str] = field(default_factory=list)
    failure_detail: str = ""
    # structural | reactive | chemistry | control. Evaluation stratifies on it
    # because the three repairable classes fail in different ways and do not
    # have the same expected yield.
    optimisation_class: str = "chemistry"
    # The measured properties of the source row, flattened onto the names the
    # reward thresholds are written against. Flat on purpose: the prompt and
    # the reward read the same keys, so they cannot quote different numbers.
    baseline: dict[str, Any] = field(default_factory=dict)
    # The published molecule this sample was generated against, when one is
    # known. Submitting it back is a flag, not a floor -- see rewards.py.
    reference_smiles: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> LinkerTask:
        fragments = list(record.get("fragment_smiles") or [])
        return cls(
            task_id=record["task_id"],
            target_id=str(record.get("target_id", "")),
            sample_id=str(record.get("sample_id", "")),
            protac_smiles=record["protac_smiles"],
            fragment_smiles=fragments,
            linker_smiles=record.get("linker_smiles"),
            linker_size=int(record.get("linker_size") or 0),
            verdict=record.get("verdict", "poor"),
            failure_codes=list(record.get("failure_codes") or []),
            failure_detail=record.get("failure_detail", "") or "",
            optimisation_class=record.get("optimisation_class", "chemistry"),
            baseline=dict(record.get("baseline") or {}),
            reference_smiles=record.get("reference_smiles"),
            metadata=dict(record.get("metadata") or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_id": self.target_id,
            "sample_id": self.sample_id,
            "protac_smiles": self.protac_smiles,
            "fragment_smiles": list(self.fragment_smiles),
            "linker_smiles": self.linker_smiles,
            "linker_size": self.linker_size,
            "verdict": self.verdict,
            "failure_codes": list(self.failure_codes),
            "failure_detail": self.failure_detail,
            "optimisation_class": self.optimisation_class,
            "baseline": dict(self.baseline),
            "reference_smiles": self.reference_smiles,
            "metadata": dict(self.metadata),
        }


# moltrain's equivalent dataclass is called `Task`, and every module ported
# from it -- grpo.py, traces.py -- spells it that way. The name here is
# `LinkerTask` because a bare `Task` says nothing about the two frozen
# fragments that define this one, but the alias keeps the ported call sites
# working unchanged rather than leaving one of the two names wrong.
Task = LinkerTask


@dataclass
class EpisodeResult:
    """Everything one rollout produced. The reward is filled in by the caller."""

    task: LinkerTask
    transcript: str
    # (transcript_before, emitted_text) at the policy boundary, in order.
    # moltrain.grpo_core.build_completion reconstructs the loss mask from these
    # by string arithmetic, so they must be recorded at the call site rather
    # than recovered from the finished transcript.
    turns: list[tuple[str, str]] = field(default_factory=list)
    # Tool outcomes in order. Duck-typed on `.name`, `.ok` and `.as_text()`,
    # which is all protocol.py and rewards.py read off them.
    calls: list[Any] = field(default_factory=list)
    submitted_smiles: str | None = None
    start_properties: LinkerProperties | None = None
    final_properties: LinkerProperties | None = None
    # Every assistant channel the model wrote, concatenated, for the claim
    # audit. Tool output is excluded on purpose: a number the environment
    # printed is not a claim the model made.
    assistant_text: str = ""
    reward: float | None = None
    truncated: bool = False

    stop_reason: str = "max_turns"
    n_turns: int = 0
    n_tool_calls: int = 0
    n_tool_failures: int = 0
    n_malformed_calls: int = 0
    wall_time_s: float = 0.0
    tool_time_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "optimisation_class": self.task.optimisation_class,
            "submitted_smiles": self.submitted_smiles,
            "final_properties": (
                self.final_properties.as_dict() if self.final_properties else None
            ),
            "reward": self.reward,
            "truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "n_turns": self.n_turns,
            "n_tool_calls": self.n_tool_calls,
            "n_tool_failures": self.n_tool_failures,
            "n_malformed_calls": self.n_malformed_calls,
            "wall_time_s": self.wall_time_s,
            "tool_time_s": self.tool_time_s,
        }


# How a threshold name is spelled in prose, and how to say its band. Keeping
# the label table here and the numbers in the config is the point: the prompt
# can be reworded without ever changing what the policy is paid for.
_GOAL_LABELS: dict[str, tuple[str, str]] = {
    "span": ("linker span", "bonds"),
    "linker_heavy_atoms": ("linker heavy atoms", ""),
    "sa_score": ("SA", ""),
    "clogp": ("cLogP", ""),
    "tpsa": ("TPSA", ""),
    "rotatable_bonds": ("rotatable bonds", ""),
    "mw": ("MW", ""),
}

# Where a threshold's starting value lives when the dataset row was flattened
# under its own names rather than the reward's. repair_tasks.jsonl reports span
# as linker.path_length_bonds and the linker atom count as linker.heavy_atoms,
# which collides with the whole-molecule heavy_atoms -- hence the explicit map.
_BASELINE_ALIASES: dict[str, tuple[str, ...]] = {
    "span": ("span", "path_length_bonds", "linker_path_length_bonds"),
    "linker_heavy_atoms": ("linker_heavy_atoms", "linker_size"),
    "sa_score": ("sa_score",),
    "clogp": ("clogp",),
    "tpsa": ("tpsa",),
    "rotatable_bonds": ("rotatable_bonds",),
    "mw": ("mw",),
}


def _number(value: Any) -> str:
    """A number as a chemist would write it: no trailing .0, no float noise."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:g}"
    return str(value)


def baseline_value(task: LinkerTask, name: str) -> Any:
    """The starting value of one threshold, under whichever key it arrived as."""
    baseline = task.baseline or {}
    for key in _BASELINE_ALIASES.get(name, (name,)):
        if baseline.get(key) is not None:
            return baseline[key]
    return None


def describe_failures(task: LinkerTask) -> str:
    """The WHAT IS WRONG block: each code, what it means, what to do about it.

    The description half comes from `protacbench.assess`, imported lazily so
    this module still imports where RDKit is absent. The remedy half is the
    local REMEDIES table, which is OPTIMISATION.md's.
    """
    try:
        from protacbench.assess import HARD_FAILURES, SOFT_FAILURES
    except Exception:  # noqa: BLE001 - a prompt must never fail to render
        HARD_FAILURES, SOFT_FAILURES = {}, {}

    lines: list[str] = []
    for code in task.failure_codes:
        what = HARD_FAILURES.get(code) or SOFT_FAILURES.get(code) or code
        remedy = REMEDIES.get(code)
        lines.append(f"- {code}: {what}." + (f" {remedy}" if remedy else ""))
    if not lines:
        # The qed-only controls land here, and deliberately so: they are told
        # that nothing is wrong, and what they do with that is the measurement.
        lines.append(
            "- nothing is recorded as wrong with this linker under the "
            "criteria you are scored on."
        )
    if task.failure_detail:
        lines.append(f"{task.failure_detail}")
    return "\n".join(lines)


def describe_measurements(task: LinkerTask) -> str:
    """The AS MEASURED block, from the source row's own measurements."""
    baseline = task.baseline or {}
    rows = [
        "  MW {mw}            SA {sa}       cLogP {clogp}".format(
            mw=_number(baseline.get("mw")),
            sa=_number(baseline.get("sa_score")),
            clogp=_number(baseline.get("clogp")),
        ),
        "  TPSA {tpsa}        rotatable {rot}".format(
            tpsa=_number(baseline.get("tpsa")),
            rot=_number(baseline.get("rotatable_bonds")),
        ),
        "  linker: {atoms} heavy atoms spanning".format(
            atoms=_number(baseline_value(task, "linker_heavy_atoms"))
        ),
        "          {span} bonds between the fragments".format(
            span=_number(baseline_value(task, "span"))
        ),
    ]
    reactive = baseline.get("reactive_groups") or []
    if reactive:
        rows.append(
            "  reactive groups in the linker: " + ", ".join(str(r) for r in reactive)
        )
    return "\n".join(rows)


def describe_goals(cfg: dict[str, Any], task: LinkerTask) -> str:
    """The WHAT TO AIM FOR block, rendered from the reward's own bands.

    Read straight out of `cfg['reward']['thresholds']` rather than written as
    prose, so the prompt and the reward can never disagree. That matters more
    than it sounds: a prompt quoting numbers the reward does not use is exactly
    how a policy learns the wrong target and then gets punished for hitting it.
    """
    lines: list[str] = []
    for name, spec in cfg["reward"]["thresholds"].items():
        if not spec.get("enabled", True):
            continue
        label, unit = _GOAL_LABELS.get(name, (name.replace("_", " "), ""))
        low, high = spec.get("low"), spec.get("high")
        if low is not None and high is not None:
            band = f"{_number(low)}-{_number(high)}"
        elif high is not None:
            band = f"at or below {_number(high)}"
        elif low is not None:
            band = f"at or above {_number(low)}"
        else:
            continue
        suffix = f" {unit}" if unit else ""
        current = baseline_value(task, name)
        lines.append(
            f"  {label} {band}{suffix} (currently {_number(current)})"
        )
    # Not a band and not negotiable: a reactive group in the linker is an
    # admissibility floor in rewards.py, not a term that can be outweighed.
    lines.append("  no reactive groups")
    return "\n".join(lines)


def build_prompt(cfg: dict[str, Any], task: LinkerTask) -> str:
    """The system and user turns, before any model output."""
    tool_names = list(cfg["episode"]["tools"])
    if SUBMIT_TOOL not in tool_names:
        # Synthesised by the toolkit rather than configured, because it is the
        # episode's terminal act rather than one of molbench's tools.
        tool_names.append(SUBMIT_TOOL)
    tools = "\n".join(
        "  " + TOOL_DESCRIPTIONS.get(name, f"{name} -- see the tool's own schema.")
        for name in tool_names
    )

    fragments = list(task.fragment_smiles) + ["(missing)", "(missing)"]
    system = SYSTEM_PROMPT.format(
        tools=tools,
        max_turns=cfg["episode"]["max_turns"],
        max_tool_calls=cfg["episode"]["max_tool_calls"],
    )
    user = USER_PROMPT.format(
        target_id=task.target_id,
        sample_id=task.sample_id,
        problems=describe_failures(task),
        protac_smiles=task.protac_smiles,
        linker_smiles=task.linker_smiles or "(not recovered)",
        fragment_1=fragments[0],
        fragment_2=fragments[1],
        measured=describe_measurements(task),
        goals=describe_goals(cfg, task),
        linker_size=task.linker_size,
    )
    return (
        f"{START}system{MESSAGE}{system}{END_TURN}"
        f"{START}user{MESSAGE}{user}{END_TURN}"
        f"{START}assistant"
    )


def _is_terminal(toolkit: Any, name: str) -> bool:
    """Does calling this tool end the episode?

    Asks the toolkit first, because it owns the fact, and falls back to the
    name protocol.py scores against so a toolkit without a spec table still
    terminates rather than looping until max_turns.
    """
    spec = getattr(toolkit, "specs", {}).get(name) if toolkit is not None else None
    if spec is not None:
        return bool(getattr(spec, "terminal", False))
    return name == SUBMIT_TOOL


async def _measure_submission(
    oracle: Any, smiles: str, fragments: Sequence[str], label: str
) -> LinkerProperties:
    """Measure a molecule, through the oracle's cache when there is one.

    `oracle` is duck-typed and may be None. The episode has to run under the
    evaluation arms and under the scripted tests, where there is no cache to
    share, and a measurement path that insists on a collaborator is a
    measurement path that cannot be exercised without one.
    """
    evaluate = None
    if oracle is not None:
        evaluate = getattr(oracle, "evaluate", None) or getattr(oracle, "measure", None)
    if evaluate is None:
        return measure(smiles, list(fragments))

    kwargs: dict[str, Any] = {}
    try:
        if "label" in inspect.signature(evaluate).parameters:
            kwargs["label"] = label
    except (TypeError, ValueError):  # builtins and C callables have no signature
        pass

    try:
        outcome = evaluate(smiles, list(fragments), **kwargs)
        if inspect.isawaitable(outcome):
            outcome = await outcome
    except Exception as exc:  # noqa: BLE001 - a measurement must never raise
        LOG.debug("oracle failed on %s: %s", label, exc)
        return LinkerProperties(
            smiles=smiles, measurable=False, error=f"{type(exc).__name__}: {exc}"
        )
    return outcome


async def run_episode(
    cfg: dict[str, Any],
    task: LinkerTask,
    generate: GenerateFn,
    toolkit: Any,
    oracle: Any = None,
) -> EpisodeResult:
    """Drive one linker-repair episode to termination.

    The reward is left unset. Scoring needs the protocol report, the synthesis
    score and the task's reference molecule, and the evaluation arms score the
    same transcript under more than one configuration, so the caller does it.
    """
    started = time.monotonic()
    episode_cfg = cfg["episode"]
    max_turns = int(episode_cfg["max_turns"])
    max_tool_calls = int(episode_cfg["max_tool_calls"])
    max_chars = int(episode_cfg["max_tool_result_chars"])

    transcript = build_prompt(cfg, task)
    result = EpisodeResult(task=task, transcript=transcript)

    # The starting molecule is measured here rather than trusted from the row.
    # The row's numbers came from a 3D structure converted by Open Babel; the
    # reward compares start against final, and comparing two measurements made
    # by different code is how a reward learns to pay for the difference
    # between the instruments.
    result.start_properties = await _measure_submission(
        oracle, task.protac_smiles, task.fragment_smiles, f"{task.task_id}_start"
    )

    assistant_parts: list[str] = []
    submitted_smiles: str | None = None
    stop_reason = "max_turns"

    while result.n_turns < max_turns:
        result.n_turns += 1
        before = transcript
        emitted = generate(transcript)
        text = await emitted if inspect.isawaitable(emitted) else emitted
        transcript += text
        result.turns.append((before, text))

        parsed = parse_assistant_text(text)
        result.n_malformed_calls += len(parsed.malformed)
        if parsed.reasoning:
            assistant_parts.append(parsed.reasoning)
        if parsed.final_text:
            assistant_parts.append(parsed.final_text)

        if not parsed.tool_calls:
            # A turn with no call is the end of the episode either way, but the
            # two endings are different failures: one answered in prose without
            # submitting, the other stalled. Worth telling apart in the logs.
            stop_reason = (
                "final_answer" if parsed.final_text is not None else "no_tool_call"
            )
            break

        terminate = False
        for call in parsed.tool_calls:
            if result.n_tool_calls >= max_tool_calls:
                stop_reason = "max_tool_calls"
                terminate = True
                break
            result.n_tool_calls += 1
            outcome = await toolkit.call(call.name, call.arguments)
            result.calls.append(outcome)
            result.tool_time_s += float(getattr(outcome, "duration_s", 0.0) or 0.0)
            if not getattr(outcome, "ok", False):
                result.n_tool_failures += 1

            reply = render_tool_result(
                call.name, outcome.as_text(), max_chars=max_chars
            )
            transcript += reply

            if _is_terminal(toolkit, call.name) and getattr(outcome, "ok", False):
                content = getattr(outcome, "content", None)
                if isinstance(content, dict):
                    submitted_smiles = str(
                        content.get("submitted_smiles", "")
                    ).strip() or None
                else:
                    # A toolkit that returns the SMILES plainly rather than in
                    # an envelope; fall back to what the policy asked for.
                    submitted_smiles = str(call.arguments.get("smiles", "")).strip() or None
                stop_reason = "submitted"
                terminate = True
                break

        if terminate:
            break
        transcript += f"{START}assistant"

    result.transcript = transcript
    result.submitted_smiles = submitted_smiles
    result.assistant_text = "\n\n".join(p for p in assistant_parts if p)
    result.stop_reason = stop_reason
    # Truncated means the policy ran out of budget rather than finishing. An
    # episode that chose to stop early without submitting is a different and
    # more interesting failure, so it is not folded in here.
    result.truncated = stop_reason in {"max_turns", "max_tool_calls"}

    if submitted_smiles:
        result.final_properties = await _measure_submission(
            oracle, submitted_smiles, task.fragment_smiles, f"{task.task_id}_final"
        )

    result.wall_time_s = round(time.monotonic() - started, 3)
    result.tool_time_s = round(result.tool_time_s, 3)
    LOG.debug(
        "%s: %s after %d turns, %d tool calls (%d failed)",
        task.task_id,
        stop_reason,
        result.n_turns,
        result.n_tool_calls,
        result.n_tool_failures,
    )
    return result
