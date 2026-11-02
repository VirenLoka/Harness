"""Stage 5: held-out evaluation, in the four arms OPTIMISATION.md specifies.

Training reward is not a result. It is measured on tasks the policy has seen,
with a cache warmed by the training loop, and both of those inflate it. This
stage re-measures against controls, because "mean reward 0.7" means nothing
without knowing what doing nothing scores -- and here doing nothing scores
surprisingly well, since the two fragments are frozen and most descriptors of a
~940 Da molecule barely move when only the linker changes.

`protacbench/OPTIMISATION.md`, "Measuring how far this can actually go", is the
authority for everything below. Its four arms:

  unchanged        submit the input untouched. The floor any arm must clear to
                   have done anything. Needs no GPU.
  one_shot         the prompt, no tools, one reply. How much comes from
                   chemistry knowledge alone.
  policy           full prompt, tools, iterate. The real number.
  policy_best_of_3 best of three independent runs. The ceiling, and how much of
                   the gap between arms is sampling luck rather than skill.

Two things this module does that `moltrain.evaluate` did not have to:

**It scores the attempt itself rather than trusting the rollout's reward.** The
episode machinery is used only to produce a transcript and an answer; the
measurement, the acceptance tests, the protocol scorecard and the reward are
recomputed here against a separate property cache. A reported number is then
never one measured during training, and the evaluation cannot silently inherit
a scoring bug from the loop it is meant to audit.

**It reports chemistry and protocol side by side and never folded together.**
That split is `protacbench.grade`'s whole design: a reply can be chemically
excellent and dishonest about it, or scrupulously truthful about a molecule it
failed to repair, and an average over the two hides exactly the failure this
training exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moltrain import logs, progress
from moltrain.policy import default_token_counter

from .config import ConfigError, load_config, log_destination, resolve
from .properties import LinkerProperties, measure
from .protocol import EDIT_TOOLS, MEASUREMENT_TOOLS, ProtocolReport, score_protocol
from .rewards import RewardBreakdown, compute_reward

LOG = logs.get_logger("evaluate")

# The arms, in the order OPTIMISATION.md lists them. The order is the reading
# order of the report: each arm is only interpretable against the one above it.
ARMS: tuple[str, ...] = ("unchanged", "one_shot", "policy", "policy_best_of_3")

# Arms that put a model on a GPU. Rule: nothing here may run a model pass
# without a visible CUDA device, the same refusal `moltrain.diffsbdd` makes.
GPU_ARMS: frozenset[str] = frozenset({"one_shot", "policy", "policy_best_of_3"})

# OPTIMISATION.md acceptance test 6: "SA has not worsened by more than 0.5".
# Deliberately NOT `grade.REGRESSION_TOLERANCE["sa_score"]` (0.15). That 0.15 is
# the drift the reward charges for; this 0.5 is the drift that makes a molecule
# not a candidate at all. Two different questions, two different numbers.
SA_WORSENING_LIMIT = 0.5

# The nine acceptance tests, in OPTIMISATION.md's order, with the short label
# the report prints. Tests 1-4 are rejections rather than deductions: a molecule
# failing one is not a worse candidate, it is not a candidate.
ACCEPTANCE_TESTS: tuple[tuple[str, str], ...] = (
    ("parses_single_molecule", "1 parses, one piece"),
    ("both_fragments_present", "2 fragments intact"),
    ("two_attachment_points", "3 two attachments"),
    ("no_reactive_group", "4 no reactive group"),
    ("span_in_range", "5 span 7-17"),
    ("sa_acceptable", "6 SA ok"),
    ("clogp_and_tpsa_ok", "7 cLogP/TPSA ok"),
    ("not_input_unchanged", "8 not the input"),
    ("not_the_reference", "9 not the reference"),
)


# --------------------------------------------------------------- the config


def evaluation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Training settings, pointed at a cache no training run has written.

    `moltrain.evaluate` turns docking up here. There is no docking in this
    package -- a PROTAC binds two proteins and a single rigid receptor does not
    model the ternary complex -- so the only thing to swap is the measurement
    cache, which matters for exactly the reason the docking swap did: a number
    read back from the loop being audited is not an independent number.
    """
    evaluated = copy.deepcopy(cfg)
    settings = evaluated["evaluate"]
    if settings.get("separate_cache", True):
        evaluated["reward"]["property_cache_path"] = settings["property_cache_path"]
    return evaluated


def _arm_config(cfg: dict[str, Any], arm: str) -> dict[str, Any]:
    """The per-arm overrides. Only `one_shot` needs any.

    One-shot is "the prompt, no tools, one reply", so it is expressed as a
    config rather than a second code path: strip the tool list and cap the
    episode at a single turn. The policy still has `submit_molecule`, because
    without it there is no way to answer at all and the arm would measure
    nothing but the absence of a submit channel.
    """
    if arm != "one_shot":
        return cfg
    narrowed = copy.deepcopy(cfg)
    narrowed["episode"]["tools"] = []
    narrowed["episode"]["max_turns"] = 1
    narrowed["episode"]["max_tool_calls"] = 1
    return narrowed


def require_cuda(arms: Sequence[str]) -> str:
    """Refuse a model pass on a machine with no GPU. Returns a reason, or "".

    Checked before any task is read, so the refusal costs nothing and is the
    first thing the operator sees rather than the last.
    """
    if not GPU_ARMS.intersection(arms):
        return ""
    try:
        import torch

        if torch.cuda.is_available():
            return ""
        detail = "torch reports no CUDA device"
    except Exception as exc:  # noqa: BLE001 - a missing torch is the same refusal
        detail = f"torch is unavailable ({type(exc).__name__}: {exc})"
    wanted = ", ".join(a for a in arms if a in GPU_ARMS)
    return (
        f"refusing to run the `{wanted}` arm(s) without a CUDA device: {detail}. "
        "Run this on the H200 box; `--arms unchanged` needs no GPU and works here."
    )


# ---------------------------------------------------- measurement, cached once


class PropertyCache:
    """A JSONL memo over `properties.measure`, keyed by molecule and fragments.

    Append-only and read whole at startup, which is the same shape
    `moltrain.oracle` uses for its docking cache and for the same reason: a run
    that dies halfway has still banked everything it measured, and a rerun pays
    only for what is new. The key carries the fragments because the identity
    checks -- the expensive and the load-bearing part -- are relative to them.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        self._entries: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            self._load(path)

    def _load(self, path: Path) -> None:
        loaded = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A truncated final line is what a killed run leaves
                        # behind. Losing one entry is cheaper than refusing to
                        # read the other ten thousand.
                        continue
                    key = row.get("key")
                    if isinstance(key, str):
                        self._entries[key] = row.get("properties", {})
                        loaded += 1
        except OSError as exc:
            LOG.warning("could not read the property cache %s: %s", path, exc)
            return
        LOG.info("property cache: %d entr(ies) from %s", loaded, path)

    @staticmethod
    def _key(smiles: str, fragments: Sequence[str]) -> str:
        return json.dumps([smiles, list(fragments)], sort_keys=True)

    def measure(self, smiles: str, fragments: Sequence[str]) -> LinkerProperties:
        key = self._key(smiles, fragments)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return LinkerProperties(**cached)

        self.misses += 1
        props = measure(smiles, list(fragments))
        payload = props.as_dict()
        self._entries[key] = payload
        self._append(key, payload)
        return props

    def _append(self, key: str, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "properties": payload}) + "\n")
        except OSError as exc:
            # An unwritable cache slows the next run down; it does not make this
            # one wrong. Warn once per failure and carry on measuring.
            LOG.warning("could not append to the property cache: %s", exc)


def synthesis_score(smiles: str, fragments: Sequence[str]) -> float | None:
    """`protacbench.score.score_sample`'s 0-100 composite, or None.

    Lazy and total: this is a reporting number, and a molecule that defeats the
    scorer should show up as a blank column rather than as a stack trace in the
    middle of a 600-episode run.
    """
    try:
        from rdkit import Chem, rdBase

        from protacbench.assess import molecule_properties
        from protacbench.repair import match_fragments
        from protacbench.score import score_sample
        from protacbench.structures import linker_view

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        matches = match_fragments(mol, [f for f in fragments if f])
        if matches is None:
            return None
        claimed = {i for match in matches for i in match}
        indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
        if not indices:
            return None
        view = linker_view(mol, indices)
        linker = {
            "heavy_atoms": view.n_heavy_atoms,
            "attachments": view.n_attachments,
            "path_length_bonds": view.path_length_bonds,
            "rotatable_bonds": view.rotatable_bonds,
            "heteroatom_fraction": view.heteroatom_fraction,
            "in_ring_fraction": view.in_ring_fraction,
        }
        scored = score_sample(mol, molecule_properties(mol), linker, indices, [])
        return None if scored.total is None else float(scored.total)
    except Exception as exc:  # noqa: BLE001 - a score is never worth an episode
        LOG.debug("synthesis score unavailable: %s: %s", type(exc).__name__, exc)
        return None


# ------------------------------------------------------- the acceptance tests


@dataclass
class Acceptance:
    """OPTIMISATION.md's nine tests, each recorded on its own.

    Individually, not as a verdict. The document asks for this explicitly and
    the reason is diagnostic: a molecule that passes eight and misses the span
    by one bond is a near-miss worth resampling, and one that fails test 2 was
    never a candidate. A single pass rate cannot tell those apart, and they ask
    for opposite responses.
    """

    parses_single_molecule: bool = False
    both_fragments_present: bool = False
    two_attachment_points: bool = False
    no_reactive_group: bool = False
    span_in_range: bool = False
    sa_acceptable: bool = False
    clogp_and_tpsa_ok: bool = False
    not_input_unchanged: bool = False
    not_the_reference: bool = True  # vacuously true when no reference is known

    reference_known: bool = False
    reactive_group_introduced: bool = False

    @property
    def all_passed(self) -> bool:
        return all(getattr(self, name) for name, _ in ACCEPTANCE_TESTS)

    @property
    def admissible(self) -> bool:
        """Tests 1-4: the rejections, as opposed to the quality gradient."""
        return (
            self.parses_single_molecule
            and self.both_fragments_present
            and self.two_attachment_points
            and self.no_reactive_group
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name) for name, _ in ACCEPTANCE_TESTS}
        out["all_passed"] = self.all_passed
        out["admissible"] = self.admissible
        out["reference_known"] = self.reference_known
        out["reactive_group_introduced"] = self.reactive_group_introduced
        return out


def check_acceptance(
    start: LinkerProperties,
    final: LinkerProperties | None,
    reference_canonical: str | None = None,
) -> Acceptance:
    """Run all nine tests. An absent answer fails everything, by construction."""
    from protacbench.grade import MAX_CLOGP, MAX_SA, MIN_TPSA, SPAN_FLOOR, SPAN_TARGET

    result = Acceptance(reference_known=bool(reference_canonical))
    if final is None:
        result.not_the_reference = True
        return result

    result.parses_single_molecule = bool(final.measurable)
    result.both_fragments_present = bool(final.fragments_intact)
    result.two_attachment_points = final.n_attachments == 2
    result.no_reactive_group = not final.reactive_groups

    # A repair that removes one reactive group and installs another passes
    # nothing useful, and against a start that already had one, test 4 alone
    # cannot say whether the policy made things better or merely different.
    result.reactive_group_introduced = bool(
        set(final.reactive_groups or []) - set(start.reactive_groups or [])
    )

    if final.span is not None:
        result.span_in_range = SPAN_FLOOR <= int(final.span) <= int(SPAN_TARGET[1])

    if final.sa_score is not None:
        within_limit = float(final.sa_score) <= MAX_SA
        drifted = (
            float(final.sa_score) - float(start.sa_score)
            if start.sa_score is not None
            else 0.0
        )
        result.sa_acceptable = within_limit and drifted <= SA_WORSENING_LIMIT

    if final.clogp is not None and final.tpsa is not None:
        result.clogp_and_tpsa_ok = (
            float(final.clogp) <= MAX_CLOGP and float(final.tpsa) >= MIN_TPSA
        )

    result.not_input_unchanged = bool(
        final.canonical and start.canonical and final.canonical != start.canonical
    )

    if reference_canonical and final.canonical:
        result.not_the_reference = final.canonical != reference_canonical

    return result


# ------------------------------------------------------- reading the rollout


@dataclass
class CallRecord:
    """One tool call, reduced to what the protocol scorer needs of it."""

    name: str
    ok: bool = True


def call_log(result: Any) -> tuple[list[CallRecord], bool]:
    """The episode's calls in order, and whether they came from the rollout.

    `EpisodeResult.calls` is the real log. When a sibling's result object does
    not carry one, the names are recovered from the transcript instead and
    every call is assumed to have succeeded -- which flatters the tool-success
    and discipline numbers, so the report says so rather than quietly printing
    a better score than it measured.
    """
    calls = getattr(result, "calls", None)
    if calls:
        return (
            [
                CallRecord(
                    name=str(getattr(c, "name", "")), ok=bool(getattr(c, "ok", False))
                )
                for c in calls
            ],
            True,
        )

    from moltrain.atem import parse_assistant_text

    recovered: list[CallRecord] = []
    for chunk in str(getattr(result, "completion", "") or "").split("<|start|>"):
        parsed = parse_assistant_text(chunk)
        recovered.extend(CallRecord(name=c.name, ok=True) for c in parsed.tool_calls)
    return recovered, False


def assistant_text(result: Any) -> str:
    """Everything the model said, which is what the claim audit reads.

    Tool results are rendered into the same stream, so they are stripped: a
    number the policy quoted back out of a tool reply is grounded by definition
    and counting it as a claim would make grounding trivially easy to farm.
    """
    from moltrain.atem import END_MESSAGE, END_TURN, MESSAGE, START

    completion = str(getattr(result, "completion", "") or "")
    kept: list[str] = []
    for chunk in completion.split(START):
        header, _, body = chunk.partition(MESSAGE)
        if not body:
            body = chunk
        elif header.strip().startswith(("tool", "functions.")):
            continue
        text = body.replace(END_TURN, " ").replace(END_MESSAGE, " ").strip()
        if text:
            kept.append(text)
    return "\n".join(kept)


def submitted_answer(result: Any) -> str | None:
    """The molecule the episode committed to, by tool call or by final line.

    `submit_molecule` is the channel the training loop uses, but the per-task
    prompt in `repair_tasks.jsonl` asks for a trailing `REPAIRED: <smiles>` line
    and the one-shot arm has no tools at all. Reading both is not leniency: an
    arm that answered in the format it was asked for has answered, and scoring
    it as a non-submission would make the one-shot column measure the submit
    channel rather than the chemistry.
    """
    direct = getattr(result, "submitted_smiles", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    try:
        from protacbench.repair import parse_answer

        found = parse_answer(str(getattr(result, "completion", "") or ""))
    except Exception as exc:  # noqa: BLE001 - never let parsing sink an attempt
        LOG.debug("answer parse failed: %s: %s", type(exc).__name__, exc)
        return None
    return found.strip() if isinstance(found, str) and found.strip() else None


# ----------------------------------------------------------- the task records


def task_value(task: Any, *names: str) -> Any:
    """A field off a Task, whether it is an attribute or lives in `metadata`.

    The task record this package builds and the raw `repair_tasks.jsonl` row it
    came from spell some of these differently, and evaluation should not be the
    thing that breaks when one of them is renamed.
    """
    metadata = getattr(task, "metadata", None) or {}
    for name in names:
        value = getattr(task, name, None)
        if value is not None:
            return value
        if isinstance(metadata, dict) and metadata.get(name) is not None:
            return metadata[name]
    return None


def optimisation_class(task: Any) -> str:
    """Which of OPTIMISATION.md's five classes this task belongs to.

    Taken from the task record when it carries one. Otherwise recomputed with
    `protacbench.prompts.classify_row`, which is the function that assigned the
    classes in the first place -- stratifying by a locally reinvented rule would
    produce strata that do not line up with the yields the document predicts.
    """
    stated = task_value(task, "optimisation_class")
    if isinstance(stated, str) and stated:
        return stated

    codes = task_value(task, "failure_codes") or []
    if isinstance(codes, str):
        codes = [c for c in codes.split(";") if c]
    row = {
        "failure_codes": ";".join(str(c) for c in codes),
        "verdict": str(task_value(task, "verdict") or ""),
    }
    try:
        from protacbench.prompts import classify_row

        return str(classify_row(row, []))
    except Exception as exc:  # noqa: BLE001 - an unclassified task is not a crash
        LOG.debug("could not classify %s: %s", getattr(task, "task_id", "?"), exc)
        return "unclassified"


def fragments_of(task: Any) -> list[str]:
    raw = task_value(task, "fragment_smiles", "fragments") or []
    if isinstance(raw, str):
        raw = [p for p in raw.split(";") if p]
    return [str(f) for f in raw if f]


def input_smiles(task: Any) -> str:
    return str(task_value(task, "smiles", "protac_smiles") or "")


def reference_canonical(task: Any) -> str | None:
    """The published molecule, canonicalised, when the corpus carried one.

    `repair_tasks.jsonl` does not ship `reference_smiles`; the CSV it was built
    from did. So acceptance test 9 is reported as "not applicable" rather than
    as "passed" wherever the reference is unknown, which is the honest reading:
    94 of 1024 samples recover the reference exactly, so a quietly vacuous test
    9 would be the most misleading number in the report.
    """
    raw = task_value(task, "reference_smiles")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        from rdkit import Chem, rdBase

        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(raw.strip())
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:  # noqa: BLE001
        return None


def stratified_sample(
    tasks: Sequence[Any], cfg: dict[str, Any], limit_per_class: int | None = None
) -> dict[str, list[Any]]:
    """At least `evaluate.min_per_class` per class, drawn with the run seed.

    `structural`, `reactive` and `chemistry` fail in different ways and will not
    have the same yield -- OPTIMISATION.md predicts 30-50%, 60-80% and 10-30%
    respectively -- so an unstratified draw would report one number dominated by
    whichever class happens to be largest, and a near-zero chemistry yield
    (which is a finding, not a bug) would vanish into the average.

    A class with fewer tasks than the minimum is kept whole, with a warning: a
    thin stratum is worth reporting as thin, not worth dropping.
    """
    settings = cfg["evaluate"]
    minimum = int(settings["min_per_class"])
    seed = int(cfg["run"]["seed"])
    rng = random.Random(seed)

    buckets: dict[str, list[Any]] = {}
    for task in tasks:
        buckets.setdefault(optimisation_class(task), []).append(task)

    wanted = limit_per_class if limit_per_class is not None else minimum
    drawn: dict[str, list[Any]] = {}
    for name in sorted(buckets):
        pool = sorted(buckets[name], key=lambda t: str(getattr(t, "task_id", "")))
        if len(pool) <= wanted:
            if len(pool) < minimum:
                LOG.warning(
                    "class `%s` has only %d task(s), below the %d minimum",
                    name,
                    len(pool),
                    minimum,
                )
            drawn[name] = pool
        else:
            drawn[name] = rng.sample(pool, wanted)
        LOG.info("class `%s`: %d of %d task(s)", name, len(drawn[name]), len(pool))
    return drawn


# ----------------------------------------------------------- the arm policies


def unchanged_factory() -> Callable[[Any], Callable[[str], str]]:
    """A policy that submits its input untouched: the do-nothing floor.

    It is not a strawman here. With both fragments frozen, the submitted
    molecule keeps every descriptor the task started with, so it collects every
    "already satisfied" hold -- and `rewards.admissibility_floor` then floors it
    on acceptance test 8 for being the input unchanged. That interaction is
    precisely what this arm is here to put a number on.
    """

    def make(task: Any) -> Callable[[str], str]:
        from moltrain.atem import render_continuation

        smiles = input_smiles(task)

        def generate(_transcript: str) -> str:
            return render_continuation(
                "Submitting the starting molecule unchanged, as a control.",
                ("submit_molecule", {"smiles": smiles, "rationale": "control arm"}),
            )

        return generate

    return make


def build_arm_factory(cfg: dict[str, Any], arm: str) -> Callable[[Any], Any]:
    """The `generate` factory for one arm."""
    if arm == "unchanged":
        return unchanged_factory()
    from .policy import vllm_factory

    return vllm_factory(cfg)


def attempts_for(cfg: dict[str, Any], arm: str) -> int:
    """How many independent runs an arm gets. Only best-of samples more than one."""
    if arm == "policy_best_of_3":
        return max(1, int(cfg["evaluate"]["attempts"]))
    return 1


# -------------------------------------------------------------- one attempt


@dataclass
class Attempt:
    """One rollout, re-measured and re-scored from the outside."""

    arm: str
    task_id: str
    optimisation_class: str
    attempt: int

    reward: float = 0.0
    floor_reason: str = ""
    flags: list[str] = field(default_factory=list)
    answered: bool = False
    answer_smiles: str | None = None
    synthesis_score: float | None = None

    acceptance: Acceptance = field(default_factory=Acceptance)
    protocol: ProtocolReport = field(default_factory=ProtocolReport)

    tool_calls: int = 0
    measurement_calls: int = 0
    edit_calls: int = 0
    failed_calls: int = 0
    call_log_recovered: bool = False

    turns: int = 0
    stop_reason: str = ""
    wall_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    terms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "task_id": self.task_id,
            "optimisation_class": self.optimisation_class,
            "attempt": self.attempt,
            "reward": round(self.reward, 4),
            "floor_reason": self.floor_reason,
            "flags": self.flags,
            "answered": self.answered,
            "answer_smiles": self.answer_smiles,
            "synthesis_score": self.synthesis_score,
            "acceptance": self.acceptance.as_dict(),
            "protocol": self.protocol.as_dict(),
            "tool_calls": self.tool_calls,
            "measurement_calls": self.measurement_calls,
            "edit_calls": self.edit_calls,
            "failed_calls": self.failed_calls,
            "call_log_recovered": self.call_log_recovered,
            "turns": self.turns,
            "stop_reason": self.stop_reason,
            "wall_time_s": round(self.wall_time_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "terms": self.terms,
        }


async def run_attempt(
    cfg: dict[str, Any],
    arm: str,
    task: Any,
    attempt_index: int,
    generate: Any,
    toolkit: Any,
    cache: PropertyCache,
    count_tokens: Callable[[str], int],
) -> Attempt:
    """Roll one episode out, then measure and score it independently."""
    started = time.monotonic()
    fragments = fragments_of(task)
    start = cache.measure(input_smiles(task), fragments)

    record = Attempt(
        arm=arm,
        task_id=str(getattr(task, "task_id", "?")),
        optimisation_class=optimisation_class(task),
        attempt=attempt_index,
    )

    try:
        result = await _invoke_episode(cfg, task, generate, toolkit)
    except Exception as exc:  # noqa: BLE001 - one dead episode is a row, not a run
        LOG.warning(
            "%s %s attempt %d failed: %s: %s",
            arm,
            record.task_id,
            attempt_index,
            type(exc).__name__,
            exc,
        )
        record.stop_reason = f"error: {type(exc).__name__}"
        record.wall_time_s = time.monotonic() - started
        record.reward = float(cfg["reward"]["no_submission_reward"])
        record.floor_reason = "the episode raised before producing an answer"
        record.acceptance = check_acceptance(start, None, reference_canonical(task))
        return record

    calls, from_rollout = call_log(result)
    record.call_log_recovered = not from_rollout
    record.tool_calls = len(calls)
    # `benchmark_transformation` is in both tables -- it validates, transforms
    # and scores in one call -- so these two counts deliberately overlap rather
    # than partitioning. An agent that only ever calls it is neither measuring
    # blind nor editing blind, and forcing it into one column would say so.
    record.measurement_calls = sum(1 for c in calls if c.name in MEASUREMENT_TOOLS)
    record.edit_calls = sum(1 for c in calls if c.name in EDIT_TOOLS)
    record.failed_calls = sum(1 for c in calls if not c.ok)
    record.turns = int(getattr(result, "n_turns", 0) or 0)
    record.stop_reason = str(getattr(result, "stop_reason", "") or "")

    completion = str(getattr(result, "completion", "") or "")
    transcript = str(getattr(result, "transcript", "") or "")
    record.completion_tokens = count_tokens(completion)
    record.prompt_tokens = max(0, count_tokens(transcript) - record.completion_tokens)

    answer = submitted_answer(result)
    record.answer_smiles = answer
    record.answered = answer is not None

    final = cache.measure(answer, fragments) if answer else None
    if final is not None and final.measurable:
        record.synthesis_score = synthesis_score(answer or "", fragments)

    record.protocol = score_protocol(
        assistant_text(result),
        calls,
        final.as_dict() if final is not None else {},
        start.as_dict(),
    )

    reference = reference_canonical(task)
    breakdown: RewardBreakdown = compute_reward(
        cfg,
        start,
        final,
        protocol=record.protocol,
        calls=calls,
        reference_smiles=task_value(task, "reference_smiles"),
        synthesis_score=record.synthesis_score,
    )
    record.reward = float(breakdown.reward)
    record.floor_reason = breakdown.floor_reason
    record.flags = list(breakdown.flags)
    record.terms = dict(breakdown.terms)
    record.acceptance = check_acceptance(start, final, reference)
    record.wall_time_s = time.monotonic() - started

    LOG.info(
        "%s %s[%d] (%s): reward %.3f, %d call(s) (%d measurement), %s",
        arm,
        record.task_id,
        attempt_index,
        record.optimisation_class,
        record.reward,
        record.tool_calls,
        record.measurement_calls,
        record.floor_reason or "admissible",
    )
    return record


async def _invoke_episode(
    cfg: dict[str, Any], task: Any, generate: Any, toolkit: Any
) -> Any:
    """Call `episode.run_episode` with whatever arguments it declares.

    The signature is bound by name rather than by position because this module
    and `episode.py` were written in parallel against one contract, and an
    evaluation harness that cannot run because a sibling grew a keyword-only
    argument is a worse failure than a slightly indirect call.
    """
    from .episode import run_episode

    available: dict[str, Any] = {
        "cfg": cfg,
        "config": cfg,
        "task": task,
        "generate": generate,
        "toolkit": toolkit,
    }
    try:
        parameters = inspect.signature(run_episode).parameters
    except (TypeError, ValueError):
        parameters = {}

    args: list[Any] = []
    for name, parameter in parameters.items():
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        if name in available:
            args.append(available[name])
        elif parameter.default is inspect.Parameter.empty:
            args.append(None)

    outcome = run_episode(*args) if args else run_episode(cfg, task, generate, toolkit)
    return await outcome if inspect.isawaitable(outcome) else outcome


# ------------------------------------------------------------- the aggregates


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _median(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


@dataclass
class CellReport:
    """One arm crossed with one class: the whole scorecard for that cell."""

    arm: str
    optimisation_class: str
    attempts: list[Attempt] = field(default_factory=list)

    def add(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def rewards(self) -> list[float]:
        return [a.reward for a in self.attempts]

    @property
    def answered(self) -> list[Attempt]:
        return [a for a in self.attempts if a.answered]

    def acceptance_rates(self) -> dict[str, float | None]:
        """Each test on its own, over every attempt including the unanswered.

        Test 9 is reported over the attempts whose task carried a reference,
        because a vacuous pass would read as "the policy never recalled the
        published molecule" when it means "nobody checked".
        """
        rates: dict[str, float | None] = {}
        for name, _ in ACCEPTANCE_TESTS:
            if name == "not_the_reference":
                judged = [a for a in self.attempts if a.acceptance.reference_known]
                rates[name] = _rate(
                    sum(1 for a in judged if a.acceptance.not_the_reference), len(judged)
                )
                continue
            rates[name] = _rate(
                sum(1 for a in self.attempts if getattr(a.acceptance, name)), self.n
            )
        return rates

    def as_dict(self) -> dict[str, Any]:
        answered = self.answered
        # Fragments lost is measured over the attempts that produced a molecule
        # at all. Folding the silent ones in would dilute exactly the number
        # OPTIMISATION.md wants reported prominently: among the answers that
        # came back, how many were a different compound.
        lost = [a for a in answered if not a.acceptance.both_fragments_present]
        introduced = [a for a in answered if a.acceptance.reactive_group_introduced]
        recalled = [a for a in self.attempts if "recalled_reference" in a.flags]
        reference_judged = [a for a in self.attempts if a.acceptance.reference_known]

        return {
            "arm": self.arm,
            "optimisation_class": self.optimisation_class,
            "attempts": self.n,
            "tasks": len({a.task_id for a in self.attempts}),
            # --- chemistry
            "reward_mean": _mean(self.rewards),
            "reward_median": _median(self.rewards),
            "reward_min": round(min(self.rewards), 4) if self.rewards else None,
            "reward_max": round(max(self.rewards), 4) if self.rewards else None,
            "answered_rate": _rate(len(answered), self.n),
            "all_tests_passed_rate": _rate(
                sum(1 for a in self.attempts if a.acceptance.all_passed), self.n
            ),
            "admissible_rate": _rate(
                sum(1 for a in self.attempts if a.acceptance.admissible), self.n
            ),
            "acceptance_rates": self.acceptance_rates(),
            "fragments_lost": len(lost),
            "fragments_lost_rate": _rate(len(lost), len(answered)),
            "reactive_group_introduced": len(introduced),
            "reactive_group_introduced_rate": _rate(len(introduced), len(answered)),
            "recalled_reference": len(recalled),
            "recalled_reference_rate": _rate(len(recalled), len(reference_judged)),
            "reference_known_for": len(reference_judged),
            "synthesis_score_mean": _mean(
                [a.synthesis_score for a in answered if a.synthesis_score is not None]
            ),
            # --- protocol, reported beside the chemistry and never inside it
            "protocol": {
                "grounding_mean": _mean([a.protocol.grounding for a in self.attempts]),
                "discipline_mean": _mean(
                    [a.protocol.discipline for a in self.attempts]
                ),
                "verified_before_submit_rate": _rate(
                    sum(1 for a in self.attempts if a.protocol.verified_before_submit),
                    self.n,
                ),
                "false_claims": sum(a.protocol.n_false_claims for a in self.attempts),
                "hedged_false_claims": sum(
                    a.protocol.n_hedged_false for a in self.attempts
                ),
                "claims": sum(a.protocol.n_claims for a in self.attempts),
                "measured_before_first_edit_rate": _rate(
                    sum(
                        1 for a in self.attempts if a.protocol.measured_before_first_edit
                    ),
                    self.n,
                ),
                "remeasured_after_edit_rate": _rate(
                    sum(1 for a in self.attempts if a.protocol.remeasured_after_edit),
                    self.n,
                ),
            },
            # --- cost
            "tool_calls_total": sum(a.tool_calls for a in self.attempts),
            "tool_calls_per_molecule": _mean([a.tool_calls for a in self.attempts]),
            "measurement_calls_per_molecule": _mean(
                [a.measurement_calls for a in self.attempts]
            ),
            "edit_calls_per_molecule": _mean([a.edit_calls for a in self.attempts]),
            "measurement_share": _rate(
                sum(a.measurement_calls for a in self.attempts),
                sum(a.tool_calls for a in self.attempts),
            ),
            "failed_call_rate": _rate(
                sum(a.failed_calls for a in self.attempts),
                sum(a.tool_calls for a in self.attempts),
            ),
            "seconds_per_molecule": _mean([a.wall_time_s for a in self.attempts]),
            "prompt_tokens_per_molecule": _mean(
                [a.prompt_tokens for a in self.attempts]
            ),
            "completion_tokens_per_molecule": _mean(
                [a.completion_tokens for a in self.attempts]
            ),
            "call_log_recovered": any(a.call_log_recovered for a in self.attempts),
        }


def collapse_best_of(attempts: Iterable[Attempt]) -> list[Attempt]:
    """One record per task: the best of that task's independent runs.

    "Best" is the reward, which already carries the admissibility floor, so a
    run that scored well by deleting a warhead cannot win a group -- it is on
    the floor with every other inadmissible answer. Ties keep the earliest
    attempt, so the arm is not quietly biased toward later samples.
    """
    best: dict[str, Attempt] = {}
    for attempt in attempts:
        incumbent = best.get(attempt.task_id)
        if incumbent is None or attempt.reward > incumbent.reward:
            best[attempt.task_id] = attempt
    return [best[key] for key in sorted(best)]


# -------------------------------------------------------------- the arm loop


async def evaluate_arm(
    cfg: dict[str, Any],
    arm: str,
    tasks: Sequence[Any],
    cache: PropertyCache,
    *,
    disable_progress: bool | None = None,
) -> list[Attempt]:
    """Every task, every attempt, for one arm."""
    from .toolkit import build_toolkit

    arm_cfg = _arm_config(cfg, arm)
    toolkit = build_toolkit(arm_cfg)
    factory = build_arm_factory(arm_cfg, arm)
    attempts = attempts_for(arm_cfg, arm)
    count_tokens = default_token_counter()

    records: list[Attempt] = []
    bar = progress.track(
        total=len(tasks) * attempts,
        desc=f"arm {arm}",
        unit="episode",
        disable=disable_progress,
    )
    for task in tasks:
        for index in range(attempts):
            bar.postfix(f"{getattr(task, 'task_id', '?')} ({index + 1}/{attempts})")
            records.append(
                await run_attempt(
                    arm_cfg,
                    arm,
                    task,
                    index,
                    factory(task),
                    # Bound to this task's fragment pair: `check_protac_linker`
                    # decides acceptance, and an unbound toolkit answers about
                    # no fragments at all.
                    toolkit.for_fragments(fragments_of(task)),
                    cache,
                    count_tokens,
                )
            )
            bar.update(1)
    bar.close()

    if arm == "policy_best_of_3":
        kept = collapse_best_of(records)
        LOG.info(
            "best-of: %d attempt(s) collapsed to %d task result(s)",
            len(records),
            len(kept),
        )
        return kept
    return records


# ----------------------------------------------------------------- the report


def _fmt(value: Any, width: int = 8, places: int = 3) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.{places}f}".rjust(width)
    return str(value).rjust(width)


def _pct(value: float | None, width: int = 7) -> str:
    return "-".rjust(width) if value is None else f"{value * 100:.1f}%".rjust(width)


def render_report(cells: Sequence[dict[str, Any]], out: Any = sys.stdout) -> None:
    """The printed scorecard. Chemistry, then protocol, then how to read it."""
    arms = []
    for cell in cells:
        if cell["arm"] not in arms:
            arms.append(cell["arm"])
    classes = sorted({cell["optimisation_class"] for cell in cells})

    def cell_of(arm: str, klass: str) -> dict[str, Any] | None:
        return next(
            (
                c
                for c in cells
                if c["arm"] == arm and c["optimisation_class"] == klass
            ),
            None,
        )

    print("\n=== CHEMISTRY: reward and acceptance, per arm, per class ===", file=out)
    print(
        f"{'arm':<17}{'class':<14}{'n':>5}{'reward':>9}{'median':>9}"
        f"{'all 9':>8}{'frag-lost':>11}{'new-rx':>8}{'recalled':>10}",
        file=out,
    )
    for arm in arms:
        for klass in classes:
            cell = cell_of(arm, klass)
            if cell is None:
                continue
            print(
                f"{arm:<17}{klass:<14}{cell['attempts']:>5}"
                f"{_fmt(cell['reward_mean'], 9)}{_fmt(cell['reward_median'], 9)}"
                f"{_pct(cell['all_tests_passed_rate'], 8)}"
                f"{_pct(cell['fragments_lost_rate'], 11)}"
                f"{_pct(cell['reactive_group_introduced_rate'], 8)}"
                f"{_pct(cell['recalled_reference_rate'], 10)}",
                file=out,
            )

    print("\n=== ACCEPTANCE TESTS, individually ===", file=out)
    print(
        "A near-miss and a molecule that was never a candidate both read as "
        "'failed' in a single pass rate. These do not.",
        file=out,
    )
    for arm in arms:
        for klass in classes:
            cell = cell_of(arm, klass)
            if cell is None:
                continue
            rates = cell["acceptance_rates"]
            print(f"{arm + ' / ' + klass:<30}", file=out)
            for name, label in ACCEPTANCE_TESTS:
                print(f"    {label:<26}{_pct(rates.get(name), 8)}", file=out)

    print("\n=== PROTOCOL: how the policy worked, never folded into the above ===",
          file=out)
    print(
        f"{'arm':<17}{'class':<14}{'ground':>9}{'disciplin':>11}"
        f"{'verified':>10}{'false':>8}{'claims':>8}{'measure%':>10}",
        file=out,
    )
    for arm in arms:
        for klass in classes:
            cell = cell_of(arm, klass)
            if cell is None:
                continue
            protocol = cell["protocol"]
            print(
                f"{arm:<17}{klass:<14}"
                f"{_fmt(protocol['grounding_mean'], 9)}"
                f"{_fmt(protocol['discipline_mean'], 11)}"
                f"{_pct(protocol['verified_before_submit_rate'], 10)}"
                f"{protocol['false_claims']:>8}{protocol['claims']:>8}"
                f"{_pct(cell['measurement_share'], 10)}",
                file=out,
            )

    print("\n=== COST per molecule ===", file=out)
    print(
        f"{'arm':<17}{'class':<14}{'calls':>8}{'measure':>9}{'edits':>8}"
        f"{'seconds':>10}{'prompt tok':>12}{'reply tok':>11}",
        file=out,
    )
    for arm in arms:
        for klass in classes:
            cell = cell_of(arm, klass)
            if cell is None:
                continue
            print(
                f"{arm:<17}{klass:<14}"
                f"{_fmt(cell['tool_calls_per_molecule'], 8, 1)}"
                f"{_fmt(cell['measurement_calls_per_molecule'], 9, 1)}"
                f"{_fmt(cell['edit_calls_per_molecule'], 8, 1)}"
                f"{_fmt(cell['seconds_per_molecule'], 10, 1)}"
                f"{_fmt(cell['prompt_tokens_per_molecule'], 12, 0)}"
                f"{_fmt(cell['completion_tokens_per_molecule'], 11, 0)}",
                file=out,
            )

    if any(cell["call_log_recovered"] for cell in cells):
        print(
            "\nNOTE: at least one arm's call log was recovered from the transcript "
            "rather than read from the rollout, so every call there is counted as "
            "successful. Tool-success and discipline are upper bounds in those rows.",
            file=out,
        )

    render_interpretation(cells, out)


def _overall(cells: Sequence[dict[str, Any]], arm: str, key: str) -> float | None:
    """One number for an arm across classes, weighted by attempts.

    Weighted rather than a mean of means, because the strata are not the same
    size once a class runs short of tasks and an unweighted average would let
    the thinnest class swing the headline. `key` may be dotted, so a protocol
    number is reachable without flattening the cell it is deliberately nested
    in.
    """
    total = 0.0
    weight = 0
    for cell in cells:
        if cell["arm"] != arm:
            continue
        value: Any = cell
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            continue
        total += float(value) * cell["attempts"]
        weight += cell["attempts"]
    return round(total / weight, 4) if weight else None


def render_interpretation(cells: Sequence[dict[str, Any]], out: Any = sys.stdout) -> None:
    """OPTIMISATION.md's "How to read it", applied to the numbers just printed.

    Printed rather than left to the reader because the three readings below all
    look like the same disappointing table and ask for opposite responses:
    rewrite the prompt, sample harder, or tighten the constraints.
    """
    print("\n=== HOW TO READ IT ===", file=out)

    arm0 = _overall(cells, "unchanged", "reward_mean")
    arm1 = _overall(cells, "one_shot", "reward_mean")
    arm2 = _overall(cells, "policy", "reward_mean")
    arm3 = _overall(cells, "policy_best_of_3", "reward_mean")

    print(
        f"  mean reward   unchanged {_fmt(arm0, 7)}   one_shot {_fmt(arm1, 7)}"
        f"   policy {_fmt(arm2, 7)}   best-of {_fmt(arm3, 7)}",
        file=out,
    )

    said = False
    if arm0 is not None and arm2 is not None:
        # "Arm 2 ~= Arm 0 -> the prompt is not working. Look at whether the
        # agent is measuring at all before blaming the model." The measurement
        # share is printed with the verdict for exactly that reason.
        if arm2 - arm0 < 0.05:
            share = _overall(cells, "policy", "measurement_share")
            print(
                "  * policy is level with the do-nothing floor: the prompt is not "
                "working. Before blaming the model, check whether it measured at "
                f"all -- {_pct(share, 6).strip()} of its calls were measurements.",
                file=out,
            )
            said = True
        else:
            print(
                f"  * policy clears the floor by {arm2 - arm0:+.3f}. The tools and "
                "the prompt are buying something.",
                file=out,
            )
            said = True

    if arm2 is not None and arm3 is not None:
        # "Arm 3 >> Arm 2 -> the model can solve these but does so unreliably.
        # Sample several times and keep the best rather than reworking it."
        if arm3 - arm2 > 0.15:
            print(
                f"  * best-of-3 beats a single run by {arm3 - arm2:+.3f}: the model "
                "can solve these but does so unreliably. Sample more and keep the "
                "best rather than reworking the prompt.",
                file=out,
            )
        else:
            print(
                f"  * best-of-3 adds only {arm3 - arm2:+.3f}: the single-run number "
                "is close to what this policy can do, so extra sampling is not "
                "where the headroom is.",
                file=out,
            )
        said = True

    if arm1 is not None and arm2 is not None:
        print(
            f"  * tools are worth {arm2 - arm1:+.3f} over one-shot chemistry "
            "knowledge alone.",
            file=out,
        )
        said = True

    # "High pass rate, high fragments-lost -> the agent is cheating and the
    # acceptance tests are earning their keep." Checked per arm, because it is
    # a property of how one arm answers rather than of the run, and only for the
    # arms a model drove: `unchanged` resubmits the input and cannot lose a
    # fragment, so a hit there would be a measurement bug, not cheating.
    seen: set[str] = set()
    for arm in [c["arm"] for c in cells if c["arm"] in GPU_ARMS]:
        if arm in seen:
            continue
        seen.add(arm)
        passed = _overall(cells, arm, "all_tests_passed_rate")
        lost = _overall(cells, arm, "fragments_lost_rate")
        if passed is not None and lost is not None and passed > 0.4 and lost > 0.1:
            print(
                f"  * `{arm}` passes {_pct(passed, 1).strip()} of the time while "
                f"losing a fragment on {_pct(lost, 1).strip()} of its answers: it is "
                "cheating, and the acceptance tests are earning their keep. "
                "Strengthen the constraint wording.",
                file=out,
            )
            said = True

    for klass in sorted({c["optimisation_class"] for c in cells}):
        policy_cell = next(
            (
                c
                for c in cells
                if c["arm"] == "policy" and c["optimisation_class"] == klass
            ),
            None,
        )
        if policy_cell is None:
            continue
        rate = policy_cell["all_tests_passed_rate"]
        if klass == "chemistry" and rate is not None and rate < 0.05:
            # OPTIMISATION.md: "If `chemistry` comes back near zero, that is a
            # finding, not a bug." SA and cLogP are whole-molecule properties
            # largely set by the frozen fragments.
            print(
                "  * the `chemistry` class is near zero. That is a finding, not a "
                "bug: those molecules need a different linker rather than a "
                "repaired one, so resample instead of optimising harder.",
                file=out,
            )
            said = True

    recalled = _overall(cells, "policy", "recalled_reference_rate")
    if recalled:
        print(
            f"  * the policy recalled the published molecule on "
            f"{_pct(recalled, 1).strip()} of the tasks where a reference was known. "
            "That is memorisation, not repair, and must not be counted as a win.",
            file=out,
        )
        said = True
    elif not any(c["reference_known_for"] for c in cells):
        print(
            "  * no task carried a `reference_smiles`, so acceptance test 9 was not "
            "run. 94 of 1024 samples recover the reference exactly, so this is a "
            "hole in the report rather than a clean result.",
            file=out,
        )

    grounding = _overall(cells, "policy", "protocol.grounding_mean")
    if grounding is not None:
        if grounding < 0.0:
            print(
                f"  * mean grounding is {grounding:+.3f}: the policy is stating more "
                "numbers than it measured. Chemistry and protocol fail "
                "independently, and this one is failing on its own.",
                file=out,
            )
        else:
            print(
                f"  * mean grounding {grounding:+.3f}, discipline "
                f"{_fmt(_overall(cells, 'policy', 'protocol.discipline_mean'), 5).strip()}, "
                "reported beside the chemistry and not inside it.",
                file=out,
            )
        said = True

    if not said:
        print("  (not enough arms were run to compare anything)", file=out)


# ---------------------------------------------------------------------- CLI


async def _run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    from .tasks import read_tasks

    refusal = require_cuda(args.arms)
    if refusal:
        print(refusal, file=sys.stderr)
        return 3

    evaluated = evaluation_config(cfg)
    tasks = read_tasks(evaluated, args.tasks)
    if not tasks:
        print("no tasks to evaluate", file=sys.stderr)
        return 1

    strata = stratified_sample(tasks, evaluated, args.per_class)
    drawn = [task for klass in sorted(strata) for task in strata[klass]]
    if not drawn:
        print("the stratified draw was empty", file=sys.stderr)
        return 1
    LOG.info("evaluating %d task(s) across %d class(es)", len(drawn), len(strata))

    cache_path = resolve(evaluated, evaluated["reward"]["property_cache_path"])
    cache = PropertyCache(cache_path)

    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for arm in args.arms:
        LOG.info("running arm `%s` over %d task(s)", arm, len(drawn))
        attempts = await evaluate_arm(
            evaluated,
            arm,
            drawn,
            cache,
            disable_progress=logs.progress_disabled(evaluated, args),
        )
        rows.extend(a.as_dict() for a in attempts)

        by_class: dict[str, CellReport] = {}
        for attempt in attempts:
            by_class.setdefault(
                attempt.optimisation_class,
                CellReport(arm=arm, optimisation_class=attempt.optimisation_class),
            ).add(attempt)
        cells.extend(by_class[k].as_dict() for k in sorted(by_class))

    out_dir = resolve(evaluated, evaluated["evaluate"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "arms": list(args.arms),
        "tasks": len(drawn),
        "classes": {k: len(v) for k, v in sorted(strata.items())},
        "seed": int(evaluated["run"]["seed"]),
        "attempts_per_task": {a: attempts_for(evaluated, a) for a in args.arms},
        "property_cache": str(cache_path),
        "property_cache_hits": cache.hits,
        "property_cache_misses": cache.misses,
        "cells": cells,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "attempts.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    render_report(cells)
    print(f"\nsummary  -> {out_dir / 'summary.json'}")
    print(f"attempts -> {out_dir / 'attempts.jsonl'}")
    print(
        f"measured {cache.misses} molecule(s), {cache.hits} cache hit(s), "
        f"in {cache_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.evaluate",
        description=(
            "Evaluate PROTAC linker repair in OPTIMISATION.md's four arms, "
            "stratified by optimisation class."
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", default=None, help="tasks.jsonl to evaluate")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["unchanged"],
        choices=list(ARMS),
        help=(
            "default is `unchanged`, which needs no GPU; the other three load "
            "the model and refuse without a CUDA device"
        ),
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="tasks per class (default: evaluate.min_per_class)",
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # `moltrain.logs.configure` resolves its default log directory against
    # moltrain's package dir, so the path is computed here instead -- otherwise
    # this package's runs would log into the sibling repository.
    log_file = args.log_file
    args.log_file = log_destination(cfg, args, "evaluate")
    logs.configure_from_args(cfg, args, stem="evaluate")

    try:
        return asyncio.run(_run(cfg, args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
