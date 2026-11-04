"""Stage 3: GRPO on the tool-using linker episode.

A GRPO step samples a group of rollouts for the same task, scores each with
the reward in `rewards.py` plus the protocol scorecard in `protocol.py`, and
pushes the policy toward the rollouts that beat their group's mean. The
baseline is the group, so there is no value network -- and so every rollout in
a group must be for the same task, which is what `rollout_tasks` guarantees.

Four things here are worth reading before the first run.

**The mask.** A rollout interleaves model output with tool results. Only the
model's tokens may carry gradient, and `moltrain.grpo_core.build_completion`
produces that mask from exact span arithmetic. `check` reports whether the
installed TRL accepts a `completion_mask` column, because if it does not,
training would learn to emit tool output and the whole pipeline is worse than
useless.

**One engine per rank.** moltrain sets TRL's `use_vllm: True` *and* starts its
own `vllm.LLM` inside `rollout_func`: two engines in one process, four under
two-rank DDP, and neither fits. See `training_settings` -- that single line is
the most important one in this file.

**What the scalar is made of.** The number GRPO maximises is the sum of a
chemistry half and a protocol half. The chemistry half pays for closing the
distance to a band the molecule started outside; the protocol half pays for
measuring before editing, re-measuring after, verifying before submitting, and
-- the term that can go negative on its own -- stating only numbers a tool
returned. An outcome-only reward is blind to all of that, because the molecule
is unaffected by what the model said about it.

**Degenerate groups.** If every rollout in a group scores the same, the
advantage is zero and the step teaches nothing. With the two fragments frozen
and the bands narrow, this is the likeliest way the run silently learns
nothing, so `rollout` prints the count before anything is committed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moltrain import logs, progress
from moltrain.grpo_core import build_batch, degenerate_groups

from . import limits
from .config import (
    ConfigError,
    latest_adapter,
    load_config,
    log_destination,
    resolve,
)
from .episode import Task, build_prompt, run_episode
from .properties import LinkerProperties
from .protocol import ProtocolReport, score_protocol
from .rewards import RewardBreakdown, compute_reward
from .toolkit import Toolkit, build_toolkit
from .traces import RecordingPolicy

LOG = logs.get_logger("protac.grpo")


# ------------------------------------------------------------------ the ranks


@dataclass(frozen=True)
class Ranks:
    """Where this process sits in the job, read once from the environment.

    Nothing in moltrain is rank-aware -- it was written for one GPU and says
    so. Two H200s under `torchrun` means every GPU-touching decision in this
    file has to ask which process it is: the engine, the printing, and the
    slice of the corpus this process rolls out.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_env(cls) -> Ranks:
        # torchrun sets all three; accelerate sets RANK and LOCAL_RANK and
        # leaves WORLD_SIZE to the launcher config. A single-process run has
        # none of them, and the defaults are then exactly right.
        def read(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except ValueError:
                return default

        rank = read("RANK", 0)
        return cls(
            rank=rank,
            local_rank=read("LOCAL_RANK", rank),
            world_size=max(1, read("WORLD_SIZE", 1)),
        )

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def is_main_process(self) -> bool:
        return self.rank == 0

    def shard(self, items: list[Any]) -> list[Any]:
        """This rank's slice, interleaved rather than contiguous.

        Interleaving matters because `tasks.jsonl` is grouped by target, and a
        contiguous split would hand one rank a different distribution of
        failure classes than the other -- which shows up as two ranks
        disagreeing about the reward scale rather than as a data bug.
        """
        if not self.distributed:
            return list(items)
        return items[self.rank :: self.world_size]


def ranks() -> Ranks:
    return Ranks.from_env()


def is_main_process() -> bool:
    return Ranks.from_env().is_main_process()


def require_cuda(what: str) -> None:
    """Refuse, visibly and immediately, rather than hang on a host with no GPU.

    Every path below this line loads a 30B model. On the laptop this package
    is written on there is nothing to load it into, and the failure mode
    without this check is a silent wait followed by an out-of-memory trace
    from inside vLLM.

    `device_count()` and not `is_available()`, and the difference is not
    cosmetic. `is_available()` falls through to `torch._C._cuda_getDeviceCount`,
    which torch's own source describes as initialising "the CUDA Driver API via
    cuInit"; that registers a `pthread_atfork` handler, and vLLM's forked
    EngineCore then dies on its first CUDA call with "Cannot re-initialize CUDA
    in forked subprocess". This gate runs immediately before every engine
    start, so for as long as it called `is_available()` it was the thing
    breaking the engine it was guarding. `device_count()` asks NVML first and
    only falls back to the poisoning call if NVML is missing entirely.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise RuntimeError(
            f"{what} needs torch with CUDA, which is not installed here. "
            "This stage runs on the two-H200 box, not on a laptop."
        ) from exc

    if torch.cuda.device_count() == 0:
        raise RuntimeError(
            f"{what} needs a visible CUDA device and torch can see none. "
            "Run `python -m protactrain.grpo check` on the training host."
        )


def should_resume_from_sft(cfg: dict[str, Any]) -> tuple[Path, bool]:
    """Where the SFT adapter is, and whether GRPO should continue from it.

    The path returned when one is found is the adapter itself, which may be a
    checkpoint inside `sft.output_dir` rather than the directory; when none is
    found it is the configured directory, so the message that reports the
    miss names the place the user configured.
    """
    configured = resolve(cfg, cfg["sft"]["output_dir"])
    if not bool(cfg["grpo"]["resume_from_sft"]):
        return configured, False
    found = latest_adapter(configured)
    return (found or configured), found is not None


def generation_adapter(cfg: dict[str, Any]) -> Path | None:
    """The adapter vLLM must sample with, or None to mean the base model.

    This refuses rather than quietly falling back, and the reason is specific
    to this episode. The prompts in `episode.py` never spell out the ATEM call
    syntax -- `<|start|>assistant to=TOOL<|message|><atem:invoke ...>` is
    learned from the SFT traces and from nothing else. So a rollout sampled
    from the base model emits prose, every turn is counted malformed, the
    episode ends at turn one with no submission, and all of it is floored at
    -1.0. In the report that is indistinguishable from a policy that tried the
    chemistry and failed, and a whole GRPO run can be spent on it.

    `enable_lora=True` was already being passed to every engine here while no
    adapter ever was, which is exactly how that happened once.
    """
    adapter, found = should_resume_from_sft(cfg)
    if found:
        return adapter
    if bool(cfg["grpo"]["resume_from_sft"]):
        raise RuntimeError(
            f"grpo.resume_from_sft is set but no adapter was found in {adapter} "
            "(neither adapter_config.json there nor in any checkpoint-N/ under "
            "it). Sampling would fall back to the base model, which has never "
            "seen the ATEM tool-call format, so every episode would be floored "
            "at -1.0 for emitting no parseable call. Finish or stop the SFT run "
            "first, or set grpo.resume_from_sft: false to say you really do "
            "mean the base model."
        )
    LOG.warning(
        "grpo.resume_from_sft is false, so generation uses the BASE model. "
        "Nothing has taught it the ATEM call format -- expect every turn to "
        "be counted malformed."
    )
    return None


def load_engine(cfg: dict[str, Any], where: Ranks) -> Any:
    """One vLLM engine in this process, bound to this rank's device.

    The engine is started WITH the policy adapter attached. Every caller of
    this function generates rollouts that are about to be scored, so there is
    no caller for which the base model is the right answer -- see
    `generation_adapter`, which refuses instead of guessing.
    """
    require_cuda("starting a vLLM engine")

    import torch
    from vllm import LLM

    from moltrain.policy import VLLMEngine, build_lora_request, lora_rank

    # Resolved before the engine is built, so a missing adapter costs a
    # one-line refusal rather than the ninety seconds it takes to load 30B of
    # weights and capture CUDA graphs first.
    adapter = generation_adapter(cfg)

    if where.distributed:
        torch.cuda.set_device(where.local_rank)

    kwargs: dict[str, Any] = {
        "model": cfg["model"]["name"],
        "gpu_memory_utilization": float(cfg["grpo"]["vllm_gpu_memory_utilization"]),
        "max_model_len": int(cfg["model"]["max_prompt_tokens"])
        + int(cfg["model"]["max_completion_tokens"]),
        "enable_lora": True,
        # vLLM's own default is 16 and these adapters are r=32; an adapter
        # above the engine's rank is rejected at load, after the weights are
        # already resident.
        "max_lora_rank": lora_rank(cfg),
        "trust_remote_code": True,
    }
    if where.distributed:
        # `external_launcher` makes vLLM adopt the process it is already in --
        # this rank's device and this job's torch.distributed world -- instead
        # of spawning its own workers. Without it, two ranks each spawn a
        # worker that assumes it owns the node, and both land on device 0.
        kwargs["distributed_executor_backend"] = "external_launcher"

    LOG.info(
        "rank %d/%d starting vLLM on local device %d at %.2f memory utilisation, "
        "sampling from %s",
        where.rank,
        where.world_size,
        where.local_rank,
        kwargs["gpu_memory_utilization"],
        adapter if adapter else "the BASE model (no adapter)",
    )
    return VLLMEngine(
        LLM(**kwargs), build_lora_request(adapter) if adapter else None
    )


# ------------------------------------------------------------------ the scalar


@dataclass(frozen=True)
class Call:
    """One tool call, reduced to what `protocol.py` reads from it."""

    name: str
    ok: bool


def normalise_calls(result: Any) -> list[Call]:
    """The episode's call log as a flat ordered list, whatever shape it came in.

    `protocol.score_discipline` reads `.name` and `.ok` and nothing else, but
    a call log travels through this package as objects on the way out of an
    episode and as dicts on the way back off disk. Normalising once here means
    the protocol score is the same number in both directions.
    """
    raw = getattr(result, "calls", None)
    if raw is None:
        raw = getattr(result, "call_log", None)
    entries = getattr(raw, "entries", raw) or []

    calls: list[Call] = []
    for entry in entries:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("tool") or "")
            ok = bool(entry.get("ok", False))
        else:
            name = str(getattr(entry, "name", "") or getattr(entry, "tool", ""))
            ok = bool(getattr(entry, "ok", False))
        if name:
            calls.append(Call(name=name, ok=ok))
    return calls


def as_properties(value: Any) -> LinkerProperties | None:
    """A `LinkerProperties`, from one or from the dict it serialises to."""
    if value is None or isinstance(value, LinkerProperties):
        return value
    if isinstance(value, dict):
        names = {f.name for f in dataclasses.fields(LinkerProperties)}
        return LinkerProperties(**{k: v for k, v in value.items() if k in names})
    return None


def assistant_text(turns: list[tuple[str, str]]) -> str:
    """Everything the model wrote, and nothing the environment appended.

    This is the string the grounding audit runs on, so the distinction is the
    whole point: auditing the transcript would find every number in every tool
    result and score the policy as perfectly grounded no matter what it said.
    """
    return "".join(emitted for _, emitted in turns)


def synthesis_score(
    smiles: str, fragment_smiles: list[str], failure_codes: list[str]
) -> float | None:
    """`protacbench.score.score_sample`, as a dense 0-100 signal.

    The band thresholds are pass/fail cliffs; this is the continuous term that
    tells a molecule which just cleared SA 4.9 apart from one that cleared it
    by a mile. Never raises: a missing synthesis term costs the episode 0.3 of
    weight, an exception costs the whole rollout.
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
        matches = match_fragments(mol, list(fragment_smiles))
        if matches is None:
            return None
        claimed = {index for match in matches for index in match}
        indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
        if not indices:
            return None

        view = linker_view(mol, indices)
        linker = {
            "heavy_atoms": view.n_heavy_atoms,
            "attachments": view.n_attachments,
            "path_length_bonds": view.path_length_bonds,
            "exit_vector_distance": view.exit_vector_distance,
            "rotatable_bonds": view.rotatable_bonds,
            "heteroatom_fraction": view.heteroatom_fraction,
            "in_ring_fraction": view.in_ring_fraction,
            "attachment_atoms": view.attachment_atoms,
            "fragment_anchor_atoms": view.fragment_anchor_atoms,
        }
        scored = score_sample(
            mol,
            dict(molecule_properties(mol)),
            linker,
            indices,
            failure_codes=list(failure_codes),
        )
        return None if scored.total is None else float(scored.total)
    except Exception as exc:  # noqa: BLE001 - a measurement never sinks a rollout
        LOG.debug("synthesis score unavailable: %s: %s", type(exc).__name__, exc)
        return None


def score_episode(
    cfg: dict[str, Any], task: Task, result: Any, turns: list[tuple[str, str]]
) -> tuple[float, RewardBreakdown, ProtocolReport]:
    """Both halves of the scorecard, collapsed into the one scalar GRPO sees.

    The scoring lives here rather than being taken from `EpisodeResult.reward`
    because this is the number the optimiser actually follows. Deriving it in
    one place, from the episode's own parts, is what stops the trained
    objective and the reported result from drifting apart -- the one failure
    this package cannot detect from the inside.
    """
    # The episode's own measurement of the starting molecule comes first, and
    # the dataset row's baseline is only the fallback. The reward is a
    # difference between start and final, and the row's numbers came from an
    # Open Babel conversion of a 3D structure while `final` comes from
    # `grade.measure`. Mixing the two would pay the policy for the gap between
    # two instruments rather than for anything it did to the molecule.
    start = as_properties(getattr(result, "start_properties", None))
    if start is None:
        start = as_properties(getattr(task, "baseline", None))
    final = as_properties(getattr(result, "final_properties", None))
    calls = normalise_calls(result)

    if start is None:
        # Without a measured starting point every threshold is a goal of
        # unknown size, and the advantages within the group would be noise.
        raise RuntimeError(
            f"{task.task_id} has no measured baseline, so a relative reward "
            "cannot be computed. Rebuild the tasks file."
        )

    protocol = score_protocol(
        assistant_text(turns),
        calls,
        final.as_dict() if final else {},
        start.as_dict(),
    )

    synthesis: float | None = None
    if final is not None and float(cfg["reward"]["weights"].get("synthesis_score", 0)):
        synthesis = synthesis_score(
            final.smiles,
            list(getattr(task, "fragment_smiles", []) or []),
            list(getattr(task, "failure_codes", []) or []),
        )

    breakdown = compute_reward(
        cfg,
        start,
        final,
        protocol=protocol,
        calls=calls,
        reference_smiles=getattr(task, "reference_smiles", None),
        synthesis_score=synthesis,
    )
    return breakdown.reward, breakdown, protocol


# ----------------------------------------------------------------- the rollout


@dataclass
class Rollout:
    """One finished episode, with everything the batch and the report need."""

    task_id: str
    turns: list[tuple[str, str]]
    transcript: str
    reward: float
    breakdown: RewardBreakdown
    protocol: ProtocolReport
    result: Any
    calls: list[Call] = field(default_factory=list)


def rollout_tasks(tasks: list[Task], num_generations: int) -> list[Task]:
    """Each task repeated `num_generations` times, grouped and contiguous."""
    expanded: list[Task] = []
    for task in tasks:
        for generation in range(num_generations):
            copy = Task.from_dict(task.as_dict())
            copy.task_id = f"{task.task_id}@{generation}"
            expanded.append(copy)
    return expanded


async def run_group(
    cfg: dict[str, Any],
    tasks: list[Task],
    policy_factory,
    toolkit: Toolkit,
    *,
    concurrency: int | None = None,
    disable_progress: bool | None = None,
    where: Ranks | None = None,
) -> list[Rollout]:
    """Run every rollout concurrently so the generator can batch their turns.

    The bar ticks as rollouts *finish*, not as they start: they all start at
    once, which is the point -- a group's turns have to be in flight together
    for the generator to batch them into one decode.

    This loop is strictly per-process. moltrain calls `asyncio.run()` from
    inside the training step, which is fine on one GPU and is not rank-safe as
    written: an event loop is not shared between ranks, so each rank must own
    its own loop over its own shard and reach no collective from inside it.
    Everything crossing ranks happens in the trainer's own step, after this
    returns.
    """
    where = where or Ranks.from_env()
    limit = asyncio.Semaphore(concurrency or len(tasks) or 1)
    bar = progress.track(
        total=len(tasks),
        desc="rollouts",
        unit="rollout",
        # Two ranks drawing two bars over each other is unreadable, and the
        # second one is not extra information: the shards are the same size.
        disable=True if not where.is_main_process() else disable_progress,
    )

    async def one(task: Task) -> Rollout:
        async with limit:
            recorder = RecordingPolicy(policy_factory(task))
            # One toolkit per task, because `check_protac_linker` is bound to
            # that task's two fragments. Sharing one across a group would let
            # the acceptance tool answer about the wrong molecule's warhead --
            # and every reward floor hangs on that answer.
            result = await run_episode(
                cfg, task, recorder, toolkit.for_fragments(task.fragment_smiles)
            )
        reward, breakdown, protocol = score_episode(cfg, task, result, recorder.turns)
        LOG.info(
            "%s: reward %.3f (grounding %+.2f, discipline %.2f, verified %s), "
            "%d turns, %d tool calls, %s%s",
            task.task_id,
            reward,
            protocol.grounding,
            protocol.discipline,
            protocol.verified_before_submit,
            getattr(result, "n_turns", 0),
            getattr(result, "n_tool_calls", 0),
            getattr(result, "stop_reason", "?"),
            f", floored: {breakdown.floor_reason}" if breakdown.floored else "",
        )
        bar.postfix(f"last {task.task_id}")
        bar.update(1)
        return Rollout(
            task_id=task.task_id,
            turns=recorder.turns,
            transcript=result.transcript,
            reward=reward,
            breakdown=breakdown,
            protocol=protocol,
            result=result,
            calls=normalise_calls(result),
        )

    try:
        return list(await asyncio.gather(*(one(task) for task in tasks)))
    finally:
        bar.close()


def summarise(rollouts: list[Rollout], batch: Any, elapsed: float) -> dict[str, Any]:
    """One step's cost and behaviour, chemistry and protocol kept apart."""
    rewards = [r.reward for r in rollouts]
    n = max(len(rollouts), 1)

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    return {
        "rollouts": len(rollouts),
        "wall_time_s": round(elapsed, 2),
        "seconds_per_rollout": round(elapsed / n, 2),
        "reward_mean": mean(rewards),
        "reward_min": round(min(rewards), 4) if rewards else 0.0,
        "reward_max": round(max(rewards), 4) if rewards else 0.0,
        "submitted": sum(1 for r in rollouts if r.result.submitted_smiles),
        "floored": sum(1 for r in rollouts if r.breakdown.floored),
        "floor_reasons": {
            reason: sum(1 for r in rollouts if r.breakdown.floor_reason == reason)
            for reason in {
                r.breakdown.floor_reason for r in rollouts if r.breakdown.floored
            }
        },
        "recalled_reference": sum(
            1 for r in rollouts if "recalled_reference" in r.breakdown.flags
        ),
        "tool_calls": sum(getattr(r.result, "n_tool_calls", 0) for r in rollouts),
        "tool_failures": sum(
            getattr(r.result, "n_tool_failures", 0) for r in rollouts
        ),
        "malformed_calls": sum(
            getattr(r.result, "n_malformed_calls", 0) for r in rollouts
        ),
        # The protocol half, reported separately because it is the half an
        # outcome-only reward cannot see and the half this package exists for.
        "grounding_mean": mean([r.protocol.grounding for r in rollouts]),
        "discipline_mean": mean([r.protocol.discipline for r in rollouts]),
        "false_claims": sum(r.protocol.n_false_claims for r in rollouts),
        "verified_before_submit": sum(
            1 for r in rollouts if r.protocol.verified_before_submit
        ),
        "measured_before_first_edit": sum(
            1 for r in rollouts if r.protocol.measured_before_first_edit
        ),
        "remeasured_after_edit": sum(
            1 for r in rollouts if r.protocol.remeasured_after_edit
        ),
        "stop_reasons": {
            reason: sum(1 for r in rollouts if r.result.stop_reason == reason)
            for reason in {r.result.stop_reason for r in rollouts}
        },
        **batch.diagnostics,
    }


def make_rollout_func(cfg: dict[str, Any], tasks_by_id: dict[str, Task]):
    """TRL's `rollout_func`: generate, score, and return masked token columns.

    TRL passes the prompts it built from the dataset; the task behind each one
    is recovered by id, because the reward needs the frozen fragments and the
    measured baseline, not just the text.

    One engine is created here, once, for this process. Under DDP each rank
    runs its own copy of this closure against its own device, and the event
    loop below never leaves the process -- see `run_group`.
    """
    from transformers import AutoTokenizer

    from moltrain.policy import build_policy, build_sampling_params

    where = Ranks.from_env()
    toolkit = build_toolkit(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    engine = load_engine(cfg, where)
    params = build_sampling_params(cfg)
    num_generations = int(cfg["grpo"]["num_generations"])

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def rollout_func(prompts: list[str], *_args: Any, **kwargs: Any) -> dict[str, Any]:
        # The task behind each prompt has to come from the dataset column, not
        # be guessed: GRPO's baseline is the group mean, so mis-grouping the
        # rollouts silently produces meaningless advantages. Better to stop.
        ids = kwargs.get("task_id")
        if not ids:
            raise RuntimeError(
                "rollout_func received no `task_id` column, so the rollouts "
                "cannot be grouped by task and the advantages would be "
                "meaningless. The train_dataset built here carries `task_id`; "
                "check whether this TRL version forwards extra dataset "
                "columns to rollout_func."
            )
        if len(ids) != len(prompts):
            raise RuntimeError(f"{len(ids)} task ids for {len(prompts)} prompts")
        tasks = [tasks_by_id[str(task_id).split("@")[0]] for task_id in ids]
        expanded = rollout_tasks(tasks, 1)  # ids already carry the generation

        policy = build_policy(cfg, engine, params, max_batch=len(expanded) or 1)
        started = time.monotonic()
        rollouts = asyncio.run(
            run_group(cfg, expanded, lambda _t: policy, toolkit, where=where)
        )
        elapsed = time.monotonic() - started

        batch = build_batch(
            [(r.task_id, r.turns, r.transcript, r.reward) for r in rollouts],
            encode,
            num_generations,
        )
        payload = batch.as_trl_payload()
        payload["_protactrain"] = summarise(rollouts, batch, elapsed)
        if where.is_main_process():
            LOG.info(
                "step: mean reward %.3f, %d/%d degenerate groups, grounding %+.2f",
                payload["_protactrain"]["reward_mean"],
                payload["_protactrain"]["degenerate_groups"],
                max(1, len(rollouts) // num_generations),
                payload["_protactrain"]["grounding_mean"],
            )
        return payload

    # The engine is handed back on the function object because TRL takes
    # `rollout_func` as a plain callable and gives nothing back. `train` needs
    # this exact engine -- the one that will generate the next group -- to
    # re-point it at the freshly trained adapter after each step.
    rollout_func.engine = engine  # type: ignore[attr-defined]
    return rollout_func


def make_engine_sync(cfg: dict[str, Any], where: Ranks, engine: Any) -> Any:
    """A TRL callback that keeps the sampling engine on the trained policy.

    GRPO's advantage is a rollout's reward minus its group's mean, which is an
    estimate of how much better THIS policy's sample was than this policy's
    average. Draw the group from a different policy than the one being updated
    and the quantity is no longer that; the step still runs, the loss still
    falls, and none of it is the algorithm.

    TRL syncs its own engine when it owns one. This package deliberately does
    not let it own one -- `training_settings` turns `use_vllm` off so a rank
    holds a single engine rather than two -- which means the sync is this
    package's job and nothing else was doing it.

    The adapter is written to a fresh per-step directory and the previous one
    removed, rather than overwritten in place: vLLM caches an adapter by its
    integer id, so the id has to change for the new weights to be read at all,
    and a path that changes with it removes any question of which tensors were
    served.
    """
    from transformers import TrainerCallback

    from moltrain.policy import build_lora_request

    every = int(cfg["grpo"].get("sync_engine_every_steps", 1) or 0)
    # Per rank: both ranks hold their own engine and their own copy of the
    # (identical, post-allreduce) weights, and one shared path would have them
    # writing over each other mid-read.
    root = resolve(cfg, cfg["grpo"]["output_dir"]) / f"_engine_rank{where.rank}"

    class SyncAdapterToEngine(TrainerCallback):
        def __init__(self) -> None:
            self.previous: Path | None = None
            self.syncs = 0

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
            step = int(getattr(state, "global_step", 0) or 0)
            if every <= 0 or step % every:
                return control

            model = kwargs.get("model")
            # accelerate hands back a wrapped model under DDP; the adapter is
            # on the peft model underneath it.
            while model is not None and not hasattr(model, "save_pretrained"):
                model = getattr(model, "module", None)
            if model is None:
                raise RuntimeError(
                    "the engine sync callback was given no model it could save "
                    "an adapter from, so vLLM would keep sampling the policy "
                    "this run started with. Set grpo.sync_engine_every_steps: 0 "
                    "to accept that deliberately."
                )

            target = root / f"step-{step}"
            target.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(target))
            # A new id, or vLLM serves the tensors it cached under the old one.
            engine.lora_request = build_lora_request(target, adapter_id=step + 1)

            if self.previous is not None and self.previous != target:
                shutil.rmtree(self.previous, ignore_errors=True)
            self.previous = target

            self.syncs += 1
            if self.syncs == 1 and where.is_main_process():
                LOG.info(
                    "engine sync is live: the adapter is written to %s and "
                    "re-registered with vLLM every %d step(s)",
                    root,
                    every,
                )
            return control

    return SyncAdapterToEngine()


# ---------------------------------------------------------------- the training


def training_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    grpo = cfg["grpo"]

    # THE line in this file. moltrain sets use_vllm True *and* builds its own
    # engine inside rollout_func, so one process holds two vLLM engines; under
    # two-rank DDP that is four engines on two H200s and neither fits.
    # rollout_func is TRL's documented place to generate yourself, which makes
    # TRL's internal engine dead weight the moment one is supplied -- it is
    # never asked for a token, it just holds a KV cache. Off means exactly one
    # engine per rank, the one `load_engine` starts above.
    use_trl_vllm = bool(grpo.get("use_trl_vllm", False))

    settings: dict[str, Any] = {
        "output_dir": str(resolve(cfg, grpo["output_dir"])),
        "learning_rate": float(grpo["learning_rate"]),
        "num_generations": int(grpo["num_generations"]),
        "per_device_train_batch_size": int(grpo["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(grpo["gradient_accumulation_steps"]),
        "max_steps": int(grpo["max_steps"]),
        "beta": float(grpo["beta"]),
        "temperature": float(grpo["temperature"]),
        "top_p": float(grpo["top_p"]),
        "top_k": int(grpo["top_k"]),
        "max_prompt_length": int(cfg["model"]["max_prompt_tokens"]),
        "max_completion_length": int(cfg["model"]["max_completion_tokens"]),
        "use_vllm": use_trl_vllm,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": int(cfg["run"]["seed"]),
    }

    if use_trl_vllm:
        # Only meaningful when TRL owns an engine; passing them otherwise just
        # reserves memory this package has already allocated elsewhere.
        settings["vllm_mode"] = grpo.get("vllm_mode", "colocate")
        settings["vllm_gpu_memory_utilization"] = float(
            grpo["vllm_gpu_memory_utilization"]
        )

    ddp = grpo.get("ddp", {})
    if ddp.get("enabled"):
        # LoRA freezes most of the model, so DDP would otherwise warn on every
        # step about parameters that never receive a gradient. The config says
        # false because the adapter's graph is static.
        settings["ddp_find_unused_parameters"] = bool(
            ddp.get("find_unused_parameters", False)
        )
    return settings


def check(cfg: dict[str, Any]) -> int:
    from moltrain import hostcheck
    from moltrain.trl_compat import accepted_names, describe

    problems: list[str] = []
    where = Ranks.from_env()

    print("imports")
    reports, import_problems = hostcheck.report_imports(
        ("torch", "transformers", "trl", "peft", "vllm", "datasets", "rdkit")
    )
    for report in reports:
        state = report.detail if report.ok else "FAILED - " + report.detail
        print(f"  {report.name}: {state}")
    problems.extend(import_problems)

    print("\ncuda")
    cuda_lines, cuda_problems = hostcheck.report_cuda()
    for line in cuda_lines:
        print(line)
    problems.extend(cuda_problems)

    # `report_cuda` prints one line per device and does not return a count, so
    # the count is read back off its own output rather than importing torch
    # into this process -- which is the whole reason it runs in a subprocess.
    visible = sum(1 for line in cuda_lines if line.startswith("  device: "))

    # ------------------------------------------------------------ fork safety
    #
    # vLLM forks its EngineCore child. A forked child may not touch CUDA if
    # this process already holds a context, and the whole run dies on engine
    # start with "Cannot re-initialize CUDA in forked subprocess" -- a long way
    # from anything that names the real cause. `is_initialized()` only reads a
    # flag; it never creates the context it is reporting on.
    print("\nfork safety")
    print(
        "  VLLM_WORKER_MULTIPROC_METHOD="
        f"{os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', '<unset>')}  "
        "PYTORCH_NVML_BASED_CUDA_CHECK="
        f"{os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK', '<unset>')}"
    )
    try:
        import torch

        initialised = torch.cuda.is_initialized()
    except Exception as exc:  # noqa: BLE001 - a probe never sinks the check
        print(f"  could not probe torch: {type(exc).__name__}: {exc}")
    else:
        print(f"  CUDA already initialised in this process: {initialised}")
        if initialised and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            problems.append(
                "this process already holds a CUDA context and vLLM would fork "
                "its EngineCore, which cannot then touch CUDA. Set "
                "VLLM_WORKER_MULTIPROC_METHOD=spawn (protactrain/__init__.py "
                "does this by default, so an explicit value has overridden it)."
            )

    ddp = cfg["grpo"].get("ddp", {})
    wanted_world = int(ddp.get("world_size", 1))

    # ---------------------------------------------------------------- threads
    #
    # A container caps how many tasks it may hold, and every thread counts.
    # Over the cap, OpenMP cannot create its pool and the rank segfaults with
    # no usable traceback -- see limits.py.
    ranks_planned = wanted_world if bool(ddp.get("enabled")) else 1
    print("\nthreads")
    thread_lines, thread_problems = limits.report_threads(ranks_planned)
    for line in thread_lines:
        print(line)
    problems.extend(thread_problems)

    # ------------------------------------------------------------ free memory
    #
    # `report_cuda` above prints each card's TOTAL memory, which says nothing
    # about whether a run can start: the H200s are shared, and a card with
    # 139.8 GB total and 200 MiB free reads identically there. Measured on the
    # box, five processes held the whole of device 1 and the OOM arrived only
    # after the dataset had been tokenised.
    print("\nfree device memory")
    gpu_lines, gpu_problems = limits.report_gpus(ranks_planned)
    for line in gpu_lines:
        print(line)
    problems.extend(gpu_problems)

    print("\ndistributed")
    print(f"  config: enabled={bool(ddp.get('enabled'))} world_size={wanted_world} "
          f"launcher={ddp.get('launcher', 'torchrun')}")
    print(f"  environment: RANK={where.rank} LOCAL_RANK={where.local_rank} "
          f"WORLD_SIZE={where.world_size}")
    print(f"  visible CUDA devices: {visible}")
    if bool(ddp.get("enabled")) and visible != wanted_world:
        problems.append(
            f"grpo.ddp.world_size is {wanted_world} but torch can see {visible} "
            f"CUDA device(s). Launch with {wanted_world} process(es) on "
            f"{wanted_world} device(s), or set world_size to {visible}; a "
            "mismatch does not fail, it just trains on a fraction of the "
            "corpus you asked for."
        )
    engines = wanted_world if bool(ddp.get("enabled")) else 1
    trl_engines = engines if bool(cfg["grpo"].get("use_trl_vllm", False)) else 0
    print(
        f"  vLLM engines planned: {engines} from this package "
        f"+ {trl_engines} from TRL (grpo.use_trl_vllm="
        f"{bool(cfg['grpo'].get('use_trl_vllm', False))})"
    )
    if trl_engines:
        problems.append(
            "grpo.use_trl_vllm is true while rollout_func supplies its own "
            "engine, so each rank would hold two engines. Set it false."
        )

    # ------------------------------------------------------- the sampled policy
    #
    # The check that would have saved a run. `enable_lora=True` was passed to
    # every engine here while no adapter ever was, so vLLM decoded from the
    # base weights -- which have never seen the ATEM call format. Every episode
    # ended at turn one with an unparseable turn and was floored at -1.0, and
    # the report said "degenerate groups", which reads as a temperature
    # problem and is not one.
    print("\nthe sampled policy")
    adapter, found = should_resume_from_sft(cfg)
    wanted_adapter = bool(cfg["grpo"]["resume_from_sft"])
    print(f"  grpo.resume_from_sft: {wanted_adapter}")
    if found:
        print(f"  adapter: {adapter}")
        config_path = adapter / "adapter_config.json"
        try:
            adapter_r = int(json.loads(config_path.read_text()).get("r", 0))
        except (OSError, ValueError, TypeError):
            adapter_r = 0
        ceiling = int(cfg["grpo"]["lora"]["r"])
        print(f"  adapter rank: {adapter_r or 'unreadable'}  engine ceiling: {ceiling}")
        if adapter_r > ceiling:
            problems.append(
                f"the adapter at {adapter} is rank {adapter_r} but the engine "
                f"would be started with max_lora_rank={ceiling}, and vLLM "
                "rejects an adapter above its ceiling after the weights are "
                "already resident. Raise grpo.lora.r to match."
            )
    elif wanted_adapter:
        problems.append(
            f"grpo.resume_from_sft is set but no adapter was found in {adapter} "
            "(no adapter_config.json there, and none in any checkpoint-N/ under "
            "it). Generation would fall back to the base model, which has never "
            "been taught the ATEM call format, so every episode would score the "
            "floor. Finish or stop the SFT run first."
        )
    else:
        print("  BASE MODEL -- nothing has taught it the ATEM call format")

    sync = int(cfg["grpo"].get("sync_engine_every_steps", 1) or 0)
    print(
        "  engine sync: "
        + (f"every {sync} step(s)" if sync > 0 else "OFF (rollouts stay on the "
           "policy this run starts with)")
    )

    if any(r.name == "trl" and r.ok for r in reports):
        try:
            from trl import GRPOConfig, GRPOTrainer

            print()
            print(describe(GRPOConfig, training_settings(cfg)))

            trainer_args = accepted_names(GRPOTrainer.__init__)
            print("\nGRPOTrainer:")
            for name in ("rollout_func", "peft_config", "reward_funcs", "args"):
                known = not trainer_args or name in trainer_args
                print(f"  {'ok      ' if known else 'ABSENT  '} {name}")
            if trainer_args and "rollout_func" not in trainer_args:
                problems.append(
                    "this TRL's GRPOTrainer has no `rollout_func`, so it cannot "
                    "run this package's multi-turn tool loop; upgrade TRL"
                )

            # The mask decides whether tool output is trained on, so its
            # support is not a nice-to-have.
            import inspect

            source = ""
            try:
                source = inspect.getsource(GRPOTrainer)
            except (OSError, TypeError):
                pass
            if source:
                honoured = "completion_mask" in source
                print(f"\n  completion_mask referenced in GRPOTrainer: {honoured}")
                if not honoured:
                    problems.append(
                        "this TRL's GRPOTrainer never mentions `completion_mask`; "
                        "confirm environment tokens are excluded from the loss "
                        "before training, or the policy will learn to write "
                        "tool output itself"
                    )
        except Exception as exc:  # noqa: BLE001 - a check reports
            print(f"\nTRL could not be inspected: {type(exc).__name__}: {exc}")
            problems.append(f"TRL is installed but unusable: {exc}")
    else:
        print("\nTRL is not usable, so its settings cannot be checked")

    print("\ntasks")
    try:
        from .tasks import read_tasks

        tasks = read_tasks(cfg)
        print(f"  tasks: {len(tasks)}")
        measured = sum(1 for t in tasks if getattr(t, "baseline", None) is not None)
        print(f"  with a measured baseline: {measured}")
        # By the time a row is a task, `tasks.clean_codes` has already
        # removed `qed_too_low` -- it is the one code this package does not
        # score -- so a control's failure_codes list is empty and looking for
        # the code itself finds none of the 952. The class is the record.
        controls = sum(
            1
            for t in tasks
            if getattr(t, "optimisation_class", "") == "control"
        )
        print(f"  qed-only controls: {controls}")
        if not tasks:
            problems.append("no tasks; run `python -m protactrain.tasks build`")
        if measured != len(tasks):
            problems.append(
                f"{len(tasks) - measured} task(s) have no measured baseline, so "
                "their thresholds cannot be split into goals and holds"
            )
    except Exception as exc:  # noqa: BLE001 - a check reports
        print(f"  {type(exc).__name__}: {exc}")
        problems.append("tasks could not be read")

    print()
    if problems:
        for problem in problems:
            print(f"! {problem}")
        return 1
    print("ready to train")
    return 0


async def _rollout_only(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Run one step's worth of rollouts and report cost, without training.

    Worth doing before committing to a run: it measures a real GRPO step with
    the real model, and it is the only way to see the degenerate-group rate
    that the configured temperature actually produces. With the fragments
    frozen and the bands narrow, degenerate groups are the likeliest way this
    run silently learns nothing, so that number is printed last and loudest.
    """
    # Before anything imports the model stack, so a laptop gets the one-line
    # refusal rather than a ModuleNotFoundError from four frames down.
    require_cuda("a GRPO rollout")

    from transformers import AutoTokenizer

    from moltrain.policy import build_policy, build_sampling_params

    from .tasks import read_tasks

    where = Ranks.from_env()
    # Unlike `train`, nothing here owns a dataloader, so this command shards
    # the corpus itself when it is launched under torchrun.
    tasks = where.shard(read_tasks(cfg, args.tasks))[: args.tasks_per_step]
    if not tasks:
        print("no tasks to roll out", file=sys.stderr)
        return 1

    num_generations = int(cfg["grpo"]["num_generations"])
    expanded = rollout_tasks(tasks, num_generations)
    toolkit = build_toolkit(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    engine = load_engine(cfg, where)
    policy = build_policy(
        cfg,
        engine,
        build_sampling_params(cfg),
        max_batch=len(expanded),
    )

    LOG.info(
        "rank %d rolling out %d tasks x %d generations = %d episodes",
        where.rank,
        len(tasks),
        num_generations,
        len(expanded),
    )
    started = time.monotonic()
    rollouts = await run_group(
        cfg,
        expanded,
        lambda _t: policy,
        toolkit,
        disable_progress=logs.progress_disabled(cfg, args),
        where=where,
    )
    elapsed = time.monotonic() - started

    batch = build_batch(
        [(r.task_id, r.turns, r.transcript, r.reward) for r in rollouts],
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        num_generations,
    )
    summary = summarise(rollouts, batch, elapsed)
    summary["generator_batches"] = policy.generator.batches
    summary["context_truncations"] = policy.truncations
    summary["rank"] = where.rank
    # Which weights produced these numbers. A report that cannot say is how a
    # base-model run was read as a degenerate-group problem for a day: every
    # field in it was real and none of them was about the trained policy.
    request = getattr(engine, "lora_request", None)
    summary["adapter"] = (
        str(getattr(request, "lora_path", request)) if request else "base model"
    )

    if not where.is_main_process():
        # The other ranks have measured the same thing on a different shard;
        # one report is the report.
        LOG.info("rank %d finished its shard", where.rank)
        return 0

    print(json.dumps(summary, indent=2))
    out = resolve(cfg, cfg["run"]["log_dir"]) / "rollout_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nreport -> {out}")

    groups = max(1, len(rollouts) // num_generations)
    degenerate = summary.get(
        "degenerate_groups", degenerate_groups([r.reward for r in rollouts], num_generations)
    )
    print(f"\ndegenerate groups: {degenerate}/{groups}")
    if degenerate:
        print(
            f"warning: {degenerate}/{groups} groups had no reward spread and "
            "will contribute no gradient. With the fragments frozen this is "
            "the usual cause of a run that trains and learns nothing: raise "
            "the temperature, or pick tasks that are neither already inside "
            "every band nor hopeless."
        )
    if summary["false_claims"]:
        print(
            f"note: {summary['false_claims']} stated number(s) no tool returned "
            "across this step. That is the habit the grounding term exists to "
            "remove; watch it fall."
        )
    return 0


def train(cfg: dict[str, Any], *, dry_run: bool = False) -> int:
    # Even --dry-run builds the trainer, which loads the model. Refuse up
    # front rather than from four frames inside peft.
    require_cuda("GRPO training")

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from moltrain.trl_compat import adapt

    from .sft import lora_settings
    from .tasks import read_tasks

    where = Ranks.from_env()
    tasks = read_tasks(cfg)
    if not tasks:
        print("no tasks to train on", file=sys.stderr)
        return 1

    from datasets import Dataset

    # NOT sharded here, deliberately. Under DDP the trainer's own dataloader
    # is wrapped by accelerate, which already hands each rank every
    # world_size-th example; sharding again would hide half the corpus from
    # the run and the loss curve would look entirely normal while it happened.
    # `Ranks.shard` is for the places that own no dataloader -- see `rollout`.
    dataset = Dataset.from_list(
        [{"prompt": build_prompt(cfg, task), "task_id": task.task_id} for task in tasks]
    )

    wanted = training_settings(cfg)
    settings, dropped = adapt(GRPOConfig, wanted)
    if dropped and where.is_main_process():
        print(f"note: this TRL does not accept {dropped}; proceeding without them")
    args = GRPOConfig(**settings)
    peft_config = LoraConfig(**lora_settings(cfg, "grpo"))

    model: Any = cfg["model"]["name"]

    # Whether to resume is decided on EVERY rank, not just the main one. It
    # used to be computed inside an `is_main_process()` block, which meant the
    # answer was never even looked at on rank 1 -- and two ranks averaging
    # gradients for two different models is a failure that shows up as a
    # slightly odd loss curve and nothing else.
    adapter, resume = should_resume_from_sft(cfg)
    if cfg["grpo"]["resume_from_sft"] and not resume and where.is_main_process():
        print(
            f"note: grpo.resume_from_sft is set but {adapter} holds no "
            "adapter_config.json, so GRPO starts from the base model and the "
            "SFT warmup is not used"
        )

    if where.is_main_process():
        print(f"model   : {model}")
        print(
            "start   : "
            + (f"SFT adapter at {adapter}" if resume else "base model, fresh LoRA")
        )
        print(f"tasks   : {len(tasks)}  x {args.num_generations} generations")
        print(f"steps   : {args.max_steps}")
        print(f"ranks   : {where.world_size}  (one vLLM engine each)")
        print(f"trl vllm: {settings.get('use_vllm')}")
        print(f"output  : {args.output_dir}")
    if dry_run:
        if where.is_main_process():
            print("dry run: not starting the trainer")
        return 0

    if resume:
        # Continue training the warmed-up adapter instead of stacking a second,
        # randomly initialised one on top of it. `peft_config` has to become
        # None here: handing GRPOTrainer both a PeftModel and a LoraConfig
        # wraps the adapter in another adapter, and SFT's work is buried under
        # a fresh layer rather than continued.
        #
        # Until this existed, `resume_from_sft: true` printed that it was
        # starting from the adapter and then passed the bare model name to the
        # trainer, so stage 2 was discarded in full every run.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        LOG.info("loading the SFT adapter from %s", adapter)
        try:
            base = AutoModelForCausalLM.from_pretrained(
                cfg["model"]["name"], dtype="auto", trust_remote_code=True
            )
        except TypeError:
            # transformers renamed `torch_dtype` to `dtype`; accept either.
            base = AutoModelForCausalLM.from_pretrained(
                cfg["model"]["name"], torch_dtype="auto", trust_remote_code=True
            )
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        peft_config = None

    rollout_func = make_rollout_func(cfg, {t.task_id: t for t in tasks})

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=rollout_func,
    )
    # Without this the engine started above samples the policy this run began
    # with, for every one of `max_steps` steps, while the gradient is applied
    # to an adapter that never generated a token.
    if int(cfg["grpo"].get("sync_engine_every_steps", 1) or 0) > 0:
        trainer.add_callback(make_engine_sync(cfg, where, rollout_func.engine))
    elif where.is_main_process():
        print(
            "note: grpo.sync_engine_every_steps is 0, so every rollout in this "
            "run is sampled from the policy it started with"
        )
    LOG.info("starting GRPO; TRL reports its own progress from here")
    trainer.train()
    LOG.info("GRPO finished; saving the adapter")
    trainer.save_model(args.output_dir)
    if where.is_main_process():
        Path(args.output_dir, "protactrain_grpo.json").write_text(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "dropped_settings": dropped,
                    "model": model,
                    "world_size": where.world_size,
                    "trl_vllm": settings.get("use_vllm"),
                },
                indent=2,
            )
        )
        print(f"adapter saved to {args.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.grpo",
        description="GRPO on the tool-using PROTAC linker episode.",
    )
    parser.add_argument("command", choices=["check", "rollout", "train"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument(
        "--tasks-per-step",
        type=int,
        default=1,
        help="tasks to roll out; 1 measures exactly one real GRPO step",
    )
    parser.add_argument("--dry-run", action="store_true")
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    where = Ranks.from_env()
    # One log file per rank: both processes writing one file interleaves
    # their lines, and the two shards are not the same work.
    stem = f"grpo-{args.command}"
    args.log_file = log_destination(
        cfg, args, stem, suffix=f"-rank{where.rank}" if where.distributed else ""
    )
    logs.configure_from_args(
        cfg,
        args,
        stem=stem if not where.distributed else f"{stem}-rank{where.rank}",
    )

    try:
        if args.command == "check":
            return check(cfg)
        if args.command == "rollout":
            return asyncio.run(_rollout_only(cfg, args))
        return train(cfg, dry_run=args.dry_run)
    except RuntimeError as exc:
        # `require_cuda` and the engine refuse this way. One line is the
        # right amount of noise for "you are on the wrong machine".
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
