"""The model side of a rollout: turning a transcript into the next turn.

Two things here earn their keep.

**The markers.** vLLM strips special tokens from its output by default, and
this format is made of them. Without `skip_special_tokens=False` every
completion comes back as prose with its channel structure deleted, and nothing
downstream can parse a call out of it.

**Stop strings.** Generation halts at the end of the assistant's TURN, not at
the end of each message. The distinction is the whole of it: `<|eom|>` ends a
*message*, and a turn is normally two of them -- reasoning, then the tool call.
Listing `<|eom|>` here stopped generation at the end of the reasoning block, so
the call was never emitted at all. The terminator is kept in the output,
because `atem.parse_assistant_text` reads it to decide whether the turn ended.

**Batching.** GRPO samples a group of rollouts for one task, and the episode
loop asks for one turn at a time. Left alone, eight concurrent episodes would
decode eight sequences one after another. `BatchingGenerator` collects the
concurrent requests and issues them to vLLM as a single batch, which is what
makes the group cost roughly one rollout's worth of decode instead of eight.

**The adapter.** A trained policy is a LoRA adapter, and vLLM applies one only
when a `LoRARequest` rides along with the call -- `enable_lora=True` on the
engine reserves slots and nothing more. `VLLMEngine` carries the request so
that every turn is decoded from the policy being trained rather than, silently,
from the base weights.

Everything is written against a tiny engine protocol (`generate(prompts,
params) -> list of texts`), so the whole layer is testable with a fake engine
and no GPU.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import atem
from .episode import Task

# `<|eom|>` is deliberately NOT here. It ends a message, and the SFT traces
# open every turn with a reasoning message (`experts._turn` takes the reasoning
# as a required positional), so stopping at the first one cut every turn off
# before its tool call. The result was an episode that parsed perfectly and
# said nothing: 0 calls, 0 malformed, `no_tool_call`, floored -- identically on
# every sample at temperature 1.0, because the reasoning prose varies and its
# terminator does not.
#
# It only became visible once the LoRA adapter reached generation. Before that
# the base model emitted prose that never parsed, so no rollout ever produced
# well-formed markup for the stop string to cut.
#
# A tool-call turn ends with the model emitting EOS -- an SFT completion is the
# whole turn, so that is what it was trained to do -- and vLLM halts there by
# itself. `<|eot|>` stays, because a final answer to the user ends with it.
STOP_STRINGS = (atem.END_TURN,)

TRUNCATION_NOTE = (
    f"{atem.START}user{atem.MESSAGE}"
    "[Earlier turns were dropped to fit the context window. The goals and the "
    "starting molecule above still stand.]"
    f"{atem.END_TURN}"
)


class Engine(Protocol):
    """What this module needs from vLLM, and all it needs."""

    def generate(self, prompts: Sequence[str], params: Any) -> list[str]: ...


@dataclass
class SamplingSettings:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 2048
    seed: int | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SamplingSettings:
        grpo = cfg["grpo"]
        return cls(
            temperature=float(grpo["temperature"]),
            top_p=float(grpo["top_p"]),
            top_k=int(grpo["top_k"]),
            max_tokens=int(cfg["model"]["max_completion_tokens"]),
        )


def default_token_counter() -> Callable[[str], int]:
    """A 4-characters-per-token estimate, for when no tokenizer is loaded."""
    return lambda text: max(1, len(text) // 4)


def split_history(transcript: str) -> tuple[str, list[str]]:
    """The opening prompt, then each later segment of the conversation.

    The prompt is everything up to and including the first `<|start|>assistant`,
    which carries the system turn, the goals and the starting molecule. Those
    are the parts that must never be dropped.
    """
    head = atem.START + "assistant"
    cut = transcript.find(head)
    if cut < 0:
        return transcript, []
    boundary = cut + len(head)
    prompt, rest = transcript[:boundary], transcript[boundary:]
    if not rest:
        return prompt, []
    pieces = rest.split(atem.START)
    return prompt, [atem.START + piece for piece in pieces if piece]


def trim_transcript(
    transcript: str,
    *,
    max_tokens: int,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, int]:
    """Fit a transcript to the context window by dropping its oldest middle.

    The system turn and the task stay; the oldest tool exchanges go first. The
    model is told that happened rather than being handed a history with a
    silent hole in it.
    """
    count = count_tokens or default_token_counter()
    if count(transcript) <= max_tokens:
        return transcript, 0

    prompt, segments = split_history(transcript)
    if not segments:
        return transcript, 0

    dropped = 0
    while segments and count(prompt + TRUNCATION_NOTE + "".join(segments)) > max_tokens:
        segments.pop(0)
        dropped += 1
    return prompt + TRUNCATION_NOTE + "".join(segments), dropped


class BatchingGenerator:
    """Collects concurrent one-turn requests into single engine batches."""

    def __init__(
        self,
        engine: Engine,
        params: Any,
        *,
        max_batch: int = 8,
        window_s: float = 0.02,
    ):
        self.engine = engine
        self.params = params
        self.max_batch = max_batch
        self.window_s = window_s
        self._queue: list[tuple[str, asyncio.Future[str]]] = []
        self._flushing = False
        self.batches: list[int] = []  # batch sizes, for cost reporting

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._queue.append((prompt, future))
        if not self._flushing:
            self._flushing = True
            loop.create_task(self._flush_soon())
        return await future

    async def _flush_soon(self) -> None:
        # One short wait lets the other episodes in this group enqueue their
        # turn before anything is sent to the engine.
        await asyncio.sleep(self.window_s)
        try:
            await self._drain()
        finally:
            # Cleared only once the queue is empty, and only here. Clearing it
            # before the drain let a turn arriving mid-generate start a SECOND
            # flusher, and two flushers call `engine.generate` from two worker
            # threads at once. vLLM's offline `LLM` is not thread-safe: the
            # calls cross, one returns another's completions ("engine returned
            # 5 completions for 1 prompts"), and the threads wedge inside the
            # engine -- VRAM full, GPU at nothing.
            #
            # Safe to clear here: between the `while` that ends the drain and
            # this line there is no await, so no `generate` can slip in,
            # append, see the flag still set, and be left with nobody to send
            # it.
            #
            # It took multi-turn episodes to expose. While every episode ended
            # at turn one, all of a group's turns arrived inside the same
            # 20ms window and there was only ever one batch to flush.
            self._flushing = False

    async def _drain(self) -> None:
        while self._queue:
            batch = self._queue[: self.max_batch]
            self._queue = self._queue[self.max_batch :]
            prompts = [prompt for prompt, _ in batch]
            self.batches.append(len(prompts))
            try:
                texts = await asyncio.to_thread(
                    self.engine.generate, prompts, self.params
                )
            except Exception as exc:  # noqa: BLE001 - every waiter must be told
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                continue
            if len(texts) != len(prompts):
                error = RuntimeError(
                    f"engine returned {len(texts)} completions for "
                    f"{len(prompts)} prompts"
                )
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                continue
            for (_, future), text in zip(batch, texts, strict=True):
                if not future.done():
                    future.set_result(text)


@dataclass
class Policy:
    """A `GenerateFn` that trims, generates and reports what it dropped."""

    generator: BatchingGenerator
    max_prompt_tokens: int
    count_tokens: Callable[[str], int] | None = None
    truncations: int = field(default=0, init=False)
    segments_dropped: int = field(default=0, init=False)

    async def __call__(self, transcript: str) -> str:
        fitted, dropped = trim_transcript(
            transcript,
            max_tokens=self.max_prompt_tokens,
            count_tokens=self.count_tokens,
        )
        if dropped:
            self.truncations += 1
            self.segments_dropped += dropped
        return await self.generator.generate(fitted)


def build_sampling_params(cfg: dict[str, Any]) -> Any:
    """vLLM `SamplingParams` carrying the markers and stops the loop needs.

    `skip_special_tokens=False` is the load-bearing one. vLLM defaults it to
    True, which strips exactly the tokens this format is made of --
    `<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>` are special tokens -- so
    every completion arrived as bare prose with the channel structure deleted.
    `parse_assistant_text` then had nothing to match: no segment, no recipient,
    no call. Every turn counted malformed, every episode ended at turn one with
    `no_tool_call`, and every rollout was floored at -1.0.

    It is also why the stop strings had never done anything: a stop string is
    matched against the same detokenized text, and `<|eom|>` cannot be found in
    output it has just been removed from. Generation ran to EOS instead, which
    is why turns were a consistent few hundred tokens and why correcting
    STOP_STRINGS changed nothing on its own. The two faults compound -- with
    the markers restored, `<|eom|>` matches again and would cut every turn off
    at the end of its reasoning, so both have to be right at once.
    """
    from vllm import SamplingParams

    settings = SamplingSettings.from_config(cfg)
    return SamplingParams(
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
        stop=list(STOP_STRINGS),
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )


def build_lora_request(path: Any, *, name: str = "policy", adapter_id: int = 1) -> Any:
    """A vLLM `LoRARequest` for the adapter at `path`.

    vLLM caches an adapter by its integer id, so swapping in newly trained
    weights means a NEW id -- reusing one serves the previously cached tensors
    and the engine goes on sampling the old policy.
    """
    from vllm.lora.request import LoRARequest

    return LoRARequest(str(name), int(adapter_id), str(path))


def lora_rank(cfg: dict[str, Any], default: int = 16) -> int:
    """The largest LoRA rank this config can ask the engine to serve.

    vLLM rejects an adapter whose rank exceeds `max_lora_rank`, and its own
    default is 16 -- below the 32 these packages train with. Taking the max
    across the sections means an engine started for one stage can still serve
    the other stage's adapter.
    """
    ranks = [default]
    for section in ("grpo", "sft"):
        # `or {}` twice: YAML writes None for a key with nothing under it, so
        # both `grpo:` and `grpo:\n  lora:` reach here as None. Raising an
        # AttributeError from inside engine construction on the GPU host is a
        # poor way to report a config typo.
        rank = ((cfg.get(section) or {}).get("lora") or {}).get("r")
        if rank:
            ranks.append(int(rank))
    return max(ranks)


class VLLMEngine:
    """Adapts vLLM's offline `LLM` to the `Engine` protocol.

    The adapter lives on the engine rather than being passed per call. It is a
    property of the process -- every turn of every episode in a rollout is
    sampled from the same policy -- and holding it here keeps `Policy` and
    `BatchingGenerator` unaware of it, so neither has to grow a field it would
    only ever forward.

    `enable_lora=True` on the `LLM` is NOT enough to apply an adapter: it only
    reserves the slots and kernels. Without a `lora_request` on the call, vLLM
    decodes from the base weights, silently.
    """

    def __init__(self, llm: Any, lora_request: Any = None):
        self.llm = llm
        self.lora_request = lora_request
        # vLLM's offline `LLM` is not thread-safe, and this engine is shared by
        # every episode in a group. `BatchingGenerator` already sends one batch
        # at a time, but the rule belongs to the object that has it rather than
        # to the one caller that currently respects it: entering twice does not
        # raise, it returns one call's completions to the other.
        self._lock = threading.Lock()

    def generate(self, prompts: Sequence[str], params: Any) -> list[str]:
        # Omitted rather than passed as None, so a fake engine in a test (and
        # any vLLM too old for the keyword) still sees the original signature.
        extra = {} if self.lora_request is None else {"lora_request": self.lora_request}
        with self._lock:
            outputs = self.llm.generate(list(prompts), params, **extra)
        return [output.outputs[0].text for output in outputs]


def load_engine(cfg: dict[str, Any], adapter: Any = None) -> VLLMEngine:
    """Start vLLM in this process, sharing the GPU with the trainer."""
    from vllm import LLM

    grpo = cfg["grpo"]
    llm = LLM(
        model=cfg["model"]["name"],
        gpu_memory_utilization=float(grpo["vllm_gpu_memory_utilization"]),
        max_model_len=int(cfg["model"]["max_prompt_tokens"])
        + int(cfg["model"]["max_completion_tokens"]),
        enable_lora=True,
        max_lora_rank=lora_rank(cfg),
        trust_remote_code=True,
    )
    return VLLMEngine(llm, build_lora_request(adapter) if adapter else None)


def build_policy(
    cfg: dict[str, Any],
    engine: Engine,
    params: Any,
    *,
    max_batch: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> Policy:
    generator = BatchingGenerator(
        engine,
        params,
        max_batch=max_batch or int(cfg["grpo"]["num_generations"]),
    )
    return Policy(
        generator=generator,
        max_prompt_tokens=int(cfg["model"]["max_prompt_tokens"]),
        count_tokens=count_tokens,
    )


def vllm_factory(cfg: dict[str, Any], adapter: Any = None) -> Callable[[Task], Policy]:
    """One shared engine and batcher across every task in a run."""
    engine = load_engine(cfg, adapter)
    params = build_sampling_params(cfg)
    policy = build_policy(cfg, engine, params)

    def make(_task: Task) -> Policy:
        return policy

    return make
