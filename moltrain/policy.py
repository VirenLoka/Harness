"""The model side of a rollout: turning a transcript into the next turn.

Two things here earn their keep.

**Stop strings.** Generation has to halt at `<|eom|>` or `<|eot|>` so the
environment can run the tool and append the result. The terminator is kept in
the output, because `atem.parse_assistant_text` reads it to decide whether the
turn ended.

**Batching.** GRPO samples a group of rollouts for one task, and the episode
loop asks for one turn at a time. Left alone, eight concurrent episodes would
decode eight sequences one after another. `BatchingGenerator` collects the
concurrent requests and issues them to vLLM as a single batch, which is what
makes the group cost roughly one rollout's worth of decode instead of eight.

Everything is written against a tiny engine protocol (`generate(prompts,
params) -> list of texts`), so the whole layer is testable with a fake engine
and no GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import atem
from .episode import Task

STOP_STRINGS = (atem.END_MESSAGE, atem.END_TURN)

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
        self._flushing = False
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
    """vLLM `SamplingParams` carrying the stop strings the loop depends on."""
    from vllm import SamplingParams

    settings = SamplingSettings.from_config(cfg)
    return SamplingParams(
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
        stop=list(STOP_STRINGS),
        include_stop_str_in_output=True,
    )


class VLLMEngine:
    """Adapts vLLM's offline `LLM` to the `Engine` protocol."""

    def __init__(self, llm: Any):
        self.llm = llm

    def generate(self, prompts: Sequence[str], params: Any) -> list[str]:
        outputs = self.llm.generate(list(prompts), params)
        return [output.outputs[0].text for output in outputs]


def load_engine(cfg: dict[str, Any]) -> VLLMEngine:
    """Start vLLM in this process, sharing the GPU with the trainer."""
    from vllm import LLM

    grpo = cfg["grpo"]
    llm = LLM(
        model=cfg["model"]["name"],
        gpu_memory_utilization=float(grpo["vllm_gpu_memory_utilization"]),
        max_model_len=int(cfg["model"]["max_prompt_tokens"])
        + int(cfg["model"]["max_completion_tokens"]),
        enable_lora=True,
        trust_remote_code=True,
    )
    return VLLMEngine(llm)


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


def vllm_factory(cfg: dict[str, Any]) -> Callable[[Task], Policy]:
    """One shared engine and batcher across every task in a run."""
    engine = load_engine(cfg)
    params = build_sampling_params(cfg)
    policy = build_policy(cfg, engine, params)

    def make(_task: Task) -> Policy:
        return policy

    return make
