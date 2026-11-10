"""Turning finished episodes into GRPO training tensors.

The delicate part is the mask. A multi-turn tool-using rollout is a single
token sequence that alternates between what the model wrote and what the
environment appended, and only the model's own tokens may carry gradient. Get
this wrong and the policy is trained to produce tool output itself, which is
how a tool-using model learns to state measurements it never made.

So the mask is built from exact string arithmetic over the recorded turns
rather than by pattern-matching the finished transcript. Because the transcript
only ever grows, the environment's contribution between two turns is just the
slice between them, and that is checked rather than assumed.

Nothing here imports torch at module level, so the masking logic is testable on
any machine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

Encoder = Callable[[str], list[int]]


class RolloutError(Exception):
    """A rollout that cannot be turned into a training example."""


@dataclass
class Span:
    """One contiguous stretch of the completion, and who wrote it."""

    text: str
    from_model: bool


def completion_spans(turns: Sequence[tuple[str, str]], transcript: str) -> list[Span]:
    """Split everything after the opening prompt into authored stretches.

    `turns` is the (transcript_before, emitted_text) pairs recorded at the
    policy boundary, and `transcript` is the episode's final transcript.
    """
    if not turns:
        return []

    spans: list[Span] = []
    for index, (before, emitted) in enumerate(turns):
        consumed = len(before) + len(emitted)
        following = turns[index + 1][0] if index + 1 < len(turns) else transcript

        if not following.startswith(before + emitted):
            raise RolloutError(
                f"turn {index} is not a prefix of what followed it; the "
                "transcript was rewritten rather than appended to"
            )
        spans.append(Span(emitted, True))
        environment = following[consumed:]
        if environment:
            spans.append(Span(environment, False))
    return spans


def build_completion(
    turns: Sequence[tuple[str, str]],
    transcript: str,
    encode: Encoder,
) -> tuple[list[int], list[int]]:
    """Completion token ids and a mask that is 1 only on the model's tokens.

    Spans are encoded one at a time. That keeps the mask exact at the cost of
    the occasional boundary token differing from encoding the whole string at
    once, which is the right trade: a mask that is off by one token trains the
    model on the first character of a tool result.
    """
    ids: list[int] = []
    mask: list[int] = []
    for span in completion_spans(turns, transcript):
        tokens = encode(span.text)
        ids.extend(tokens)
        mask.extend([1 if span.from_model else 0] * len(tokens))
    if len(ids) != len(mask):
        raise RolloutError("mask and ids diverged while encoding")
    return ids, mask


def prompt_text(turns: Sequence[tuple[str, str]]) -> str:
    """The opening prompt, which is the transcript before the first turn."""
    if not turns:
        raise RolloutError("a rollout with no turns has no prompt")
    return turns[0][0]


def group_advantages(
    rewards: Sequence[float],
    num_generations: int,
    *,
    scale_by_std: bool = True,
    epsilon: float = 1e-4,
) -> list[float]:
    """GRPO's advantage: a reward's standing within its own group.

    There is no value network; the baseline is the group mean, which is why the
    rollouts in a group have to be for the same task. A group whose rewards are
    all equal yields zero advantage, so it contributes no gradient -- worth
    watching, because it means the task is either solved or impossible.
    """
    if num_generations <= 0:
        raise RolloutError("num_generations must be positive")
    if len(rewards) % num_generations != 0:
        raise RolloutError(
            f"{len(rewards)} rewards do not divide into groups of {num_generations}"
        )

    advantages: list[float] = []
    for start in range(0, len(rewards), num_generations):
        group = [float(value) for value in rewards[start : start + num_generations]]
        mean = sum(group) / len(group)
        centred = [value - mean for value in group]
        if scale_by_std and len(group) > 1:
            variance = sum(value * value for value in centred) / len(group)
            deviation = variance**0.5
            centred = [value / (deviation + epsilon) for value in centred]
        advantages.extend(centred)
    return advantages


def degenerate_groups(rewards: Sequence[float], num_generations: int) -> int:
    """How many groups gave identical rewards, and so no learning signal."""
    count = 0
    for start in range(0, len(rewards), num_generations):
        group = rewards[start : start + num_generations]
        if group and max(group) - min(group) < 1e-9:
            count += 1
    return count


def index_by_prompt(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Map each prompt back to the task it was built from.

    TRL 1.17 calls `rollout_func(prompts, trainer)` -- its own type is
    `Callable[[list[str], GRPOTrainer], dict]` -- so a `task_id` column on the
    train_dataset never reaches the rollout at all. The prompt text is built
    from the task and is the only thing that does arrive, so it is what the
    task is recovered from.

    A collision is refused rather than resolved. Two tasks rendering to one
    prompt would put their rollouts in a single group, and GRPO's baseline is
    the group mean -- so every advantage in that group would be measured
    against the wrong set, quietly.
    """
    by_prompt: dict[str, str] = {}
    for prompt, task_id in pairs:
        existing = by_prompt.setdefault(prompt, task_id)
        if existing != task_id:
            raise RolloutError(
                f"tasks {existing!r} and {task_id!r} build the same prompt, so "
                "a rollout cannot be attributed to either. GRPO's baseline is "
                "the group mean, so grouping them together would make every "
                "advantage in that group meaningless."
            )
    return by_prompt


def recover_task_ids(prompts: Sequence[str], by_prompt: dict[str, str]) -> list[str]:
    """The task id behind each prompt TRL handed to the rollout."""
    ids = [by_prompt.get(prompt) for prompt in prompts]
    missing = sum(1 for task_id in ids if task_id is None)
    if missing:
        raise RolloutError(
            f"{missing} of {len(prompts)} prompts matched no task. The prompt "
            "is how a rollout is attributed to its task now that TRL forwards "
            "no dataset columns, so a prompt altered on the way through cannot "
            "be scored against the right baseline."
        )
    return [str(task_id) for task_id in ids]


@dataclass
class RolloutBatch:
    """One GRPO step's worth of rollouts, ready to hand to a trainer."""

    prompts: list[str]
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    rewards: list[float]
    advantages: list[float]
    task_ids: list[str]
    diagnostics: dict[str, Any]

    def as_trl_payload(self) -> dict[str, Any]:
        """The column layout TRL's `rollout_func` must return.

        TRL 1.17 requires `prompt_ids`, `completion_ids` and `logprobs` and
        refuses the rollout outright without all three. Every other key is an
        "extra field", merged into the rows the reward functions are given --
        which is how `reward` reaches `episode_reward`.

        `env_mask` is the column that keeps environment tokens out of the
        loss: 1 marks a model token, 0 marks tool output, the same convention
        `completion_mask` has always used here. TRL pops `env_mask` out of the
        extra fields and treats it as its tool mask. `completion_mask` is sent
        as well because it was the name when this was written; whichever one
        the installed TRL reads, the other costs one ignored column. Sending
        neither means the policy is trained on text it did not write.

        `logprobs` is None rather than zeros, because None is the truth. The
        value is the SAMPLING policy's per-token logprobs, and the completion
        here is re-encoded from an assembled multi-turn transcript, so vLLM's
        token boundaries do not line up with these ids -- there is nothing
        honest to put in it. TRL only reads it for the vLLM importance-sampling
        correction and for off-policy masking, and None makes both behave as
        though no sampling logprobs existed, which is the case. Zeros would
        instead claim every token was sampled with probability 1.
        """
        return {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "logprobs": None,
            "env_mask": self.completion_mask,
            "completion_mask": self.completion_mask,
            "reward": self.rewards,
            "task_id": self.task_ids,
        }


def build_batch(
    episodes: Sequence[tuple[str, Sequence[tuple[str, str]], str, float]],
    encode: Encoder,
    num_generations: int,
) -> RolloutBatch:
    """Assemble a batch from (task_id, turns, transcript, reward) tuples."""
    prompts: list[str] = []
    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    completion_mask: list[list[int]] = []
    rewards: list[float] = []
    task_ids: list[str] = []

    for task_id, turns, transcript, reward in episodes:
        prompt = prompt_text(turns)
        ids, mask = build_completion(turns, transcript, encode)
        prompts.append(prompt)
        prompt_ids.append(encode(prompt))
        completion_ids.append(ids)
        completion_mask.append(mask)
        rewards.append(float(reward))
        task_ids.append(task_id)

    advantages = group_advantages(rewards, num_generations)
    model_tokens = sum(sum(mask) for mask in completion_mask)
    total_tokens = sum(len(mask) for mask in completion_mask)
    return RolloutBatch(
        prompts=prompts,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rewards=rewards,
        advantages=advantages,
        task_ids=task_ids,
        diagnostics={
            "rollouts": len(rewards),
            "degenerate_groups": degenerate_groups(rewards, num_generations),
            "model_tokens": model_tokens,
            "environment_tokens": total_tokens - model_tokens,
            "trained_token_fraction": (
                round(model_tokens / total_tokens, 4) if total_tokens else 0.0
            ),
            "mean_reward": (round(sum(rewards) / len(rewards), 4) if rewards else 0.0),
        },
    )
