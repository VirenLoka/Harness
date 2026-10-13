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

from collections.abc import Callable, Sequence
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
        """The column layout TRL's `rollout_func` is expected to return.

        `completion_mask` is the column that keeps environment tokens out of
        the loss. Confirm the installed TRL honours it before training -- see
        `python -m moltrain.grpo check`.
        """
        return {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
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
