"""The reward TRL is handed, and where it comes from.

`GRPOTrainer(...)` was built with a `rollout_func` and no `reward_funcs`,
because an older TRL treated a rollout carrying its own rewards as a complete
reward source. TRL 0.33 refuses to construct the trainer at all:

    ValueError: No reward source provided. Pass `reward_funcs`, or an
    `environment_factory` whose environment defines a `get_reward` method.

The scores were never missing -- `as_trl_payload()` has returned a `reward`
column throughout. What was missing is the function that reads it back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protactrain.grpo import episode_reward  # noqa: E402


def test_the_rollouts_own_score_is_what_comes_back():
    """TRL forwards a rollout_func's extra columns to the reward functions.

    The score depends on tool results, the frozen fragments and the measured
    baseline, none of which survive into the completion text a reward
    function is handed -- so it is computed during the rollout and read back
    here rather than recomputed from nothing.
    """
    assert episode_reward(
        prompts=["p1", "p2"],
        completions=["c1", "c2"],
        reward=[0.25, -1.0],
    ) == [0.25, -1.0]


def test_the_scores_come_back_as_floats():
    assert episode_reward(reward=[1, 0]) == [1.0, 0.0]
    assert all(isinstance(value, float) for value in episode_reward(reward=[1, 0]))


def test_prompts_and_completions_may_arrive_positionally():
    """TRL has passed them both ways across versions."""
    assert episode_reward(["p"], ["c"], reward=[0.5]) == [0.5]


def test_a_missing_reward_column_refuses_rather_than_scoring_nothing():
    """The quiet version of this trains on advantages that mean nothing.

    If a TRL release stops forwarding the column, every episode's measured
    score is silently dropped and the run looks healthy throughout. Better to
    stop at the first step.
    """
    with pytest.raises(RuntimeError) as exc:
        episode_reward(prompts=["p"], completions=["c"])
    assert "reward" in str(exc.value)


def test_an_empty_group_is_not_an_error():
    assert episode_reward(reward=[]) == []


def test_the_trainer_is_given_a_reward_source():
    """Read as source: `train` calls require_cuda on its first line.

    Reaching the GRPOTrainer construction in a test would mean standing up
    peft, trl, datasets and a GPU. The regression that matters is someone
    dropping `reward_funcs` again, and reading the one function for it is
    honest about what is and is not being checked.
    """
    import inspect

    from protactrain import grpo

    source = inspect.getsource(grpo.train)
    assert "reward_funcs=" in source
    assert "episode_reward" in source
