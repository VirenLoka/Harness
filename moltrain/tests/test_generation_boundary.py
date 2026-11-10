"""Where generation is allowed to stop, and why it is not each message.

A GRPO run on the H200 box produced this, identically, on every rollout of
every task at temperature 1.0:

    reward -1.000, 1 turns, 0 tool calls, no_tool_call,
    floored: the episode ended without submitting a molecule
    step: mean reward -1.000, 1/1 degenerate groups

The cause was `STOP_STRINGS` listing `<|eom|>`. That marker ends a *message*,
and a turn is normally two of them -- the reasoning block, then the tool call
(see `atem`'s module docstring). vLLM stops at the FIRST occurrence, so every
turn was cut off at the end of its reasoning and the call was never emitted.
The result parsed perfectly and contained nothing: zero calls, zero malformed.

It is deterministic because `experts._turn` takes the reasoning as a required
positional, so every SFT trace opens with one, and a policy at token accuracy
0.9995 reproduces that opening every time. The prose varies with temperature;
its terminator does not.

And it was invisible until the LoRA adapter reached generation: before that the
base model emitted prose that never parsed, so no rollout ever produced
well-formed markup for the stop string to cut.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import atem  # noqa: E402
from moltrain.policy import STOP_STRINGS, build_sampling_params  # noqa: E402


def generated(turn: str) -> str:
    """What vLLM returns: the prompt already ends with `<|start|>assistant`."""
    return turn.removeprefix(atem.START + "assistant")


def cut_at_first_stop(text: str) -> str:
    """vLLM halts at the first stop string and keeps it in the output."""
    hits = [(text.find(s), s) for s in STOP_STRINGS if text.find(s) >= 0]
    if not hits:
        return text
    where, marker = min(hits)
    return text[: where + len(marker)]


# --------------------------------------------------- the turn must survive


def test_a_reasoning_block_does_not_end_the_turn():
    """The regression. This is the whole bug, in one assertion."""
    turn = generated(
        atem.render_reasoning("Measure before editing.")
        + atem.render_tool_call("validate_smiles", {"smiles": "CCO"})
    )
    kept = atem.parse_assistant_text(cut_at_first_stop(turn))
    assert len(kept.tool_calls) == 1
    assert kept.tool_calls[0].name == "validate_smiles"


def test_every_call_in_a_turn_survives():
    """`_turn(reasoning, *calls)` emits as many calls as it was given."""
    turn = generated(
        atem.render_reasoning("Two things to check.")
        + atem.render_tool_call("validate_smiles", {"smiles": "CCO"})
        + atem.render_tool_call("get_molecule_metrics", {"smiles": "CCO"})
    )
    kept = atem.parse_assistant_text(cut_at_first_stop(turn))
    assert [call.name for call in kept.tool_calls] == [
        "validate_smiles",
        "get_molecule_metrics",
    ]


def test_the_message_terminator_is_not_a_stop_string():
    """Stated directly, because the comment explaining it is the fix.

    `<|eom|>` ends a message; `<|eot|>` ends the turn. Only the second one is
    a boundary the environment is waiting for.
    """
    assert atem.END_MESSAGE not in STOP_STRINGS
    assert atem.END_TURN in STOP_STRINGS


def test_a_final_answer_still_halts():
    """The environment must get the turn back when the model addresses user."""
    turn = generated(
        atem.render_reasoning("Nothing left to do.")
        + atem.render_final("The linker is already admissible.")
    )
    cut = cut_at_first_stop(turn)
    assert cut.endswith(atem.END_TURN)
    kept = atem.parse_assistant_text(cut)
    assert kept.tool_calls == []
    assert kept.final_text == "The linker is already admissible."


def test_a_turn_that_really_is_reasoning_only_still_reads_as_one():
    """The honest version of the failure, so it stays distinguishable.

    After the fix, zero calls means the policy produced none -- not that the
    harness cut them off.
    """
    kept = atem.parse_assistant_text(
        cut_at_first_stop(generated(atem.render_reasoning("Thinking.")))
    )
    assert kept.tool_calls == []
    assert kept.malformed == []
    assert kept.final_text is None


def test_a_tool_result_the_model_invented_is_ignored_not_counted():
    """Without `<|eom|>` halting it, a drifting policy may write one.

    The parser drops environment-authored segments, so the real calls still
    stand and the invention costs tokens rather than correctness.
    """
    turn = generated(
        atem.render_reasoning("Check it.")
        + atem.render_tool_call("validate_smiles", {"smiles": "CCO"})
        + atem.render_tool_result("validate_smiles", "ok")
    )
    kept = atem.parse_assistant_text(cut_at_first_stop(turn))
    assert len(kept.tool_calls) == 1
    assert kept.malformed == []


# ------------------------------------------------- and it must reach vLLM


def test_the_stop_strings_reach_the_sampling_params(monkeypatch):
    """STOP_STRINGS can be right while not being passed to anything."""
    seen: dict = {}

    module = types.ModuleType("vllm")
    module.SamplingParams = lambda **kwargs: seen.update(kwargs) or object()
    monkeypatch.setitem(sys.modules, "vllm", module)

    cfg = {
        "grpo": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "model": {"max_completion_tokens": 2048, "max_prompt_tokens": 8192},
    }
    build_sampling_params(cfg)

    assert seen["stop"] == list(STOP_STRINGS)
    assert atem.END_MESSAGE not in seen["stop"]
    # The terminator has to come back in the text: `parse_assistant_text`
    # reads it to decide whether the turn ended.
    assert seen["include_stop_str_in_output"] is True
