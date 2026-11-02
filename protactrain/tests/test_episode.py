"""The whole episode loop, driven by a scripted policy.

No model and no GPU: `run_episode` takes a `generate` callable, so a plain
function standing in for the policy exercises the environment end to end --
the prompt, the ATEM parsing, the real molbench tools, the terminal submit and
the two measurements. That is the property that makes this package verifiable
on a laptop, so it is worth a test of its own.

The tools are the real ones. Stubbing them would leave the thing most likely
to break -- the shape of what `check_protac_linker` hands back to the loop --
untested.

Turns are built with `atem.render_*` rather than written out as literal
markup, so a change to the wire format breaks these tests at the format
rather than leaving them asserting against a string the parser no longer
produces.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import atem  # noqa: E402
from moltrain.grpo_core import build_completion, completion_spans  # noqa: E402

from protactrain.config import load_config  # noqa: E402
from protactrain.episode import (  # noqa: E402
    LinkerTask,
    Task,
    build_prompt,
    describe_goals,
    run_episode,
)
from protactrain.protocol import score_protocol  # noqa: E402
from protactrain.tests import fixtures  # noqa: E402
from protactrain.toolkit import build_toolkit  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    config = load_config()
    config["reward"]["property_cache_path"] = "cache/test_linker_properties.jsonl"
    return config


def _task() -> LinkerTask:
    return LinkerTask(
        task_id=fixtures.TASK_102_0,
        target_id="102",
        sample_id="0",
        protac_smiles=fixtures.PROTAC,
        fragment_smiles=list(fixtures.FRAGMENTS),
        verdict="poor",
        failure_codes=["qed_too_low"],
        optimisation_class="control",
        baseline={"span": 10, "clogp": 6.67, "sa_score": 4.94},
    )


@pytest.fixture(scope="module")
def task():
    return _task()


@pytest.fixture(scope="module")
def toolkit(cfg, task):
    # Bound to this task's two fragments, the way every driver binds it:
    # an unbound toolkit cannot answer what `check_protac_linker` is for.
    return build_toolkit(cfg, task.fragment_smiles)


def _scripted(*turns: str):
    """A policy that reads from a fixed list, one turn per call."""
    remaining = list(turns)

    def generate(_transcript: str) -> str:
        return remaining.pop(0) if remaining else atem.render_final("done")

    return generate


def _measure_turn(smiles: str) -> str:
    return atem.render_reasoning(
        "Measure the starting molecule before changing anything."
    ) + atem.render_tool_call("check_protac_linker", {"smiles": smiles})


def _submit_turn(smiles: str, rationale: str) -> str:
    return atem.render_tool_call(
        "submit_molecule", {"smiles": smiles, "rationale": rationale}
    )


# ----------------------------------------------------------------- the prompt


def test_the_prompt_names_the_frozen_fragments_and_the_bands(cfg, task):
    prompt = build_prompt(cfg, task)
    for fragment in fixtures.FRAGMENTS:
        assert fragment in prompt
    assert fixtures.PROTAC in prompt
    assert "submit_molecule" in prompt
    # The six configured tools, and nothing that docks.
    for tool in cfg["episode"]["tools"]:
        assert tool in prompt
    assert "dock" not in prompt.lower()


def test_the_prompt_does_not_ask_the_policy_to_improve_qed(cfg, task):
    """952 control tasks exist precisely because QED is not a target here."""
    prompt = build_prompt(cfg, task).lower()
    assert "qed" in prompt  # it is mentioned, to say not to chase it
    assert "do not optimis" in prompt or "do not optimiz" in prompt


def test_goals_are_described_without_a_baseline_rather_than_raising(cfg):
    bare = _task()
    bare.baseline = {}
    assert "unknown" in describe_goals(cfg, bare)


# ------------------------------------------------------------- the whole loop


def test_a_scripted_episode_measures_edits_and_submits(cfg, task, toolkit):
    """Measure, change one thing, re-measure, verify, submit."""
    script = _scripted(
        _measure_turn(fixtures.PROTAC),
        _measure_turn(fixtures.PROTAC_PEG),
        _submit_turn(fixtures.PROTAC_PEG, "swapped the diamide bridge for PEG3"),
    )
    result = asyncio.run(run_episode(cfg, task, script, toolkit))

    assert result.stop_reason == "submitted"
    assert result.truncated is False
    assert result.n_malformed_calls == 0
    assert result.n_tool_failures == 0
    assert result.n_tool_calls == 3
    assert result.submitted_smiles == fixtures.PROTAC_PEG

    # Both ends measured, by this package's own instrument.
    assert result.start_properties is not None
    assert result.start_properties.admissible
    assert result.final_properties is not None
    assert result.final_properties.admissible
    assert result.final_properties.span == 11


def test_the_episode_measures_the_start_rather_than_trusting_the_row(cfg, toolkit):
    """The row's numbers came from an Open Babel conversion of a 3D
    structure and `final` comes from grade.measure. Scoring a difference
    between the two pays the policy for the gap between two instruments."""
    lying = _task()
    lying.baseline = {"span": 999, "clogp": -5.0}
    result = asyncio.run(
        run_episode(cfg, lying, _scripted(_submit_turn(fixtures.PROTAC_PEG, "x")), toolkit)
    )
    assert result.start_properties.span == 10  # measured, not 999


def test_the_call_log_feeds_score_protocol_unchanged(cfg, task, toolkit):
    """`EpisodeResult.calls` holds the toolkit's own result objects, which is
    exactly what protocol.py reads `.name` and `.ok` off."""
    script = _scripted(
        _measure_turn(fixtures.PROTAC),
        _submit_turn(fixtures.PROTAC_PEG, "swapped the bridge"),
    )
    result = asyncio.run(run_episode(cfg, task, script, toolkit))
    report = score_protocol(
        result.assistant_text,
        result.calls,
        result.final_properties.as_dict(),
        result.start_properties.as_dict(),
    )
    assert report.verified_before_submit is True


def test_recorded_turns_satisfy_the_completion_mask_invariant(cfg, task, toolkit):
    """GRPO's loss mask is reconstructed from these pairs by string
    arithmetic: each recorded prefix must really be a prefix of the final
    transcript, or the mask silently covers the wrong tokens."""
    script = _scripted(
        _measure_turn(fixtures.PROTAC),
        _submit_turn(fixtures.PROTAC_PEG, "swapped the bridge"),
    )
    result = asyncio.run(run_episode(cfg, task, script, toolkit))

    for before, _emitted in result.turns:
        assert result.transcript.startswith(before)

    # A character-per-token stand-in for the tokenizer: this test is about
    # whether the spans reconstruct, not about how they encode.
    ids, mask = build_completion(
        result.turns, result.transcript, lambda text: [ord(c) for c in text]
    )
    assert ids
    assert len(mask) == len(ids)
    assert any(mask) and not all(mask)  # model turns masked in, tools out

    spans = completion_spans(result.turns, result.transcript)
    model_text = "".join(s.text for s in spans if s.from_model)
    assert "tool_output" not in model_text


def test_assistant_text_excludes_everything_the_environment_appended(
    cfg, task, toolkit
):
    """The grounding audit runs on this string. Auditing the transcript would
    find every number in every tool result and score the policy as perfectly
    grounded no matter what it said."""
    script = _scripted(
        _measure_turn(fixtures.PROTAC),
        _submit_turn(fixtures.PROTAC_PEG, "swapped the bridge"),
    )
    result = asyncio.run(run_episode(cfg, task, script, toolkit))
    assert "tool_output" not in result.assistant_text
    assert "Measure the starting molecule" in result.assistant_text
    assert len(result.assistant_text) < len(result.transcript)


def test_an_episode_that_answers_in_prose_submits_nothing(cfg, task, toolkit):
    result = asyncio.run(
        run_episode(cfg, task, _scripted(atem.render_final("I cannot fix this.")), toolkit)
    )
    assert result.stop_reason == "final_answer"
    assert result.submitted_smiles is None
    assert result.final_properties is None
    # Not "truncated": stopping early of its own accord is a different
    # failure from running out of budget, and the two want different fixes.
    assert result.truncated is False


def test_running_out_of_turns_is_marked_truncated(cfg, task, toolkit):
    budget = dict(cfg)
    budget["episode"] = {**cfg["episode"], "max_turns": 2}
    endless = lambda _t: _measure_turn(fixtures.PROTAC)  # noqa: E731
    result = asyncio.run(run_episode(budget, task, endless, toolkit))
    assert result.stop_reason == "max_turns"
    assert result.truncated is True
    assert result.n_turns == 2


def test_a_malformed_turn_is_counted_and_does_not_crash_the_loop(cfg, task, toolkit):
    """Never teach a format the parser rejects -- but never die on one."""
    broken = f"{atem.START}assistant to=check_protac_linker{atem.MESSAGE}" \
             f"<function_calls><invoke name=\"check_protac_linker\">" \
             f"{atem.END_MESSAGE}"
    result = asyncio.run(
        run_episode(cfg, task, _scripted(broken, atem.render_final("giving up")), toolkit)
    )
    assert result.n_malformed_calls >= 1


def test_the_toolkit_refuses_to_let_the_policy_choose_its_own_fragments(
    cfg, task, toolkit
):
    """The one move that would let a policy fake chemical correctness.

    `check_protac_linker` is bound to the task's pair, so fragments passed in
    the call are ignored and the result says so.
    """
    outcome = asyncio.run(
        toolkit.call(
            "check_protac_linker",
            {"smiles": fixtures.PROTAC, "fragment_1": "CC", "fragment_2": "CN"},
        )
    )
    assert outcome.ok
    assert outcome.content.get("fragments_intact") is True
    assert "note" in outcome.content


def test_submitting_a_deleted_warhead_still_produces_a_measurable_result(
    cfg, task, toolkit
):
    """The episode must not raise on the worst submission it can receive;
    the floor is the reward's job, not the loop's."""
    result = asyncio.run(
        run_episode(
            cfg,
            task,
            _scripted(_submit_turn(fixtures.PROTAC_FRAGMENT_DELETED, "trimmed it")),
            toolkit,
        )
    )
    assert result.stop_reason == "submitted"
    assert result.final_properties is not None
    assert result.final_properties.measurable is False
    assert result.final_properties.admissible is False


# ------------------------------------------------------------------ the task


def test_task_round_trips_through_its_own_dict(task):
    assert LinkerTask.from_dict(task.as_dict()) == task


def test_task_is_the_name_the_ported_modules_use():
    """grpo.py and traces.py were ported from moltrain, which calls it Task."""
    assert Task is LinkerTask
