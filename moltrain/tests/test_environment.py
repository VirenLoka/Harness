"""Environment tests: ATEM parsing, tools, reward shaping and whole episodes.

Episodes run against a scripted policy, so the full loop is exercised with no
GPU and no model. Docking tests skip when the engines are absent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import atem
from moltrain.config import ConfigError, load_config
from moltrain.episode import Task, build_prompt, run_episode
from moltrain.oracle import Oracle, Properties
from moltrain.rewards import compute_reward, score_thresholds
from moltrain.toolkit import build_toolkit

ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def cfg():
    config = load_config()
    config["reward"]["docking"]["mode"] = "off"  # most tests need no engine
    config["reward"]["property_cache_path"] = "cache/test_properties.jsonl"
    config["reward"]["docking"]["cache_path"] = "cache/test_docking.jsonl"
    return config


@pytest.fixture(scope="module")
def toolkit(cfg):
    return build_toolkit(cfg)


@pytest.fixture(scope="module")
def oracle(cfg, toolkit):
    return Oracle(cfg, toolkit)


# ----------------------------------------------------------------- ATEM format


def test_parses_a_reasoning_then_tool_call_turn():
    text = (
        " to=self<|message|>I should measure the molecule first.<|eom|>"
        "<|start|>assistant to=get_molecule_metrics<|message|><atem:function_calls>\n"
        '<atem:invoke name="get_molecule_metrics">\n'
        '<atem:parameter name="smiles">CCO</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    )
    turn = atem.parse_assistant_text(text)
    assert "measure the molecule" in turn.reasoning
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "get_molecule_metrics"
    assert turn.tool_calls[0].arguments == {"smiles": "CCO"}
    assert turn.malformed == []


def test_parses_a_final_answer_turn():
    turn = atem.parse_assistant_text(
        " to=user<|message|>Done; the analog is better.<|eot|>"
    )
    assert turn.final_text == "Done; the analog is better."
    assert turn.ended_turn and not turn.wants_tools


def test_coerces_json_shaped_parameters_but_leaves_smiles_alone():
    text = (
        " to=dock_and_score<|message|><atem:function_calls>\n"
        '<atem:invoke name="dock_and_score">\n'
        '<atem:parameter name="smiles">C1=CC=CC=C1</atem:parameter>\n'
        '<atem:parameter name="exhaustiveness">4</atem:parameter>\n'
        '<atem:parameter name="center">[1.0, 2.0, 3.0]</atem:parameter>\n'
        '<atem:parameter name="relax">true</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    )
    args = atem.parse_assistant_text(text).tool_calls[0].arguments
    assert args == {
        "smiles": "C1=CC=CC=C1",  # stays a string, not a number or list
        "exhaustiveness": 4,
        "center": [1.0, 2.0, 3.0],
        "relax": True,
    }


def test_reports_an_unterminated_invoke_as_malformed():
    text = ' to=x<|message|><atem:function_calls>\n<atem:invoke name="x">\n<|eom|>'
    turn = atem.parse_assistant_text(text)
    assert turn.tool_calls == []
    assert turn.malformed


def test_round_trips_a_rendered_call():
    rendered = atem.render_tool_call("get_molecule_metrics", {"smiles": ASPIRIN})
    parsed = atem.parse_assistant_text(rendered)
    assert parsed.tool_calls[0].arguments["smiles"] == ASPIRIN


def test_tool_results_are_truncated_not_dropped():
    rendered = atem.render_tool_result("x", "y" * 500, max_chars=100)
    assert "truncated, 500 chars total" in rendered
    assert len(rendered) < 400


# --------------------------------------------------------------------- config


def test_unknown_config_keys_fail_loudly(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("episode:\n  max_turnz: 3\n")
    with pytest.raises(ConfigError, match="unknown config key: episode.max_turnz"):
        load_config(bad)


def test_thresholds_are_user_defined_and_not_schema_checked(tmp_path):
    custom = tmp_path / "ok.yaml"
    custom.write_text(
        "reward:\n  thresholds:\n    tpsa:\n      direction: decrease\n      goal_delta: 5\n"
    )
    config = load_config(custom)
    assert config["reward"]["thresholds"] == {
        "tpsa": {"direction": "decrease", "goal_delta": 5}
    }


# ---------------------------------------------------------------------- tools


def test_toolkit_exposes_the_configured_surface_plus_submit(toolkit):
    names = set(toolkit.names())
    assert {"validate_smiles", "get_molecule_metrics", "dock_and_score"} <= names
    assert "submit_molecule" in names
    assert toolkit.specs["submit_molecule"].terminal


def test_tools_are_the_same_functions_the_mcp_server_registers(toolkit):
    assert (
        toolkit.specs["get_molecule_metrics"].fn is toolkit.server.get_molecule_metrics
    )


def test_a_tool_call_returns_a_value(toolkit):
    result = asyncio.run(toolkit.call("get_molecule_metrics", {"smiles": ASPIRIN}))
    assert result.ok and result.content["sa_score"] > 0
    assert result.duration_s >= 0


def test_a_failing_tool_becomes_data_not_an_exception(toolkit):
    result = asyncio.run(toolkit.call("get_molecule_metrics", {"smiles": "c1ccccc"}))
    assert not result.ok
    assert "invalid SMILES" in result.error


def test_unknown_tools_and_bad_arguments_are_explained(toolkit):
    unknown = asyncio.run(toolkit.call("teleport", {}))
    assert not unknown.ok and "Available tools" in unknown.error

    bad_args = asyncio.run(toolkit.call("get_molecule_metrics", {"smiley": "CCO"}))
    assert not bad_args.ok and "invalid arguments" in bad_args.error


# --------------------------------------------------------------------- reward


def test_partial_progress_earns_partial_credit():
    start = Properties(smiles="A", valid=True, sa_score=4.0, qed=0.30)
    final = Properties(smiles="B", valid=True, sa_score=3.75, qed=0.30)
    outcomes = score_thresholds(
        start, final, {"sa_score": {"direction": "decrease", "goal_delta": 0.5}}
    )
    assert outcomes[0].progress == pytest.approx(0.5)
    assert outcomes[0].met is False


def test_a_do_not_get_worse_rule_tolerates_small_regressions():
    start = Properties(smiles="A", valid=True, binding_affinity=-9.0)
    worse = Properties(smiles="B", valid=True, binding_affinity=-8.7)
    rule = {
        "binding_affinity": {
            "direction": "decrease",
            "goal_delta": 0.0,
            "tolerance": 0.5,
        }
    }
    assert score_thresholds(start, worse, rule)[0].met is True

    much_worse = Properties(smiles="C", valid=True, binding_affinity=-7.0)
    assert score_thresholds(start, much_worse, rule)[0].met is False


def test_an_unmeasured_property_scores_zero_rather_than_crashing():
    start = Properties(smiles="A", valid=True, sa_score=4.0)
    final = Properties(smiles="B", valid=True, sa_score=3.0)  # no affinity measured
    outcomes = score_thresholds(
        start, final, {"binding_affinity": {"direction": "decrease", "goal_delta": 0.0}}
    )
    assert outcomes[0].delta is None and outcomes[0].progress == 0.0


def test_failing_to_submit_is_worse_than_submitting_something_mediocre(cfg):
    start = Properties(smiles="A", valid=True, sa_score=4.0, qed=0.3)
    mediocre = Properties(smiles="B", valid=True, sa_score=3.95, qed=0.3)
    gave_up = compute_reward(
        cfg,
        start=start,
        final=None,
        tool_calls=3,
        tool_failures=0,
        malformed_calls=0,
        submitted=False,
    )
    tried = compute_reward(
        cfg,
        start=start,
        final=mediocre,
        tool_calls=3,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    assert gave_up.total < tried.total


def test_meeting_every_threshold_pays_the_bonus(cfg):
    start = Properties(
        smiles="A", valid=True, sa_score=4.0, qed=0.30, binding_affinity=-9.0
    )
    final = Properties(
        smiles="B", valid=True, sa_score=3.2, qed=0.40, binding_affinity=-9.4
    )
    breakdown = compute_reward(
        cfg,
        start=start,
        final=final,
        tool_calls=5,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    assert breakdown.terms["all_met_bonus"] > 0
    assert all(t.met for t in breakdown.thresholds)


def test_resubmitting_the_input_unchanged_forfeits_novelty(cfg):
    start = Properties(smiles="A", valid=True, sa_score=4.0, qed=0.3)
    breakdown = compute_reward(
        cfg,
        start=start,
        final=start,
        tool_calls=2,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    assert breakdown.terms["novelty"] == 0.0
    assert "unchanged" in " ".join(breakdown.notes)


def test_broken_tool_calls_reduce_the_reward(cfg):
    start = Properties(smiles="A", valid=True, sa_score=4.0, qed=0.3)
    final = Properties(smiles="B", valid=True, sa_score=3.5, qed=0.35)
    clean = compute_reward(
        cfg,
        start=start,
        final=final,
        tool_calls=4,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    messy = compute_reward(
        cfg,
        start=start,
        final=final,
        tool_calls=4,
        tool_failures=2,
        malformed_calls=1,
        submitted=True,
    )
    assert messy.total < clean.total


# -------------------------------------------------------------------- episodes


def scripted(turns: list[str]):
    """A stand-in policy that replays fixed turns."""
    remaining = list(turns)

    def generate(_transcript: str) -> str:
        return (
            remaining.pop(0) if remaining else atem.render_final("I am out of ideas.")
        )

    return generate


def test_a_full_episode_measures_then_submits(cfg, toolkit, oracle):
    task = Task(task_id="t1", smiles=ASPIRIN, target="test")
    policy = scripted(
        [
            atem.render_reasoning("Measure the input first.")
            + atem.render_tool_call("get_molecule_metrics", {"smiles": ASPIRIN}),
            # Methyl ester: simpler than the acid and a real change.
            atem.render_tool_call(
                "submit_molecule",
                {
                    "smiles": "COC(=O)c1ccccc1OC(C)=O",
                    "rationale": "ester instead of acid",
                },
            ),
        ]
    )
    result = asyncio.run(run_episode(cfg, task, policy, toolkit, oracle))

    assert result.stop_reason == "submitted"
    assert result.submitted_smiles == "COC(=O)c1ccccc1OC(C)=O"
    assert result.tool_calls == 2 and result.tool_failures == 0
    assert result.final_properties["valid"] is True
    assert "thresholds" in result.breakdown["terms"]
    assert result.wall_time_s > 0


def test_an_episode_that_never_submits_is_penalized(cfg, toolkit, oracle):
    task = Task(task_id="t2", smiles=ETHANOL, target="test")
    policy = scripted([atem.render_final("I think ethanol is fine as it is.")])
    result = asyncio.run(run_episode(cfg, task, policy, toolkit, oracle))

    assert result.stop_reason == "final_answer"
    assert result.submitted_smiles is None
    assert result.reward == cfg["reward"]["no_submission_reward"]


def test_an_invalid_submission_is_penalized(cfg, toolkit, oracle):
    task = Task(task_id="t3", smiles=ETHANOL, target="test")
    policy = scripted([atem.render_tool_call("submit_molecule", {"smiles": "c1ccccc"})])
    result = asyncio.run(run_episode(cfg, task, policy, toolkit, oracle))
    assert result.reward == cfg["reward"]["invalid_submission_reward"]


def test_tool_errors_are_fed_back_so_the_policy_can_recover(cfg, toolkit, oracle):
    task = Task(task_id="t4", smiles=ASPIRIN, target="test")
    policy = scripted(
        [
            atem.render_tool_call("get_molecule_metrics", {"smiles": "not-a-molecule"}),
            atem.render_tool_call("submit_molecule", {"smiles": ETHANOL}),
        ]
    )
    result = asyncio.run(run_episode(cfg, task, policy, toolkit, oracle))
    assert result.tool_failures == 1
    assert "ERROR: invalid SMILES" in result.transcript  # the model sees the failure
    assert result.stop_reason == "submitted"


def test_the_tool_call_budget_is_enforced(cfg, toolkit, oracle):
    tight = dict(cfg)
    tight["episode"] = {**cfg["episode"], "max_tool_calls": 2}
    task = Task(task_id="t5", smiles=ASPIRIN, target="test")
    policy = scripted(
        [atem.render_tool_call("get_molecule_metrics", {"smiles": ASPIRIN})] * 5
    )
    result = asyncio.run(run_episode(tight, task, policy, toolkit, oracle))
    assert result.tool_calls == 2
    assert result.stop_reason == "max_tool_calls"


def test_the_prompt_states_the_goals_and_the_molecule(cfg):
    task = Task(
        task_id="t6",
        smiles=ASPIRIN,
        target="trypsin",
        baseline=Properties(smiles=ASPIRIN, valid=True, sa_score=2.33, qed=0.55),
    )
    prompt = build_prompt(cfg, task)
    assert ASPIRIN in prompt and "trypsin" in prompt
    assert "sa_score: lower it by at least 0.5" in prompt
    assert "sa_score: 2.33" in prompt
    assert prompt.endswith("<|start|>assistant")  # ready for the model to continue
