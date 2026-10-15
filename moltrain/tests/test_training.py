"""Training-side tests: the expert, trace shape, batching, masking, advantages.

No GPU and no model. The expert's tool calls are executed for real, so these
also check that the vetted SMARTS library still applies to the molecules it was
characterised on.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import atem, grpo_core, hostcheck
from moltrain import policy as policy_mod
from moltrain.config import load_config
from moltrain.episode import Task, run_episode
from moltrain.experts import MOVES, ScriptedExpert, last_tool_output
from moltrain.grpo_core import (
    RolloutError,
    build_batch,
    completion_spans,
    degenerate_groups,
    group_advantages,
)
from moltrain.oracle import Oracle
from moltrain.toolkit import build_toolkit
from moltrain.traces import RecordingPolicy, collect, scripted_factory

# Probes carrying the groups the move library was characterised against.
CF3_SPIRO = "FC(F)(F)c1ccc(cc1)C(=O)N[C@H]1CC2(CC2)C[C@@H]1O"
NITROARYL = "O=[N+]([O-])c1ccc(cc1)C(=O)NCc1ccccc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CHLOROARYL = "Clc1ccc(cc1)c1ccc(cc1)C(=O)NC1CCCCC1"
THIOETHER = "CSCc1ccc(cc1)C(=O)NC1CCCC1"


@pytest.fixture(scope="module")
def cfg():
    config = load_config()
    config["reward"]["docking"]["mode"] = "off"
    config["reward"]["property_cache_path"] = "cache/test_train_properties.jsonl"
    config["reward"]["docking"]["cache_path"] = "cache/test_train_docking.jsonl"
    config["episode"]["max_turns"] = 40
    config["episode"]["max_tool_calls"] = 60
    return config


@pytest.fixture(scope="module")
def toolkit(cfg):
    return build_toolkit(cfg)


@pytest.fixture(scope="module")
def oracle(cfg, toolkit):
    return Oracle(cfg, toolkit)


def run(cfg, toolkit, oracle, smiles, task_id="probe-00"):
    task = Task(task_id=task_id, smiles=smiles, target="probe")
    policy = RecordingPolicy(ScriptedExpert(task, cfg["reward"]["thresholds"]))
    result = asyncio.run(run_episode(cfg, task, policy, toolkit, oracle))
    return task, policy, result


# ----------------------------------------------------------------- the library


def test_move_library_is_ordered_by_measured_benefit():
    deltas = [move.measured_delta_sa for move in MOVES]
    assert deltas == sorted(deltas), "greedy search should try the best moves first"
    assert all(move.measured_delta_sa < 0 for move in MOVES), (
        "a move that did not lower SA when measured should not be in the library"
    )


def test_every_move_is_a_single_reactant_template():
    for move in MOVES:
        assert move.smarts.count(">>") == 1
        assert "." not in move.smarts.split(">>")[0], (
            f"{move.name} expects more than one reactant"
        )


def test_the_moves_still_apply_to_what_they_were_measured_on(toolkit):
    """Guards against an RDKit upgrade silently breaking the library."""
    server = toolkit.server
    applied = set()
    for probe in (CF3_SPIRO, NITROARYL, CHLOROARYL, THIOETHER):
        for move in MOVES:
            products = server.apply_smarts_reaction(
                smiles=probe, reaction_smarts=move.smarts
            )["products"]
            if products:
                applied.add(move.name)
    assert {
        "remove_cyclopropane",
        "remove_hydroxyl",
        "remove_cf3",
        "nitro_to_amine",
        "dehalogenate_aryl",
        "thioether_to_ether",
    } <= applied


# ------------------------------------------------------------------ the expert


def test_expert_finishes_an_episode_cleanly(cfg, toolkit, oracle):
    _, _, result = run(cfg, toolkit, oracle, CF3_SPIRO, "cf3-00")
    assert result.stop_reason == "submitted"
    assert result.tool_failures == 0
    assert result.malformed_calls == 0, "the expert must emit parseable turns"
    assert result.submitted_smiles


def test_expert_lowers_sa_without_giving_up_qed(cfg, toolkit, oracle):
    """Accepting on SA alone used to trade QED away; the reward scores both."""
    task, _, result = run(cfg, toolkit, oracle, CF3_SPIRO, "cf3-01")
    final = result.final_properties
    assert final["sa_score"] < task.baseline.sa_score
    assert final["qed"] >= task.baseline.qed - 0.05


def test_expert_submits_a_canonical_smiles(cfg, toolkit, oracle):
    """Reaction products carry explicit [H]; the trace should not."""
    _, _, result = run(cfg, toolkit, oracle, NITROARYL, "nitro-00")
    assert "[H]" not in result.submitted_smiles
    canonical = toolkit.server.validate_smiles(result.submitted_smiles)
    assert canonical["canonical_smiles"] == result.submitted_smiles


def test_expert_leaves_an_already_simple_molecule_alone(cfg, toolkit, oracle):
    _, _, result = run(cfg, toolkit, oracle, ASPIRIN, "aspirin-00")
    assert result.submitted_smiles is not None
    assert all(
        outcome["progress"] == 0.0 for outcome in result.breakdown["thresholds"]
    ), "nothing in the library helps aspirin, so no threshold should move"


def test_last_tool_output_reads_the_most_recent_result():
    transcript = atem.render_tool_result("a", '{"value": 1}') + atem.render_tool_result(
        "b", '{"value": 2}'
    )
    name, payload = last_tool_output(transcript)
    assert name == "b"
    assert payload == {"value": 2}


def test_last_tool_output_survives_a_non_json_result():
    name, payload = last_tool_output(atem.render_tool_result("a", "ERROR: nope"))
    assert (name, payload) == ("a", "ERROR: nope")


# ------------------------------------------------------------------ SFT traces


def test_collected_traces_never_put_tool_output_in_a_completion(cfg):
    """The invariant that keeps SFT from teaching fabricated measurements."""
    tasks = [Task(task_id="t0", smiles=CF3_SPIRO, target="probe")]
    records, _, report = asyncio.run(collect(cfg, tasks, scripted_factory(cfg)))
    assert records, "the expert's episode should clear the reward cutoff"
    assert all("<tool_output" not in r["completion"] for r in records)
    assert any("<tool_output" in r["prompt"] for r in records)
    assert report.kept == 1


def test_episodes_that_changed_nothing_are_not_imitated(cfg):
    """Diligent but fruitless is honest work; imitating it teaches inaction."""
    tasks = [Task(task_id="plain", smiles=ASPIRIN, target="probe")]
    records, episodes, report = asyncio.run(collect(cfg, tasks, scripted_factory(cfg)))
    assert report.episodes == 1
    assert report.kept == 0
    assert records == []
    # It is filtered for making no progress, not for failing.
    assert episodes[0].stop_reason == "submitted"
    assert episodes[0].tool_failures == 0


def test_every_completion_parses_as_an_assistant_turn(cfg):
    tasks = [Task(task_id="t0", smiles=NITROARYL, target="probe")]
    records, _, _ = asyncio.run(collect(cfg, tasks, scripted_factory(cfg)))
    for record in records:
        turn = atem.parse_assistant_text(record["completion"])
        assert turn.malformed == []
        assert turn.reasoning or turn.tool_calls


def test_the_expert_only_docks_when_the_reward_does(cfg):
    """Under final_only the reward docks once; the expert should not dock at all."""
    from copy import deepcopy

    every = deepcopy(cfg)
    every["reward"]["docking"]["mode"] = "every_candidate"
    task = Task(task_id="t", smiles=CF3_SPIRO, receptor_file="/tmp/r.pdbqt")
    assert scripted_factory(cfg)(task).dock is False
    assert scripted_factory(every)(task).dock is True


# -------------------------------------------------------------- the generator


class FakeEngine:
    """Records the batch sizes it was asked for."""

    def __init__(self, reply: str = " to=self<|message|>ok<|eom|>"):
        self.reply = reply
        self.batches: list[int] = []

    def generate(self, prompts, params):
        self.batches.append(len(prompts))
        return [self.reply for _ in prompts]


def test_concurrent_turns_become_one_engine_batch(cfg):
    """A GRPO group costs one batched decode, not one per rollout."""
    engine = FakeEngine()
    policy = policy_mod.build_policy(cfg, engine, params=None, max_batch=8)

    async def main():
        return await asyncio.gather(*(policy("<|start|>assistant") for _ in range(8)))

    assert len(asyncio.run(main())) == 8
    assert engine.batches == [8]


def test_a_batch_larger_than_the_cap_is_split(cfg):
    engine = FakeEngine()
    policy = policy_mod.build_policy(cfg, engine, params=None, max_batch=3)

    async def main():
        await asyncio.gather(*(policy("<|start|>assistant") for _ in range(7)))

    asyncio.run(main())
    assert engine.batches == [3, 3, 1]


def test_an_engine_failure_reaches_every_waiter(cfg):
    class Broken:
        def generate(self, prompts, params):
            raise RuntimeError("engine died")

    policy = policy_mod.build_policy(cfg, Broken(), params=None, max_batch=4)

    async def main():
        return await asyncio.gather(
            *(policy("<|start|>assistant") for _ in range(3)),
            return_exceptions=True,
        )

    assert all(isinstance(r, RuntimeError) for r in asyncio.run(main()))


def test_a_short_count_of_completions_is_an_error_not_a_hang(cfg):
    class Stingy:
        def generate(self, prompts, params):
            return ["only one"]

    policy = policy_mod.build_policy(cfg, Stingy(), params=None, max_batch=4)

    async def main():
        return await asyncio.gather(
            *(policy("<|start|>assistant") for _ in range(3)),
            return_exceptions=True,
        )

    assert all(isinstance(r, RuntimeError) for r in asyncio.run(main()))


# ------------------------------------------------------- context-window trimming


def long_transcript(turns: int) -> str:
    head = "<|start|>system<|message|>GOALS<|eot|><|start|>assistant"
    body = "".join(
        atem.render_tool_result("t", "x" * 400)
        + atem.render_continuation(
            "thinking", ("get_molecule_metrics", {"smiles": "C"})
        )
        for _ in range(turns)
    )
    return head + body


def test_trimming_keeps_the_goals_and_says_what_it_dropped():
    transcript = long_transcript(40)
    fitted, dropped = policy_mod.trim_transcript(transcript, max_tokens=1000)
    assert dropped > 0
    assert "GOALS" in fitted
    assert "Earlier turns were dropped" in fitted
    assert policy_mod.default_token_counter()(fitted) <= 1000


def test_trimming_leaves_a_short_transcript_untouched():
    transcript = long_transcript(1)
    fitted, dropped = policy_mod.trim_transcript(transcript, max_tokens=100_000)
    assert (fitted, dropped) == (transcript, 0)


def test_the_policy_counts_its_own_truncations(cfg):
    engine = FakeEngine()
    policy = policy_mod.build_policy(cfg, engine, params=None, max_batch=1)
    policy.max_prompt_tokens = 500

    async def main():
        await policy(long_transcript(40))

    asyncio.run(main())
    assert policy.truncations == 1
    assert policy.segments_dropped > 0


# ---------------------------------------------------------------- masking


def test_spans_alternate_between_model_and_environment(cfg, toolkit, oracle):
    _, policy, result = run(cfg, toolkit, oracle, NITROARYL, "nitro-01")
    spans = completion_spans(policy.turns, result.transcript)
    assert spans
    assert [span.from_model for span in spans[:4]] == [True, False, True, False]


def test_no_model_span_contains_a_tool_result(cfg, toolkit, oracle):
    """If this fails, training teaches the model to write its own measurements."""
    _, policy, result = run(cfg, toolkit, oracle, CF3_SPIRO, "cf3-02")
    authored = "".join(
        span.text
        for span in completion_spans(policy.turns, result.transcript)
        if span.from_model
    )
    assert "<tool_output" not in authored


def test_a_rewritten_transcript_is_rejected():
    turns = [("prompt", "turn-a"), ("prompt-DIFFERENT", "turn-b")]
    with pytest.raises(RolloutError, match="prefix"):
        completion_spans(turns, "whatever")


def test_mask_marks_exactly_the_model_tokens():
    turns = [("P", "MMMM")]  # one model turn, then the environment appends
    transcript = "PMMMMEEEEEEEE"
    ids, mask = grpo_core.build_completion(turns, transcript, lambda t: [0] * len(t))
    assert len(ids) == len(mask) == 12
    assert mask == [1] * 4 + [0] * 8


def test_environment_tokens_are_a_large_share_of_the_sequence(cfg, toolkit, oracle):
    """Reported because it is how much signal the mask is protecting."""
    _, policy, result = run(cfg, toolkit, oracle, CF3_SPIRO, "cf3-03")
    batch = build_batch(
        [("cf3-03", policy.turns, result.transcript, result.reward)],
        lambda text: [0] * max(1, len(text) // 4),
        num_generations=1,
    )
    fraction = batch.diagnostics["trained_token_fraction"]
    assert 0.2 < fraction < 0.9
    assert batch.diagnostics["environment_tokens"] > 0


# -------------------------------------------------------------- advantages


def test_advantages_centre_within_each_group():
    advantages = group_advantages([0.0, 1.0, 2.0, 3.0], num_generations=2)
    assert advantages[0] == pytest.approx(-advantages[1])
    assert advantages[2] == pytest.approx(-advantages[3])


def test_a_group_with_no_spread_gives_no_gradient():
    """A deterministic policy makes every group degenerate, so GRPO learns nothing."""
    advantages = group_advantages([0.7, 0.7, 0.7, 0.7], num_generations=4)
    assert advantages == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert degenerate_groups([0.7, 0.7, 0.7, 0.7], 4) == 1


def test_unscaled_advantages_keep_the_reward_units():
    advantages = group_advantages([0.0, 2.0], num_generations=2, scale_by_std=False)
    assert advantages == pytest.approx([-1.0, 1.0])


def test_rewards_that_do_not_fill_whole_groups_are_rejected():
    with pytest.raises(RolloutError, match="groups of"):
        group_advantages([1.0, 2.0, 3.0], num_generations=2)


def test_the_deterministic_expert_is_flagged_as_degenerate(cfg, toolkit, oracle):
    """The expert is for SFT; GRPO on it would be a no-op, so prove it shows up."""
    episodes = []
    for index in range(2):
        _, policy, result = run(cfg, toolkit, oracle, NITROARYL, f"group-{index}")
        episodes.append(
            (f"group-{index}", policy.turns, result.transcript, result.reward)
        )
    batch = build_batch(
        episodes, lambda t: [0] * max(1, len(t) // 4), num_generations=2
    )
    assert batch.diagnostics["degenerate_groups"] == 1
    assert batch.advantages == pytest.approx([0.0, 0.0])


def test_trl_payload_carries_the_mask():
    batch = build_batch(
        [("t", [("P", "MM")], "PMMEE", 1.0)],
        lambda text: [0] * len(text),
        num_generations=1,
    )
    payload = batch.as_trl_payload()
    assert set(payload) == {
        "prompt_ids",
        "completion_ids",
        "completion_mask",
        "reward",
        "task_id",
    }
    assert payload["completion_mask"] == [[1, 1, 0, 0]]


# ------------------------------------------------- the reward's incentives


def test_doing_nothing_scores_the_floor(cfg, toolkit, oracle):
    """A blind submission collects validity and nothing else."""
    from moltrain.evaluate import unchanged_factory

    task = Task(task_id="floor-00", smiles=CF3_SPIRO, target="probe")
    result = asyncio.run(
        run_episode(cfg, task, unchanged_factory(cfg)(task), toolkit, oracle)
    )
    terms = result.breakdown["terms"]
    assert terms["tool_success"] == 0.0, "no measurement, no tool-success credit"
    assert terms["novelty"] == 0.0
    assert terms["turn_penalty"] == 0.0, "one call is inside the free budget"
    assert result.reward == pytest.approx(cfg["reward"]["weights"]["validity"])


def test_trying_beats_doing_nothing(cfg, toolkit, oracle):
    """The regression that mattered: a flat per-call penalty made measuring
    irrational, so the reward's optimum was to submit without using a tool."""
    from moltrain.evaluate import unchanged_factory

    task = Task(task_id="cmp-00", smiles=CF3_SPIRO, target="probe")
    floor = asyncio.run(
        run_episode(cfg, task, unchanged_factory(cfg)(task), toolkit, oracle)
    )
    _, _, worked = run(cfg, toolkit, oracle, CF3_SPIRO, "cmp-01")
    assert worked.tool_calls > floor.tool_calls
    assert worked.reward > floor.reward


def test_the_penalty_cannot_outweigh_the_progress_it_buys(cfg, toolkit, oracle):
    _, _, result = run(cfg, toolkit, oracle, CF3_SPIRO, "cmp-02")
    terms = result.breakdown["terms"]
    assert abs(terms["turn_penalty"]) < terms["thresholds"] + terms["novelty"]


def test_calls_inside_the_free_budget_are_not_charged(cfg):
    from moltrain.oracle import Properties
    from moltrain.rewards import compute_reward

    start = Properties(smiles="CCO", valid=True, sa_score=4.0, qed=0.4)
    final = Properties(smiles="CCC", valid=True, sa_score=3.4, qed=0.5)
    free = int(cfg["reward"]["turn_penalty_free_calls"])
    inside = compute_reward(
        cfg,
        start=start,
        final=final,
        tool_calls=free,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    outside = compute_reward(
        cfg,
        start=start,
        final=final,
        tool_calls=free + 4,
        tool_failures=0,
        malformed_calls=0,
        submitted=True,
    )
    assert inside.terms["turn_penalty"] == 0.0
    assert outside.terms["turn_penalty"] == pytest.approx(
        -4 * cfg["reward"]["weights"]["turn_penalty"]
    )


# ------------------------------------------------- goals versus constraints

GOALS_AND_CONSTRAINT = {
    "sa_score": {"direction": "decrease", "goal_delta": 0.5},
    "binding_affinity": {"direction": "decrease", "goal_delta": 0.0, "tolerance": 0.5},
}


def reward_for(cfg, *, start_kwargs, final_kwargs, calls=12, submitted=True):
    from copy import deepcopy

    from moltrain.oracle import Properties
    from moltrain.rewards import compute_reward

    config = deepcopy(cfg)
    config["reward"]["thresholds"] = GOALS_AND_CONSTRAINT
    return compute_reward(
        config,
        start=Properties(valid=True, **start_kwargs),
        final=Properties(valid=True, **final_kwargs),
        tool_calls=calls,
        tool_failures=0,
        malformed_calls=0,
        submitted=submitted,
    )


def test_a_constraint_satisfied_by_inaction_earns_nothing(cfg):
    """The regression: submitting the input scored full marks on "maintain
    binding affinity" and outranked an episode that improved something."""
    unchanged = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
    )
    assert unchanged.terms["thresholds"] == 0.0
    assert unchanged.terms["constraint_violation"] == 0.0
    assert unchanged.terms["novelty"] == 0.0
    assert "unchanged" in " ".join(unchanged.notes)


def test_improving_a_goal_outscores_leaving_it_alone(cfg):
    unchanged = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
    )
    improved = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.5, "binding_affinity": -8.0},
    )
    assert improved.total > unchanged.total


def test_breaking_a_constraint_is_charged_by_how_badly(cfg):
    """A flat charge treated a 0.6 kcal/mol slip like a 2.0 collapse."""
    slip = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.5, "binding_affinity": -7.4},
    )
    collapse = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.5, "binding_affinity": -6.0},
    )
    assert slip.terms["constraint_violation"] < 0
    assert collapse.terms["constraint_violation"] < slip.terms["constraint_violation"]
    assert "tolerances past its allowance" in " ".join(collapse.notes)


def test_the_constraint_charge_is_capped(cfg):
    """One ruinous rollout must not swamp its group's advantage."""
    weights = cfg["reward"]["weights"]["constraint_violation"]
    cap = cfg["reward"]["constraint_violation_cap"]
    ruinous = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -12.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.5, "binding_affinity": -1.0},
    )
    assert ruinous.terms["constraint_violation"] == pytest.approx(-weights * cap)


def test_buying_a_goal_by_breaking_a_constraint_is_not_worth_it(cfg):
    """Trading binding away for synthesisability should score below inaction."""
    traded = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.6, "binding_affinity": -6.0},
    )
    unchanged = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
    )
    assert traded.total < unchanged.total


def test_the_bonus_needs_the_goals_met_and_the_constraints_kept(cfg):
    clean = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.4, "binding_affinity": -8.2},
    )
    assert clean.terms["all_met_bonus"] == cfg["reward"]["weights"]["all_met_bonus"]

    dirty = reward_for(
        cfg,
        start_kwargs={"smiles": "CCO", "sa_score": 4.0, "binding_affinity": -8.0},
        final_kwargs={"smiles": "CCC", "sa_score": 3.4, "binding_affinity": -7.0},
    )
    assert dirty.terms["all_met_bonus"] == 0.0


def test_an_episode_that_moved_no_goal_is_not_collected(cfg):
    """`require_improvement` must read goals, not constraints."""
    from moltrain.traces import _improved

    constraint_only = SimpleNamespace(
        breakdown={
            "thresholds": [
                # Satisfied by changing nothing, so it must not count.
                {
                    "name": "binding_affinity",
                    "delta": 0.0,
                    "goal_delta": 0.0,
                    "progress": 1.0,
                },
                {"name": "sa_score", "delta": 0.0, "goal_delta": 0.5, "progress": 0.0},
            ]
        }
    )
    assert _improved(constraint_only) is False


# --------------------------------------------------- host diagnostics


def test_importing_the_package_caps_the_thread_pools():
    """A container can expose 64 cores while capping threads per process, and
    every BLAS-backed import then dies before it starts."""
    import moltrain
    from moltrain.hostcheck import THREAD_VARS

    for name in THREAD_VARS:
        assert os.environ.get(name), f"{name} should have a default"
    assert os.environ["OPENBLAS_NUM_THREADS"] == moltrain.NUM_THREADS


def test_an_explicit_thread_setting_wins(monkeypatch):
    monkeypatch.setenv("MOLTRAIN_NUM_THREADS", "3")
    env = hostcheck.thread_limited_env()
    assert all(env[name] == "3" for name in hostcheck.THREAD_VARS)


def test_a_thread_limit_is_passed_explicitly():
    env = hostcheck.thread_limited_env(2)
    assert all(env[name] == "2" for name in hostcheck.THREAD_VARS)


def test_the_blas_failure_is_recognised():
    noise = (
        "OpenBLAS blas_thread_init: pthread_create failed for thread 63 of 64: "
        "Resource temporarily unavailable"
    )
    assert hostcheck.looks_like_blas_thread_failure(noise)
    assert not hostcheck.looks_like_blas_thread_failure("ImportError: no module")
    assert "MOLTRAIN_NUM_THREADS" in hostcheck.BLAS_HINT


def test_a_module_that_imports_is_reported_with_its_version():
    report = hostcheck.probe_module("json")
    assert report.ok
    assert report.detail


def test_a_missing_module_is_a_line_not_an_exception():
    report = hostcheck.probe_module("definitely_not_a_real_module_xyz")
    assert not report.ok
    assert "ModuleNotFoundError" in report.detail


def test_a_hanging_import_times_out_rather_than_blocking():
    """The first real run hung inside peft's import chain until interrupted."""
    report = hostcheck.probe_module("json", timeout_s=0.001)
    assert not report.ok
    assert report.timed_out


def test_one_broken_dependency_does_not_end_the_report():
    reports, problems = hostcheck.report_imports(
        ("json", "definitely_not_a_real_module_xyz", "pathlib")
    )
    assert [r.name for r in reports] == [
        "json",
        "definitely_not_a_real_module_xyz",
        "pathlib",
    ]
    assert [r.ok for r in reports] == [True, False, True]
    assert len(problems) == 1


def test_missing_binaries_name_the_conda_packages():
    lines, problems = hostcheck.report_binaries(("definitely_not_a_binary_xyz",))
    assert "NOT FOUND" in lines[0]
    assert "conda install -c conda-forge" in problems[0]


def test_obabel_is_installed_under_its_package_name():
    """`conda install obabel` does not exist; the package is openbabel."""
    _, problems = hostcheck.report_binaries(("obabel",))
    if problems:  # only when it is genuinely absent on this host
        assert "openbabel" in problems[0]


def test_an_override_counts_as_finding_the_binary(monkeypatch):
    monkeypatch.setenv("MOLBENCH_VINA_BIN", "/opt/vina/bin/vina")
    lines, problems = hostcheck.report_binaries(("vina",))
    assert "/opt/vina/bin/vina" in lines[0]
    assert problems == []
