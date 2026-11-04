"""The seams between the modules, which is where this package broke.

Eight modules were written in parallel against one contract, and almost
every defect found while putting them together was a seam: one module
calling another by a name it does not have, reading an attribute that was
spelled differently, or passing arguments in the other order. None of those
raise at import time, and several of them -- the grounding field name above
all -- do not raise at all. They return a default and the training silently
optimises the wrong thing.

So the seams are asserted here rather than left to be rediscovered. These
tests are deliberately shallow: they check that the call is well formed and
the names line up, not what the call computes. The depth is in the other
files.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protactrain import (  # noqa: E402
    config,
    episode,
    evaluate,
    experts,
    oracle,
    policy,
    properties,
    protocol,
    rewards,
    tasks,
    toolkit,
    traces,
)
from protactrain.tests import fixtures  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return config.load_config()


# --------------------------------------------- the package imports on a laptop


def test_every_module_imports_without_torch_trl_or_vllm():
    """The training modules must be readable on a machine with no GPU stack,
    or none of the rest of this suite can run."""
    for name in ("sft", "grpo"):
        module = __import__(f"protactrain.{name}", fromlist=[name])
        assert module is not None
    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules


def test_the_gpu_paths_refuse_rather_than_hang_without_cuda():
    """Rule 6: nothing runs a model pass on this machine."""
    from protactrain import grpo

    for refuse in (grpo.require_cuda, policy.require_cuda):
        with pytest.raises(RuntimeError) as exc:
            refuse("a test")
        assert "CUDA" in str(exc.value) or "torch" in str(exc.value)


def test_the_gpu_evaluation_arms_refuse_and_the_control_arm_does_not():
    assert evaluate.require_cuda(["unchanged"]) == ""
    for arm in sorted(evaluate.GPU_ARMS):
        assert evaluate.require_cuda([arm])


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--dry-run"],
        ["rollout", "--tasks-per-step", "1"],
    ],
)
def test_grpo_commands_refuse_in_one_line_rather_than_a_traceback(argv, capsys):
    """Even --dry-run builds the trainer, which loads a 30B model. Without
    the gate the failure is a ModuleNotFoundError from inside peft, four
    frames down, which reads like a broken package rather than a wrong host.
    """
    from protactrain import grpo

    assert grpo.main(argv) == 3
    assert "two-H200 box" in capsys.readouterr().err


def test_sft_train_refuses_in_one_line_too(capsys):
    from protactrain import sft

    assert sft.main(["train", "--dry-run"]) == 3
    assert "two-H200 box" in capsys.readouterr().err


# ----------------------------------------------------- names the siblings use


def test_task_is_reachable_under_both_names():
    """`tasks.py` and `episode.py` say LinkerTask; `grpo.py` and `traces.py`
    were ported from moltrain and say Task."""
    assert episode.Task is episode.LinkerTask
    assert traces.Task is episode.LinkerTask


def test_read_tasks_has_moltrains_signature():
    parameters = list(inspect.signature(tasks.read_tasks).parameters)
    assert parameters[:2] == ["cfg", "path"]


def test_run_episode_takes_the_arguments_its_four_callers_pass():
    parameters = list(inspect.signature(episode.run_episode).parameters)
    assert parameters[:4] == ["cfg", "task", "generate", "toolkit"]
    # experts.py passes a fifth; everyone else omits it, so it must default.
    assert inspect.signature(episode.run_episode).parameters["oracle"].default is None


def test_policy_exposes_the_factory_traces_and_evaluate_reach_for():
    assert callable(policy.vllm_factory)
    parameters = inspect.signature(policy.vllm_factory).parameters
    assert list(parameters) == ["cfg", "adapter"]
    # Defaulting to None keeps `evaluate`'s GPU-free `unchanged` arm working,
    # but the callers that sample a trained policy must pass one: an engine
    # built without an adapter decodes from the base model, which has never
    # seen the ATEM call format. See test_adapter.py.
    assert parameters["adapter"].default is None


def test_episode_result_attribute_names_are_the_ones_readers_use():
    """grpo, traces and evaluate all read these off the result; moltrain
    spells four of them without the `n_` prefix."""
    names = {f.name for f in dataclasses.fields(episode.EpisodeResult)}
    for required in (
        "task",
        "turns",
        "calls",
        "transcript",
        "assistant_text",
        "submitted_smiles",
        "start_properties",
        "final_properties",
        "stop_reason",
        "n_turns",
        "n_tool_calls",
        "n_tool_failures",
        "n_malformed_calls",
    ):
        assert required in names, required


def test_no_module_still_reads_a_moltrain_only_attribute():
    """A grep, because these reads are all `getattr(..., default)` and so
    fail by returning zero rather than by raising."""
    package = Path(__file__).resolve().parents[1]
    stale = {
        'getattr(result, "tool_calls"',
        'getattr(result, "turns"',
        "result.tool_failures",
        "result.malformed_calls",
        "result.breakdown",
        'getattr(claim, "status"',
    }
    offenders = []
    for source in package.glob("*.py"):
        text = source.read_text()
        offenders += [f"{source.name}: {s}" for s in stale if s in text]
    assert not offenders, offenders


# ------------------------------------------------- the measurement boundary


def test_properties_covers_every_key_the_claim_audit_tolerates():
    """`audit_claims` is handed `LinkerProperties.as_dict()` straight, with
    no key translation, so a missing key here is a claim that can never be
    judged."""
    from protacbench.grade import _CLAIM_TOLERANCE

    measured = properties.measure(fixtures.PROTAC, fixtures.FRAGMENTS).as_dict()
    missing = sorted(set(_CLAIM_TOLERANCE) - set(measured))
    assert not missing, missing


def test_every_threshold_name_is_an_attribute_on_the_properties(cfg):
    """`rewards._value` reaches fields by getattr, and a getattr miss returns
    None, which is skipped -- so a typo deletes a reward term in silence."""
    blank = properties.LinkerProperties(smiles="")
    for name in cfg["reward"]["thresholds"]:
        assert hasattr(blank, name), name
    for name in cfg["reward"]["regression_tolerance"]:
        assert hasattr(blank, name), name


def test_the_weights_the_reward_reads_all_exist(cfg):
    weights = cfg["reward"]["weights"]
    for name in (
        "thresholds",
        "all_met_bonus",
        "hold_violation",
        "regression",
        "synthesis_score",
        "grounding",
        "discipline",
        "verification",
        "tool_success",
        "turn_penalty",
    ):
        assert name in weights, name


# ----------------------------------------------------------------- the tools


def test_the_toolkit_registers_the_configured_tools_and_one_terminal(cfg):
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    assert set(kit.names()) == set(cfg["episode"]["tools"]) | {protocol.SUBMIT_TOOL}
    terminal = [n for n, spec in kit.specs.items() if spec.terminal]
    assert terminal == [protocol.SUBMIT_TOOL]


def test_no_docking_tool_is_reachable(cfg):
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    assert not [n for n in kit.names() if "dock" in n.lower()]


def test_the_protocol_tool_tables_name_real_tools(cfg):
    """A measurement tool that no toolkit registers scores nothing forever."""
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    registered = set(kit.names())
    assert protocol.MEASUREMENT_TOOLS & registered
    assert protocol.ACCEPTANCE_TOOL in registered
    assert protocol.SUBMIT_TOOL in registered
    # benchmark_transformation is the only edit tool on the policy's surface;
    # apply_smarts_reaction is reachable only inside the composite, so that
    # every product gets measured.
    assert protocol.EDIT_TOOLS & registered == {"benchmark_transformation"}


def test_a_toolkit_rebinds_to_another_task_without_rebuilding(cfg):
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    other = kit.for_fragments(fixtures.FRAGMENTS_1112)
    assert other.fragments == fixtures.FRAGMENTS_1112
    assert kit.fragments == fixtures.FRAGMENTS  # the original is untouched


def test_a_failed_tool_call_returns_an_error_and_does_not_raise(cfg):
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    outcome = asyncio.run(kit.call("check_protac_linker", {"smiles": "not a molecule"}))
    assert outcome.ok is False or outcome.content
    assert isinstance(outcome.as_text(), str)


def test_an_unknown_tool_name_is_reported_not_raised(cfg):
    kit = toolkit.build_toolkit(cfg, fixtures.FRAGMENTS)
    outcome = asyncio.run(kit.call("run_docking", {"smiles": fixtures.PROTAC}))
    assert outcome.ok is False


# ---------------------------------------------------------------- the oracle


def test_the_oracle_caches_a_measurement_and_returns_the_same_answer(cfg, tmp_path):
    linker = oracle.LinkerOracle(cfg, cache_path=tmp_path / "cache.jsonl")
    first = linker.evaluate(fixtures.PROTAC, fixtures.FRAGMENTS)
    second = linker.evaluate(fixtures.PROTAC, fixtures.FRAGMENTS)
    assert first == second
    assert linker.stats["property_hits"] >= 1


def test_the_oracle_does_not_cache_an_unmeasurable_molecule(cfg, tmp_path):
    linker = oracle.LinkerOracle(cfg, cache_path=tmp_path / "cache.jsonl")
    bad = linker.evaluate("not a molecule", fixtures.FRAGMENTS)
    assert bad.measurable is False
    assert bad.error


def test_a_missing_synthesis_score_is_none_rather_than_zero(cfg, tmp_path):
    """None is a missing term worth 0.3 of weight; 0.0 is a bad molecule."""
    linker = oracle.LinkerOracle(cfg, cache_path=tmp_path / "cache.jsonl")
    assert linker.synthesis_score("CCO", fixtures.FRAGMENTS) is None
    real = linker.synthesis_score(fixtures.PROTAC, fixtures.FRAGMENTS)
    assert real is None or real > 0


# ---------------------------------------------------------------- the expert


def test_the_expert_takes_the_two_arguments_traces_constructs_it_with():
    parameters = list(inspect.signature(experts.ScriptedExpert.__init__).parameters)
    assert parameters[:3] == ["self", "task", "thresholds"]


def test_every_expert_move_is_a_parsable_reaction_smarts():
    from rdkit.Chem import rdChemReactions

    assert len(experts.MOVES) >= 10
    for move in experts.MOVES:
        assert rdChemReactions.ReactionFromSmarts(move.smarts) is not None, move.name
        # Each move records what it was measured to do on real molecules.
        assert move.measured and move.rationale
        assert move.addresses


def test_the_scripted_factory_builds_an_expert_per_task(cfg):
    factory = traces.scripted_factory(cfg)
    task = episode.LinkerTask(
        task_id="102-0",
        target_id="102",
        sample_id="0",
        protac_smiles=fixtures.PROTAC,
        fragment_smiles=list(fixtures.FRAGMENTS),
    )
    assert callable(factory(task))


# ------------------------------------------------------- the evaluation arms


def test_the_acceptance_tests_are_optimisation_mds_nine():
    assert len(evaluate.ACCEPTANCE_TESTS) == 9


def test_the_unchanged_arm_fails_the_not_the_input_test():
    """The floor every other arm must clear. With the fragments frozen most
    descriptors barely move, so this is a real floor and not a formality."""
    start = properties.measure(fixtures.PROTAC, fixtures.FRAGMENTS)
    verdict = evaluate.check_acceptance(start, start)
    assert verdict.admissible is True  # it is a valid PROTAC
    assert verdict.all_passed is False  # and it is not an answer


def test_the_arms_are_the_four_the_config_names(cfg):
    assert set(evaluate.ARMS) == set(cfg["evaluate"]["arms"])
    assert "unchanged" not in evaluate.GPU_ARMS


# --------------------------------------------------------------- end to end


def test_reward_and_protocol_compose_over_one_real_episode(cfg):
    """The whole scoring chain, on a real molecule, with no model.

    This is the assertion that the eight modules add up: an episode run
    against a scripted policy, scored by the same function GRPO uses, has to
    produce a finite reward with both halves of the scorecard present.
    """
    from protactrain.grpo import score_episode

    from moltrain import atem

    task = episode.LinkerTask(
        task_id="102-0",
        target_id="102",
        sample_id="0",
        protac_smiles=fixtures.PROTAC,
        fragment_smiles=list(fixtures.FRAGMENTS),
        optimisation_class="control",
        failure_codes=["qed_too_low"],
    )
    kit = toolkit.build_toolkit(cfg, task.fragment_smiles)

    turns = [
        atem.render_reasoning("Measure before editing.")
        + atem.render_tool_call("check_protac_linker", {"smiles": fixtures.PROTAC}),
        atem.render_tool_call("check_protac_linker", {"smiles": fixtures.PROTAC_PEG}),
        atem.render_tool_call(
            "submit_molecule",
            {"smiles": fixtures.PROTAC_PEG, "rationale": "PEG3 bridge"},
        ),
    ]
    remaining = list(turns)
    recorded: list[tuple[str, str]] = []

    def generate(transcript: str) -> str:
        text = remaining.pop(0) if remaining else atem.render_final("done")
        recorded.append((transcript, text))
        return text

    result = asyncio.run(episode.run_episode(cfg, task, generate, kit))
    assert result.stop_reason == "submitted"

    reward, breakdown, report = score_episode(cfg, task, result, recorded)
    assert isinstance(reward, float)
    assert reward == breakdown.reward
    assert not breakdown.floored
    # Both halves present: chemistry scored, and protocol actually judged.
    assert breakdown.thresholds
    assert report.verified_before_submit is True
    assert report.discipline >= 0.0
    assert "grounding" in breakdown.terms


def test_the_grounding_term_is_not_silently_zero(cfg):
    """The canary for the `claim.status` defect.

    A reply quoting numbers the tools returned must move the grounding term.
    If this reads 0.0, grounding has been disconnected again and the reward
    is outcome-only -- which trains a model that is good at proposing SMILES
    and no better at knowing when to measure.
    """
    measured = properties.measure(fixtures.PROTAC, fixtures.FRAGMENTS).as_dict()
    baseline = properties.measure(fixtures.PROTAC_PEG, fixtures.FRAGMENTS).as_dict()
    report = protocol.score_protocol(
        f"check_protac_linker gives a span of {measured['span']} bonds "
        f"and {measured['linker_heavy_atoms']} linker heavy atoms.",
        [],
        measured,
        baseline,
    )
    assert report.n_ok_claims >= 1
    assert report.grounding > 0.0

    weighted = float(cfg["reward"]["weights"]["grounding"]) * report.grounding
    assert weighted > 0.0
