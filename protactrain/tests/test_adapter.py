"""The policy that generates has to be the policy that was trained.

A GRPO rollout on the H200 box came back like this:

    "tool_calls": 0, "malformed_calls": 8,
    "stop_reasons": {"no_tool_call": 8},
    "floor_reasons": {"the episode ended without submitting a molecule": 8},
    "degenerate_groups": 1

Eight identical failures at turn one. The cause was not temperature and not
the task: no LoRA adapter was ever attached to the vLLM engine. `enable_lora=
True` was passed at construction, which reserves the slots and loads nothing,
and no `LoRARequest` existed anywhere in either package -- so every token came
from the base model. The ATEM call syntax appears nowhere in `episode.py`'s
prompts; it is learned from the SFT traces and from nothing else. A base-model
rollout therefore cannot produce a parseable call, and the report that results
is indistinguishable from a policy that tried the chemistry and failed.

The same `load_engine` is used by `make_rollout_func`, so a full GRPO run
would have sampled every episode from the untrained base model for all 500
steps while the gradient was applied to an adapter that never generated a
token.

These tests hold the two halves of the fix: the adapter reaches the engine,
and the engine follows the adapter as it trains.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from protactrain.config import DEFAULTS, latest_adapter, load_config, trained_adapter


def _adapter(directory: Path, *, rank: int = 32) -> Path:
    """What `save_pretrained` leaves behind, reduced to what is looked for."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank}))
    (directory / "adapter_model.safetensors").write_bytes(b"")
    return directory


# ------------------------------------------------------------ finding the thing


def test_a_saved_adapter_is_found_where_it_was_saved(tmp_path):
    _adapter(tmp_path / "sft")
    assert latest_adapter(tmp_path / "sft") == tmp_path / "sft"


def test_a_directory_without_the_config_file_is_not_an_adapter(tmp_path):
    """An output directory exists as soon as a log is written into it."""
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "training.log").write_text("...")
    assert latest_adapter(tmp_path / "sft") is None


def test_an_interrupted_warmup_is_found_in_its_newest_checkpoint(tmp_path):
    """The case this actually has to handle.

    `sft.max_steps` exists because the scripted traces are formulaic and the
    warmup is meant to be stopped once entropy collapses. A stopped run never
    reaches `save_model`, so the only adapter on disk is inside checkpoint-N/.
    Without this fallback the next stage starts from the base model, which is
    the exact failure above.
    """
    root = tmp_path / "sft"
    for step in (100, 300, 200):
        _adapter(root / f"checkpoint-{step}")
    assert latest_adapter(root) == root / "checkpoint-300"


def test_checkpoints_are_ordered_by_step_not_by_name(tmp_path):
    """`checkpoint-1000` sorts before `checkpoint-300` as a string."""
    root = tmp_path / "sft"
    for step in (300, 1000):
        _adapter(root / f"checkpoint-{step}")
    assert latest_adapter(root) == root / "checkpoint-1000"


def test_a_finished_run_beats_its_own_checkpoints(tmp_path):
    root = tmp_path / "sft"
    _adapter(root / "checkpoint-300")
    _adapter(root)
    assert latest_adapter(root) == root


def test_a_checkpoint_directory_with_no_adapter_in_it_is_skipped(tmp_path):
    root = tmp_path / "sft"
    (root / "checkpoint-400").mkdir(parents=True)  # optimiser state only
    _adapter(root / "checkpoint-100")
    assert latest_adapter(root) == root / "checkpoint-100"


def test_a_missing_directory_is_not_an_error(tmp_path):
    """A diagnostic that raises while diagnosing is the worst outcome."""
    assert latest_adapter(tmp_path / "never-created") is None


def test_grpos_output_is_preferred_over_sfts(tmp_path):
    """Stage 3 continues stage 2's adapter, so it is strictly the later policy."""
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(_adapter(tmp_path / "grpo"))
    cfg["sft"]["output_dir"] = str(_adapter(tmp_path / "sft"))
    found, searched = trained_adapter(cfg)
    assert found == tmp_path / "grpo"
    assert len(searched) == 2


def test_sft_is_used_when_grpo_has_not_run_yet(tmp_path):
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    cfg["sft"]["output_dir"] = str(_adapter(tmp_path / "sft"))
    found, _ = trained_adapter(cfg)
    assert found == tmp_path / "sft"


def test_nothing_trained_yet_reports_both_places_it_looked(tmp_path):
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    cfg["sft"]["output_dir"] = str(tmp_path / "sft")
    found, searched = trained_adapter(cfg)
    assert found is None
    assert [path.name for path in searched] == ["grpo", "sft"]


# ------------------------------------------------- refusing rather than guessing


def test_a_missing_adapter_refuses_instead_of_sampling_the_base_model(tmp_path):
    """The whole point. Falling back here is what cost the run."""
    from protactrain.grpo import generation_adapter

    cfg = load_config()
    cfg["sft"]["output_dir"] = str(tmp_path / "sft")
    cfg["grpo"]["resume_from_sft"] = True

    with pytest.raises(RuntimeError) as exc:
        generation_adapter(cfg)
    message = str(exc.value)
    # The message has to name the fix, because it is read by someone whose
    # last run produced a report full of -1.0 and no explanation.
    assert "resume_from_sft" in message
    assert "checkpoint-N" in message
    assert str(tmp_path / "sft") in message


def test_the_base_model_is_allowed_when_it_is_asked_for(tmp_path):
    from protactrain.grpo import generation_adapter

    cfg = load_config()
    cfg["sft"]["output_dir"] = str(tmp_path / "sft")
    cfg["grpo"]["resume_from_sft"] = False
    assert generation_adapter(cfg) is None


def test_an_interrupted_warmup_satisfies_the_gate(tmp_path):
    from protactrain.grpo import generation_adapter, should_resume_from_sft

    cfg = load_config()
    cfg["sft"]["output_dir"] = str(tmp_path / "sft")
    _adapter(tmp_path / "sft" / "checkpoint-300")

    adapter, resume = should_resume_from_sft(cfg)
    assert resume and adapter == tmp_path / "sft" / "checkpoint-300"
    assert generation_adapter(cfg) == adapter


# ------------------------------------------------------- reaching the engine


class _FakeLLM:
    """Records how it was called, which is the whole question here."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompts, params, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})

        class _Out:
            def __init__(self, text):
                self.outputs = [type("O", (), {"text": text})()]

        return [_Out(f"reply to {prompt}") for prompt in prompts]


def test_the_adapter_rides_along_with_every_generate_call():
    from moltrain.policy import VLLMEngine

    llm = _FakeLLM()
    engine = VLLMEngine(llm, lora_request="REQ")
    engine.generate(["a", "b"], params=None)

    assert llm.calls[0]["lora_request"] == "REQ"


def test_no_adapter_means_the_keyword_is_omitted_not_passed_as_none():
    """So a fake engine, or a vLLM too old for it, sees the old signature."""
    from moltrain.policy import VLLMEngine

    llm = _FakeLLM()
    VLLMEngine(llm).generate(["a"], params=None)
    assert "lora_request" not in llm.calls[0]


def test_swapping_the_request_changes_what_the_next_call_carries():
    """How the sync works: the engine is re-pointed between steps."""
    from moltrain.policy import VLLMEngine

    llm = _FakeLLM()
    engine = VLLMEngine(llm, lora_request="step-1")
    engine.generate(["a"], params=None)
    engine.lora_request = "step-2"
    engine.generate(["b"], params=None)

    assert [call["lora_request"] for call in llm.calls] == ["step-1", "step-2"]


def test_the_engine_ceiling_covers_the_ranks_this_package_trains():
    """vLLM's own default is 16 and these adapters are r=32.

    An adapter above the engine's `max_lora_rank` is rejected at load, which
    happens after 30B of weights are already resident.
    """
    from moltrain.policy import lora_rank

    cfg = load_config()
    assert cfg["grpo"]["lora"]["r"] == 32
    assert lora_rank(cfg) >= 32
    assert lora_rank({"grpo": {}, "sft": {}}) == 16


def test_a_bigger_rank_in_either_section_raises_the_ceiling():
    from moltrain.policy import lora_rank

    assert lora_rank({"grpo": {"lora": {"r": 8}}, "sft": {"lora": {"r": 64}}}) == 64


# --------------------------------------------------------- staying on policy


def test_the_sync_interval_defaults_to_every_step():
    """Anything else is off-policy, and GRPO's advantage assumes it is not."""
    assert DEFAULTS["grpo"]["sync_engine_every_steps"] == 1
    assert load_config()["grpo"]["sync_engine_every_steps"] == 1


class _FakePeftModel:
    def __init__(self):
        self.saved: list[str] = []

    def save_pretrained(self, path):
        self.saved.append(str(path))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}")


class _State:
    def __init__(self, step):
        self.global_step = step


def _stub_the_heavy_imports(monkeypatch):
    """Keep `transformers` and vLLM out of this process.

    `make_engine_sync` imports `TrainerCallback` from transformers, which
    imports torch -- and `test_integration` asserts torch never reaches
    sys.modules, an invariant any earlier test can break. `TrainerCallback` is
    only a base class here; every behaviour under test is this package's.
    """
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "moltrain.policy.build_lora_request",
        lambda path, *, name="policy", adapter_id=1: (str(path), adapter_id),
    )


def _sync(cfg, engine, monkeypatch):
    import protactrain.grpo as grpo

    _stub_the_heavy_imports(monkeypatch)
    return grpo.make_engine_sync(cfg, grpo.Ranks(), engine)


class _Engine:
    def __init__(self):
        self.lora_request = ("original", 0)


def test_each_step_repoints_the_engine_at_newly_saved_weights(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    engine = _Engine()
    callback = _sync(cfg, engine, monkeypatch)
    model = _FakePeftModel()

    callback.on_step_end(None, _State(1), None, model=model)
    first = engine.lora_request
    callback.on_step_end(None, _State(2), None, model=model)
    second = engine.lora_request

    assert len(model.saved) == 2
    assert first != second
    # A NEW id, or vLLM serves the tensors cached under the old one and the
    # engine goes on sampling the policy from before the step.
    assert first[1] != second[1]


def test_the_previous_steps_adapter_is_removed(tmp_path, monkeypatch):
    """Hundreds of MB per step; keeping them all fills the disk mid-run."""
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    callback = _sync(cfg, _Engine(), monkeypatch)
    model = _FakePeftModel()

    callback.on_step_end(None, _State(1), None, model=model)
    callback.on_step_end(None, _State(2), None, model=model)

    kept = sorted(p.name for p in (tmp_path / "grpo" / "_engine_rank0").iterdir())
    assert kept == ["step-2"]


def test_an_interval_above_one_syncs_only_on_its_own_steps(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    cfg["grpo"]["sync_engine_every_steps"] = 3
    engine = _Engine()
    callback = _sync(cfg, engine, monkeypatch)
    model = _FakePeftModel()

    for step in (1, 2, 3, 4, 5, 6):
        callback.on_step_end(None, _State(step), None, model=model)

    assert model.saved and len(model.saved) == 2  # steps 3 and 6


def test_a_wrapped_model_is_unwrapped_to_find_the_adapter(tmp_path, monkeypatch):
    """accelerate hands the callback a DDP wrapper under torchrun."""
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    callback = _sync(cfg, _Engine(), monkeypatch)
    inner = _FakePeftModel()
    wrapper = type("DDP", (), {"module": inner})()

    callback.on_step_end(None, _State(1), None, model=wrapper)
    assert inner.saved


def test_no_saveable_model_refuses_rather_than_silently_going_stale(
    tmp_path, monkeypatch
):
    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    callback = _sync(cfg, _Engine(), monkeypatch)

    with pytest.raises(RuntimeError) as exc:
        callback.on_step_end(None, _State(1), None, model=None)
    assert "sync_engine_every_steps" in str(exc.value)


def test_the_ranks_do_not_write_over_each_other(tmp_path, monkeypatch):
    """Both ranks hold their own engine and save their own copy."""
    import protactrain.grpo as grpo

    cfg = load_config()
    cfg["grpo"]["output_dir"] = str(tmp_path / "grpo")
    _stub_the_heavy_imports(monkeypatch)

    for rank in (0, 1):
        callback = grpo.make_engine_sync(
            cfg, grpo.Ranks(rank=rank, world_size=2), _Engine()
        )
        callback.on_step_end(None, _State(1), None, model=_FakePeftModel())

    written = sorted(p.name for p in (tmp_path / "grpo").iterdir())
    assert written == ["_engine_rank0", "_engine_rank1"]
