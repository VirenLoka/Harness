"""Which weights the launcher actually serves.

`--enable-lora` reserves slots and kernels and nothing else: without an
adapter named on the command line and requested by name, vLLM decodes from
the base model and says nothing about it. Every test here exists because that
failure is invisible from the outside -- the server is healthy, the responses
are fluent, and the policy never trained.

So these assert on the command that gets run, not on the pieces that build
it. Deleting the `lora_cli_args` call from `build_vllm_command` leaves every
helper below passing and the server serving base weights, which is exactly
the bug; the tests that would fail are the ones that read the argv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve  # noqa: E402
from serve import (  # noqa: E402
    ServeError,
    adapter_rank,
    build_vllm_command,
    latest_adapter,
    load_config,
    lora_cli_args,
    resolve_lora_modules,
)


def write_adapter(directory: Path, rank: int = 32) -> Path:
    """An adapter on disk is `adapter_config.json`, not a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank}))
    return directory


def make_config(tmp_path: Path, **sections) -> dict:
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "vllm_args": {"served_model_name": "muse-glimmer-30b"},
    }
    body.update(sections)
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(body))
    return load_config(path)


def flags(cfg: dict) -> list[str]:
    found, _ = resolve_lora_modules(cfg)
    return build_vllm_command(cfg, found)


# ------------------------------------------------- finding the adapter on disk


def test_a_saved_adapter_is_the_adapter(tmp_path):
    write_adapter(tmp_path / "grpo")
    assert latest_adapter(tmp_path / "grpo") == tmp_path / "grpo"


def test_an_empty_run_directory_holds_no_adapter(tmp_path):
    """A run directory exists as soon as anything writes a log into it."""
    (tmp_path / "grpo").mkdir()
    (tmp_path / "grpo" / "trainer_state.json").write_text("{}")
    assert latest_adapter(tmp_path / "grpo") is None


def test_a_stopped_run_is_found_through_its_checkpoint(tmp_path):
    """A warmup stopped on entropy never reaches `save_model`."""
    write_adapter(tmp_path / "sft" / "checkpoint-300")
    assert latest_adapter(tmp_path / "sft") == tmp_path / "sft" / "checkpoint-300"


def test_checkpoints_are_ordered_by_step_not_by_name(tmp_path):
    """`checkpoint-1000` sorts before `checkpoint-300` as a string."""
    write_adapter(tmp_path / "sft" / "checkpoint-300")
    write_adapter(tmp_path / "sft" / "checkpoint-1000")
    assert latest_adapter(tmp_path / "sft").name == "checkpoint-1000"


def test_a_finished_adapter_beats_its_own_checkpoints(tmp_path):
    write_adapter(tmp_path / "sft")
    write_adapter(tmp_path / "sft" / "checkpoint-300")
    assert latest_adapter(tmp_path / "sft") == tmp_path / "sft"


def test_the_engine_sync_copy_is_the_last_resort(tmp_path):
    """An interrupted GRPO run's only weights are the ones vLLM sampled."""
    write_adapter(tmp_path / "grpo" / "_engine_rank0" / "step-120")
    write_adapter(tmp_path / "grpo" / "_engine_rank0" / "step-40")
    found = latest_adapter(tmp_path / "grpo")
    assert found == tmp_path / "grpo" / "_engine_rank0" / "step-120"


def test_rank_one_is_not_consulted(tmp_path):
    """The ranks hold identical weights; picking one on purpose beats luck."""
    write_adapter(tmp_path / "grpo" / "_engine_rank1" / "step-120")
    assert latest_adapter(tmp_path / "grpo") is None


def test_a_checkpoint_without_an_adapter_is_skipped(tmp_path):
    (tmp_path / "sft" / "checkpoint-1000").mkdir(parents=True)
    write_adapter(tmp_path / "sft" / "checkpoint-300")
    assert latest_adapter(tmp_path / "sft").name == "checkpoint-300"


# ------------------------------------------------ the copy cannot drift silently


def test_this_resolver_agrees_with_protactrains(tmp_path):
    """serve.py keeps its own copy; a copy that drifts is worse than none.

    It is a copy because importing anything under `protactrain` runs that
    package's `__init__`, which sets OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF
    and the vLLM multiprocessing method through `os.environ.setdefault` -- and
    the launcher hands its own environment to the vLLM subprocess, so the
    import alone would re-tune the server's CPU threading. That import is
    fine here, inside a test process that serves nothing.
    """
    from protactrain.config import latest_adapter as theirs

    trees = []

    empty = tmp_path / "empty"
    empty.mkdir()
    trees.append(empty)

    saved = write_adapter(tmp_path / "saved").parent
    trees.append(saved)

    stepped = tmp_path / "stepped"
    write_adapter(stepped / "checkpoint-300")
    write_adapter(stepped / "checkpoint-1000")
    trees.append(stepped)

    synced = tmp_path / "synced"
    write_adapter(synced / "_engine_rank0" / "step-7")
    trees.append(synced)

    both = tmp_path / "both"
    write_adapter(both)
    write_adapter(both / "checkpoint-9")
    trees.append(both)

    trees.append(tmp_path / "does-not-exist")

    for tree in trees:
        assert latest_adapter(tree) == theirs(tree), tree


# -------------------------------------------------------------- the module list


def test_no_modules_means_no_lora_flags_at_all(tmp_path):
    """The base-only server is still the base-only server."""
    cfg = make_config(tmp_path, lora={"modules": {}})
    assert not [arg for arg in flags(cfg) if "lora" in arg]


def test_a_module_reaches_the_command_line(tmp_path):
    """The adapter is nested on purpose.

    With `adapter_config.json` at the top of the run directory, the directory
    searched and the adapter resolved are the same path, and the suite cannot
    tell them apart -- `found.append((name, directory))` would pass every test
    while handing vLLM a path it cannot load from, at the rank-16 floor
    because `adapter_rank` finds no config there either.
    """
    run = tmp_path / "sft"
    write_adapter(run / "checkpoint-300", rank=32)
    cfg = make_config(tmp_path, lora={"modules": {"protac-policy": str(run)}})
    argv = flags(cfg)
    assert "--enable-lora" in argv
    assert f"protac-policy={run / 'checkpoint-300'}" in argv
    assert argv[argv.index("--max-lora-rank") + 1] == "32"


def test_the_module_list_is_the_last_thing_on_the_command_line(tmp_path):
    """`--lora-modules` takes a variable number of values."""
    adapter = write_adapter(tmp_path / "grpo")
    cfg = make_config(
        tmp_path,
        vllm_args={"served_model_name": "muse-glimmer-30b", "dtype": "bfloat16"},
        lora={"modules": {"protac-policy": str(adapter)}},
    )
    argv = flags(cfg)
    assert argv.index("--lora-modules") == len(argv) - 2
    assert argv[-1].startswith("protac-policy=")


def test_a_list_of_directories_is_tried_in_order(tmp_path):
    """One name follows the policy across stages without a config edit."""
    write_adapter(tmp_path / "sft")
    cfg = make_config(
        tmp_path,
        lora={"modules": {"protac-policy": [str(tmp_path / "grpo"), str(tmp_path / "sft")]}},
    )
    assert f"protac-policy={tmp_path / 'sft'}" in flags(cfg)


def test_the_earlier_directory_wins_once_it_has_an_adapter(tmp_path):
    write_adapter(tmp_path / "grpo")
    write_adapter(tmp_path / "sft")
    cfg = make_config(
        tmp_path,
        lora={"modules": {"protac-policy": [str(tmp_path / "grpo"), str(tmp_path / "sft")]}},
    )
    assert f"protac-policy={tmp_path / 'grpo'}" in flags(cfg)


def test_several_modules_are_served_together(tmp_path):
    write_adapter(tmp_path / "grpo")
    write_adapter(tmp_path / "sft")
    cfg = make_config(
        tmp_path,
        lora={
            "modules": {
                "protac-policy": str(tmp_path / "grpo"),
                "protac-warmup": str(tmp_path / "sft"),
            }
        },
    )
    argv = flags(cfg)
    assert f"protac-policy={tmp_path / 'grpo'}" in argv
    assert f"protac-warmup={tmp_path / 'sft'}" in argv


def test_a_relative_path_resolves_against_the_repository(tmp_path):
    """Not the cwd: the launcher is started from wherever is convenient."""
    cfg = make_config(tmp_path, lora={"modules": {"p": "protactrain/runs/grpo"}, "required": False})
    _, missing = resolve_lora_modules(cfg)
    assert missing[0][1] == [serve.SCRIPT_DIR / "protactrain" / "runs" / "grpo"]


# ------------------------------------------------------------ refusing to serve


def test_a_missing_adapter_refuses_and_names_everywhere_it_looked(tmp_path):
    cfg = make_config(
        tmp_path,
        lora={"modules": {"protac-policy": [str(tmp_path / "grpo"), str(tmp_path / "sft")]}},
    )
    with pytest.raises(ServeError) as exc:
        resolve_lora_modules(cfg)
    assert str(tmp_path / "grpo") in str(exc.value)
    assert str(tmp_path / "sft") in str(exc.value)


def test_required_false_comes_up_without_it(tmp_path):
    cfg = make_config(
        tmp_path, lora={"modules": {"protac-policy": str(tmp_path / "grpo")}, "required": False}
    )
    found, missing = resolve_lora_modules(cfg)
    assert found == []
    assert [name for name, _ in missing] == ["protac-policy"]
    assert not [arg for arg in build_vllm_command(cfg, found) if "lora" in arg]


def test_required_false_still_serves_what_did_resolve(tmp_path):
    write_adapter(tmp_path / "sft")
    cfg = make_config(
        tmp_path,
        lora={
            "modules": {
                "protac-policy": str(tmp_path / "grpo"),
                "protac-warmup": str(tmp_path / "sft"),
            },
            "required": False,
        },
    )
    argv = flags(cfg)
    assert f"protac-warmup={tmp_path / 'sft'}" in argv
    assert not [arg for arg in argv if arg.startswith("protac-policy=")]


def test_the_dry_run_refuses_exactly_what_the_real_run_refuses(tmp_path, capsys):
    """A dry run that prints a command the real run rejects is a lie."""
    path = tmp_path / "serve.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "meta-models/Muse-Glimmer-30B",
                "lora": {"modules": {"protac-policy": str(tmp_path / "grpo")}},
            }
        )
    )
    assert serve.main(["--config", str(path), "--dry-run"]) == 2
    assert "vllm" not in capsys.readouterr().out


def test_the_dry_run_prints_the_adapter_it_resolved(tmp_path, capsys):
    """The refusal test covers main's no; this covers main's yes.

    Everything else asserting on argv calls `build_vllm_command` itself, so
    dropping `found` at main's one call site -- the only path that ever
    reaches a real `vllm serve` -- left the whole suite green while the
    startup log, the endpoint file and `models` all went on advertising an
    adapter the server was not loading.
    """
    adapter = write_adapter(tmp_path / "grpo")
    path = tmp_path / "serve.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": "meta-models/Muse-Glimmer-30B",
                "lora": {"modules": {"protac-policy": str(adapter)}},
            }
        )
    )
    assert serve.main(["--config", str(path), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "--enable-lora" in printed
    assert f"protac-policy={adapter}" in printed


def test_a_name_that_would_break_the_pair_is_refused(tmp_path):
    """Entries are `name=path`; a name with '=' parses as a different pair."""
    adapter = write_adapter(tmp_path / "grpo")
    for bad in ("po=licy", "pol,icy", "pol icy", "runs/policy", "pol\tis", "-policy"):
        cfg = make_config(tmp_path, lora={"modules": {bad: str(adapter)}})
        with pytest.raises(ServeError):
            resolve_lora_modules(cfg)


def test_an_adapter_path_that_would_break_the_pair_is_refused(tmp_path):
    """vLLM reads `name=path` only when the item holds no comma.

    A comma anywhere sends the item down its JSON branch and exits 2; a second
    '=' reaches `name, path = item.split("=")` and raises inside argparse.
    Both land after the base model is resident, so they are caught here where
    `--dry-run` can report them. `lr=3e-6,r32` is an ordinary sweep directory.
    """
    for bad in ("lr=3e-6", "bs8,r32"):
        run = write_adapter(tmp_path / bad / "grpo")
        cfg = make_config(tmp_path, lora={"modules": {"p": str(run)}})
        with pytest.raises(ServeError) as exc:
            resolve_lora_modules(cfg)
        assert str(run) in str(exc.value)


def test_an_empty_name_is_refused(tmp_path):
    adapter = write_adapter(tmp_path / "grpo")
    cfg = make_config(tmp_path, lora={"modules": {"": str(adapter)}})
    with pytest.raises(ServeError):
        resolve_lora_modules(cfg)


def test_a_name_colliding_with_the_base_model_is_refused(tmp_path):
    """Clients could not address one without the other."""
    adapter = write_adapter(tmp_path / "grpo")
    cfg = make_config(
        tmp_path, lora={"modules": {"muse-glimmer-30b": str(adapter)}}
    )
    with pytest.raises(ServeError) as exc:
        resolve_lora_modules(cfg)
    assert "base model" in str(exc.value)


def test_the_collision_check_reads_a_list_of_served_names(tmp_path):
    adapter = write_adapter(tmp_path / "grpo")
    cfg = make_config(
        tmp_path,
        vllm_args={"served_model_name": ["glimmer", "muse"]},
        lora={"modules": {"muse": str(adapter)}},
    )
    with pytest.raises(ServeError):
        resolve_lora_modules(cfg)


def test_modules_must_be_a_mapping(tmp_path):
    cfg = make_config(tmp_path, lora={"modules": ["protac-policy"]})
    with pytest.raises(ServeError):
        resolve_lora_modules(cfg)


def test_a_module_naming_no_directory_is_refused(tmp_path):
    cfg = make_config(tmp_path, lora={"modules": {"protac-policy": []}})
    with pytest.raises(ServeError):
        resolve_lora_modules(cfg)


def test_the_launcher_owns_the_lora_flags(tmp_path):
    """A hand-written `--lora-modules` would win over the resolved one."""
    for flag in ("enable_lora", "lora_modules", "max_lora_rank", "max_loras"):
        path = tmp_path / "serve.yaml"
        path.write_text(yaml.safe_dump({"vllm_args": {flag: "whatever"}}))
        with pytest.raises(ServeError) as exc:
            load_config(path)
        assert flag in str(exc.value)


# ------------------------------------------------------------------- the rank


def test_the_rank_is_read_off_the_adapter(tmp_path):
    """vLLM rejects an adapter above the ceiling; its own default is 16."""
    adapter = write_adapter(tmp_path / "grpo", rank=32)
    cfg = make_config(tmp_path, lora={"modules": {"p": str(adapter)}})
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "32"


def test_the_ceiling_never_drops_below_vllms_own_default(tmp_path):
    adapter = write_adapter(tmp_path / "grpo", rank=8)
    cfg = make_config(tmp_path, lora={"modules": {"p": str(adapter)}})
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "16"


def test_the_largest_rank_across_adapters_wins(tmp_path):
    write_adapter(tmp_path / "grpo", rank=32)
    write_adapter(tmp_path / "sft", rank=64)
    cfg = make_config(
        tmp_path,
        lora={"modules": {"a": str(tmp_path / "grpo"), "b": str(tmp_path / "sft")}},
    )
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "64"


def test_an_explicit_ceiling_overrides_what_is_on_disk(tmp_path):
    adapter = write_adapter(tmp_path / "grpo", rank=32)
    cfg = make_config(
        tmp_path, lora={"modules": {"p": str(adapter)}, "max_rank": 64}
    )
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "64"


def test_an_explicit_ceiling_never_caps_below_the_weights(tmp_path):
    """The knob exists to raise the ceiling, not to make the adapter unloadable.

    vLLM rejects a rank-32 adapter against a rank-16 ceiling only once the
    base model is resident, and the rank is sitting in adapter_config.json
    where this module already reads it.
    """
    adapter = write_adapter(tmp_path / "grpo", rank=32)
    cfg = make_config(tmp_path, lora={"modules": {"p": str(adapter)}, "max_rank": 16})
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "32"


def test_an_unreadable_adapter_config_does_not_take_the_server_down(tmp_path):
    """It is metadata for sizing a flag, not the weights."""
    adapter = tmp_path / "grpo"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{not json")
    assert adapter_rank(adapter) is None
    cfg = make_config(tmp_path, lora={"modules": {"p": str(adapter)}})
    argv = flags(cfg)
    assert argv[argv.index("--max-lora-rank") + 1] == "16"


def test_max_loras_reaches_the_command_line(tmp_path):
    adapter = write_adapter(tmp_path / "grpo")
    cfg = make_config(
        tmp_path, lora={"modules": {"p": str(adapter)}, "max_loras": 4}
    )
    argv = flags(cfg)
    assert argv[argv.index("--max-loras") + 1] == "4"


def test_nothing_resolved_means_no_rank_flag_either(tmp_path):
    assert lora_cli_args(make_config(tmp_path, lora={"modules": {}}), []) == []


# ------------------------------------------------------- the packaged config


def test_the_packaged_config_serves_the_trained_policy(tmp_path):
    """The default is not the base model.

    If this ever reads as an empty mapping again, the server that gets run on
    the box is the one that silently serves untrained weights.
    """
    cfg = load_config(serve.DEFAULT_CONFIG)
    modules = cfg["lora"]["modules"]
    assert modules, "configs/default.yaml serves no adapter"
    targets = [t for value in modules.values() for t in
               (value if isinstance(value, list) else [value])]
    assert any("grpo" in t for t in targets)
    assert any("sft" in t for t in targets)
    assert cfg["lora"]["required"] is True


def test_the_packaged_config_lets_vllm_read_the_model(tmp_path):
    """Without it vLLM cannot inspect the architecture, let alone load it."""
    cfg = load_config(serve.DEFAULT_CONFIG)
    assert cfg["vllm_args"]["trust_remote_code"] is True
