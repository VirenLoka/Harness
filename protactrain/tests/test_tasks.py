"""Building the corpus: classification, ids, and what gets dropped.

These run against the real `repair_tasks.jsonl`, not a fixture file. The
class counts are the ones the reward design was argued from -- 952 controls
is why QED is excluded rather than scored -- so a test that invented its own
rows would check the code and not the claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protactrain.config import load_config, resolve  # noqa: E402
from protactrain.episode import LinkerTask  # noqa: E402
from protactrain.tasks import (  # noqa: E402
    CLASS_ORDER,
    UNGATED_CODES,
    baseline_from_row,
    build_tasks,
    classify,
    clean_codes,
    read_source,
    read_tasks,
    select_rows,
    source_path,
    task_id_for,
    unknown_codes,
    write_tasks,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def rows(cfg):
    path = source_path(cfg)
    if not path.is_file():
        pytest.skip(f"the repair corpus is not on this host: {path}")
    return read_source(path)


# -------------------------------------------------------------- the corpus


def test_the_corpus_is_the_one_the_design_was_argued_from(rows):
    assert len(rows) == 2229
    assert len({r["target_id"] for r in rows}) == 346
    assert {r["stage"] for r in rows} == {"generated"}


def test_every_failure_code_in_the_corpus_is_classified(rows):
    """An unclassified code would land its tasks in the wrong stratum
    silently, and the evaluation arms are reported per stratum."""
    assert unknown_codes(rows) == []


def test_the_class_histogram_accounts_for_every_row(cfg, rows):
    selection = select_rows(cfg, rows)
    assert selection.read == 2229
    assert len(selection.rows) + sum(selection.dropped.values()) == 2229

    counts = {name: 0 for name in CLASS_ORDER}
    for row in selection.rows:
        counts[classify(row.get("failure_codes"))] += 1
    assert sum(counts.values()) == len(selection.rows)
    # The number the reward design turns on: 952 tasks whose only recorded
    # failure is qed_too_low, which this package does not score, so they are
    # controls with genuinely nothing to fix.
    assert counts["control"] == 952
    assert counts == {
        "structural": 230,
        "reactive": 369,
        "chemistry": 639,
        "control": 952,
    }


def test_rows_without_a_usable_fragment_pair_are_dropped_and_counted(cfg, rows):
    """Every reward floor hangs on matching both fragments back, so a row
    with none or three can only ever score the floor. Dropping them is a
    decision, so it is reported rather than silently filtered."""
    selection = select_rows(cfg, rows)
    assert selection.dropped["no_molecule"] == 33
    assert selection.dropped["fragment_count_3"] == 6
    assert len(selection.rows) == 2190
    assert all(len(r["fragment_smiles"]) == 2 for r in selection.rows)


def test_a_qed_only_row_is_a_control_and_anything_else_is_not():
    assert classify(["qed_too_low"]) == "control"
    assert classify([]) == "control"
    assert classify(["qed_too_low", "clogp_too_high"]) == "chemistry"
    assert classify(["reactive_linker_group"]) == "reactive"
    assert classify(["disconnected"]) == "structural"
    # Structural beats everything: a molecule in pieces is not a chemistry
    # problem, and no SMARTS edit reconnects it.
    assert classify(["disconnected", "reactive_linker_group"]) == "structural"
    assert UNGATED_CODES == {"qed_too_low"}


def test_clean_codes_survives_the_shapes_the_corpus_actually_contains():
    assert clean_codes(None) == []
    assert clean_codes([]) == []
    assert clean_codes(["a", "", None, "a", "b"]) == ["a", "b"]


# ------------------------------------------------------------------ task ids


def test_the_generated_suffix_is_stripped_from_the_task_id(cfg):
    """GRPO rewrites ids to '{id}@{generation}' and traces to '{id}#{attempt}'.

    A third suffix already on the id makes group membership ambiguous, and
    that does not fail -- it silently destroys the advantages, because a
    group whose members do not share an id has nothing to be centred against.
    """
    row = {"task_id": "102-0-generated", "target_id": "102", "sample_id": "0"}
    assert task_id_for(cfg, row) == "102-0"
    assert "generated" not in task_id_for(cfg, row)


def test_stripping_can_be_turned_off_by_config(cfg):
    keep = {**cfg, "dataset": {**cfg["dataset"], "strip_stage_suffix": False}}
    row = {"task_id": "102-0-generated", "target_id": "102", "sample_id": "0"}
    assert task_id_for(keep, row) == "102-0-generated"


def test_task_ids_carry_neither_grpo_nor_trace_separator(cfg, rows):
    selection = select_rows(cfg, rows)
    ids = [task_id_for(cfg, row) for row in selection.rows]
    assert not [i for i in ids if "@" in i or "#" in i]
    assert len(set(ids)) == len(ids), "task ids must be unique to group rollouts"


def test_a_grpo_suffixed_id_still_names_its_own_group(cfg):
    """What the stripping is for, stated as the property it protects."""
    base = task_id_for(cfg, {"task_id": "102-0-generated", "target_id": "102",
                             "sample_id": "0"})
    rollouts = [f"{base}@{n}" for n in range(8)]
    assert {r.split("@")[0] for r in rollouts} == {base}


# ------------------------------------------------------------ the built tasks


def test_a_built_task_round_trips_through_disk(cfg, tmp_path):
    tasks, report = build_tasks(cfg, limit=5, disable_progress=True)
    assert len(tasks) == 5
    assert report.read == 2229

    path = write_tasks(tmp_path / "tasks.jsonl", tasks)
    back = read_tasks(cfg, path)
    assert [t.as_dict() for t in back] == [t.as_dict() for t in tasks]
    assert all(isinstance(t, LinkerTask) for t in back)


def test_every_built_task_carries_what_the_reward_reads(cfg):
    tasks, _report = build_tasks(cfg, limit=20, disable_progress=True)
    for task in tasks:
        assert task.task_id and task.protac_smiles
        assert len(task.fragment_smiles) == 2
        assert task.optimisation_class in CLASS_ORDER
        assert isinstance(task.baseline, dict)


def test_an_unmeasurable_baseline_is_none_everywhere_never_zero():
    """236 corpus rows have an empty properties{}. A None start is skipped by
    the reward; a zero start would be scored as progress from nothing."""
    baseline = baseline_from_row(
        {"protac_smiles": "", "properties": {}, "linker": {}, "fragment_smiles": []}
    )
    assert baseline.measurable is False
    for name in ("span", "clogp", "sa_score", "tpsa", "mw", "linker_heavy_atoms"):
        assert getattr(baseline, name) is None


def test_the_build_is_deterministic_for_one_seed(cfg):
    first, _ = build_tasks(cfg, limit=40, disable_progress=True)
    second, _ = build_tasks(cfg, limit=40, disable_progress=True)
    assert [t.task_id for t in first] == [t.task_id for t in second]


def test_read_tasks_takes_the_config_first_like_moltrain(cfg, tmp_path):
    """Four call sites across grpo, traces, evaluate and experts spell this
    the same way; the signature is what keeps them doing so."""
    tasks, _ = build_tasks(cfg, limit=2, disable_progress=True)
    path = write_tasks(tmp_path / "t.jsonl", tasks)
    assert len(read_tasks(cfg, path)) == 2
    assert len(read_tasks(cfg, path, limit=1)) == 1


def test_the_tasks_file_on_disk_matches_what_the_build_reports(cfg):
    """`tasks build` has been run, so the committed artefact is checkable."""
    path = resolve(cfg, cfg["dataset"]["tasks_file"])
    if not path.is_file():
        pytest.skip("data/tasks.jsonl has not been built on this host")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2190
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["optimisation_class"]] = counts.get(row["optimisation_class"], 0) + 1
    assert counts["control"] == 952
