"""Assessment and triage tests. No GPU, no model pass.

Samples are built to Pro-GAT's layout and broken in the ways the diffusion
model breaks them, then run through the real Open Babel and RDKit path.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protacbench import structures as S
from protacbench.assess import HARD_FAILURES, assess_sample, molecule_properties
from protacbench.config import load_config
from protacbench.prompts import build_prompt, build_task
from protacbench.tests import fixtures
from protacbench.triage import (
    assess_all,
    discover,
    discover_flat,
    discover_nested,
    write_report,
)

pytestmark = pytest.mark.skipif(
    S.shutil.which("obabel") is None
    and "MOLBENCH_OBABEL_BIN" not in __import__("os").environ,
    reason="assessment needs Open Babel to perceive bonds",
)

LINKER_LEN = len(fixtures.protac_like()[1])
HEAVY_ATOMS = 43  # the fixture PROTAC


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["progat"]["output_dir"] = str(tmp_path / "samples")
    config["progat"]["linker_size"] = LINKER_LEN
    config["assess"]["repaired_dir"] = str(tmp_path / "repaired")
    config["assess"]["sdf_dir"] = str(tmp_path / "sdf")
    config["assess"]["stages"] = ["generated"]
    config["report"]["output_dir"] = str(tmp_path / "report")
    config["run"]["log_to_file"] = False
    return config


@pytest.fixture
def good_sample(tmp_path):
    smiles, linker = fixtures.protac_like()
    mol = fixtures.ordered_mol(smiles, linker)
    xyz = fixtures.write_xyz(tmp_path / "T000" / "0_.xyz", mol)
    return xyz, tmp_path / "sdf" / "0_.sdf"


# ----------------------------------------------------------------- the reader


def test_a_clean_sample_is_read_and_measured(cfg, good_sample):
    xyz, sdf = good_sample
    result = assess_sample(
        "T000", "0", xyz, sdf, linker_size=LINKER_LEN, thresholds=cfg["thresholds"]
    )
    assert result.verdict == "ok", result.failure_detail
    assert result.protac_smiles
    assert "[C]" not in result.protac_smiles, "hydrogens were not restored"
    assert result.properties["heavy_atoms"] == HEAVY_ATOMS
    assert result.properties["sa_score"] is not None
    assert result.properties["qed"] > 0


def test_hydrogens_are_restored_so_descriptors_are_real(good_sample):
    """Open Babel writes heavy atoms only, which RDKit reads as radicals."""
    from rdkit import Chem

    xyz, sdf = good_sample
    result = S.load_molecule(xyz, sdf)
    assert result.ok
    smiles = Chem.MolToSmiles(result.mol)
    assert "[N]" not in smiles and "[C]" not in smiles
    reference = Chem.MolFromSmiles(fixtures.protac_like()[0])
    assert smiles == Chem.MolToSmiles(reference)
    # The radical-laden read comes back tens of hydrogens short.
    assert molecule_properties(result.mol)["mw"] > 580


def test_atom_order_survives_conversion(good_sample):
    xyz, sdf = good_sample
    frame = S.read_xyz(xyz)
    result = S.load_molecule(xyz, sdf, expected_symbols=frame.symbols)
    assert result.ok
    assert result.warnings == []
    assert [a.GetSymbol() for a in result.mol.GetAtoms()] == frame.symbols
    assert len(frame.symbols) == HEAVY_ATOMS


# ---------------------------------------------------------------- the linker


def test_the_linker_is_recovered_by_index(cfg, good_sample):
    xyz, sdf = good_sample
    result = assess_sample(
        "T000", "0", xyz, sdf, linker_size=LINKER_LEN, thresholds=cfg["thresholds"]
    )
    assert result.linker["heavy_atoms"] == LINKER_LEN
    assert result.linker["attachments"] == 2, "a linker bridges exactly two fragments"
    assert result.linker["path_length_bonds"] is not None
    assert result.linker["exit_vector_distance"] > 0
    assert "O" in result.linker_smiles, "the PEG bridge should carry its oxygens"


def test_the_bond_span_is_reported_not_just_the_atom_count(cfg, good_sample):
    """Atom count and bond span differ, and the span is what sets geometry."""
    xyz, sdf = good_sample
    result = assess_sample(
        "T000", "0", xyz, sdf, linker_size=LINKER_LEN, thresholds=cfg["thresholds"]
    )
    assert result.linker["path_length_bonds"] >= LINKER_LEN


# --------------------------------------------------------- the failure modes


def broken(tmp_path, mode, cfg, linker_size=LINKER_LEN):
    smiles, linker = fixtures.protac_like()
    mol = fixtures.ordered_mol(smiles, linker)
    xyz = fixtures.write_xyz(tmp_path / "T000" / "9_.xyz", mol, break_mode=mode)
    return assess_sample(
        "T000",
        "9",
        xyz,
        tmp_path / "sdf" / "9_.sdf",
        linker_size=linker_size,
        thresholds=cfg["thresholds"],
    )


def test_a_disconnected_sample_says_so(tmp_path, cfg):
    result = broken(tmp_path, "disconnected", cfg)
    assert result.verdict == "broken"
    assert "disconnected" in result.failure_codes
    assert (
        "more than one piece" in result.failure_detail
        or "pieces" in result.failure_detail
    )


def test_a_collapsed_geometry_is_caught_before_bond_perception(tmp_path, cfg):
    """Open Babel infers bonds from distance, so overlapping atoms would
    otherwise produce a confidently wrong molecule rather than an error."""
    result = broken(tmp_path, "clash", cfg)
    assert result.verdict == "broken"
    assert "atom_clash" in result.failure_codes
    assert result.geometry["clashing_atom_pairs"] >= 1


def test_every_hard_failure_carries_a_reason(tmp_path, cfg):
    for mode in ("disconnected", "clash", "overcrowded"):
        result = broken(tmp_path, mode, cfg)
        if result.verdict != "broken":
            continue  # not every perturbation is guaranteed to break perception
        assert result.failure_codes
        assert result.failure_detail, f"{mode} produced no explanation"
        assert all(c in HARD_FAILURES for c in result.failure_codes)


def test_a_missing_file_is_a_verdict_not_a_crash(tmp_path, cfg):
    result = assess_sample(
        "T000",
        "7",
        tmp_path / "absent" / "7_.xyz",
        tmp_path / "sdf" / "7_.sdf",
        linker_size=LINKER_LEN,
        thresholds=cfg["thresholds"],
    )
    assert result.verdict == "broken"
    assert "missing_xyz" in result.failure_codes


def test_a_linker_longer_than_the_molecule_is_rejected(tmp_path, cfg):
    result = broken(tmp_path, None, cfg, linker_size=500)
    assert "linker_size_mismatch" in result.failure_codes


def test_soft_gates_make_a_sample_poor_not_broken(cfg, good_sample):
    xyz, sdf = good_sample
    strict = dict(cfg["thresholds"])
    strict["min_mw"] = 10_000.0
    strict["max_mw"] = 20_000.0
    result = assess_sample(
        "T000", "0", xyz, sdf, linker_size=LINKER_LEN, thresholds=strict
    )
    assert result.verdict == "poor"
    assert "mw_out_of_range" in result.failure_codes
    assert result.protac_smiles, "a poor sample is still a real molecule"


# ----------------------------------------------------------- the two layouts


def test_sampling_layout_is_discovered(tmp_path):
    root = fixtures.build_run(tmp_path / "samples", targets=2, samples=3)
    found = discover_nested(root)
    assert len(found) == 6
    assert {f.target_id for f in found} == {"T000", "T001"}
    # true_.xyz and frag_.xyz are not samples.
    assert all(f.sample_id.isdigit() for f in found)


def test_repair_layout_is_discovered_with_its_linker_length(tmp_path):
    """The repair model writes flat files, not a directory per PROTAC."""
    root = fixtures.build_repaired(tmp_path / "repaired", linker_len=LINKER_LEN)
    found = discover_flat(root.parent)
    assert len(found) == 1
    assert found[0].target_id == "T000"
    assert found[0].sample_id == "0"
    assert found[0].linker_size == LINKER_LEN


def test_discover_accepts_either_layout(tmp_path):
    nested = fixtures.build_run(tmp_path / "a", targets=1, samples=1)
    flat = fixtures.build_repaired(tmp_path / "b")
    assert len(discover(nested)) == 1
    assert len(discover(flat.parent)) == 1


# -------------------------------------------------------------- the whole run


def test_a_run_is_assessed_and_written_out(cfg, tmp_path):
    fixtures.build_run(
        Path(cfg["progat"]["output_dir"]),
        targets=2,
        samples=2,
        break_modes={(0, 1): "disconnected", (1, 1): "clash"},
    )
    assessments, report = assess_all(cfg, disable_progress=True)
    assert len(assessments) == 4

    summary = report.as_dict()
    assert summary["total_samples"] == 4
    assert summary["by_verdict"].get("ok") == 2
    assert summary["by_verdict"].get("broken") == 2
    assert "disconnected" in summary["failure_counts"]
    assert "atom_clash" in summary["failure_counts"]

    paths = write_report(cfg, assessments, report)
    rows = list(csv.DictReader(paths["samples_csv"].open()))
    assert len(rows) == 4
    assert {"target_id", "verdict", "protac_smiles", "linker_smiles"} <= set(rows[0])

    rejects = list(csv.DictReader(paths["rejects_csv"].open()))
    assert len(rejects) == 2
    assert all(r["failure_detail"] for r in rejects)
    assert all(r["repair_prompt"] for r in rejects)

    tasks = [
        json.loads(line) for line in paths["repair_tasks"].read_text().splitlines()
    ]
    assert len(tasks) == 2
    assert all("prompt" in t and t["failure_codes"] for t in tasks)

    loaded = json.loads(paths["summary_json"].read_text())
    assert loaded["total_samples"] == 4


def test_the_reference_molecule_enables_a_recovery_rate(cfg):
    """Pro-GAT hardcodes recovery to 0.0 though true_.xyz is right there."""
    fixtures.build_run(Path(cfg["progat"]["output_dir"]), targets=1, samples=1)
    assessments, report = assess_all(cfg, disable_progress=True)
    assert assessments[0].reference_smiles, "true_.xyz should have been read"
    # The fixture's sample is the reference, so it recovers it exactly.
    assert assessments[0].recovers_reference is True
    assert report.as_dict()["recovered_reference"] == 1


def test_the_repair_delta_is_counted(cfg, tmp_path):
    """A sample broken as generated and fine after repair is a rescue."""
    fixtures.build_run(
        Path(cfg["progat"]["output_dir"]),
        targets=1,
        samples=1,
        break_modes={(0, 0): "disconnected"},
    )
    fixtures.build_repaired(
        Path(cfg["assess"]["repaired_dir"]),
        target="T000",
        sample=0,
        linker_len=LINKER_LEN,
    )
    cfg["assess"]["stages"] = ["generated", "repaired"]
    _, report = assess_all(cfg, disable_progress=True)
    summary = report.as_dict()
    assert summary["by_stage"]["generated"]["broken"] == 1
    assert summary["by_stage"]["repaired"]["ok"] == 1
    assert summary["rescued_by_progat"] == 1


# ------------------------------------------------------------- the handoff


def test_the_repair_prompt_is_self_contained(tmp_path, cfg):
    result = broken(tmp_path, "disconnected", cfg)
    prompt = build_prompt(result)
    assert "disconnected" in prompt
    assert "Do not change either fragment" in prompt
    assert str(LINKER_LEN) in prompt


def test_the_prompt_only_names_tools_that_exist(tmp_path, cfg):
    """`submit_molecule` is moltrain's, not molbench's. A prompt that asks for
    it dead-ends every repair, and MCP tools arrive namespaced."""
    prompt = build_prompt(broken(tmp_path, "disconnected", cfg))
    assert "submit_molecule" not in prompt
    assert "mcp__molbench__validate_smiles" in prompt
    assert "mcp__molbench__get_molecule_metrics" in prompt


def test_the_prompt_states_a_parseable_answer_contract(tmp_path, cfg):
    """`repair.py` reads the answer off a REPAIRED: line, so the prompt has to
    ask for exactly that, including the way out when repair is impossible."""
    prompt = build_prompt(broken(tmp_path, "disconnected", cfg))
    assert "REPAIRED:" in prompt
    assert "REPAIRED: none" in prompt


def test_the_linker_is_salvaged_from_a_broken_sample(tmp_path, cfg):
    """A disconnected molecule still has its linker, and naming it is the most
    useful thing the repair agent can be handed."""
    result = broken(tmp_path, "disconnected", cfg)
    assert result.verdict == "broken"
    assert result.linker_smiles, "the linker should be recoverable by index"
    assert "O" in result.linker_smiles
    assert "could not be read" not in build_prompt(result)


def test_the_prompt_forbids_touching_the_fragments(tmp_path, cfg):
    """An agent that 'fixes' a disconnected PROTAC by deleting a fragment has
    destroyed the molecule, so the constraint has to be explicit."""
    result = broken(tmp_path, "disconnected", cfg)
    task = build_task(result)
    assert "warhead" in task["prompt"]
    assert "E3" in task["prompt"]
    assert task["linker_size"] == LINKER_LEN


def test_a_task_has_everything_needed_without_the_csv(tmp_path, cfg):
    result = broken(tmp_path, "clash", cfg)
    task = build_task(result)
    assert set(task) >= {
        "task_id",
        "target_id",
        "failure_codes",
        "failure_detail",
        "linker_size",
        "prompt",
    }


# ------------------------------------------------------------- sample reuse


def test_leftover_directories_do_not_count_as_samples(tmp_path, cfg):
    """`sample` must reuse only what `assess` can actually read.

    An interrupted run leaves the per-PROTAC directories behind without
    writing an XYZ into them. Testing the output directory for any content at
    all then reports "reusing them" for a directory assessment reads as empty,
    and the pipeline stalls with no way to tell which half is wrong.
    """
    import asyncio

    from protacbench import runner

    out = tmp_path / "samples"
    (out / "T000").mkdir(parents=True)
    (out / "T001").mkdir(parents=True)
    cfg["progat"]["output_dir"] = str(out)
    cfg["progat"]["allow_cpu"] = False

    assert discover(out, "generated") == []
    # Having found nothing reusable it goes on to sample, and refuses here
    # only because this host has no GPU. Returning the directory instead
    # would be the bug.
    with pytest.raises(runner.RunnerError, match="CUDA"):
        asyncio.run(runner.sample(cfg, refresh=False))


def test_a_directory_with_real_samples_is_reused(tmp_path, cfg):
    import asyncio

    from protacbench import runner

    out = tmp_path / "samples"
    (out / "T000").mkdir(parents=True)
    (out / "T000" / "0_.xyz").write_text("1\n\nC 0.0 0.0 0.0\n")
    cfg["progat"]["output_dir"] = str(out)
    cfg["progat"]["allow_cpu"] = False

    assert asyncio.run(runner.sample(cfg, refresh=False)) == out


# ------------------------------------------------------- the manual handoff


def _row(**overrides):
    row = {
        "target_id": "T1", "sample_id": "0", "verdict": "poor",
        "failure_codes": "qed_too_low", "failure_detail": "QED 0.03 below 0.05",
        "protac_smiles": "CCOCC", "linker_smiles": "CCOCC",
        "fragment_smiles": "c1ccccc1.c1ccncc1", "linker_size": "5",
        "reference_smiles": "SECRET-REFERENCE-SMILES",
        "recovers_reference": "False", "score_total": "71.5",
        "prop_mw": "900", "prop_sa_score": "4.1", "prop_clogp": "4.4",
        "prop_tpsa": "180", "prop_rotatable_bonds": "15",
        "linker_heavy_atoms": "5", "linker_path_length_bonds": "6",
    }
    row.update(overrides)
    return row


def test_the_published_answer_never_reaches_the_prompt():
    """`reference_smiles` is the real PROTAC that target was built from. An
    agent that sees it reads the answer off instead of repairing anything,
    and the salvage rate then measures recall."""
    from protacbench.prompts import build_prompt_from_row

    text = build_prompt_from_row(_row(failure_codes="disconnected"))
    assert "SECRET-REFERENCE-SMILES" not in text


def test_the_score_is_withheld_too():
    """A composite is gameable in a way raw descriptors are not."""
    from protacbench.prompts import build_prompt_from_row

    text = build_prompt_from_row(_row(failure_codes="disconnected"))
    assert "71.5" not in text


def test_qed_is_not_asked_about():
    """The remedy for qed_too_low is "improve drug-likeness", which directly
    contradicts the system prompt's instruction not to. The instruction is
    the correct one -- 34% of real PROTACs fail a QED gate of 0.05 -- so a
    stale code from an older CSV must be dropped, not rendered."""
    from protacbench.prompts import build_prompt_from_row

    text = build_prompt_from_row(_row())
    assert "qed" not in text.lower()
    assert "drug-likeness" not in text
    assert "QED" not in text


def test_classification_matches_the_optimisation_classes():
    from protacbench.prompts import classify_row

    assert classify_row(_row(verdict="broken", failure_codes="disconnected")) == "structural"
    assert classify_row(_row(), reactive=["aldehyde"]) == "reactive"
    assert classify_row(_row(failure_codes="pains_alert")) == "chemistry"
    assert classify_row(_row(failure_codes="qed_too_low")) == "qed_only"
    assert classify_row(_row(verdict="ok", failure_codes="")) == "clean"


def test_a_prompt_states_the_fragment_constraint():
    from protacbench.prompts import build_prompt_from_row

    text = build_prompt_from_row(_row(failure_codes="disconnected"))
    assert text.count("must survive unchanged") == 2


def test_a_prompt_without_isolable_fragments_still_warns():
    from protacbench.prompts import build_prompt_from_row

    text = build_prompt_from_row(_row(fragment_smiles="", failure_codes="disconnected"))
    assert "warhead" in text and "E3 ligand" in text


def test_the_system_prompt_forbids_the_two_things_that_matter():
    from protacbench.prompts import SYSTEM_PROMPT

    assert "NOT YOURS TO CHANGE" in SYSTEM_PROMPT
    assert "Do not\noptimise QED" in SYSTEM_PROMPT or "optimise QED" in SYSTEM_PROMPT
    assert "REPAIRED:" in SYSTEM_PROMPT


def test_prompts_command_skips_and_clears(tmp_path, cfg):
    """Stale prompts from an earlier run are indistinguishable from current
    ones once they are sitting in a folder you are working through, so the
    directory is cleared rather than added to."""
    import csv as _csv

    from protacbench.__main__ import main

    source = tmp_path / "samples.csv"
    rows = [
        _row(target_id="A", failure_codes="disconnected", verdict="broken"),
        _row(target_id="B", failure_codes="qed_too_low"),              # skipped
        _row(target_id="C", failure_codes="pains_alert"),
        _row(target_id="D", failure_codes="pains_alert",
             recovers_reference="True"),                                # skipped
    ]
    with source.open("w", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "prompts"
    (out / "tasks").mkdir(parents=True)
    (out / "tasks" / "STALE-9.txt").write_text("from a previous run")

    config = tmp_path / "cfg.yaml"
    import yaml

    config.write_text(
        yaml.safe_dump({
            "promptout": {"output_dir": str(out)},
            "run": {"log_to_file": False},
        })
    )
    assert main(["prompts", "--config", str(config), "--source", str(source),
                 "--no-progress"]) == 0

    written = sorted(p.stem for p in (out / "tasks").glob("*.txt"))
    assert written == ["A-0", "C-0"], written
    assert not (out / "tasks" / "STALE-9.txt").exists()
    assert (out / "system.txt").read_text().startswith("You are a medicinal chemist")
