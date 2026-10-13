"""Dataset tests: the CrossDocked split, DiffSBDD's argv, and task building.

Nothing here runs a diffusion pass or touches a GPU. The split fixture is a
hand-built torch-style archive, so these tests do not need torch either.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moltrain import crossdocked, diffsbdd
from moltrain import tasks as taskmod
from moltrain.config import load_config
from moltrain.crossdocked import CrossDockedError, Pocket
from moltrain.episode import Task, build_prompt
from moltrain.oracle import Properties

EXAMPLES = Path(__file__).resolve().parents[2] / "DiffSBDD" / "example"
CAFFEINE_SDF = EXAMPLES / "3rfm_B_CFF.sdf"
MACROCYCLE_SDF = EXAMPLES / "5ndu_C_8V2.sdf"

# Real entry shapes, taken from the listing of crossdocked_pocket10.tar.gz.
# The receptor is the 10 Angstrom pocket shell, and in a cross-docked pair the
# ligand's own PDB id differs from the receptor's.
SPLIT = {
    "train": [
        (
            "SYH_THET8_1_421_0/1ady_A_rec_1ady_ham_lig_tt_min_0_pocket10.pdb",
            "SYH_THET8_1_421_0/1ady_A_rec_1ady_ham_lig_tt_min_0.sdf",
        )
    ],
    "test": [
        (
            "ABL1_HUMAN_25_300_0/1iep_A_rec_1iep_sti_lig_tt_min_0_pocket10.pdb",
            "ABL1_HUMAN_25_300_0/1iep_A_rec_1iep_sti_lig_tt_min_0.sdf",
        ),
        (
            "PB2_INBLE_319_484_0/5efc_A_rec_4q46_gdp_lig_tt_min_0_pocket10.pdb",
            "PB2_INBLE_319_484_0/5efc_A_rec_4q46_gdp_lig_tt_min_0.sdf",
        ),
        (
            "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0_pocket10.pdb",
            "1B57_HUMAN_25_300_0/5u98_D_rec_5u98_1kx_lig_tt_min_0.sdf",
        ),
    ],
}

# index.pkl rows carry the full receptor path and an RMSD as well.
INDEX_ROWS = [
    (receptor, ligand, "full/" + receptor.split("/")[-1], 0.37)
    for receptor, ligand in SPLIT["test"]
]


def write_zip_split(path: Path, payload: object) -> Path:
    """A torch-style zip archive: the payload lives at `archive/data.pkl`."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(payload))
    return path


@pytest.fixture
def split_root(tmp_path: Path) -> Path:
    root = tmp_path / "crossdocked"
    (root / crossdocked.POCKET_SUBDIR).mkdir(parents=True)
    write_zip_split(root / crossdocked.SPLIT_FILE, SPLIT)
    return root


@pytest.fixture
def cfg(split_root: Path, tmp_path: Path):
    config = load_config()
    config["dataset"]["pockets"]["crossdocked_root"] = str(split_root)
    config["dataset"]["pockets"]["staging_dir"] = str(tmp_path / "pockets")
    config["dataset"]["diffsbdd"]["samples_dir"] = str(tmp_path / "samples")
    config["dataset"]["tasks_file"] = str(tmp_path / "tasks.jsonl")
    return config


# ------------------------------------------------------------ the split reader


def test_reads_a_zip_split(split_root):
    splits = crossdocked.load_split(split_root / crossdocked.SPLIT_FILE)
    assert sorted(splits) == ["test", "train"]
    assert len(splits["test"]) == 3
    assert splits["test"][0][0].endswith("1iep_sti_lig_tt_min_0_pocket10.pdb")


def test_reads_a_legacy_split(tmp_path):
    """Pre-1.6 torch wrote three headers ahead of the payload."""
    path = tmp_path / "legacy.pt"
    with path.open("wb") as handle:
        for header in (b"magic", 1001, {"protocol_version": 1001}):
            pickle.dump(header, handle)
        pickle.dump(SPLIT, handle)
    assert len(crossdocked.load_split(path)["test"]) == 3


def test_refuses_to_build_arbitrary_classes(tmp_path):
    """A split of file names has no reason to construct anything else."""
    path = write_zip_split(tmp_path / "hostile.pt", {"test": [Counter("abc")]})
    with pytest.raises(CrossDockedError, match="Refusing to unpickle"):
        crossdocked.load_split(path)


def test_missing_split_file_names_the_download(tmp_path):
    with pytest.raises(CrossDockedError, match="Pocket2Mol"):
        crossdocked.load_split(tmp_path / "absent.pt")


# ----------------------------------------------------------------------- naming


def test_pocket_naming_recovers_pdb_ligand_and_family():
    pocket = crossdocked.pocket_from_entry(Path("/root"), 7, *SPLIT["test"][0])
    assert pocket.name == "1iep-007"
    assert "1IEP" in pocket.target
    assert "ABL1_HUMAN" in pocket.target
    assert pocket.metadata["ligand_code"] == "STI"
    assert pocket.pdb_file.endswith("_pocket10.pdb")


def test_pocket_naming_handles_a_cross_docked_pair():
    """The receptor's PDB id is what was docked into, not the ligand's."""
    pocket = crossdocked.pocket_from_entry(Path("/root"), 1, *SPLIT["test"][1])
    assert pocket.metadata["pdb_id"] == "5EFC"
    assert pocket.metadata["ligand_code"] == "GDP"
    assert pocket.metadata["family"] == "PB2_INBLE"


def test_index_file_yields_receptor_ligand_pairs(tmp_path):
    path = tmp_path / crossdocked.INDEX_FILE
    path.write_bytes(pickle.dumps(INDEX_ROWS))
    pairs = crossdocked.load_index(path)
    assert len(pairs) == len(INDEX_ROWS)
    assert pairs[0] == SPLIT["test"][0]


def test_index_split_is_selectable_without_the_split_file(cfg, split_root):
    """The archive ships index.pkl but not split_by_name.pt."""
    (split_root / crossdocked.SPLIT_FILE).unlink()
    index = split_root / crossdocked.POCKET_SUBDIR / crossdocked.INDEX_FILE
    index.write_bytes(pickle.dumps(INDEX_ROWS))
    cfg["dataset"]["pockets"]["split"] = "index"
    cfg["dataset"]["pockets"]["shuffle_seed"] = None
    cfg["dataset"]["pockets"]["limit"] = None
    assert len(crossdocked.select_pockets(cfg)) == len(INDEX_ROWS)


def test_verify_root_reports_a_still_packed_archive(tmp_path):
    (tmp_path / crossdocked.ARCHIVE).write_bytes(b"not really a tarball")
    assert any("tar -xzf" in p for p in crossdocked.verify_root(tmp_path))


def test_verify_root_reports_a_doubly_nested_unpack(tmp_path):
    (tmp_path / crossdocked.POCKET_SUBDIR / crossdocked.POCKET_SUBDIR).mkdir(
        parents=True
    )
    assert any("one level too deep" in p for p in crossdocked.verify_root(tmp_path))


def test_verify_root_points_at_the_index_when_the_split_is_absent(split_root):
    (split_root / crossdocked.SPLIT_FILE).unlink()
    structures = split_root / crossdocked.POCKET_SUBDIR
    (structures / "SOME_POCKET_1_2_0").mkdir()
    (structures / crossdocked.INDEX_FILE).write_bytes(pickle.dumps(INDEX_ROWS))
    assert any("split to `index`" in p for p in crossdocked.verify_root(split_root))


def test_verify_root_is_quiet_on_a_good_layout(split_root):
    (split_root / crossdocked.POCKET_SUBDIR / "SOME_POCKET_1_2_0").mkdir()
    assert crossdocked.verify_root(split_root) == []


def test_pocket_names_stay_unique_across_the_split():
    pockets = [
        crossdocked.pocket_from_entry(Path("/root"), i, *entry)
        for i, entry in enumerate(SPLIT["test"])
    ]
    assert len({p.name for p in pockets}) == len(pockets)


# -------------------------------------------------------------- pocket selection


def test_select_pockets_applies_limit_and_is_deterministic(cfg):
    cfg["dataset"]["pockets"]["limit"] = 2
    first = crossdocked.select_pockets(cfg)
    second = crossdocked.select_pockets(cfg)
    assert len(first) == 2
    assert [p.name for p in first] == [p.name for p in second]


def test_select_pockets_without_shuffle_keeps_split_order(cfg):
    cfg["dataset"]["pockets"]["shuffle_seed"] = None
    cfg["dataset"]["pockets"]["limit"] = None
    names = [p.name for p in crossdocked.select_pockets(cfg)]
    assert names == ["1iep-000", "5efc-001", "5u98-002"]


def test_unset_crossdocked_root_explains_the_download(cfg):
    cfg["dataset"]["pockets"]["crossdocked_root"] = None
    with pytest.raises(CrossDockedError, match="crossdocked_root"):
        crossdocked.select_pockets(cfg)


def test_explicit_pockets_bypass_crossdocked(cfg):
    cfg["dataset"]["pockets"]["source"] = "explicit"
    cfg["dataset"]["pockets"]["shuffle_seed"] = None
    cfg["dataset"]["pockets"]["explicit"] = [
        {"name": "demo", "pdb_file": "/tmp/demo.pdb", "ref_ligand": "/tmp/demo.sdf"}
    ]
    pockets = crossdocked.select_pockets(cfg)
    assert [p.name for p in pockets] == ["demo"]
    assert pockets[0].ref_ligand_file == "/tmp/demo.sdf"


def test_missing_files_reports_absent_structures(cfg):
    absent = crossdocked.missing_files(crossdocked.select_pockets(cfg))
    assert absent, "the fixture has no structures on disk, so all should be listed"


# --------------------------------------------------------------- DiffSBDD argv


def pocket_with_reference() -> Pocket:
    return Pocket(
        name="demo",
        pdb_file=str(EXAMPLES / "3rfm.pdb"),
        ref_ligand_file=str(CAFFEINE_SDF),
        target="3RFM",
    )


def test_build_command_matches_the_documented_interface(cfg, tmp_path):
    command = diffsbdd.build_command(cfg, pocket_with_reference(), tmp_path / "out.sdf")
    assert command[1] == "generate_ligands.py"
    assert command[2].endswith("crossdocked_fullatom_cond.ckpt")
    assert "--ref_ligand" in command
    assert command[command.index("--n_samples") + 1] == "20"
    assert "--sanitize" in command
    assert "--relax" not in command


def test_build_command_passes_a_residue_list_when_there_is_no_ligand(cfg, tmp_path):
    pocket = Pocket(
        name="demo",
        pdb_file=str(EXAMPLES / "3rfm.pdb"),
        metadata={"resi_list": ["A:1", "A:2"]},
    )
    command = diffsbdd.build_command(cfg, pocket, tmp_path / "out.sdf")
    assert command[
        command.index("--resi_list") + 1 : command.index("--resi_list") + 3
    ] == [
        "A:1",
        "A:2",
    ]


def test_build_command_refuses_a_pocket_with_no_definition(cfg, tmp_path):
    pocket = Pocket(name="demo", pdb_file=str(EXAMPLES / "3rfm.pdb"))
    with pytest.raises(diffsbdd.DiffSBDDError, match="resi_list"):
        diffsbdd.build_command(cfg, pocket, tmp_path / "out.sdf")


def test_sampling_refuses_without_a_gpu(cfg, monkeypatch):
    """A diffusion pass on CPU takes hours; failing fast beats looking hung."""

    async def no_gpu(_cfg):
        return diffsbdd.Runtime(
            python="python",
            repo=".",
            ok=False,
            cuda_available=False,
            problems=["no CUDA device"],
        )

    monkeypatch.setattr(diffsbdd, "probe", no_gpu)
    with pytest.raises(diffsbdd.DiffSBDDError, match="CUDA"):
        asyncio.run(diffsbdd.sample_pocket(cfg, pocket_with_reference()))


def test_sampling_reuses_an_existing_sdf(cfg):
    """Resuming a long run must not re-sample pockets already on disk."""
    directory = diffsbdd.samples_dir(cfg, create=True)
    existing = directory / "demo.sdf"
    shutil.copy(CAFFEINE_SDF, existing)
    assert asyncio.run(diffsbdd.sample_pocket(cfg, pocket_with_reference())) == existing


def test_read_sdf_smiles_keeps_the_largest_fragment():
    accepted, rejected = diffsbdd.read_sdf_smiles(CAFFEINE_SDF)
    assert accepted == ["Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
    assert rejected == []


# ------------------------------------------------------------------- the boxes


def test_box_clamps_a_small_ligand_to_the_minimum(cfg):
    box = taskmod.ligand_box(cfg, CAFFEINE_SDF)
    assert box.size == (18.0, 18.0, 18.0)
    assert box.ligand_atoms > 0


def test_box_wraps_a_large_ligand(cfg):
    box = taskmod.ligand_box(cfg, MACROCYCLE_SDF)
    assert max(box.size) > 18.0
    assert max(box.size) <= cfg["dataset"]["box"]["max_size"]


def test_box_respects_the_ceiling(cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["dataset"]["box"]["max_size"] = 19.0
    assert taskmod.ligand_box(cfg, MACROCYCLE_SDF).size == (19.0, 19.0, 19.0)


def test_box_rejects_a_file_with_no_structure(cfg, tmp_path):
    empty = tmp_path / "empty.sdf"
    empty.write_text("")
    with pytest.raises(taskmod.TaskBuildError, match="reference ligand"):
        taskmod.ligand_box(cfg, empty)


def test_reference_smiles_reads_the_crystal_ligand():
    assert taskmod.reference_smiles(CAFFEINE_SDF) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


# ----------------------------------------------------------------- the filters


def test_filters_reject_by_size_ring_and_duplication(cfg):
    kept, rejected = taskmod.filter_candidates(
        cfg,
        [
            "CCO",  # too small
            "c1ccccc1C(=O)Nc1ccccc1",  # keeps
            "c1ccccc1C(=O)Nc1ccccc1",  # duplicate
            "CCCCCCCCCCCCCCCC",  # no ring
            "not-a-molecule",
        ],
        server=None,
    )
    assert kept == ["O=C(Nc1ccccc1)c1ccccc1"]
    assert rejected["too_small"] == 1
    assert rejected["duplicate"] == 1
    assert rejected["no_ring"] == 1
    assert rejected["unparseable"] == 1


def test_filters_cap_the_starts_per_pocket(cfg):
    cfg["dataset"]["filters"]["starts_per_pocket"] = 1
    kept, _ = taskmod.filter_candidates(
        cfg, ["c1ccccc1C(=O)Nc1ccccc1", "c1ccccc1C(=O)Nc1ccccn1"], server=None
    )
    assert len(kept) == 1


def test_smiles_file_skips_comments_and_trailing_names(cfg, tmp_path):
    path = tmp_path / "starts.smi"
    path.write_text("# header\n\nCCO ethanol\nc1ccccc1\n")
    cfg["dataset"]["smiles_file"] = str(path)
    assert taskmod.load_smiles_file(cfg) == ["CCO", "c1ccccc1"]


# ------------------------------------------------------------- the tasks file


def test_tasks_round_trip_through_jsonl(cfg):
    original = Task(
        task_id="demo-00",
        smiles="CCO",
        target="3RFM",
        receptor_file="/tmp/r.pdbqt",
        origin="diffsbdd",
        baseline=Properties(smiles="CCO", valid=True, sa_score=2.1, qed=0.4),
        metadata={"pocket": {"name": "demo"}},
    )
    taskmod.write_tasks(cfg, [original])
    restored = taskmod.read_tasks(cfg)
    assert len(restored) == 1
    assert restored[0].as_dict() == original.as_dict()
    assert restored[0].baseline.sa_score == 2.1


def test_reading_a_missing_tasks_file_says_how_to_make_one(cfg, tmp_path):
    cfg["dataset"]["tasks_file"] = str(tmp_path / "absent.jsonl")
    with pytest.raises(taskmod.TaskBuildError, match="moltrain.tasks build"):
        taskmod.read_tasks(cfg)


# ------------------------------------------------------------ the whole build


@pytest.mark.skipif(
    shutil.which("obabel") is None, reason="receptor preparation needs Open Babel"
)
def test_builds_tasks_from_the_example_pockets(cfg, tmp_path):
    """End to end with no GPU: real pockets, real receptors, no docking."""
    cfg["dataset"]["source"] = "reference_ligand"
    cfg["dataset"]["baseline"]["dock"] = False
    cfg["dataset"]["pockets"]["source"] = "explicit"
    cfg["dataset"]["pockets"]["shuffle_seed"] = None
    cfg["dataset"]["pockets"]["explicit"] = [
        {
            "name": "3rfm-cff",
            "pdb_file": str(EXAMPLES / "3rfm.pdb"),
            "ref_ligand": str(CAFFEINE_SDF),
            "target": "3RFM",
        }
    ]
    cfg["molbench"]["receptor_dir"] = str(tmp_path / "receptors")
    cfg["reward"]["property_cache_path"] = str(tmp_path / "properties.jsonl")
    cfg["reward"]["docking"]["cache_path"] = str(tmp_path / "docking.jsonl")

    built, report = asyncio.run(taskmod.build_tasks(cfg))
    assert report.pockets_failed == []
    assert len(built) == 1
    task = built[0]
    assert task.smiles == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine, 3RFM's CFF
    assert task.baseline.valid and task.baseline.sa_score is not None
    assert Path(task.receptor_file).is_file()
    assert task.metadata["receptor"]["box_size"] == [18.0, 18.0, 18.0]

    # The built task has to survive the round trip and drive an episode.
    taskmod.write_tasks(cfg, built)
    restored = taskmod.read_tasks(cfg)[0]
    prompt = build_prompt(cfg, restored)
    assert restored.smiles in prompt
    assert "3RFM" in prompt
    assert str(restored.baseline.sa_score) in prompt


def test_a_failing_pocket_does_not_abort_the_rest(cfg, tmp_path):
    cfg["dataset"]["source"] = "reference_ligand"
    cfg["dataset"]["baseline"]["dock"] = False
    cfg["dataset"]["pockets"]["source"] = "explicit"
    cfg["dataset"]["pockets"]["shuffle_seed"] = None
    cfg["dataset"]["pockets"]["explicit"] = [
        {"name": "gone", "pdb_file": str(tmp_path / "absent.pdb"), "ref_ligand": None}
    ]
    cfg["reward"]["property_cache_path"] = str(tmp_path / "properties.jsonl")
    cfg["reward"]["docking"]["cache_path"] = str(tmp_path / "docking.jsonl")

    built, report = asyncio.run(taskmod.build_tasks(cfg))
    assert built == []
    assert report.pockets_failed[0]["pocket"] == "gone"
    assert "absent.pdb" in report.pockets_failed[0]["error"]
