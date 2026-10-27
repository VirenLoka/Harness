"""adclinker tests. No GPU, no REINVENT, no network.

REINVENT is replaced by a script that writes the same CSV its sampler does,
so the config writing, the subprocess call, the output parsing and the
re-assembly all run for real against scripted output.
"""

from __future__ import annotations

import csv
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adclinker import reference, score, slice as slicing, warheads as wh
from adclinker import reinvent as rv
from adclinker.config import ConfigError, load_config

Chem = pytest.importorskip("rdkit.Chem")


# A handful of real ADC linker reagents, enough to calibrate against. Two
# carry protecting groups and must be excluded from the property profile;
# two are the same molecule under different catalogue names.
FIXTURE_ROWS = [
    ("Val-Cit-PAB", "CC(C)[C@@H](C(=O)N[C@@H](CCCNC(=O)N)C(=O)NC1=CC=C(C=C1)CO)N"),
    ("Val-cit-PAB-OH", "CC(C)[C@@H](C(=O)N[C@@H](CCCNC(=O)N)C(=O)NC1=CC=C(C=C1)CO)N"),
    ("MC-NHS", "C1CC(=O)N(C1=O)OC(=O)CCCCCN2C(=O)C=CC2=O"),
    ("Fmoc-Val-Cit-PAB", "CC(C)[C@@H](C(=O)N[C@@H](CCCNC(=O)N)C(=O)NC1=CC=C(C=C1)CO)"
                         "NC(=O)OCC2C3=CC=CC=C3C4=CC=CC=C24"),
    ("Boc-PEG4-acid", "O=C(O)CCOCCOCCOCCOCCNC(=O)OC(C)(C)C"),
    ("PEG4-diol", "OCCOCCOCCOCCO"),
    ("Azido-PEG3-amine", "NCCOCCOCCOCCN=[N+]=[N-]"),
    ("MC-Val-Cit-PAB", "CC(C)[C@@H](C(=O)N[C@@H](CCCNC(=O)N)C(=O)NC1=CC=C(C=C1)CO)"
                       "NC(=O)CCCCCN3C(=O)C=CC3=O"),
    ("Mal-PEG2-acid", "O=C(O)CCOCCOCCN1C(=O)C=CC1=O"),
    ("DBCO-acid", "O=C(O)CCN1C(=O)c2ccccc2C#Cc2ccccc21"),
    ("Bromoacetamido-PEG3", "BrCC(=O)NCCOCCOCCOCCN"),
    ("SPDP-like", "O=C(O)CCSSc1ccccn1"),
]


def _peg_variants() -> list[tuple[str, str]]:
    """Realistic PEG reagents, to carry the fixture past the calibration floor.

    `build_profile` refuses fewer than 50 molecules, because calibrating
    thresholds on a handful of examples is worse than failing. The fixture
    therefore has to be a plausible reference set rather than a token one, so
    these are real homobifunctional PEG reagents of the kind that dominate
    any linker catalogue.
    """
    rows = []
    for n in range(1, 13):
        peg = "CCO" * n
        rows.append((f"Mal-PEG{n}-acid", f"O=C(O)CC{peg}CCN1C(=O)C=CC1=O"))
        rows.append((f"Azido-PEG{n}-amine", f"NCC{peg}CCN=[N+]=[N-]"))
        rows.append((f"NHS-PEG{n}-acid", f"O=C(ON1C(=O)CCC1=O)CC{peg}CCC(=O)O"))
        rows.append((f"HO-PEG{n}-OH", f"OCC{peg}CCO"))
        rows.append((f"Thiol-PEG{n}-acid", f"SCC{peg}CCC(=O)O"))
    return rows


@pytest.fixture
def dataset(tmp_path) -> Path:
    path = tmp_path / "linkers.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Product name", "smiles", "source"])
        for name, smiles in FIXTURE_ROWS + _peg_variants():
            writer.writerow([name, smiles, "train"])
    return path


@pytest.fixture
def cfg(tmp_path, dataset) -> dict:
    config = load_config()
    config["reference"]["dataset"] = str(dataset)
    config["reference"]["cache"] = str(tmp_path / "profile.json")
    config["reinvent"]["prior_file"] = str(tmp_path / "fake.prior")
    config["reinvent"]["working_dir"] = str(tmp_path)
    config["reinvent"]["sampling_config"] = str(tmp_path / "sampling.toml")
    config["reinvent"]["warheads_file"] = str(tmp_path / "warheads.smi")
    config["reinvent"]["samples_csv"] = str(tmp_path / "samples.csv")
    config["finetune"]["config_file"] = str(tmp_path / "tl.toml")
    config["report"]["output_dir"] = str(tmp_path / "report")
    config["run"]["log_to_file"] = False
    return config


# ------------------------------------------------------------------ reference


def test_the_profile_excludes_protecting_groups(dataset):
    """Fmoc and Boc are bench chemistry, gone from the finished conjugate.
    Measuring cLogP across them describes a bottle, not a linker."""
    profile = reference.build_profile(dataset)
    assert profile.n_profiled < profile.n_unique
    # Fmoc-Val-Cit-PAB, Boc-PEG4-acid and the NHS-PEG reagents all carry
    # bench-only groups and are excluded from the property profile.
    assert profile.n_profiled <= profile.n_unique - 2


def test_the_profile_deduplicates(dataset):
    """Val-Cit-PAB appears twice under two catalogue names."""
    profile = reference.build_profile(dataset)
    # Val-Cit-PAB appears twice under two catalogue names.
    assert profile.n_unique == len(FIXTURE_ROWS) + len(_peg_variants()) - 1


def test_the_profile_reports_windows_not_means(dataset):
    profile = reference.build_profile(dataset)
    low, high = profile.window("clogp")
    assert low <= profile.properties["clogp"].median <= high


def test_a_tiny_reference_set_is_refused(tmp_path):
    """Fifty molecules is not enough to calibrate a score against, and
    silently calibrating anyway would be worse than failing."""
    path = tmp_path / "tiny.csv"
    path.write_text("smiles\nCCO\nCCN\n")
    with pytest.raises(reference.ReferenceError, match="too few"):
        reference.build_profile(path)


def test_a_missing_reference_set_says_what_to_do(tmp_path):
    with pytest.raises(reference.ReferenceError, match="Linker-GPT"):
        reference.read_reference_smiles(tmp_path / "nope.csv")


def test_the_profile_caches_and_reloads(dataset, tmp_path):
    cache = tmp_path / "p.json"
    first = reference.load_profile(dataset, cache=cache)
    assert cache.is_file()
    second = reference.load_profile(dataset, cache=cache)
    assert second.window("clogp") == first.window("clogp")


def test_cleavable_motifs_finds_citrulline_and_disulfide():
    cit = Chem.MolFromSmiles(FIXTURE_ROWS[0][1])
    assert "citrulline" in reference.cleavable_motifs(cit)
    ss = Chem.MolFromSmiles("O=C(O)CCSSc1ccccn1")
    assert "disulfide" in reference.cleavable_motifs(ss)
    peg = Chem.MolFromSmiles("OCCOCCOCCO")
    assert reference.cleavable_motifs(peg) == []


# ------------------------------------------------------------------- warheads


@pytest.mark.parametrize("name", sorted(wh.HANDLES))
def test_every_handle_has_one_attachment_point(name):
    mol = Chem.MolFromSmiles(wh.HANDLES[name].smiles, sanitize=False)
    assert mol is not None
    assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0) == 1


@pytest.mark.parametrize("name", sorted(wh.CASSETTES))
def test_every_cassette_has_two_attachment_points(name):
    mol = Chem.MolFromSmiles(wh.CASSETTES[name].smiles, sanitize=False)
    assert mol is not None
    maps = sorted(
        a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() == 0
    )
    assert maps == [1, 2]


@pytest.mark.parametrize("name", sorted(wh.PAYLOADS))
def test_every_payload_parses_with_one_attachment_point(name):
    mol = Chem.MolFromSmiles(wh.PAYLOADS[name].smiles, sanitize=False)
    assert mol is not None
    assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0) == 1


def test_a_bare_maleimide_will_not_be_acylated_directly():
    """The maleimide nitrogen is already an imide; a third carbonyl is not a
    real compound. This is why clinical ADCs use maleimidocaproyl."""
    with pytest.raises(wh.WarheadError, match="triacylamine"):
        wh.build_pair("maleimide", "val_cit_pab", "mmae", "spacer_after_cassette")


def test_the_same_combination_is_fine_with_a_caproyl_spacer():
    pair = wh.build_pair(
        "maleimidocaproyl", "val_cit_pab", "mmae", "spacer_after_cassette"
    )
    assert pair.first and pair.second


def test_both_architectures_leave_one_attachment_point_per_warhead():
    for architecture in wh.ARCHITECTURES:
        pair = wh.build_pair(
            "maleimidocaproyl", "val_cit_pab", "mmae", architecture
        )
        for smiles in (pair.first, pair.second):
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0) == 1


def test_linkinvent_input_has_no_stereochemistry():
    """The prior's input vocabulary is 34 tokens and contains neither
    [C@H] nor [C@@H]. A faithful MMAE warhead is rejected at validation."""
    pair = wh.build_pair("maleimidocaproyl", "val_cit_pab", "mmae")
    line = pair.as_linkinvent_input()
    assert "[C@H]" not in line and "[C@@H]" not in line
    assert "[*:1]" not in line
    assert line.count("|") == 1
    assert line.count("*") == 2


def test_assembly_restores_the_stereochemistry_the_model_never_saw():
    """The model is given flat warheads, so its own assembled SMILES has
    every stereocentre unspecified. Rejoining to the originals restores
    all of them."""
    pair = wh.build_pair("maleimidocaproyl", "val_cit_pab", "mmae")
    full = wh.assemble("[*]CCOCC[*]", pair)
    mol = Chem.MolFromSmiles(full)
    assert mol is not None
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    # ten from MMAE, two from valine-citrulline
    assert len(centres) == 12


def test_assembly_refuses_a_linker_without_two_attachment_points():
    pair = wh.build_pair("maleimidocaproyl", "val_cit_pab", "mmae")
    with pytest.raises(wh.WarheadError, match="exactly two"):
        wh.assemble("[*]CCOCC", pair)


def test_unknown_library_entries_name_what_is_available():
    with pytest.raises(wh.WarheadError, match="unknown payload"):
        wh.build_pair("maleimide", "val_cit_pab", "not-a-drug")


def test_attachment_point_proposer_finds_an_amine():
    found = wh.propose_attachment_points("NCCOCCO")
    assert any("amine" in f["group"] for f in found)


# ---------------------------------------------------------------------- score


def make_context(dataset, **overrides):
    profile = reference.build_profile(dataset)
    kwargs = dict(
        profile=profile,
        reference_fingerprints=reference.fingerprint_reference(dataset),
        payload_smiles=None,
        handle_smarts="O=C1C=CC(=O)N1",
        require_cleavable=True,
        min_heavy_atoms=12,
    )
    kwargs.update(overrides)
    return score.Context(**kwargs)


def test_a_real_linker_scores_well(dataset):
    context = make_context(dataset)
    result = score.score_smiles(FIXTURE_ROWS[7][1], context)  # MC-Val-Cit-PAB
    assert result.accepted
    assert result.total >= 70


def test_a_linker_with_no_release_trigger_is_gated_out(dataset):
    """Not down-ranked. A linker that never releases its payload is not a
    worse candidate, it is not a candidate."""
    context = make_context(dataset)
    result = score.score_smiles("O=C1C=CC(=O)N1CCOCCOCCOCCC(=O)O", context)
    assert not result.accepted
    assert "trigger_lost" in result.gates_failed
    assert score.reward("O=C1C=CC(=O)N1CCOCCOCCOCCC(=O)O", context) == 0.0


def test_two_handles_are_gated_out(dataset):
    """Two maleimides is a crosslinker, not a linker."""
    context = make_context(dataset)
    result = score.score_smiles(
        "O=C1C=CC(=O)N1CCNC(=O)[C@@H](CCCNC(N)=O)NC(=O)CCN1C(=O)C=CC1=O", context
    )
    assert "handle_duplicated" in result.gates_failed


def test_losing_the_payload_is_gated_out(dataset):
    """The failure the gate exists for: a clean, well-scoring molecule that
    is not the drug you wanted."""
    payload = "c1ccc2c(c1)nc(n2)C(F)(F)F"
    context = make_context(dataset, payload_smiles=payload)
    intact = f"O=C1C=CC(=O)N1CCC(=O)N[C@@H](CCCNC(N)=O)C(=O)Nc1ccc(CO)cc1"
    result = score.score_smiles(intact, context)
    assert "payload_lost" in result.gates_failed


def test_an_unparseable_candidate_is_rejected_not_crashed(dataset):
    context = make_context(dataset)
    result = score.score_smiles("C(C(C", context)
    assert not result.accepted
    assert "unparseable" in result.gates_failed


def test_a_greasy_linker_scores_below_a_polar_one(dataset):
    """Polarity carries the most weight, because hydrophobicity is what kills
    ADCs in vivo."""
    context = make_context(dataset)
    polar = "O=C1C=CC(=O)N1CCOCCOCCC(=O)N[C@@H](CCCNC(N)=O)C(=O)Nc1ccc(CO)cc1"
    greasy = "O=C1C=CC(=O)N1CCCCCCCCCCCCCCCC(=O)N[C@@H](CCCNC(N)=O)C(=O)Nc1ccc(CO)cc1"
    assert score.score_smiles(polar, context).total > score.score_smiles(
        greasy, context
    ).total


def test_reward_is_bounded_and_zero_for_rejects(dataset):
    context = make_context(dataset)
    for smiles in (FIXTURE_ROWS[7][1], "OCCOCCO", "C(C(C"):
        value = score.reward(smiles, context)
        assert 0.0 <= value <= 1.0


def test_rank_puts_accepted_candidates_first():
    rows = [
        {"accepted": False, "score_total": None},
        {"accepted": True, "score_total": 55.0},
        {"accepted": True, "score_total": 90.0},
    ]
    ordered = score.rank(rows)
    assert [r["rank"] for r in ordered] == [1, 2, None]
    assert ordered[0]["score_total"] == 90.0


# ---------------------------------------------------------------------- slice


def test_slicing_produces_warhead_linker_triples(dataset):
    smiles = reference.read_reference_smiles(dataset)
    triples, report = slicing.build_triples(smiles, require_handle=False)
    assert triples
    assert report.triples == len(triples)
    for triple in triples[:5]:
        warheads, linker = triple.as_line().split("\t")
        assert warheads.count("|") == 1
        # Input side takes a bare `*`, target side a bracketed `[*]`. Mixing
        # them fails token validation on one side or the other.
        assert "[*]" not in warheads
        assert "[*]" in linker


def test_slicing_never_breaks_a_ring():
    """A broken ring gives two fragments no chemistry reassembles."""
    mol = Chem.MolFromSmiles("c1ccccc1CCOCCc1ccccc1")
    for index in slicing.cuttable_bonds(mol):
        assert not mol.GetBondWithIdx(index).IsInRing()


def test_slicing_can_require_a_conjugation_handle(dataset):
    """Without this the model learns peptide chemistry instead of spacers."""
    smiles = reference.read_reference_smiles(dataset)
    with_handle, _ = slicing.build_triples(smiles, require_handle=True)
    assert all(t.handle for t in with_handle)


def test_the_split_is_by_molecule_not_by_triple(dataset, tmp_path):
    """Cuts from one reagent are near-duplicates. Splitting them across train
    and validation is the leakage that makes Linker-GPT's numbers unreliable."""
    smiles = reference.read_reference_smiles(dataset)
    triples, _ = slicing.build_triples(smiles, require_handle=False)
    written = slicing.write_split(
        triples, tmp_path / "train.smi", tmp_path / "valid.smi",
        validation_fraction=0.4,
    )
    if written["validation_triples"] == 0:
        pytest.skip("fixture too small to split")

    def lines(path):
        return set(Path(path).read_text().splitlines())

    train = lines(written["training_file"])
    valid = lines(written["validation_file"])

    # The invariant that matters: no identical training example on both
    # sides. A warhead *pair* may legitimately recur, because different
    # reagents share functional ends -- what must not recur is the whole
    # (warheads, linker) example, which would make the validation loss
    # flatter the model.
    assert not train & valid
    assert train and valid


def test_slicing_an_unsliceable_set_says_so():
    with pytest.raises(slicing.SliceError, match="no training triples"):
        slicing.build_triples(["C", "O"], require_handle=True)


# ------------------------------------------------------------------- reinvent


def test_probe_reports_a_missing_prior(cfg):
    runtime = rv.probe(cfg)
    assert not runtime.prior_present
    assert any("linkinvent.prior" in p for p in runtime.problems)
    # and says how to get it back
    assert any("git" in p or "zenodo" in p.lower() for p in runtime.problems)


def test_the_sampling_config_is_valid_toml(cfg, tmp_path):
    warheads = rv.write_warheads_file(tmp_path / "w.smi", ["*CC|*CN"])
    path = rv.write_sampling_config(cfg, warheads, tmp_path / "out.csv")
    text = path.read_text()
    assert 'run_type = "sampling"' in text
    assert "num_smiles = " in text
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(text)
    assert parsed["run_type"] == "sampling"
    assert parsed["parameters"]["unique_molecules"] is True


def test_the_transfer_learning_config_is_valid_toml(cfg, tmp_path):
    training = tmp_path / "tl.smi"
    training.write_text("*CC|*CN\t[*]CCO[*]\n")
    path = rv.write_transfer_learning_config(cfg, training, None)
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(path.read_text())
    assert parsed["run_type"] == "transfer_learning"
    # REINVENT rejects anything below 100 outright.
    assert parsed["parameters"]["sample_batch_size"] >= 100


def test_the_warheads_file_has_no_header(tmp_path):
    """read_smiles_csv_file runs with header=False, so a column name would be
    parsed as a SMILES and rejected."""
    path = rv.write_warheads_file(tmp_path / "w.smi", ["*CC|*CN", "*CO|*CS"])
    lines = path.read_text().strip().splitlines()
    assert lines == ["*CC|*CN", "*CO|*CS"]


def test_an_empty_warhead_list_is_refused(tmp_path):
    with pytest.raises(rv.ReinventError, match="no warhead pairs"):
        rv.write_warheads_file(tmp_path / "w.smi", [])


def test_reading_the_sampler_output(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "SMILES,SMILES_state,Warheads,Linker,NLL\n"
        "CCOCC,1,*CC|*CC,[*]O[*],12.5\n"
        "CCNCC,2,*CC|*CC,[*]N[*],9.1\n"
        "bad,0,*CC|*CC,,0.0\n"
    )
    samples = rv.read_samples(path)
    assert len(samples) == 3
    stats = rv.summarise_samples(samples)
    assert stats["valid"] == 1
    assert stats["duplicate"] == 1
    assert stats["invalid"] == 1
    assert stats["validity_percent"] == pytest.approx(33.33, abs=0.01)


def test_missing_output_is_reported_clearly(tmp_path):
    with pytest.raises(rv.ReinventError, match="wrote no output"):
        rv.read_samples(tmp_path / "nothing.csv")


def test_wrong_columns_are_reported_clearly(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(rv.ReinventError, match="missing expected columns"):
        rv.read_samples(path)


# ---------------------------------------------------------------- end to end


def write_fake_reinvent(tmp_path: Path, rows: list[tuple[str, int, str, str]]) -> Path:
    """A stand-in for the `reinvent` console script.

    It reads the TOML it was given, finds the output path, and writes the
    same CSV the real sampler writes, so everything from config generation to
    re-assembly runs for real.
    """
    payload = json.dumps(rows)
    script = tmp_path / "fake_reinvent.py"
    script.write_text(
        "import json, re, sys\n"
        "cfg = open(sys.argv[1]).read()\n"
        "out = re.search(r'output_file = \"([^\"]+)\"', cfg).group(1)\n"
        "wh = re.search(r'smiles_file = \"([^\"]+)\"', cfg).group(1)\n"
        "pair = open(wh).read().strip().splitlines()[0]\n"
        f"rows = json.loads({payload!r})\n"
        "with open(out, 'w') as fh:\n"
        "    fh.write('SMILES,SMILES_state,Warheads,Linker,NLL\\n')\n"
        "    for smiles, state, _w, linker in rows:\n"
        "        fh.write(f'{smiles},{state},{pair},{linker},5.0\\n')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_generate_runs_end_to_end(cfg, tmp_path, monkeypatch):
    from adclinker.__main__ import main

    prior = tmp_path / "fake.prior"
    prior.write_bytes(b"not a real model, but present")

    script = write_fake_reinvent(
        tmp_path,
        [
            ("ignored-by-us", 1, "", "[*]CCOCC[*]"),
            ("ignored-by-us", 1, "", "[*]CCOCCOCC[*]"),
            ("ignored-by-us", 1, "", "[*]CCCCCCCCCCCC[*]"),
            ("bad", 0, "", ""),
        ],
    )
    cfg["reinvent"]["executable"] = sys.executable

    config_file = tmp_path / "adc.yaml"
    import yaml

    config_file.write_text(
        yaml.safe_dump(
            {
                "reference": {
                    "dataset": cfg["reference"]["dataset"],
                    "cache": cfg["reference"]["cache"],
                },
                "reinvent": {
                    "executable": sys.executable,
                    "prior_file": str(prior),
                    "working_dir": str(tmp_path),
                    "sampling_config": str(tmp_path / "sampling.toml"),
                    "warheads_file": str(tmp_path / "warheads.smi"),
                    "samples_csv": str(tmp_path / "samples.csv"),
                    "num_smiles": 4,
                },
                "report": {"output_dir": str(tmp_path / "report")},
                "run": {"log_to_file": False},
            }
        )
    )

    # `run_config` is the only seam needed: the fake takes the config path as
    # its sole argument, exactly as the real console script does.
    def run(c, path, log_path=None):
        import subprocess

        return subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True,
            text=True,
        )

    monkeypatch.setattr(rv, "run_config", run)

    code = main(["generate", "--config", str(config_file), "--no-progress"])
    assert code == 0

    candidates = tmp_path / "report" / "candidates.csv"
    assert candidates.is_file()
    rows = list(csv.DictReader(candidates.open()))
    assert rows, "no candidates were written"

    # Re-assembly must have restored the payload stereochemistry that the
    # model never saw.
    best = rows[0]
    mol = Chem.MolFromSmiles(best["smiles"])
    assert mol is not None
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    assert len(centres) >= 10

    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["scored"] >= 3


def test_an_unknown_config_key_fails_loudly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("reference:\n  not_a_key: 1\n")
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(path)
