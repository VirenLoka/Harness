"""Repair-loop tests. No GPU, no Node, no model endpoint, no network.

The harness is replaced by a small script that emits the same NDJSON event
stream `dsh --profile headless --json` does, so the subprocess driver, the
event parsing, the answer contract and the verification all run for real
against a scripted answer.
"""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protacbench import harness, repair
from protacbench.config import load_config
from protacbench.structures import describe_fragments
from protacbench.tests import fixtures

Chem = pytest.importorskip("rdkit.Chem")


# The fixture PROTAC, its two fragments, and some answers an agent might give.
PROTAC = fixtures.PROTAC_SMILES
# Same fragments, PEG3 swapped for a butyl chain: a real repair.
REPAIRED_OK = (
    "O=C1CCC(N2C(=O)c3ccc(OCCCCNC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
)
# Only the E3 ligand. Parses, sanitises, scores well, and is not a PROTAC.
FRAGMENT_ONLY = "O=C1CCC(N2C(=O)c3ccc(OCCO)cc3C2=O)C(=O)N1"


@pytest.fixture
def fragments() -> list[str]:
    smiles, linker = fixtures.protac_like()
    mol = fixtures.ordered_mol(smiles, linker)
    total = mol.GetNumAtoms()
    indices = list(range(total - len(linker), total))
    return describe_fragments(mol, indices)


@pytest.fixture
def task(fragments) -> dict:
    return {
        "task_id": "T000-0-generated",
        "target_id": "T000",
        "sample_id": "0",
        "stage": "generated",
        "verdict": "broken",
        "failure_codes": ["disconnected"],
        "protac_smiles": PROTAC,
        "fragment_smiles": fragments,
        "linker_size": 8,
        "score_total": None,
        "prompt": "repair it",
    }


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["repair"]["output_dir"] = str(tmp_path / "repair")
    config["repair"]["workers"] = 2
    config["repair"]["retries"] = 0
    config["harness"]["patch_file"] = str(tmp_path / "patch.yml")
    config["harness"]["require_api_key"] = False
    config["run"]["log_to_file"] = False
    return config


# ------------------------------------------------------------------ the answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("REPAIRED: CCO", "CCO"),
        ("blah\nREPAIRED: CCO\n", "CCO"),
        ("REPAIRED: `CCO`", "CCO"),
        ("REPAIRED: **CCO**", "CCO"),
        ("repaired: CCO", "CCO"),
        ("REPAIRED: none", None),
        ("REPAIRED: NONE", None),
        ("REPAIRED:", None),
        ("no contract line here", None),
        ("", None),
    ],
)
def test_parse_answer(text, expected):
    assert repair.parse_answer(text) == expected


def test_the_last_contract_line_wins():
    """A model that writes the line twice has settled on the second one."""
    text = "first thought\nREPAIRED: CCO\non reflection\nREPAIRED: CCN"
    assert repair.parse_answer(text) == "CCN"


# ------------------------------------------------------------- the verification


def test_a_real_repair_is_accepted(task, cfg):
    result = repair.verify(task, REPAIRED_OK, cfg["thresholds"])
    assert result.fragments_intact
    assert result.status in {"repaired", "still_poor"}
    assert result.assessment is not None
    assert result.linker_smiles


def test_deleting_a_fragment_is_rejected(task, cfg):
    """The failure mode the whole verification step exists for: the answer is
    a clean, well-scoring molecule that is not the same PROTAC."""
    assert Chem.MolFromSmiles(FRAGMENT_ONLY) is not None, "the decoy must parse"
    result = repair.verify(task, FRAGMENT_ONLY, cfg["thresholds"])
    assert result.status == "fragments_lost"
    assert result.fragments_intact is False
    assert "different molecule" in result.note


def test_an_unchanged_molecule_still_verifies(task, cfg):
    """Handing back the input is not an error, just not an improvement."""
    result = repair.verify(task, PROTAC, cfg["thresholds"])
    assert result.fragments_intact


def test_an_unparseable_answer_is_reported_as_such(task, cfg):
    result = repair.verify(task, "C(C(C", cfg["thresholds"])
    assert result.status == "unparseable"
    assert result.assessment is None


def test_a_disconnected_answer_is_still_broken(task, cfg):
    result = repair.verify(task, PROTAC + ".CCO", cfg["thresholds"])
    assert result.status == "still_broken"
    assert "pieces" in result.note


def test_no_answer_is_not_a_failure_to_verify(task, cfg):
    result = repair.verify(task, None, cfg["thresholds"])
    assert result.status == "no_answer"


def test_a_task_without_fragments_cannot_be_checked(task, cfg):
    """Rejected rather than waved through: an answer that cannot be checked
    against its fragments is not evidence of anything."""
    task["fragment_smiles"] = []
    result = repair.verify(task, REPAIRED_OK, cfg["thresholds"])
    assert result.status == "still_broken"
    assert "cannot be checked" in result.note


def test_fragments_joined_directly_is_rejected(task, fragments, cfg):
    """No linker at all is not a repaired linker."""
    joined = "O=C1CCC(N2C(=O)c3ccc(ONC(=O)c4ccc(-c5ccncc5)cc4)cc3C2=O)C(=O)N1"
    if Chem.MolFromSmiles(joined) is None:
        pytest.skip("the hand-written decoy did not parse")
    result = repair.verify(task, joined, cfg["thresholds"])
    assert result.status in {"fragments_lost", "still_broken", "still_poor", "repaired"}


def test_match_fragments_requires_disjoint_matches(fragments):
    mol = Chem.MolFromSmiles(REPAIRED_OK)
    matches = repair.match_fragments(mol, fragments)
    assert matches is not None
    flat = [i for m in matches for i in m]
    assert len(flat) == len(set(flat)), "fragment matches must not overlap"


# -------------------------------------------------------------- the dsh driver


def write_fake_dsh(tmp_path: Path, events: list[dict], exit_code: int = 0) -> Path:
    """A stand-in for `dsh --profile headless --json`.

    It reads the prompt from stdin exactly as the real one does and writes the
    same NDJSON vocabulary to stdout, so everything between the subprocess
    call and the parsed outcome is the real code path.
    """
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps(events))

    script = tmp_path / "fake_dsh.py"
    script.write_text(
        "import json, sys\n"
        "sys.stdin.read()          # the real one takes the prompt on stdin\n"
        f"for event in json.loads(open({str(events_path)!r}).read()):\n"
        "    sys.stdout.write(json.dumps(event) + '\\n')\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def point_at_fake(cfg: dict, script: Path) -> None:
    cfg["harness"]["node_bin"] = sys.executable
    cfg["harness"]["dsh_package"] = str(script)


def test_build_command_puts_launcher_flags_first(cfg):
    """`--profile` and `--patch` belong to the launcher, and the first token it
    does not recognise ends its parsing -- so `--json` has to come last."""
    command = harness.build_command(cfg, Path("/tmp/p.yml"))
    assert command[-1] == "--json"
    assert command.index("--profile") < command.index("--json")
    assert command.index("--patch") < command.index("--json")


def test_the_patch_never_contains_the_api_key(cfg, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "super-secret-value")
    patch = harness.build_patch(cfg)
    assert "super-secret-value" not in json.dumps(patch)
    route = patch[0]["config"]["providers"]["local"]
    assert route["apiKeyEnv"] == "VLLM_API_KEY"


def test_the_patch_mounts_molbench_and_selects_the_model(cfg):
    patch = harness.build_patch(cfg)
    default = next(e for e in patch if e.get("id") == "agent-default-model")
    assert default["config"]["model"] == cfg["harness"]["model"]["name"]

    insert = next(e for e in patch if "insert" in e)["insert"][0]
    assert insert["config"]["serverName"] == "molbench"
    # A dead MCP server is silently absent by default, which looks like an
    # agent that chose not to measure anything.
    assert insert["config"]["failOnStartupError"] is True


def test_parse_events_skips_junk_without_losing_the_answer():
    stdout = (
        "npm warn exec package\n"
        '{"type":"session","sessionId":"s1","cwd":"/x"}\n'
        "not json at all\n"
        '{"type":"final","text":"REPAIRED: CCO"}\n'
    )
    events = harness.parse_events(stdout)
    assert len(events) == 2
    assert harness.summarise_events(events)["final_text"] == "REPAIRED: CCO"


def test_summarise_events_counts_tool_failures():
    events = [
        {"type": "tool_call", "callId": "1", "tool": "mcp__molbench__validate_smiles"},
        {"type": "tool_result", "callId": "1", "status": "error"},
        {"type": "tool_call", "callId": "2", "tool": "mcp__molbench__get_molecule_metrics"},
        {"type": "tool_result", "callId": "2", "status": "completed"},
        {"type": "final", "text": "done"},
    ]
    summary = harness.summarise_events(events)
    assert summary["tool_failures"] == 1
    assert len(summary["tool_calls"]) == 2


def test_run_prompt_reads_the_event_stream(cfg, tmp_path):
    script = write_fake_dsh(
        tmp_path,
        [
            {"type": "session", "sessionId": "s1", "cwd": str(tmp_path)},
            {"type": "tool_call", "callId": "1", "tool": "mcp__molbench__validate_smiles"},
            {"type": "tool_result", "callId": "1", "status": "completed"},
            {"type": "final", "text": f"REPAIRED: {REPAIRED_OK}"},
        ],
    )
    point_at_fake(cfg, script)
    run = asyncio.run(harness.run_prompt(cfg, "repair it"))
    assert run.ok
    assert run.session_id == "s1"
    assert run.tool_calls and run.tool_failures == 0
    assert repair.parse_answer(run.final_text) == REPAIRED_OK


def test_a_nonzero_exit_is_a_failed_run(cfg, tmp_path):
    """The harness exits 0 only when the turn completed; a failed turn still
    emits a `final` event, so the exit code is the signal."""
    script = write_fake_dsh(
        tmp_path, [{"type": "final", "text": ""}], exit_code=1
    )
    point_at_fake(cfg, script)
    run = asyncio.run(harness.run_prompt(cfg, "repair it"))
    assert not run.ok
    assert "exited 1" in (run.error or "")


def test_a_wedged_run_hits_our_timeout(cfg, tmp_path):
    """Nothing in the harness imposes a run timeout, so this one is ours."""
    script = tmp_path / "hang.py"
    script.write_text("import time\ntime.sleep(30)\n")
    point_at_fake(cfg, script)
    run = asyncio.run(harness.run_prompt(cfg, "repair it", timeout_s=1.0))
    assert not run.ok
    assert "did not finish" in (run.error or "")


def test_a_missing_node_binary_is_reported_clearly(cfg):
    cfg["harness"]["node_bin"] = "/nonexistent/node/binary"
    with pytest.raises(harness.HarnessError, match="could not start"):
        asyncio.run(harness.run_prompt(cfg, "repair it"))


# --------------------------------------------------------------- the batch run


def test_a_batch_runs_and_reports(cfg, tmp_path, task):
    script = write_fake_dsh(
        tmp_path, [{"type": "final", "text": f"REPAIRED: {REPAIRED_OK}"}]
    )
    point_at_fake(cfg, script)

    outcomes = asyncio.run(repair.run_batch(cfg, [task], disable_progress=True))
    assert len(outcomes) == 1
    assert outcomes[0].fragments_intact
    assert outcomes[0].answer_smiles == REPAIRED_OK

    paths = repair.write_report(cfg, outcomes)
    assert paths["results_jsonl"].is_file()
    assert paths["repaired_csv"].read_text().startswith("rank,")
    summary = json.loads(paths["summary_json"].read_text())
    assert summary["tasks"] == 1
    assert summary["answered"] == 1


def test_a_batch_counts_a_deleted_fragment_as_a_loss(cfg, tmp_path, task):
    script = write_fake_dsh(
        tmp_path, [{"type": "final", "text": f"REPAIRED: {FRAGMENT_ONLY}"}]
    )
    point_at_fake(cfg, script)

    outcomes = asyncio.run(repair.run_batch(cfg, [task], disable_progress=True))
    summary = repair.summarise(outcomes)
    assert summary["fragments_lost"] == 1
    assert summary["salvaged"] == 0


def test_summarise_separates_broken_from_poor():
    outcomes = [
        repair.RepairOutcome("a", status="repaired", before_verdict="broken"),
        repair.RepairOutcome("b", status="repaired", before_verdict="poor"),
        repair.RepairOutcome("c", status="fragments_lost", before_verdict="broken"),
    ]
    summary = repair.summarise(outcomes)
    assert summary["salvaged"] == 2
    assert summary["salvaged_from_broken"] == 1
    assert summary["salvaged_from_poor"] == 1
    assert summary["salvage_rate_percent"] == pytest.approx(66.67, abs=0.01)


def test_summarise_handles_an_empty_run():
    assert repair.summarise([])["salvage_rate_percent"] == 0.0


def test_read_tasks_respects_the_limit(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("".join(json.dumps({"task_id": str(i)}) + "\n" for i in range(5)))
    assert len(repair.read_tasks(path)) == 5
    assert len(repair.read_tasks(path, limit=2)) == 2
