"""LLM repair of rejected samples, and an honest account of what it salvaged.

`assess` writes one repair task per rejected sample. This module runs them
through the DeepSeek harness, then **verifies the answer rather than
believing it**, which is most of the work here and the reason the module is
not three functions long.

An agent asked to repair a disconnected PROTAC has an easy way out: delete one
fragment. The result is a single connected molecule that sanitises, scores
well on every descriptor, and is not a PROTAC at all. Nothing in the chemistry
of the answer gives this away -- the molecule is fine, it is just the wrong
molecule. So the check cannot be "does it parse"; it has to be "are both
fragments still there".

That is what `assess.fragment_smiles` is for. The two fragments are recorded
before the repair, and afterwards each one is matched back into whatever came
out. Both present and disjoint means the agent edited the linker. Either one
missing means it did something else, and the answer is rejected whatever its
descriptors say.

The linker is then whatever is left over, which is how the repaired molecule
gets measured on the same footing as the generated one: same gates, same
score, same thresholds. Only then is "improved" a claim worth making.
"""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moltrain import logs, progress

from . import harness
from .assess import Assessment, apply_soft_gates, molecule_properties
from .config import resolve
from .score import rank, score_sample
from .structures import linker_view

LOG = logs.get_logger("protacbench.repair")

ANSWER_PREFIX = "REPAIRED:"
NO_ANSWER = "none"

# What became of one task. Ordered roughly best to worst, which is also the
# order the summary prints them in.
STATUSES = {
    "repaired": "a valid PROTAC with both fragments intact",
    "still_poor": "both fragments intact and chemically real, but soft gates fail",
    "still_broken": "both fragments intact, but it is not a usable molecule",
    "fragments_lost": "a fragment was deleted or altered; not the same PROTAC",
    "unparseable": "the answer was not a readable SMILES",
    "no_answer": "the agent did not propose a molecule",
    "harness_failed": "the harness run did not complete",
}


@dataclass
class RepairOutcome:
    """One task: what went in, what came back, and whether it is better."""

    task_id: str
    target_id: str = ""
    sample_id: str = ""
    stage: str = ""
    status: str = "harness_failed"

    before_verdict: str = ""
    before_score: float | None = None
    before_smiles: str = ""
    before_failures: list[str] = field(default_factory=list)

    answer_smiles: str | None = None
    after_verdict: str | None = None
    after_score: float | None = None
    after_failures: list[str] = field(default_factory=list)

    fragments_intact: bool | None = None
    linker_smiles: str | None = None
    notes: str = ""
    run: dict[str, Any] = field(default_factory=dict)

    @property
    def salvaged(self) -> bool:
        """A broken or poor sample that came back as a usable molecule."""
        return self.status == "repaired"

    @property
    def score_delta(self) -> float | None:
        if self.after_score is None or self.before_score is None:
            return None
        return round(self.after_score - self.before_score, 2)

    def as_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_id": self.target_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "status": self.status,
            "salvaged": self.salvaged,
            "fragments_intact": self.fragments_intact,
            "before_verdict": self.before_verdict,
            "verdict": self.after_verdict or "",
            "before_score": self.before_score,
            "score_total": self.after_score,
            "score_delta": self.score_delta,
            "before_failures": ";".join(self.before_failures),
            "after_failures": ";".join(self.after_failures),
            "before_smiles": self.before_smiles,
            "repaired_smiles": self.answer_smiles or "",
            "linker_smiles": self.linker_smiles or "",
            "notes": self.notes,
            **{f"run_{k}": v for k, v in self.run.items()},
        }


# ------------------------------------------------------------------ the answer


def parse_answer(final_text: str) -> str | None:
    """Read the SMILES off the `REPAIRED:` line.

    The last such line wins, because a model that reasons in its final message
    may well write the contract line more than once, and the last one is its
    settled answer. Returns None for `REPAIRED: none` and for no line at all.
    """
    found: str | None = None
    for line in final_text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith(ANSWER_PREFIX):
            continue
        value = stripped[len(ANSWER_PREFIX) :].strip()
        # Models like to wrap SMILES in backticks or bold markers.
        value = value.strip("`*_ ").strip()
        found = value or None
    if found is None or found.lower() == NO_ANSWER:
        return None
    return found


# ---------------------------------------------------------------- the verifying


@dataclass
class Verification:
    """Whether a repaired SMILES is the same PROTAC, and how good it is."""

    status: str
    fragments_intact: bool
    note: str = ""
    assessment: Assessment | None = None
    linker_smiles: str | None = None


def match_fragments(mol: Any, fragment_smiles: list[str]) -> list[tuple[int, ...]] | None:
    """Find each fragment in the repaired molecule, without overlap.

    Returns one match per fragment, or None when any fragment is missing or
    the matches cannot be made disjoint. Greedy and first-fit: the fragments
    are large and distinctive, so an exhaustive assignment buys nothing.
    """
    from rdkit import Chem

    claimed: set[int] = set()
    matches: list[tuple[int, ...]] = []
    for smiles in fragment_smiles:
        query = Chem.MolFromSmiles(smiles, sanitize=False)
        if query is None:
            return None
        try:
            Chem.SanitizeMol(query, Chem.SanitizeFlags.SANITIZE_ALL)
        except Exception:  # noqa: BLE001 - an unsanitisable query still matches
            pass
        candidates = mol.GetSubstructMatches(query, uniquify=True)
        chosen = next(
            (m for m in candidates if not claimed.intersection(m)),
            None,
        )
        if chosen is None:
            return None
        claimed.update(chosen)
        matches.append(chosen)
    return matches


def verify(
    task: dict[str, Any], answer: str | None, thresholds: dict[str, Any]
) -> Verification:
    """Judge a repaired SMILES on the same terms as a generated sample."""
    from rdkit import Chem, rdBase

    if answer is None:
        return Verification("no_answer", fragments_intact=False,
                            note="the agent proposed no molecule")

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(answer)
    if mol is None:
        return Verification("unparseable", fragments_intact=False,
                            note="RDKit could not read the proposed SMILES")

    pieces = Chem.GetMolFrags(mol, asMols=False)
    if len(pieces) > 1:
        return Verification(
            "still_broken",
            fragments_intact=False,
            note=f"the proposed molecule is in {len(pieces)} pieces",
        )

    fragments = [f for f in (task.get("fragment_smiles") or []) if f]
    if not fragments:
        return Verification(
            "still_broken",
            fragments_intact=False,
            note="no fragments were recorded for this sample, so the answer "
            "cannot be checked against them",
        )

    matches = match_fragments(mol, fragments)
    if matches is None:
        return Verification(
            "fragments_lost",
            fragments_intact=False,
            note="at least one of the two fragments is no longer present; the "
            "answer is a different molecule, not a repaired linker",
        )

    claimed = {index for match in matches for index in match}
    linker_indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
    if not linker_indices:
        return Verification(
            "fragments_lost",
            fragments_intact=True,
            note="the fragments were joined directly, leaving no linker",
        )

    assessment = Assessment(
        target_id=str(task.get("target_id", "")),
        sample_id=str(task.get("sample_id", "")),
        stage="repaired_by_llm",
        linker_size=len(linker_indices),
    )
    assessment.protac_smiles = Chem.MolToSmiles(mol)
    assessment.fragment_smiles = fragments
    assessment.properties = molecule_properties(mol)

    view = linker_view(mol, linker_indices)
    assessment.linker_smiles = view.smiles
    assessment.linker = {
        "heavy_atoms": view.n_heavy_atoms,
        "attachments": view.n_attachments,
        "path_length_bonds": view.path_length_bonds,
        "rotatable_bonds": view.rotatable_bonds,
        "heteroatom_fraction": view.heteroatom_fraction,
        "in_ring_fraction": view.in_ring_fraction,
    }

    if view.n_attachments != 2:
        assessment.add_failure(
            "linker_not_bridging",
            f"the linker touches a fragment at {view.n_attachments} atom(s)",
        )
        return Verification(
            "still_broken",
            fragments_intact=True,
            note="both fragments survived, but the linker does not bridge them",
            assessment=assessment,
            linker_smiles=view.smiles,
        )

    apply_soft_gates(assessment, thresholds, mol, view)
    assessment.score = score_sample(
        mol,
        assessment.properties,
        assessment.linker,
        linker_indices,
        assessment.failure_codes,
    )

    status = {"ok": "repaired", "poor": "still_poor", "broken": "still_broken"}[
        assessment.verdict
    ]
    return Verification(
        status,
        fragments_intact=True,
        note="",
        assessment=assessment,
        linker_smiles=view.smiles,
    )


# ------------------------------------------------------------------- the runner


def read_tasks(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if limit and len(tasks) >= limit:
                break
    return tasks


async def repair_one(
    cfg: dict[str, Any],
    task: dict[str, Any],
    *,
    patch_path: Path,
    retries: int = 1,
) -> RepairOutcome:
    """Run one task, retrying only when the run itself failed to answer."""
    outcome = RepairOutcome(
        task_id=str(task.get("task_id", "")),
        target_id=str(task.get("target_id", "")),
        sample_id=str(task.get("sample_id", "")),
        stage=str(task.get("stage", "")),
        before_verdict=str(task.get("verdict", "")),
        before_smiles=str(task.get("protac_smiles") or ""),
        before_failures=list(task.get("failure_codes") or []),
    )
    before = task.get("score_total")
    outcome.before_score = float(before) if before is not None else None

    prompt = task.get("prompt")
    if not prompt:
        outcome.status = "no_answer"
        outcome.notes = "the task carried no prompt"
        return outcome

    attempt = 0
    while True:
        run = await harness.run_prompt(cfg, prompt, patch_path=patch_path)
        outcome.run = run.as_dict()

        if not run.ok:
            # A harness failure is worth one more go: it is usually a cold
            # npx start or a transient endpoint error, not a bad task.
            if attempt < retries:
                attempt += 1
                LOG.info("retrying %s after: %s", outcome.task_id, run.error)
                continue
            outcome.status = "harness_failed"
            outcome.notes = run.error or "the harness run did not complete"
            return outcome

        answer = parse_answer(run.final_text)
        if answer is None and attempt < retries:
            attempt += 1
            LOG.info("retrying %s: no REPAIRED: line in the answer", outcome.task_id)
            continue
        break

    outcome.answer_smiles = answer
    result = verify(task, answer, cfg["thresholds"])
    outcome.status = result.status
    outcome.fragments_intact = result.fragments_intact
    outcome.notes = result.note
    outcome.linker_smiles = result.linker_smiles

    if result.assessment is not None:
        outcome.after_verdict = result.assessment.verdict
        outcome.after_failures = list(result.assessment.failure_codes)
        if result.assessment.score is not None:
            outcome.after_score = result.assessment.score.total
    return outcome


async def run_batch(
    cfg: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    disable_progress: bool | None = None,
) -> list[RepairOutcome]:
    """Run every task, bounded by `repair.workers` concurrent harness runs."""
    settings = cfg["repair"]
    patch_path = harness.write_patch(cfg)
    limit = asyncio.Semaphore(max(1, int(settings["workers"])))
    bar = progress.track(
        len(tasks), desc="repairing", unit="task", disable=disable_progress
    )

    async def worker(task: dict[str, Any]) -> RepairOutcome:
        async with limit:
            try:
                outcome = await repair_one(
                    cfg, task, patch_path=patch_path, retries=int(settings["retries"])
                )
            except harness.HarnessError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad task is not the run
                outcome = RepairOutcome(
                    task_id=str(task.get("task_id", "")),
                    status="harness_failed",
                    notes=f"{type(exc).__name__}: {exc}",
                )
            bar.update()
            return outcome

    try:
        outcomes = await asyncio.gather(*(worker(t) for t in tasks))
    finally:
        bar.close()
    return list(outcomes)


# ------------------------------------------------------------------- the report


def summarise(outcomes: list[RepairOutcome]) -> dict[str, Any]:
    """Counts, and the one number the whole exercise exists to produce."""
    by_status: dict[str, int] = {}
    for outcome in outcomes:
        by_status[outcome.status] = by_status.get(outcome.status, 0) + 1

    salvaged = [o for o in outcomes if o.salvaged]
    deltas = [o.score_delta for o in outcomes if o.score_delta is not None]
    answered = [o for o in outcomes if o.answer_smiles]

    from_broken = sum(1 for o in salvaged if o.before_verdict == "broken")
    from_poor = sum(1 for o in salvaged if o.before_verdict == "poor")

    return {
        "tasks": len(outcomes),
        "by_status": dict(sorted(by_status.items(), key=lambda kv: -kv[1])),
        "answered": len(answered),
        "salvaged": len(salvaged),
        "salvage_rate_percent": (
            round(len(salvaged) / len(outcomes) * 100, 2) if outcomes else 0.0
        ),
        "salvaged_from_broken": from_broken,
        "salvaged_from_poor": from_poor,
        # Counted separately because it is the failure mode that matters: an
        # agent that deletes a fragment produces a molecule that passes every
        # descriptor check and is not a PROTAC.
        "fragments_lost": by_status.get("fragments_lost", 0),
        "mean_score_delta": (
            round(sum(deltas) / len(deltas), 2) if deltas else None
        ),
        "unique_repaired_smiles": len(
            {o.answer_smiles for o in salvaged if o.answer_smiles}
        ),
    }


def write_report(
    cfg: dict[str, Any], outcomes: list[RepairOutcome]
) -> dict[str, Path]:
    settings = cfg["repair"]
    out_dir = resolve(cfg, settings["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = out_dir / settings["results_jsonl"]
    with results.open("w") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.as_row()) + "\n")

    rows = rank([o.as_row() for o in outcomes])
    repaired_csv = out_dir / settings["repaired_csv"]
    if rows:
        columns = ["rank"] + [k for k in rows[0] if k != "rank"]
        with repaired_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
    else:
        repaired_csv.write_text("")

    summary = out_dir / settings["summary_json"]
    summary.write_text(json.dumps(summarise(outcomes), indent=2) + "\n")

    LOG.info("wrote %d repair results to %s", len(outcomes), out_dir)
    return {
        "results_jsonl": results,
        "repaired_csv": repaired_csv,
        "summary_json": summary,
    }
