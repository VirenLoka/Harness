"""Walking a Pro-GAT run, assessing every sample, and writing the CSVs.

The output is three files that answer three different questions:

  `samples.csv`      every sample with its verdict and all measurements
  `rejects.csv`      only the ones that failed, with the reason and a prompt
  `repair_tasks.jsonl`  the same rejects as structured tasks for the agent

Assessing both Pro-GAT stages is what makes the repair delta visible: a
sample that is `broken` as generated and `ok` after repair is one Pro-GAT
rescued, and one that is broken in both is what the harness agent gets.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moltrain import logs, progress

from .assess import Assessment, assess_sample
from .config import resolve
from .prompts import build_prompt, write_tasks
from .score import rank

LOG = logs.get_logger("protacbench.triage")

# The two stages use different layouts, and getting this wrong means finding
# nothing. Sampling writes a directory per PROTAC -- `{uuid}/{sample}_.xyz`,
# alongside `true_.xyz` (the reference molecule) and `frag_.xyz` (the input
# fragments), neither of which is a sample. The repair model instead writes a
# flat directory, where `save_as_xyz` names each file from the composite uuid
# `append_to_dataset` built: `{uuid}-{linker_len}_{sample}_.xyz`.
NESTED_SAMPLE = re.compile(r"^(?P<sample>\d+)_\.xyz$")
FLAT_SAMPLE = re.compile(r"^(?P<target>.+?)-(?P<linker>\d+)_(?P<sample>\d+)_\.xyz$")
RESERVED = {"true", "frag"}


@dataclass
class Discovered:
    """One sample file found on disk."""

    target_id: str
    sample_id: str
    xyz_path: Path
    stage: str
    # The repaired layout carries the linker length in the filename, so it
    # does not need looking up in the dataset.
    linker_size: int | None = None


def discover_nested(root: str | Path, stage: str = "generated") -> list[Discovered]:
    """Samples from the sampling layout: one directory per PROTAC."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Discovered] = []
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for xyz in sorted(target_dir.glob("*_.xyz")):
            match = NESTED_SAMPLE.match(xyz.name)
            if match is None:
                continue  # true_.xyz and frag_.xyz are not samples
            found.append(
                Discovered(
                    target_id=target_dir.name,
                    sample_id=match.group("sample"),
                    xyz_path=xyz,
                    stage=stage,
                )
            )
    return found


def discover_flat(root: str | Path, stage: str = "repaired") -> list[Discovered]:
    """Samples from the repair layout, searched recursively.

    Pro-GAT puts repaired output under `repaired/<run_tag>/`, where the run tag
    is the basename of the sampling output directory, so the exact depth
    depends on how sampling was invoked.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[Discovered] = []
    for xyz in sorted(root.rglob("*_.xyz")):
        match = FLAT_SAMPLE.match(xyz.name)
        if match is None:
            continue
        target = match.group("target")
        if target in RESERVED:
            continue
        found.append(
            Discovered(
                target_id=target,
                sample_id=match.group("sample"),
                xyz_path=xyz,
                stage=stage,
                linker_size=int(match.group("linker")),
            )
        )
    return found


def discover(root: str | Path, stage: str = "generated") -> list[Discovered]:
    """Whichever layout this directory uses."""
    nested = discover_nested(root, stage)
    flat = discover_flat(root, stage)
    # A sampling directory can contain both, since the composite names are
    # distinguishable; prefer whichever actually matched.
    return nested + flat


def reference_smiles_for(target_dir: Path, sdf_dir: Path) -> str | None:
    """The reference molecule, from the `true_.xyz` Pro-GAT writes alongside.

    This is what makes a recovery rate computable. Pro-GAT's own
    `compute_repair.py` hardcodes `recovery = 0.0`, even though the reference
    is sitting right there.
    """
    from rdkit import Chem

    true_xyz = target_dir / "true_.xyz"
    if not true_xyz.is_file():
        return None
    from .structures import load_molecule

    result = load_molecule(true_xyz, sdf_dir / target_dir.name / "true_.sdf")
    if not result.ok or result.mol is None:
        return None
    try:
        return Chem.MolToSmiles(result.mol)
    except Exception:  # noqa: BLE001
        return None


def native_linker_sizes(cfg: dict[str, Any]) -> dict[str, int]:
    """Each entry's own linker length, read from the Pro-GAT dataset.

    Needed whenever sampling ran with `linker_size: 0`, because then every
    PROTAC has a different linker and the assessment has to know which.
    """
    import torch

    repo = resolve(cfg, cfg["progat"]["repo_path"])
    path = repo / cfg["progat"]["data_path"] / f"{cfg['progat']['test_data_prefix']}.pt"
    if not path.is_file():
        LOG.warning("Pro-GAT dataset not found at %s", path)
        return {}
    try:
        entries = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - fall back to the configured size
        LOG.warning("could not read %s: %s", path, exc)
        return {}
    sizes: dict[str, int] = {}
    for entry in entries:
        mask = entry.get("linker_mask")
        uuid = entry.get("uuid")
        if mask is None or uuid is None:
            continue
        sizes[str(uuid)] = int(mask.sum())
    LOG.info("read native linker sizes for %d targets", len(sizes))
    return sizes


@dataclass
class TriageReport:
    """Counts worth printing, and the breakdown the CSV cannot show at a glance."""

    total: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_stage: dict[str, dict[str, int]] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    rescued_by_progat: int = 0
    broken_after_repair: int = 0
    recovered_reference: int = 0
    unique_smiles: int = 0
    scores: list[float] = field(default_factory=list)

    def record(self, assessment: Assessment) -> None:
        self.total += 1
        self.by_verdict[assessment.verdict] = (
            self.by_verdict.get(assessment.verdict, 0) + 1
        )
        stage = self.by_stage.setdefault(assessment.stage, {})
        stage[assessment.verdict] = stage.get(assessment.verdict, 0) + 1
        for code in assessment.failure_codes:
            self.failure_counts[code] = self.failure_counts.get(code, 0) + 1
        if assessment.recovers_reference:
            self.recovered_reference += 1
        if assessment.score is not None and assessment.score.total is not None:
            self.scores.append(float(assessment.score.total))

    def score_summary(self) -> dict[str, Any]:
        """The distribution, not just a mean.

        A mean alone hides the shape that matters: a run can average 70 either
        because everything is mediocre or because half the samples are good
        and half are not worth opening.
        """
        if not self.scores:
            return {"scored": 0}
        ordered = sorted(self.scores)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "scored": len(ordered),
            "min": round(ordered[0], 2),
            "median": round(median, 2),
            "mean": round(sum(ordered) / len(ordered), 2),
            "max": round(ordered[-1], 2),
            "at_least_80": sum(1 for s in ordered if s >= 80.0),
            "at_least_60": sum(1 for s in ordered if s >= 60.0),
        }

    def as_dict(self) -> dict[str, Any]:
        def rate(stage: str, verdict: str) -> float | None:
            counts = self.by_stage.get(stage)
            if not counts:
                return None
            total = sum(counts.values())
            return round(counts.get(verdict, 0) / total * 100, 2) if total else None

        return {
            "total_samples": self.total,
            "by_verdict": self.by_verdict,
            "by_stage": self.by_stage,
            "validity_percent": {
                stage: rate(stage, "ok") for stage in sorted(self.by_stage)
            },
            "failure_counts": dict(
                sorted(self.failure_counts.items(), key=lambda kv: -kv[1])
            ),
            "rescued_by_progat": self.rescued_by_progat,
            "broken_after_repair": self.broken_after_repair,
            "recovered_reference": self.recovered_reference,
            "unique_valid_smiles": self.unique_smiles,
            "synthesis_score": self.score_summary(),
        }


def assess_all(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    disable_progress: bool | None = None,
) -> tuple[list[Assessment], TriageReport]:
    """Assess every sample in every configured stage."""
    thresholds = cfg["thresholds"]
    sdf_dir = resolve(cfg, cfg["assess"]["sdf_dir"])
    configured_size = int(cfg["progat"]["linker_size"])
    native = native_linker_sizes(cfg) if configured_size <= 0 else {}

    generated_root = resolve(cfg, cfg["progat"]["output_dir"])
    stage_roots: list[tuple[str, Path]] = []
    for stage in cfg["assess"]["stages"]:
        if stage == "generated":
            stage_roots.append((stage, generated_root))
        elif stage == "repaired":
            stage_roots.append((stage, resolve(cfg, cfg["assess"]["repaired_dir"])))
        else:
            raise ValueError(f"unknown stage in assess.stages: {stage}")

    discovered: list[Discovered] = []
    for stage, root in stage_roots:
        found = discover(root, stage)
        LOG.info("%s: %d samples under %s", stage, len(found), root)
        if not found and root.is_dir() is False:
            LOG.warning("%s directory does not exist: %s", stage, root)
        discovered.extend(found)

    if limit is not None:
        discovered = discovered[:limit]

    references: dict[str, str | None] = {}
    assessments: list[Assessment] = []
    report = TriageReport()

    bar = progress.track(
        total=len(discovered),
        desc="assessing samples",
        unit="sample",
        disable=disable_progress,
    )
    for item in discovered:
        bar.postfix(f"{item.stage} {item.target_id}/{item.sample_id}")
        if item.target_id not in references:
            # `true_.xyz` sits beside the generated samples. The repair layout
            # is flat and has no reference of its own, so look it up under the
            # sampling directory by target id.
            target_dir = (
                item.xyz_path.parent
                if item.stage == "generated"
                else generated_root / item.target_id
            )
            references[item.target_id] = reference_smiles_for(target_dir, sdf_dir)

        if item.linker_size is not None:
            linker_size = item.linker_size
        elif configured_size > 0:
            linker_size = configured_size
        else:
            linker_size = native.get(item.target_id, 0)
        sdf_path = sdf_dir / item.stage / item.target_id / f"{item.sample_id}_.sdf"
        assessment = assess_sample(
            item.target_id,
            item.sample_id,
            item.xyz_path,
            sdf_path,
            linker_size=linker_size,
            thresholds=thresholds,
            stage=item.stage,
            reference_smiles=references[item.target_id],
        )
        assessments.append(assessment)
        report.record(assessment)
        LOG.debug(
            "%s %s/%s -> %s %s",
            item.stage,
            item.target_id,
            item.sample_id,
            assessment.verdict,
            assessment.failure_codes,
        )
        bar.update(1)
    bar.close()

    # What Pro-GAT's repair actually bought: samples broken as generated that
    # came back usable, and the ones still broken afterwards.
    generated = {
        (a.target_id, a.sample_id): a for a in assessments if a.stage == "generated"
    }
    for assessment in assessments:
        if assessment.stage != "repaired":
            continue
        before = generated.get((assessment.target_id, assessment.sample_id))
        if before is not None and before.is_broken and not assessment.is_broken:
            report.rescued_by_progat += 1
        if assessment.is_broken:
            report.broken_after_repair += 1

    report.unique_smiles = len(
        {a.protac_smiles for a in assessments if a.verdict == "ok" and a.protac_smiles}
    )
    return assessments, report


# ------------------------------------------------------------------- the output


def column_order(rows: list[dict[str, Any]]) -> list[str]:
    """Identity and verdict first, then measurements, then paths and prompt."""
    leading = [
        "rank",
        "score_total",
        "score_reason",
        "target_id",
        "sample_id",
        "stage",
        "verdict",
        "linker_size",
        "failure_codes",
        "failure_detail",
        "protac_smiles",
        "linker_smiles",
    ]
    trailing = ["xyz_path", "sdf_path", "repair_prompt"]
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    middle = [k for k in seen if k not in leading and k not in trailing]
    return (
        [k for k in leading if k in seen] + middle + [k for k in trailing if k in seen]
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    columns = column_order(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_report(
    cfg: dict[str, Any], assessments: list[Assessment], report: TriageReport
) -> dict[str, Path]:
    """Write every output file and return where each landed."""
    settings = cfg["report"]
    out_dir = resolve(cfg, settings["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(settings["repair_verdicts"])

    # Ranked, so the CSV opens on the molecules worth making rather than on
    # whatever order the filesystem walk produced.
    all_rows = rank([a.as_row() for a in assessments])
    samples_csv = write_csv(out_dir / settings["samples_csv"], all_rows)

    rejected = [a for a in assessments if a.verdict in wanted]
    reject_rows = []
    for assessment in rejected:
        row = assessment.as_row()
        row["repair_prompt"] = build_prompt(assessment)
        reject_rows.append(row)
    rejects_csv = write_csv(out_dir / settings["rejects_csv"], reject_rows)

    tasks_path = out_dir / settings["repair_tasks"]
    write_tasks(tasks_path, rejected)

    summary_path = out_dir / settings["summary_json"]
    summary_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")

    LOG.info(
        "wrote %d samples, %d rejects, %d repair tasks",
        len(all_rows),
        len(reject_rows),
        len(rejected),
    )
    return {
        "samples_csv": samples_csv,
        "rejects_csv": rejects_csv,
        "repair_tasks": tasks_path,
        "summary_json": summary_path,
    }
