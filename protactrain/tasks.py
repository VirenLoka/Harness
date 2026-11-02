"""Building `tasks.jsonl`: the dataset every later stage trains and evaluates on.

One task is a PROTAC whose linker is wrong, the two fragments that must
survive the repair, and the measured baseline the reward is scored relative
to. Unlike `moltrain.tasks` there is no pocket, no receptor and no sampling
stage: `repair_tasks.jsonl` was produced by protacbench's own assessment pass,
so every number this module needs has already been measured once. Nothing here
touches a GPU, and nothing here re-measures.

Not re-measuring is a decision, not an economy. The reward pays for the
*distance closed* between the start and the submission, so the start has to be
the same value the corpus was graded against. Re-deriving it here with a
different RDKit build would move the baseline under the reward and silently
change what every episode is paid for.

What this module does add is the classification. OPTIMISATION.md splits the
corpus into five classes with very different expected yields -- a reactive
group is one local swap, a tangled linker is a redesign -- and the evaluation
is stratified by that class, so it has to be decided once, here, and carried
on the task.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from moltrain import hostcheck, logs, progress

from .config import (
    ConfigError,
    load_config,
    log_destination,
    resolve,
    resolve_sibling,
)
from .episode import LinkerTask
from .properties import LinkerProperties

# moltrain.logs roots every logger under `moltrain`, which is what lets one
# configure() call cover both packages. The explicit `protactrain.` prefix is
# what keeps this module's lines apart from moltrain's own tasks module in a
# shared log file.
LOG = logs.get_logger("protactrain.tasks")


class TaskBuildError(Exception):
    """A dataset problem worth reporting verbatim."""


# ------------------------------------------------------------- classification

# OPTIMISATION.md's five classes, as failure codes. The codes are protacbench's
# assessment vocabulary; the grouping is OPTIMISATION.md's table of what is
# worth sending to the model and what each class is expected to yield.

# Broken as a molecule or as a PROTAC: it does not parse, it is in pieces, or
# the linker is not a bridge. Expected yield 30-50%.
STRUCTURAL_CODES = frozenset(
    {
        "disconnected",
        "sanitization_failed",
        "linker_not_bridging",
        "impossible_valence",
    }
)

# A group in the linker that reacts with protein lysines and cysteines in vivo
# or does not survive as drawn. One functional-group swap each, so this is the
# highest-yield class: OPTIMISATION.md expects 60-80%.
REACTIVE_CODES = frozenset({"reactive_linker_group"})

# Descriptor and medicinal-chemistry complaints. Lowest expected yield
# (10-30%), because SA and cLogP are whole-molecule properties largely set by
# the two fragments the policy may not touch.
CHEMISTRY_CODES = frozenset(
    {
        "bad_linker_ring",
        "pains_alert",
        "sa_too_high",
        "clogp_too_high",
        "too_flexible",
        "macrocycle",
        "mw_out_of_range",
        "linker_too_short",
        "linker_too_long",
    }
)

# Dropped from every task's failure list, everywhere. QED is not in the reward
# and not in the prompt: 34% of the 128 published PROTACs behind this corpus
# fail the 0.05 gate that produced these codes, so the complaint describes the
# gate rather than the molecule. A row left with nothing after this is a
# control -- 952 of the 2,229 -- and what the policy does with a task that has
# genuinely nothing to fix is itself a measurement.
UNGATED_CODES = frozenset({"qed_too_low"})

# Most structural first. A row carrying codes from several classes takes the
# most structural one, because that is the one that has to be fixed first: a
# disconnected molecule has no cLogP worth arguing about.
CLASS_ORDER = ("structural", "reactive", "chemistry", "control")

_CLASS_CODES: dict[str, frozenset[str]] = {
    "structural": STRUCTURAL_CODES,
    "reactive": REACTIVE_CODES,
    "chemistry": CHEMISTRY_CODES,
}


def clean_codes(codes: Any) -> list[str]:
    """The row's failure codes with the ungated ones removed, order kept."""
    if not codes:
        return []
    seen: set[str] = set()
    kept: list[str] = []
    for raw in codes:
        # `str(None)` is "None", which would classify as an unrecognised code
        # and quietly promote a control row into the chemistry stratum. Blanks
        # go the same way.
        if raw is None:
            continue
        code = str(raw).strip()
        if not code or code in UNGATED_CODES or code in seen:
            continue
        seen.add(code)
        kept.append(code)
    return kept


def classify(codes: Any) -> str:
    """Which of OPTIMISATION.md's classes this row belongs to.

    An unrecognised code counts as `chemistry` rather than being ignored: a new
    descriptor complaint upstream should land in the hardest class and show up
    in the histogram, not quietly turn a real task into a control.
    """
    remaining = clean_codes(codes)
    if not remaining:
        return "control"
    for name in ("structural", "reactive", "chemistry"):
        if any(code in _CLASS_CODES[name] for code in remaining):
            return name
    return "chemistry"


def unknown_codes(rows: list[dict[str, Any]]) -> list[str]:
    """Failure codes this module has no class for, so a drift is visible."""
    known = STRUCTURAL_CODES | REACTIVE_CODES | CHEMISTRY_CODES | UNGATED_CODES
    seen = {str(code) for row in rows for code in (row.get("failure_codes") or [])}
    return sorted(seen - known)


# ----------------------------------------------------------------- baselines


def _number(value: Any) -> Any:
    """Keep a measured value only if it is one. Never raises."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


# `properties{}` in repair_tasks.jsonl already uses LinkerProperties' names.
_PROPERTY_FIELDS = (
    "mw",
    "exact_mw",
    "clogp",
    "tpsa",
    "sa_score",
    "qed",
    "rotatable_bonds",
    "hbd",
    "hba",
    "heavy_atoms",
    "rings",
    "aromatic_rings",
    "aliphatic_rings",
    "fraction_csp3",
    "stereocentres",
    "formal_charge",
)

# `linker{}` does not. These three are the renames that matter, and getting one
# wrong costs a reward term silently: `rewards._value` reaches fields by
# getattr, a miss returns None, and a None threshold is skipped.
_LINKER_FIELDS = {
    "path_length_bonds": "span",
    "heavy_atoms": "linker_heavy_atoms",
    "rotatable_bonds": "linker_rotatable_bonds",
    "attachments": "n_attachments",
}


def baseline_from_row(row: dict[str, Any]) -> LinkerProperties:
    """The assessed baseline, flattened onto LinkerProperties' field names.

    This is a measurement path, so it never raises: a row that carries no
    descriptors at all (the 236 `broken` rows, whose molecule never parsed)
    comes back with `measurable` False and every number None, which the reward
    reads as "no start value, so no progress to pay for" rather than as zero.
    """
    properties = row.get("properties") or {}
    linker = row.get("linker") or {}

    props = LinkerProperties(
        smiles=str(row.get("protac_smiles") or ""),
        # The assessor records descriptors only for a molecule it could
        # sanitize, so an empty properties block IS the parse failure.
        measurable=bool(properties),
        error="" if properties else "no descriptors recorded by the assessor",
    )

    for name in _PROPERTY_FIELDS:
        value = _number(properties.get(name))
        if value is not None:
            setattr(props, name, value)

    for source, name in _LINKER_FIELDS.items():
        value = _number(linker.get(source))
        if value is not None:
            setattr(props, name, value)

    props.linker_smiles = row.get("linker_smiles")
    props.reactive_groups = [str(g) for g in (properties.get("reactive_groups") or [])]
    props.liability_groups = [str(g) for g in (properties.get("liable_groups") or [])]

    # `fragments_intact` is the identity check, not a quality score, so it is
    # asserted only where the assessor actually matched both fragments back:
    # two attachment points and a linker it could name. A `disconnected` row
    # has one attachment and a dotted linker SMILES, and must not claim this.
    props.fragments_intact = bool(
        props.linker_smiles
        and props.n_attachments == 2
        and "." not in str(props.linker_smiles)
    )
    return props


# -------------------------------------------------------------------- source


def read_source(path: str | Path) -> list[dict[str, Any]]:
    """Read repair_tasks.jsonl, reporting the line a bad record is on."""
    source = Path(path)
    if not source.is_file():
        raise TaskBuildError(
            f"repair tasks file not found: {source}. It is protacbench's "
            "assessment output; set dataset.repair_tasks to where it lives."
        )
    rows: list[dict[str, Any]] = []
    with source.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskBuildError(f"{source}:{number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TaskBuildError(f"{source}:{number}: record is not an object")
            rows.append(record)
    if not rows:
        raise TaskBuildError(f"{source} is empty")
    return rows


def source_path(cfg: dict[str, Any]) -> Path:
    source = cfg["dataset"]["source"]
    if source != "repair_tasks":
        raise TaskBuildError(
            f"unknown dataset.source: {source} (this package only builds from "
            "protacbench's repair_tasks.jsonl)"
        )
    return resolve(cfg, cfg["dataset"]["repair_tasks"])


@dataclass
class Selection:
    """Which rows survived the filters, and why the others did not."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    read: int = 0
    eligible: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "eligible": self.eligible,
            "selected": len(self.rows),
            "dropped": dict(sorted(self.dropped.items())),
        }


def select_rows(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> Selection:
    """Apply the dataset filters, then cut the deterministic window."""
    dataset = cfg["dataset"]
    stages = {str(s) for s in (dataset.get("stages") or [])}
    verdicts = {str(v) for v in (dataset.get("verdicts") or [])}
    include_controls = bool(dataset.get("include_controls", True))

    selection = Selection(read=len(rows))
    eligible: list[dict[str, Any]] = []

    for row in rows:
        if stages and str(row.get("stage")) not in stages:
            selection.drop("stage_not_selected")
            continue
        if verdicts and str(row.get("verdict")) not in verdicts:
            selection.drop("verdict_not_selected")
            continue
        if not str(row.get("protac_smiles") or "").strip():
            selection.drop("no_molecule")
            continue

        # Exactly two fragments or the task is unscoreable: the whole reward
        # hangs on matching both of them back into the submission, and with
        # none recorded (the 33 `sanitization_failed` rows) or three (6 rows)
        # there is nothing to match against. The floor would reject every
        # submission, so the episode could only ever pay -1.
        fragments = row.get("fragment_smiles") or []
        if not isinstance(fragments, list) or len(fragments) != 2:
            selection.drop(f"fragment_count_{len(fragments) if fragments else 0}")
            continue

        if not include_controls and classify(row.get("failure_codes")) == "control":
            selection.drop("control_excluded")
            continue
        eligible.append(row)

    selection.eligible = len(eligible)
    if not eligible:
        raise TaskBuildError(
            "no row survived the dataset filters; check dataset.stages, "
            "dataset.verdicts and dataset.include_controls"
        )

    seed = dataset.get("shuffle_seed")
    if seed is not None:
        random.Random(int(seed)).shuffle(eligible)

    # `offset` then `limit` cut a deterministic window out of the shuffled
    # list, so a training set and a held-out set are disjoint by construction.
    # Changing the seed instead would reshuffle the same rows and hand back an
    # overlapping subset, leaking training molecules into the evaluation.
    offset = int(dataset.get("offset") or 0)
    if offset < 0:
        raise TaskBuildError("dataset.offset cannot be negative")
    if offset >= len(eligible):
        raise TaskBuildError(
            f"dataset.offset is {offset} but only {len(eligible)} rows survive "
            "the filters"
        )
    windowed = eligible[offset:]

    limit = dataset.get("limit")
    if limit is not None:
        windowed = windowed[: int(limit)]

    selection.rows = windowed
    return selection


# -------------------------------------------------------------------- tasks

# Field names this module may be handing an episode.LinkerTask that spells them
# differently. Every alias present on the dataclass is set to the same value,
# and one of each group must exist or the build stops: a silently dropped
# molecule or fragment pair would produce a dataset that looks fine and scores
# -1 on every episode.
_ALIASES: dict[str, tuple[str, ...]] = {
    "smiles": ("smiles", "protac_smiles"),
    "fragment_smiles": ("fragment_smiles", "fragments"),
    "optimisation_class": ("optimisation_class", "optimization_class"),
}
_REQUIRED = ("smiles", "fragment_smiles")


def _task_field_names() -> set[str]:
    if not is_dataclass(LinkerTask):
        return set()
    return {f.name for f in dataclass_fields(LinkerTask)}


def _make_task(record: dict[str, Any]) -> LinkerTask:
    """Build a LinkerTask from a flat record, tolerating its exact field set.

    `episode.LinkerTask` is the neighbouring module's contract and this one is
    its only writer. Keys it does not declare are tucked into `metadata`
    instead of raising, so a field renamed next door costs provenance rather
    than the whole build -- but a missing molecule or fragment pair still stops
    everything, because that one cannot be worked around.
    """
    names = _task_field_names()
    if not names:  # not a dataclass; let its own signature do the complaining
        return LinkerTask(**record)

    payload: dict[str, Any] = {}
    leftover: dict[str, Any] = {}

    for key, value in record.items():
        targets = [name for name in _ALIASES.get(key, (key,)) if name in names]
        if targets:
            for name in targets:
                payload[name] = value
        else:
            leftover[key] = value

    missing = [key for key in _REQUIRED if not any(n in payload for n in _ALIASES[key])]
    if missing:
        raise TaskBuildError(
            f"episode.LinkerTask declares no field for {', '.join(missing)}; "
            f"it has {sorted(names)}"
        )

    if leftover:
        if "metadata" in names:
            metadata = dict(payload.get("metadata") or {})
            metadata.update(leftover)
            payload["metadata"] = metadata
        else:
            LOG.warning(
                "episode.LinkerTask has no home for %s; dropping it from the "
                "task record",
                ", ".join(sorted(leftover)),
            )
    return LinkerTask(**payload)


def _is_measurable(task: LinkerTask) -> bool:
    """Did the assessor get descriptors for this task's molecule?

    `LinkerTask.baseline` is a flat dict, but the tolerant constructor above
    means it could be handed the dataclass instead, so both are read.
    """
    baseline = getattr(task, "baseline", None)
    if isinstance(baseline, dict):
        return bool(baseline.get("measurable", True))
    return bool(getattr(baseline, "measurable", True))


def task_id_for(cfg: dict[str, Any], row: dict[str, Any]) -> str:
    """The row's id, with the stage suffix stripped when configured.

    GRPO rewrites ids to `{id}@{generation}` and the trace collector to
    `{id}#{attempt}`. A third suffix makes group membership ambiguous, and an
    ambiguous group does not fail -- it quietly produces advantages computed
    over the wrong set of rollouts, which is the kind of bug that costs a
    training run rather than a traceback.
    """
    task_id = str(row.get("task_id") or "").strip()
    stage = str(row.get("stage") or "").strip()
    if not task_id:
        target = str(row.get("target_id") or "unknown")
        sample = str(row.get("sample_id") or "0")
        task_id = f"{target}-{sample}" + (f"-{stage}" if stage else "")
    if cfg["dataset"].get("strip_stage_suffix", True):
        # The row's own stage first, then the configured ones. A row that
        # arrives without a `stage` key would otherwise keep its suffix while
        # every row beside it lost one, and two spellings of the same task id
        # split one GRPO group in half without failing anything.
        for candidate in [stage, *cfg["dataset"].get("stages", [])]:
            if not candidate:
                continue
            suffix = f"-{candidate}"
            if task_id.endswith(suffix):
                task_id = task_id[: -len(suffix)]
                break
    return task_id


def task_from_row(cfg: dict[str, Any], row: dict[str, Any]) -> LinkerTask:
    """One repair row as a training task."""
    codes = clean_codes(row.get("failure_codes"))
    fragments = [str(f) for f in row["fragment_smiles"]]
    baseline = baseline_from_row(row)

    record: dict[str, Any] = {
        "task_id": task_id_for(cfg, row),
        "target_id": str(row.get("target_id") or "unknown"),
        "sample_id": str(row.get("sample_id") or ""),
        "verdict": str(row.get("verdict") or ""),
        "optimisation_class": classify(row.get("failure_codes")),
        "failure_codes": codes,
        "failure_detail": str(row.get("failure_detail") or ""),
        "smiles": str(row["protac_smiles"]),
        "linker_smiles": row.get("linker_smiles"),
        "fragment_smiles": fragments,
        # The length the repair must keep. It comes from the ternary-complex
        # geometry, not from preference, which is why it travels with the task
        # rather than being re-derived from the submission.
        "linker_size": int(_number(row.get("linker_size")) or 0),
        # Flat, and a plain dict rather than the dataclass, because that is
        # what LinkerTask holds: the prompt and the reward read the same keys
        # off it, so they cannot end up quoting different numbers.
        "baseline": baseline.as_dict(),
        "metadata": {
            "origin": "repair_tasks",
            "stage": str(row.get("stage") or ""),
            "source_task_id": row.get("task_id"),
            "source_verdict": row.get("verdict"),
            "source_failure_codes": list(row.get("failure_codes") or []),
            "score_total": _number(row.get("score_total")),
            "linker": dict(row.get("linker") or {}),
            # Remote-box paths, recorded for provenance only. Nothing in this
            # package may require them to exist -- they point at /home/jovyan.
            "paths": dict(row.get("paths") or {}),
        },
    }
    # The row's own `prompt` is deliberately not carried. The episode renders
    # its prompt from OPTIMISATION.md's template, and keeping the harness's
    # older text alongside it would give the package two prompts that can
    # drift apart without anything noticing.
    return _make_task(record)


@dataclass
class BuildReport:
    """What the build did, so the dataset can be audited without reading it."""

    source: str = ""
    read: int = 0
    eligible: int = 0
    selected: int = 0
    tasks: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_failure_code: dict[str, int] = field(default_factory=dict)
    targets: int = 0
    unknown_codes: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    unmeasurable_baselines: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "read": self.read,
            "eligible": self.eligible,
            "selected": self.selected,
            "tasks": self.tasks,
            "dropped": dict(sorted(self.dropped.items())),
            "by_class": {k: self.by_class.get(k, 0) for k in CLASS_ORDER},
            "by_verdict": dict(sorted(self.by_verdict.items())),
            "by_failure_code": dict(
                sorted(self.by_failure_code.items(), key=lambda kv: -kv[1])
            ),
            "targets": self.targets,
            "unknown_codes": self.unknown_codes,
            "unmeasurable_baselines": self.unmeasurable_baselines,
            "failures": self.failures,
        }


def build_tasks(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    disable_progress: bool | None = None,
) -> tuple[list[LinkerTask], BuildReport]:
    """Read the repair corpus and turn the selected rows into tasks."""
    path = source_path(cfg)
    rows = read_source(path)
    selection = select_rows(cfg, rows)
    chosen = selection.rows[:limit] if limit is not None else selection.rows

    report = BuildReport(
        source=str(path),
        read=selection.read,
        eligible=selection.eligible,
        selected=len(chosen),
        dropped=dict(selection.dropped),
        unknown_codes=unknown_codes(rows),
    )
    if report.unknown_codes:
        # Worth a warning rather than a failure: an unclassified code still
        # lands in `chemistry`, but the histogram the evaluation is stratified
        # on would be quietly describing the wrong thing.
        LOG.warning(
            "failure codes with no class in this module: %s "
            "(they are being treated as `chemistry`)",
            ", ".join(report.unknown_codes),
        )

    LOG.info(
        "building tasks from %d of %d rows in %s",
        len(chosen),
        selection.read,
        path,
    )

    tasks: list[LinkerTask] = []
    seen_ids: set[str] = set()
    targets: set[str] = set()
    classes: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    codes: Counter[str] = Counter()

    bar = progress.track(
        total=len(chosen),
        desc="building tasks",
        unit="row",
        disable=disable_progress,
    )
    for row in chosen:
        try:
            task = task_from_row(cfg, row)
        except TaskBuildError:
            raise  # a contract breach, not one bad row
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop 2,228
            LOG.warning(
                "row %s failed: %s: %s", row.get("task_id"), type(exc).__name__, exc
            )
            report.failures.append(
                {
                    "task_id": row.get("task_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            bar.update(1)
            continue

        identifier = str(getattr(task, "task_id", ""))
        if identifier in seen_ids:
            # Two rows sharing an id would be pooled into one GRPO group and
            # have their advantages computed against each other's rollouts.
            report.dropped["duplicate_task_id"] = (
                report.dropped.get("duplicate_task_id", 0) + 1
            )
            LOG.warning("duplicate task id %s; keeping the first", identifier)
            bar.update(1)
            continue
        seen_ids.add(identifier)

        classes[classify(row.get("failure_codes"))] += 1
        verdicts[str(row.get("verdict"))] += 1
        for code in clean_codes(row.get("failure_codes")):
            codes[code] += 1
        targets.add(str(row.get("target_id")))
        if not _is_measurable(task):
            report.unmeasurable_baselines += 1

        tasks.append(task)
        bar.update(1)
    bar.close()

    report.tasks = len(tasks)
    report.by_class = dict(classes)
    report.by_verdict = dict(verdicts)
    report.by_failure_code = dict(codes)
    report.targets = len(targets)

    LOG.info(
        "built %d tasks over %d targets (%s) in %s",
        report.tasks,
        report.targets,
        ", ".join(f"{name} {classes.get(name, 0)}" for name in CLASS_ORDER),
        progress.format_seconds(bar.elapsed_s),
    )
    return tasks, report


def write_tasks(path: str | Path, tasks: list[LinkerTask]) -> Path:
    """Write the tasks as JSONL, one record per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(_task_as_dict(task)) + "\n")
    return target


def read_tasks(
    cfg: dict[str, Any],
    path: str | Path | None = None,
    limit: int | None = None,
) -> list[LinkerTask]:
    """Read back what `write_tasks` wrote.

    Takes the config first and the path second, the way `moltrain.tasks`
    does, so a caller that knows only the config need not reach into
    `dataset.tasks_file` and resolve it itself -- and, more to the point, so
    the four call sites across grpo, traces, evaluate and experts all spell
    this one call the same way.
    """
    source = resolve(cfg, path or cfg["dataset"]["tasks_file"])
    if not source.is_file():
        raise TaskBuildError(
            f"tasks file not found: {source}. "
            "Run `python -m protactrain.tasks build`."
        )
    tasks: list[LinkerTask] = []
    with source.open() as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and len(tasks) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskBuildError(f"{source}:{number}: {exc}") from exc
            tasks.append(_task_from_dict(record))
    return tasks


def build(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    disable_progress: bool | None = None,
) -> tuple[Path, BuildReport]:
    """Build the dataset and write it to `dataset.tasks_file`."""
    tasks, report = build_tasks(cfg, limit=limit, disable_progress=disable_progress)
    if not tasks:
        raise TaskBuildError("no task survived the build; see the report above")
    path = write_tasks(resolve(cfg, cfg["dataset"]["tasks_file"]), tasks)
    return path, report


def _task_as_dict(task: LinkerTask) -> dict[str, Any]:
    """A task as JSON, using the dataclass' own serializer when it has one."""
    serialize = getattr(task, "as_dict", None)
    if callable(serialize):
        return serialize()

    record: dict[str, Any] = {}
    for item in dataclass_fields(task):
        value = getattr(task, item.name)
        if isinstance(value, LinkerProperties):
            value = value.as_dict()
        record[item.name] = value
    return record


def _task_from_dict(record: dict[str, Any]) -> LinkerTask:
    parse = getattr(LinkerTask, "from_dict", None)
    if callable(parse):
        return parse(record)

    payload = dict(record)
    names = _task_field_names()
    if "baseline" in names and isinstance(payload.get("baseline"), dict):
        payload["baseline"] = LinkerProperties(**payload["baseline"])
    return _make_task(payload)


# ------------------------------------------------------------------------- CLI


def _probe_environment(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Import the three dependencies in subprocesses, as hostcheck insists.

    molbench and protacbench are imported from a path rather than installed,
    so the path goes on PYTHONPATH for the probe's own subprocesses. Importing
    them here instead would mean one broken install takes the whole report
    down with it -- which is the failure hostcheck exists to prevent.
    """
    molbench_dir = resolve_sibling(cfg, "molbench")
    protacbench_dir = resolve_sibling(cfg, "protacbench")

    lines = [
        f"  molbench path: {molbench_dir}"
        f"{'' if (molbench_dir / 'server.py').is_file() else '  (no server.py)'}",
        f"  protacbench path: {protacbench_dir}"
        f"{'' if (protacbench_dir / 'grade.py').is_file() else '  (no grade.py)'}",
    ]

    previous = os.environ.get("PYTHONPATH")
    entries = [str(molbench_dir), str(protacbench_dir.parent)]
    if previous:
        entries.append(previous)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)
    try:
        reports, problems = hostcheck.report_imports(
            ("rdkit", "server", "protacbench.grade")
        )
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous

    labels = {"server": "molbench (server.py)", "protacbench.grade": "protacbench"}
    for report in reports:
        label = labels.get(report.name, report.name)
        lines.append(f"  {label}: {'ok ' + report.detail if report.ok else 'FAILED'}")
        if not report.ok:
            lines.append(f"    {report.detail}")
    return lines, problems


def _cmd_probe(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    print("imports")
    lines, problems = _probe_environment(cfg)
    for line in lines:
        print(line)
    for problem in problems:
        print(f"  ! {problem}")

    print("\nrepair corpus")
    try:
        path = source_path(cfg)
        rows = read_source(path)
    except TaskBuildError as exc:
        print(f"  ! {exc}")
        return 1

    print(f"  file: {path}")
    print(f"  rows: {len(rows)}")
    print(f"  targets: {len({str(r.get('target_id')) for r in rows})}")
    stages = Counter(str(r.get("stage")) for r in rows)
    verdicts = Counter(str(r.get("verdict")) for r in rows)
    print(f"  stages: {dict(stages)}")
    print(f"  verdicts: {dict(verdicts)}")
    missing = unknown_codes(rows)
    print(f"  unclassified failure codes: {', '.join(missing) if missing else 'none'}")

    try:
        selection = select_rows(cfg, rows)
    except TaskBuildError as exc:
        print(f"  ! {exc}")
        return 1
    print(f"  selected by the current config: {len(selection.rows)}")
    for reason, count in sorted(selection.dropped.items()):
        print(f"    dropped, {reason}: {count}")

    # Said plainly because the sibling package's probe cannot say it: this
    # stage is RDKit-free and CPU-only, and will run anywhere.
    print("\nthis stage needs no GPU, no docking binary and no model")
    return 0


def _cmd_classes(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    rows = read_source(source_path(cfg))
    selection = select_rows(cfg, rows)
    chosen = selection.rows[: args.limit] if args.limit else selection.rows

    classes: Counter[str] = Counter()
    codes_by_class: dict[str, Counter[str]] = {name: Counter() for name in CLASS_ORDER}
    targets_by_class: dict[str, set[str]] = {name: set() for name in CLASS_ORDER}
    for row in chosen:
        name = classify(row.get("failure_codes"))
        classes[name] += 1
        targets_by_class[name].add(str(row.get("target_id")))
        for code in clean_codes(row.get("failure_codes")):
            codes_by_class[name][code] += 1

    total = sum(classes.values()) or 1
    minimum = int(cfg["evaluate"].get("min_per_class") or 0)

    print(f"{len(chosen)} tasks selected of {selection.read} rows\n")
    print(f"{'class':<12}{'tasks':>7}{'share':>8}{'targets':>9}  top failure codes")
    for name in CLASS_ORDER:
        count = classes.get(name, 0)
        top = ", ".join(
            f"{code} {n}" for code, n in codes_by_class[name].most_common(3)
        ) or ("nothing to fix" if name == "control" else "-")
        print(
            f"{name:<12}{count:>7}{count / total:>7.0%}"
            f"{len(targets_by_class[name]):>9}  {top}"
        )

    if minimum:
        short = [n for n in CLASS_ORDER if classes.get(n, 0) < minimum]
        print(
            f"\nevaluate.min_per_class is {minimum}: "
            + (
                f"short in {', '.join(short)}"
                if short
                else "every class can be stratified"
            )
        )
    if args.json:
        print(
            json.dumps(
                {
                    "selection": selection.as_dict(),
                    "by_class": {n: classes.get(n, 0) for n in CLASS_ORDER},
                    "codes_by_class": {
                        n: dict(codes_by_class[n].most_common()) for n in CLASS_ORDER
                    },
                },
                indent=2,
            )
        )
    return 0


def _cmd_build(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    path, report = build(
        cfg,
        limit=args.limit,
        disable_progress=logs.progress_disabled(cfg, args),
    )
    report_path = path.with_name(path.stem + "_report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2))

    print(f"wrote {report.tasks} tasks over {report.targets} targets -> {path}")
    print(f"build report -> {report_path}")
    print(
        "  by class: "
        + ", ".join(f"{name} {report.by_class.get(name, 0)}" for name in CLASS_ORDER)
    )
    print(
        "  by verdict: "
        + ", ".join(f"{k} {v}" for k, v in sorted(report.by_verdict.items()))
    )
    if report.unmeasurable_baselines:
        print(
            f"  baselines with no recorded descriptors: "
            f"{report.unmeasurable_baselines}"
        )
    for reason, count in sorted(report.dropped.items()):
        print(f"  dropped, {reason}: {count}")
    for failure in report.failures:
        print(f"  ! {failure['task_id']}: {failure['error']}")

    example = read_tasks(cfg, path, limit=1)[0]
    print(f"\nfirst task: {example.task_id}  class {example.optimisation_class}")
    print(f"  codes: {', '.join(example.failure_codes) or 'none (control)'}")
    print(f"  start: {example.protac_smiles}")
    baseline = example.baseline or {}
    print(
        f"  baseline: span {baseline.get('span')} "
        f"SA {baseline.get('sa_score')} "
        f"cLogP {baseline.get('clogp')} "
        f"TPSA {baseline.get('tpsa')}"
    )
    return 0


COMMANDS = {
    "probe": _cmd_probe,
    "classes": _cmd_classes,
    "build": _cmd_build,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m protactrain.tasks",
        description="Build the PROTAC linker optimisation dataset.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--limit", type=int, default=None, help="cap the task count")
    parser.add_argument("--json", action="store_true", help="also dump JSON")
    parser.add_argument(
        "--tasks-file", default=None, help="override dataset.tasks_file"
    )
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    args.log_file = log_destination(cfg, args, f"tasks-{args.command}")
    logs.configure_from_args(cfg, args, stem=f"tasks-{args.command}")

    if args.tasks_file:
        cfg["dataset"]["tasks_file"] = args.tasks_file

    try:
        return COMMANDS[args.command](cfg, args)
    except TaskBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
