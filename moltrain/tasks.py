"""Building `tasks.jsonl`: the dataset every later stage trains and evaluates on.

One task is a pocket plus a starting molecule plus that molecule's measured
baseline. The reward thresholds are relative to the baseline, so it is recorded
once here instead of being re-derived on every rollout.

The stages are separate and resumable because they need different machines'
worth of resources: `pockets` is pure file reading, `sample` needs the GPU, and
`build` needs Open Babel and Vina. Each skips work it already has on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import crossdocked, diffsbdd, logs, progress
from .config import ConfigError, load_config, resolve
from .crossdocked import Pocket
from .episode import Task
from .oracle import Oracle
from .toolkit import Toolkit, import_molbench

DEFAULT_BOX = 20.0

LOG = logs.get_logger("tasks")


class TaskBuildError(Exception):
    """A dataset problem worth reporting verbatim."""


# --------------------------------------------------------------- docking boxes


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    ligand_atoms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "center": [round(v, 3) for v in self.center],
            "size": [round(v, 3) for v in self.size],
            "ligand_atoms": self.ligand_atoms,
        }


def ligand_box(cfg: dict[str, Any], sdf_path: str | Path) -> Box:
    """A box wrapping the reference ligand, padded and clamped per axis.

    Sizing the box to the ligand rather than using a fixed cube keeps Vina's
    search space close to the pocket it was measured in, which matters because
    binding affinity is the threshold most sensitive to search volume.
    """
    from rdkit import Chem

    box_cfg = cfg["dataset"]["box"]
    padding = float(box_cfg["padding"])
    low, high = float(box_cfg["min_size"]), float(box_cfg["max_size"])

    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        mol = next((m for m in supplier if m is not None), None)
    except OSError as exc:
        raise TaskBuildError(
            f"reference ligand unreadable: {sdf_path} ({exc})"
        ) from exc
    if mol is None or mol.GetNumConformers() == 0:
        raise TaskBuildError(f"no readable 3D reference ligand in {sdf_path}")

    positions = mol.GetConformer().GetPositions()
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = tuple(float(v) for v in (minimum + maximum) / 2.0)
    size = tuple(
        float(min(max(float(extent) + 2 * padding, low), high))
        for extent in (maximum - minimum)
    )
    return Box(center=center, size=size, ligand_atoms=mol.GetNumAtoms())


def reference_smiles(sdf_path: str | Path) -> str | None:
    """The crystal ligand as SMILES, when it can be sanitized."""
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
        for mol in supplier:
            if mol is not None:
                return Chem.MolToSmiles(mol)
    return None


# ------------------------------------------------------------- receptor set-up


def stage_pocket_pdb(cfg: dict[str, Any], pocket: Pocket) -> Path:
    """Copy the pocket PDB to a file named after the pocket.

    molbench names the prepared receptor after the PDB file's stem, and
    CrossDocked reuses stems like `1a0q_A_rec.pdb` across directories. Staging
    under the pocket's own unique name keeps two pockets from overwriting each
    other's receptor and, worse, each other's docking box.
    """
    source = Path(pocket.pdb_file)
    if not source.is_file():
        raise TaskBuildError(f"pocket PDB missing: {source}")
    staging = resolve(cfg, cfg["dataset"]["pockets"]["staging_dir"])
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / f"{pocket.name}.pdb"
    staged.write_bytes(source.read_bytes())
    return staged


async def prepare_receptor(cfg: dict[str, Any], server: Any, pocket: Pocket) -> dict:
    """Prepare the pocket for docking and record the box alongside it."""
    staged = stage_pocket_pdb(cfg, pocket)

    if pocket.ref_ligand_file and Path(pocket.ref_ligand_file).is_file():
        box = ligand_box(cfg, pocket.ref_ligand_file)
    else:
        raise TaskBuildError(
            f"pocket {pocket.name} has no reference ligand on disk, so the "
            "docking box cannot be placed"
        )

    payload = await server.prepare_target_receptor(
        local_pdb_path=str(staged),
        binding_site_center=list(box.center),
        box_size=list(box.size),
    )
    payload["box"] = box.as_dict()
    payload["staged_pdb_file"] = str(staged)
    return payload


# --------------------------------------------------------- starting molecules


def filter_candidates(
    cfg: dict[str, Any], smiles_list: list[str], server: Any
) -> tuple[list[str], dict[str, int]]:
    """Keep the generated molecules that are worth spending an episode on."""
    from rdkit import Chem, rdBase

    filters = cfg["dataset"]["filters"]
    minimum = int(filters["min_heavy_atoms"])
    maximum = int(filters["max_heavy_atoms"])
    require_ring = bool(filters["require_ring"])
    max_sa = filters["max_sa_score"]

    rejected = {
        "unparseable": 0,
        "too_small": 0,
        "too_large": 0,
        "no_ring": 0,
        "sa_too_high": 0,
        "duplicate": 0,
    }
    kept: list[str] = []
    seen: set[str] = set()

    with rdBase.BlockLogs():
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rejected["unparseable"] += 1
                continue
            canonical = Chem.MolToSmiles(mol)
            if filters["dedupe"] and canonical in seen:
                rejected["duplicate"] += 1
                continue
            heavy = mol.GetNumHeavyAtoms()
            if heavy < minimum:
                rejected["too_small"] += 1
                continue
            if heavy > maximum:
                rejected["too_large"] += 1
                continue
            if require_ring and mol.GetRingInfo().NumRings() == 0:
                rejected["no_ring"] += 1
                continue
            if max_sa is not None:
                sa = server.get_molecule_metrics(smiles=canonical)["sa_score"]
                if sa > float(max_sa):
                    rejected["sa_too_high"] += 1
                    continue
            seen.add(canonical)
            kept.append(canonical)

    limit = filters["starts_per_pocket"]
    if limit is not None:
        kept = kept[: int(limit)]
    return kept, rejected


async def starting_molecules(
    cfg: dict[str, Any],
    pocket: Pocket,
    server: Any,
    *,
    smiles_pool: list[str] | None = None,
    refresh: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """The molecules this pocket's episodes begin from, per the configured source."""
    source = cfg["dataset"]["source"]
    provenance: dict[str, Any] = {"source": source}

    if source == "diffsbdd":
        sdf = await diffsbdd.sample_pocket(cfg, pocket, refresh=refresh)
        raw, rejected_records = diffsbdd.read_sdf_smiles(sdf)
        provenance["sdf_file"] = str(sdf)
        provenance["generated"] = len(raw)
        provenance["unreadable_records"] = len(rejected_records)
    elif source == "reference_ligand":
        if not pocket.ref_ligand_file:
            raise TaskBuildError(f"pocket {pocket.name} has no reference ligand")
        smiles = reference_smiles(pocket.ref_ligand_file)
        if smiles is None:
            raise TaskBuildError(
                f"reference ligand for {pocket.name} would not sanitize"
            )
        raw = [smiles]
        provenance["generated"] = 1
    elif source == "smiles_file":
        if smiles_pool is None:
            raise TaskBuildError("dataset.smiles_file produced no molecules")
        raw = list(smiles_pool)
        provenance["generated"] = len(raw)
    else:
        raise TaskBuildError(
            f"unknown dataset.source: {source} "
            "(use diffsbdd, reference_ligand or smiles_file)"
        )

    kept, rejected = filter_candidates(cfg, raw, server)
    provenance["kept"] = len(kept)
    provenance["rejected"] = {k: v for k, v in rejected.items() if v}
    return kept, provenance


def load_smiles_file(cfg: dict[str, Any]) -> list[str]:
    path = cfg["dataset"]["smiles_file"]
    if not path:
        raise TaskBuildError(
            "dataset.source is `smiles_file` but dataset.smiles_file is unset"
        )
    resolved = resolve(cfg, path)
    if not resolved.is_file():
        raise TaskBuildError(f"dataset.smiles_file not found: {resolved}")
    molecules: list[str] = []
    for line in resolved.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        molecules.append(text.split()[0])
    return molecules


# -------------------------------------------------------------------- building


@dataclass
class BuildReport:
    """What the build did, so a long GPU run leaves an audit trail."""

    tasks: int = 0
    pockets_used: int = 0
    pockets_failed: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    baselines_docked: int = 0
    baselines_missing_affinity: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": self.tasks,
            "pockets_used": self.pockets_used,
            "pockets_failed": self.pockets_failed,
            "baselines_docked": self.baselines_docked,
            "baselines_missing_affinity": self.baselines_missing_affinity,
            "provenance": self.provenance,
        }


async def build_tasks(
    cfg: dict[str, Any],
    *,
    refresh: bool = False,
    limit: int | None = None,
    disable_progress: bool | None = None,
) -> tuple[list[Task], BuildReport]:
    """Pockets plus starting molecules plus baselines, ready to write out."""
    server = import_molbench(cfg)
    toolkit = Toolkit(cfg, server)
    oracle = Oracle(cfg, toolkit)
    dock_baseline = bool(cfg["dataset"]["baseline"]["dock"])

    pockets = crossdocked.select_pockets(cfg)
    if limit is not None:
        pockets = pockets[:limit]
    if not pockets:
        raise TaskBuildError("no pockets selected; check dataset.pockets")

    pool = load_smiles_file(cfg) if cfg["dataset"]["source"] == "smiles_file" else None

    LOG.info(
        "building tasks for %d pockets (source=%s, baseline docking=%s)",
        len(pockets),
        cfg["dataset"]["source"],
        "on" if dock_baseline else "off",
    )

    report = BuildReport()
    tasks: list[Task] = []

    # The bar tracks pockets, but the phase within a pocket is what makes it
    # legible: protonating a receptor takes about a minute, so a counter alone
    # still looks stuck.
    bar = progress.track(
        total=len(pockets),
        desc="building tasks",
        unit="pocket",
        disable=disable_progress,
    )
    for pocket in pockets:
        try:
            bar.postfix(f"{pocket.name} preparing receptor")
            LOG.debug("preparing receptor for %s", pocket.name)
            receptor = await prepare_receptor(cfg, server, pocket)

            bar.postfix(f"{pocket.name} reading start molecules")
            starts, provenance = await starting_molecules(
                cfg, pocket, server, smiles_pool=pool, refresh=refresh
            )
        except Exception as exc:  # noqa: BLE001 - one bad pocket must not stop 99
            LOG.warning("%s failed: %s: %s", pocket.name, type(exc).__name__, exc)
            report.pockets_failed.append(
                {
                    "pocket": pocket.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            bar.update(1)
            continue

        if not starts:
            LOG.warning("%s: no starting molecule survived the filters", pocket.name)
            report.pockets_failed.append(
                {
                    "pocket": pocket.name,
                    "error": "no starting molecule survived filters",
                }
            )
            bar.update(1)
            continue

        provenance["pocket"] = pocket.name
        report.provenance.append(provenance)
        report.pockets_used += 1
        LOG.info(
            "%s: %d start molecules kept of %d generated",
            pocket.name,
            provenance.get("kept", len(starts)),
            provenance.get("generated", len(starts)),
        )

        for index, smiles in enumerate(starts):
            phase = "docking baseline" if dock_baseline else "measuring baseline"
            bar.postfix(f"{pocket.name} {phase} {index + 1}/{len(starts)}")
            baseline = await oracle.evaluate(
                smiles,
                receptor["receptor_file"] if dock_baseline else None,
                is_final=True,
                label=f"baseline-{pocket.name}-{index}",
            )
            if dock_baseline:
                if baseline.binding_affinity is None:
                    report.baselines_missing_affinity += 1
                    LOG.warning(
                        "%s start %d: docking produced no affinity", pocket.name, index
                    )
                else:
                    report.baselines_docked += 1
                    LOG.debug(
                        "%s start %d: affinity %.3f kcal/mol",
                        pocket.name,
                        index,
                        baseline.binding_affinity,
                    )

            tasks.append(
                Task(
                    task_id=f"{pocket.name}-{index:02d}",
                    smiles=smiles,
                    target=pocket.target,
                    receptor_file=receptor["receptor_file"],
                    origin=provenance["source"],
                    baseline=baseline,
                    metadata={
                        "pocket": pocket.as_dict(),
                        "receptor": {
                            "protein_pdb_file": receptor.get("protein_pdb_file"),
                            "binding_site_center": receptor.get("binding_site_center"),
                            "box_size": receptor.get("box_size"),
                        },
                        "provenance": {
                            k: v for k, v in provenance.items() if k != "pocket"
                        },
                    },
                )
            )

        bar.update(1)

    bar.close()
    report.tasks = len(tasks)
    LOG.info(
        "built %d tasks from %d pockets (%d pockets failed) in %s",
        len(tasks),
        report.pockets_used,
        len(report.pockets_failed),
        progress.format_seconds(bar.elapsed_s),
    )
    return tasks, report


def write_tasks(cfg: dict[str, Any], tasks: list[Task]) -> Path:
    path = resolve(cfg, cfg["dataset"]["tasks_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task.as_dict()) + "\n")
    return path


def read_tasks(cfg: dict[str, Any], path: str | Path | None = None) -> list[Task]:
    resolved = resolve(cfg, path or cfg["dataset"]["tasks_file"])
    if not resolved.is_file():
        raise TaskBuildError(
            f"tasks file not found: {resolved}. Run `python -m moltrain.tasks build`."
        )
    tasks: list[Task] = []
    for line in resolved.read_text().splitlines():
        if line.strip():
            tasks.append(Task.from_dict(json.loads(line)))
    return tasks


# ------------------------------------------------------------------------- CLI


async def _cmd_probe(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    print("molbench")
    try:
        server = import_molbench(cfg)
        print(f"  imported from {resolve(cfg, cfg['molbench']['path'])}")
        missing = [
            name
            for name in cfg["episode"]["tools"]
            if not hasattr(server, name) and not name.startswith("dock_and_score")
        ]
        print(f"  tools missing from server.py: {missing or 'none'}")
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\ndocking binaries")
    from . import hostcheck

    binary_lines, binary_problems = hostcheck.report_binaries(
        ("vina", "smina", "obabel")
    )
    for line in binary_lines:
        print(line)
    for problem in binary_problems:
        print(f"  ! {problem}")

    print("\nDiffSBDD")
    runtime = await diffsbdd.probe(cfg)
    for key, value in runtime.as_dict().items():
        if key != "problems":
            print(f"  {key}: {value}")
    for problem in runtime.problems:
        print(f"  ! {problem}")

    print("\npockets")
    root = cfg["dataset"]["pockets"].get("crossdocked_root")
    if cfg["dataset"]["pockets"]["source"] == "crossdocked" and root:
        for problem in crossdocked.verify_root(root):
            print(f"  ! {problem}")
    try:
        pockets = crossdocked.select_pockets(cfg)
        print(f"  selected: {len(pockets)}")
        absent = crossdocked.missing_files(pockets)
        print(f"  files missing on disk: {len(absent)}")
        for path in absent[:5]:
            print(f"    - {path}")
        if len(absent) > 5:
            print(f"    ... and {len(absent) - 5} more")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


async def _cmd_pockets(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    pockets = crossdocked.select_pockets(cfg)
    for pocket in pockets:
        on_disk = "ok" if Path(pocket.pdb_file).is_file() else "MISSING"
        print(f"{pocket.name:>16}  {on_disk:>7}  {pocket.target}")
    print(f"\n{len(pockets)} pockets")
    if args.json:
        print(json.dumps([p.as_dict() for p in pockets], indent=2))
    return 0


async def _cmd_plan(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Print the DiffSBDD commands without running any of them."""
    pockets = crossdocked.select_pockets(cfg)
    if args.limit:
        pockets = pockets[: args.limit]
    directory = diffsbdd.samples_dir(cfg)
    print(f"# cwd: {diffsbdd.repo_path(cfg)}")
    for pocket in pockets:
        outfile = directory / f"{pocket.name}.sdf"
        state = "exists, would skip" if outfile.is_file() else "would run"
        command = diffsbdd.build_command(cfg, pocket, outfile)
        print(f"\n# {pocket.name}: {state}")
        print(" ".join(command))
    return 0


async def _cmd_sample(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    pockets = crossdocked.select_pockets(cfg)
    if args.limit:
        pockets = pockets[: args.limit]
    runtime = await diffsbdd.probe(cfg)
    if not runtime.cuda_available and not cfg["dataset"]["diffsbdd"]["allow_cpu"]:
        print("refusing to sample: no CUDA device visible to DiffSBDD")
        for problem in runtime.problems:
            print(f"  ! {problem}")
        return 2

    LOG.info(
        "sampling %d pockets with %s on %s",
        len(pockets),
        Path(runtime.checkpoint or "?").name,
        runtime.device_name or "cpu",
    )

    failures = 0
    bar = progress.track(
        total=len(pockets),
        desc="sampling pockets",
        unit="pocket",
        disable=logs.progress_disabled(cfg, args),
    )
    for pocket in pockets:
        bar.postfix(f"{pocket.name} generating")
        started = time.monotonic()
        try:
            path = await diffsbdd.sample_pocket(
                cfg, pocket, refresh=args.refresh, runtime=runtime
            )
            accepted, rejected = diffsbdd.read_sdf_smiles(path)
            LOG.info(
                "%s: %d usable, %d unreadable in %s -> %s",
                pocket.name,
                len(accepted),
                len(rejected),
                progress.format_seconds(time.monotonic() - started),
                path,
            )
        except Exception as exc:  # noqa: BLE001 - keep going through the set
            failures += 1
            LOG.error("%s: FAILED %s", pocket.name, exc)
        bar.update(1)
    bar.close()

    print(f"\n{len(pockets) - failures}/{len(pockets)} pockets sampled")
    return 1 if failures == len(pockets) else 0


async def _cmd_build(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    tasks, report = await build_tasks(
        cfg,
        refresh=args.refresh,
        limit=args.limit,
        disable_progress=logs.progress_disabled(cfg, args),
    )
    if not tasks:
        print("no tasks were built")
        print(json.dumps(report.as_dict(), indent=2))
        return 1

    path = write_tasks(cfg, tasks)
    report_path = path.with_name(path.stem + "_report.json")
    report_path.write_text(json.dumps(report.as_dict(), indent=2))

    print(f"wrote {report.tasks} tasks from {report.pockets_used} pockets -> {path}")
    print(f"build report -> {report_path}")
    if cfg["dataset"]["baseline"]["dock"]:
        print(
            f"baselines docked: {report.baselines_docked}, "
            f"without an affinity: {report.baselines_missing_affinity}"
        )
    for failure in report.pockets_failed:
        print(f"  ! {failure['pocket']}: {failure['error']}")

    example = tasks[0]
    print(f"\nfirst task: {example.task_id}  {example.target}")
    print(f"  start: {example.smiles}")
    if example.baseline:
        print(
            f"  baseline: SA {example.baseline.sa_score} "
            f"QED {example.baseline.qed} "
            f"affinity {example.baseline.binding_affinity}"
        )
    return 0


COMMANDS = {
    "probe": _cmd_probe,
    "pockets": _cmd_pockets,
    "plan": _cmd_plan,
    "sample": _cmd_sample,
    "build": _cmd_build,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m moltrain.tasks",
        description="Build the molecular-optimization dataset.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--limit", type=int, default=None, help="cap the pocket count")
    parser.add_argument(
        "--refresh", action="store_true", help="redo work that is already on disk"
    )
    parser.add_argument("--json", action="store_true", help="also dump JSON")
    parser.add_argument(
        "--source",
        default=None,
        choices=["diffsbdd", "reference_ligand", "smiles_file"],
        help="override dataset.source; reference_ligand needs no GPU",
    )
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

    logs.configure_from_args(cfg, args, stem=f"tasks-{args.command}")

    if args.source:
        cfg["dataset"]["source"] = args.source
    if args.tasks_file:
        cfg["dataset"]["tasks_file"] = args.tasks_file

    try:
        return asyncio.run(COMMANDS[args.command](cfg, args))
    except (
        TaskBuildError,
        crossdocked.CrossDockedError,
        diffsbdd.DiffSBDDError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
