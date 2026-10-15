"""Running DiffSBDD to produce each episode's starting molecule.

DiffSBDD is driven as a subprocess rather than imported, for two reasons: its
dependency set (torch, pytorch-lightning, torch-scatter, openbabel) conflicts
with molbench's, so it wants its own environment; and `generate_ligands.py`
resolves `import utils` against the repository root, so it has to run with that
as the working directory anyway.

Sampling needs the GPU. Nothing in this module runs a diffusion pass unless a
CUDA device is actually present, because on CPU a pocket takes hours instead of
seconds and a silent fallback would look like a hang.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import hostcheck, logs
from .config import resolve
from .crossdocked import Pocket

LOG = logs.get_logger("diffsbdd")


class DiffSBDDError(Exception):
    """A sampling problem worth reporting verbatim."""


@dataclass
class Runtime:
    """What the DiffSBDD environment can actually do, as reported by itself."""

    python: str
    repo: str
    ok: bool
    torch_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    imports_model: bool = False
    checkpoint: str | None = None
    checkpoint_present: bool = False
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "repo": self.repo,
            "ok": self.ok,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "imports_model": self.imports_model,
            "checkpoint": self.checkpoint,
            "checkpoint_present": self.checkpoint_present,
            "problems": self.problems,
        }


# Printed by the DiffSBDD interpreter, parsed back here. Kept as a single
# literal so there is one place to read what the probe actually asks.
_PROBE = """
import json
report = {}
try:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
except Exception as exc:
    report["error"] = f"torch: {type(exc).__name__}: {exc}"
try:
    import lightning_modules  # noqa: F401
    report["imports_model"] = True
except Exception as exc:
    report["imports_model"] = False
    report["import_error"] = f"{type(exc).__name__}: {exc}"
print("MOLTRAIN_PROBE " + json.dumps(report))
"""


def repo_path(cfg: dict[str, Any]) -> Path:
    path = resolve(cfg, cfg["dataset"]["diffsbdd"]["repo_path"])
    if not (path / "generate_ligands.py").is_file():
        raise DiffSBDDError(
            f"generate_ligands.py not found under {path}; set "
            "dataset.diffsbdd.repo_path to the DiffSBDD checkout"
        )
    return path


def checkpoint_path(cfg: dict[str, Any]) -> Path:
    raw = cfg["dataset"]["diffsbdd"]["checkpoint"]
    if not raw:
        raise DiffSBDDError("dataset.diffsbdd.checkpoint is unset")
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (repo_path(cfg) / candidate)


def interpreter(cfg: dict[str, Any]) -> str:
    """The python that has DiffSBDD's dependencies, defaulting to this one."""
    configured = cfg["dataset"]["diffsbdd"]["python"]
    if not configured:
        return sys.executable
    found = shutil.which(configured) or (
        str(Path(configured).expanduser())
        if Path(configured).expanduser().is_file()
        else None
    )
    if not found:
        raise DiffSBDDError(f"dataset.diffsbdd.python is not executable: {configured}")
    return found


async def probe(cfg: dict[str, Any]) -> Runtime:
    """Ask the DiffSBDD environment about itself. Runs no diffusion pass."""
    problems: list[str] = []
    try:
        repo = repo_path(cfg)
    except DiffSBDDError as exc:
        return Runtime(python="?", repo="?", ok=False, problems=[str(exc)])
    try:
        python = interpreter(cfg)
    except DiffSBDDError as exc:
        return Runtime(python="?", repo=str(repo), ok=False, problems=[str(exc)])

    checkpoint = checkpoint_path(cfg)
    present = checkpoint.is_file()
    if not present:
        problems.append(f"checkpoint not found: {checkpoint}")

    # A probe that prints a version string has no business allocating one
    # thread per core, and in a thread-capped container trying to would stop
    # it importing torch at all.
    process = await asyncio.create_subprocess_exec(
        python,
        "-c",
        _PROBE,
        cwd=str(repo),
        env=hostcheck.thread_limited_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    report: dict[str, Any] = {}
    for line in stdout.decode(errors="replace").splitlines():
        if line.startswith("MOLTRAIN_PROBE "):
            report = json.loads(line[len("MOLTRAIN_PROBE ") :])
    if not report:
        text = stderr.decode(errors="replace")
        tail = [
            line
            for line in text.strip().splitlines()
            if line.strip() and not line.startswith("OpenBLAS ")
        ][-4:]
        problems.append(
            f"{python} could not report on its environment: "
            + (" / ".join(tail) or "no output")
        )
        if hostcheck.looks_like_blas_thread_failure(text):
            problems.append(hostcheck.BLAS_HINT)

    for key in ("error", "import_error"):
        if report.get(key):
            problems.append(report[key])
    if report and not report.get("cuda_available"):
        problems.append(
            "no CUDA device visible to DiffSBDD; sampling will refuse to run "
            "unless dataset.diffsbdd.allow_cpu is set"
        )
    if not report.get("imports_model"):
        problems.append("DiffSBDD's lightning_modules did not import")

    return Runtime(
        python=python,
        repo=str(repo),
        ok=not problems,
        torch_version=report.get("torch_version"),
        cuda_available=bool(report.get("cuda_available")),
        device_name=report.get("device_name"),
        imports_model=bool(report.get("imports_model")),
        checkpoint=str(checkpoint),
        checkpoint_present=present,
        problems=problems,
    )


def samples_dir(cfg: dict[str, Any], *, create: bool = False) -> Path:
    path = resolve(cfg, cfg["dataset"]["diffsbdd"]["samples_dir"])
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def build_command(cfg: dict[str, Any], pocket: Pocket, outfile: Path) -> list[str]:
    """The exact argv for one pocket, so it can be printed before it is run."""
    settings = cfg["dataset"]["diffsbdd"]
    command = [
        interpreter(cfg),
        "generate_ligands.py",
        str(checkpoint_path(cfg)),
        "--pdbfile",
        str(Path(pocket.pdb_file).resolve()),
        "--outfile",
        str(outfile),
        "--n_samples",
        str(int(settings["n_samples"])),
    ]

    if pocket.ref_ligand_file:
        command += ["--ref_ligand", str(Path(pocket.ref_ligand_file).resolve())]
    else:
        resi_list = pocket.metadata.get("resi_list")
        if not resi_list:
            raise DiffSBDDError(
                f"pocket {pocket.name} has neither a reference ligand nor a "
                "resi_list, so DiffSBDD cannot be told where the pocket is"
            )
        command += ["--resi_list", *[str(r) for r in resi_list]]

    for flag, key in (
        ("--batch_size", "batch_size"),
        ("--timesteps", "timesteps"),
        ("--num_nodes_lig", "num_nodes_lig"),
    ):
        value = settings.get(key)
        if value is not None:
            command += [flag, str(int(value))]
    if settings["sanitize"]:
        command.append("--sanitize")
    if settings["all_frags"]:
        command.append("--all_frags")
    if settings["relax"]:
        command.append("--relax")
    return command


async def sample_pocket(
    cfg: dict[str, Any],
    pocket: Pocket,
    *,
    refresh: bool = False,
    runtime: Runtime | None = None,
) -> Path:
    """Generate ligands for one pocket, reusing an existing SDF when present."""
    settings = cfg["dataset"]["diffsbdd"]
    outfile = samples_dir(cfg, create=True) / f"{pocket.name}.sdf"
    if outfile.is_file() and outfile.stat().st_size > 0 and not refresh:
        LOG.info("%s already sampled, reusing %s", pocket.name, outfile.name)
        return outfile

    if not settings["allow_cpu"]:
        runtime = runtime or await probe(cfg)
        if not runtime.cuda_available:
            raise DiffSBDDError(
                "refusing to sample DiffSBDD without a CUDA device. Run this on "
                "the GPU host, or set dataset.diffsbdd.allow_cpu to accept hours "
                "per pocket.\n  " + "\n  ".join(runtime.problems)
            )

    command = build_command(cfg, pocket, outfile)
    LOG.debug("running: %s", " ".join(command))
    LOG.info("sampling %d ligands for %s", int(settings["n_samples"]), pocket.name)
    environment = hostcheck.thread_limited_env()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo_path(cfg)),
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=float(settings["timeout_s"])
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise DiffSBDDError(
            f"DiffSBDD timed out after {settings['timeout_s']}s on {pocket.name}"
        ) from None

    if process.returncode != 0:
        tail = "\n".join(stderr.decode(errors="replace").strip().splitlines()[-12:])
        raise DiffSBDDError(
            f"DiffSBDD exited {process.returncode} on {pocket.name}:\n{tail}"
        )
    if not outfile.is_file() or outfile.stat().st_size == 0:
        raise DiffSBDDError(
            f"DiffSBDD reported success but wrote no molecules for {pocket.name}"
        )
    return outfile


def read_sdf_smiles(path: str | Path) -> tuple[list[str], list[str]]:
    """Canonical SMILES from a generated SDF, plus a note per rejected entry.

    Generated 3D structures are frequently chemically impossible, so each
    record is sanitized on its own and the failures are counted rather than
    aborting the file.
    """
    from rdkit import Chem, rdBase

    accepted: list[str] = []
    rejected: list[str] = []
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=True)
        for index, mol in enumerate(supplier):
            if mol is None:
                rejected.append(f"record {index}: unreadable")
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception as exc:  # noqa: BLE001 - a bad geometry is data
                rejected.append(f"record {index}: {type(exc).__name__}")
                continue
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if not fragments:
                rejected.append(f"record {index}: no atoms")
                continue
            largest = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
            smiles = Chem.MolToSmiles(largest)
            if smiles:
                accepted.append(smiles)
            else:
                rejected.append(f"record {index}: produced an empty SMILES")
    return accepted, rejected
