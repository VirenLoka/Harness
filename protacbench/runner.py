"""Driving Pro-GAT's sampling as a subprocess.

Pro-GAT is run rather than imported, for the same reasons DiffSBDD is in
`moltrain`: its dependency set (torch, the openbabel Python bindings) wants its
own environment, and `run_progat.py` resolves `import edm`, `import utils` and
`import compute_repair` against the repository root, so it has to run with that
as the working directory.

Sampling is 500 diffusion steps per molecule and needs the GPU. Nothing here
runs a model pass unless a CUDA device is actually present, because on CPU it
would take hours and look like a hang.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from moltrain import hostcheck, logs

from .config import resolve

LOG = logs.get_logger("protacbench.runner")


class RunnerError(Exception):
    """A sampling problem worth reporting verbatim."""


@dataclass
class Runtime:
    """What the Pro-GAT environment can do, as reported by itself."""

    python: str
    repo: str
    ok: bool
    torch_version: str | None = None
    cuda_available: bool = False
    device_name: str | None = None
    imports_model: bool = False
    has_openbabel: bool = False
    model_path: str | None = None
    model_present: bool = False
    repair_model_present: bool = False
    dataset_present: bool = False
    dataset_entries: int | None = None
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload.pop("problems", None)
        return payload


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
    from edm import EDM  # noqa: F401
    import compute_repair  # noqa: F401
    report["imports_model"] = True
except Exception as exc:
    report["imports_model"] = False
    report["import_error"] = f"{type(exc).__name__}: {exc}"
try:
    from openbabel import openbabel  # noqa: F401
    report["has_openbabel"] = True
except Exception as exc:
    report["has_openbabel"] = False
    report["openbabel_error"] = f"{type(exc).__name__}: {exc}"
print("PROTACBENCH_PROBE " + json.dumps(report))
"""


def repo_path(cfg: dict[str, Any]) -> Path:
    path = resolve(cfg, cfg["progat"]["repo_path"])
    if not (path / "run_progat.py").is_file():
        raise RunnerError(
            f"run_progat.py not found under {path}; set progat.repo_path to "
            "the Pro-GAT checkout"
        )
    return path


def interpreter(cfg: dict[str, Any]) -> str:
    configured = cfg["progat"]["python"]
    if not configured:
        return sys.executable
    found = shutil.which(configured) or (
        str(Path(configured).expanduser())
        if Path(configured).expanduser().is_file()
        else None
    )
    if not found:
        raise RunnerError(f"progat.python is not executable: {configured}")
    return found


def dataset_path(cfg: dict[str, Any]) -> Path:
    settings = cfg["progat"]
    return repo_path(cfg) / settings["data_path"] / f"{settings['test_data_prefix']}.pt"


async def probe(cfg: dict[str, Any]) -> Runtime:
    """Ask the Pro-GAT environment about itself. Runs no diffusion pass."""
    problems: list[str] = []
    try:
        repo = repo_path(cfg)
        python = interpreter(cfg)
    except RunnerError as exc:
        return Runtime(python="?", repo="?", ok=False, problems=[str(exc)])

    settings = cfg["progat"]
    model = repo / settings["model_path"]
    repair = repo / settings["repair_model"]
    data = dataset_path(cfg)

    for label, path in (
        ("diffusion checkpoint", model),
        ("Pro-GAT repair checkpoint", repair),
        ("test dataset", data),
    ):
        if not path.is_file():
            problems.append(f"{label} not found: {path}")

    entries: int | None = None
    if data.is_file():
        try:
            import torch

            entries = len(torch.load(data, map_location="cpu", weights_only=True))
        except Exception as exc:  # noqa: BLE001 - a probe reports
            problems.append(f"could not read {data}: {type(exc).__name__}: {exc}")

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
        if line.startswith("PROTACBENCH_PROBE "):
            report = json.loads(line[len("PROTACBENCH_PROBE ") :])

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

    for key in ("error", "import_error", "openbabel_error"):
        if report.get(key):
            problems.append(report[key])
    if report and not report.get("cuda_available"):
        problems.append(
            "no CUDA device visible to Pro-GAT; sampling will refuse to run "
            "unless progat.allow_cpu is set"
        )
    if report and not report.get("imports_model"):
        problems.append("Pro-GAT's edm/compute_repair modules did not import")

    return Runtime(
        python=python,
        repo=str(repo),
        ok=not problems,
        torch_version=report.get("torch_version"),
        cuda_available=bool(report.get("cuda_available")),
        device_name=report.get("device_name"),
        imports_model=bool(report.get("imports_model")),
        has_openbabel=bool(report.get("has_openbabel")),
        model_path=str(model),
        model_present=model.is_file(),
        repair_model_present=repair.is_file(),
        dataset_present=data.is_file(),
        dataset_entries=entries,
        problems=problems,
    )


def build_command(cfg: dict[str, Any], output_dir: Path) -> list[str]:
    """The exact argv for one Pro-GAT run, so it can be printed before running.

    `linker_size 0` is deliberate and load-bearing: `run_progat.py` only
    installs its fixed-size `sample_fn` when the value is positive, and
    `Trainer.sample_chain` then falls back to each entry's own
    `linker_mask.sum()`. So zero means "use the reference linker length for
    every PROTAC", which is what the test set was built around.
    """
    settings = cfg["progat"]
    return [
        interpreter(cfg),
        "run_progat.py",
        "--model_path",
        str(settings["model_path"]),
        "--data_path",
        str(settings["data_path"]),
        "--test_data_prefix",
        str(settings["test_data_prefix"]),
        "--linker_size",
        str(int(settings["linker_size"])),
        "--n_samples",
        str(int(settings["n_samples"])),
        "--batch_size",
        str(int(settings["batch_size"])),
        "--output_dir",
        str(output_dir),
    ]


async def sample(
    cfg: dict[str, Any], *, runtime: Runtime | None = None, refresh: bool = False
) -> Path:
    """Run Pro-GAT end to end: diffusion, conversion, metrics and repair."""
    settings = cfg["progat"]
    output_dir = resolve(cfg, settings["output_dir"])

    # Reuse only what `assess` would actually find. Testing the directory for
    # any content at all is not the same question: an interrupted run leaves
    # the per-PROTAC subdirectories behind without writing an XYZ into them,
    # and the old check then reported "reusing them" for a directory that
    # assessment reads as empty. One definition of "has samples", shared.
    from .triage import discover

    if not refresh:
        existing = discover(output_dir, "generated")
        if existing:
            LOG.info(
                "%s already holds %d sample(s), reusing them",
                output_dir,
                len(existing),
            )
            return output_dir
        if output_dir.is_dir() and any(output_dir.iterdir()):
            LOG.info(
                "%s has leftover files but no readable samples, so sampling "
                "will run again",
                output_dir,
            )

    if not settings["allow_cpu"]:
        runtime = runtime or await probe(cfg)
        if not runtime.cuda_available:
            raise RunnerError(
                "refusing to sample without a CUDA device. Run this on the GPU "
                "host, or set progat.allow_cpu to accept hours per batch.\n  "
                + "\n  ".join(runtime.problems)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(cfg, output_dir)
    LOG.debug("running: %s", " ".join(command))
    LOG.info(
        "sampling %d per PROTAC at linker size %s",
        int(settings["n_samples"]),
        settings["linker_size"] or "native",
    )

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo_path(cfg)),
        env=hostcheck.thread_limited_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=float(settings["timeout_s"])
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RunnerError(f"Pro-GAT timed out after {settings['timeout_s']}s") from None

    for line in stdout.decode(errors="replace").splitlines():
        if line.strip():
            LOG.debug("progat: %s", line.rstrip())

    if process.returncode != 0:
        tail = "\n".join(stderr.decode(errors="replace").strip().splitlines()[-15:])
        raise RunnerError(f"Pro-GAT exited {process.returncode}:\n{tail}")
    return output_dir
