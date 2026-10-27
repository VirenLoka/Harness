"""Driving REINVENT as a subprocess: configs in, samples out.

REINVENT is invoked through its `reinvent` console script rather than
imported. It pins `torch==2.12.0` and `rdkit>=2025.09.1`, which is a different
environment from the one this package needs, and shelling out keeps the two
apart -- `adclinker` needs only RDKit, so everything except the generation
step itself runs and tests anywhere.

Four details of REINVENT's interface are load-bearing and were established by
reading its source rather than its docs.

**Warheads are pipe-separated with bare `*`.** `reinvent/chemistry/tokens.py`
sets the separator to `|`, and LinkInvent's *input* vocabulary tokenises `*`
as a single character. There is no `[*:1]` token, so a mapped dummy fails
validation.

**`num_smiles` is per input line, not per run.** The sampler multiplies it by
the number of warhead pairs, so a file of twelve pairs with `num_smiles = 500`
produces six thousand molecules.

**The transfer-learning file is TAB separated**, two columns, warheads then
linker. The docstring in `runmodes/TL/linkinvent.py` says commas or spaces are
accepted; the code passes no delimiter and `read_smiles_csv_file` defaults to
a tab, so anything else collapses into one field.

**A GPU is optional.** `device` defaults to `cpu` and nothing in the execution
path hardcodes CUDA, so sampling the LinkInvent RNN runs on a laptop. It is
only slow, not impossible.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import resolve

# `reinvent/runmodes/samplers/run_sampling.py` HEADERS.
SAMPLE_COLUMNS = ("SMILES", "SMILES_state", "Warheads", "Linker", "NLL")

# `reinvent/models/model_factory/sample_batch.py` SmilesState.
STATE_INVALID = 0
STATE_VALID = 1
STATE_DUPLICATE = 2


# REINVENT reports peak memory after a run finishes, and in a restricted
# container `resource.getrusage` raises PermissionError. It guards the import
# against Windows but not the call, so a completed sampling run dies on the
# way out with a non-zero exit and nothing to show for it.
#
# This shim neutralises that one call and then hands straight over to
# REINVENT's own entry point, so nothing else about the run changes. It is
# used in place of the console script when `reinvent.sandbox_safe` is set and
# the environment's interpreter can be located.
RESOURCE_SHIM = """
import sys, resource as _resource

_original = _resource.getrusage


class _Zero:
    def __getattr__(self, name):
        return 0


def _tolerant(who):
    try:
        return _original(who)
    except Exception:
        # A sandbox that refuses getrusage should not lose a finished run.
        return _Zero()


_resource.getrusage = _tolerant

from reinvent.Reinvent import main_script

sys.argv = ["reinvent"] + sys.argv[1:]
sys.exit(main_script())
"""


class ReinventError(RuntimeError):
    """REINVENT could not be run, or did not produce usable output."""


@dataclass
class Runtime:
    """What this host can do, established without running a model."""

    executable: str | None = None
    prior: str | None = None
    prior_present: bool = False
    device: str = "cpu"
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "executable": self.executable or "NOT FOUND",
            "prior": self.prior or "NOT SET",
            "prior_present": self.prior_present,
            "device": self.device,
        }


def probe(cfg: dict[str, Any]) -> Runtime:
    """Check the executable and the prior. Runs no model pass."""
    settings = cfg["reinvent"]
    runtime = Runtime(device=str(settings["device"]))

    executable = settings["executable"]
    found = shutil.which(executable) or (
        executable if Path(executable).is_file() else None
    )
    runtime.executable = found
    if not found:
        runtime.problems.append(
            f"the `{executable}` command is not on PATH. Install REINVENT "
            "into an environment and point reinvent.executable at its "
            "console script."
        )

    prior = resolve(cfg, settings["prior_file"])
    runtime.prior = str(prior)
    runtime.prior_present = prior.is_file()
    if not runtime.prior_present:
        runtime.problems.append(
            f"the LinkInvent prior is missing: {prior}\n"
            "    Recover it from REINVENT4's own git history, which still "
            "has it:\n"
            "      git -C REINVENT4 show "
            "1561377efd74e7954fb4a87f88a4b81a0eb6e208~1:priors/linkinvent.prior "
            "> REINVENT4/priors/linkinvent.prior\n"
            "    or download it from https://doi.org/10.5281/zenodo.15641296"
        )
    return runtime


# -------------------------------------------------------------- the TOML files


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_table(name: str, values: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines)


def write_warheads_file(path: Path, pairs: Iterable[str]) -> Path:
    """One `warhead1|warhead2` pair per line.

    No header: `read_smiles_csv_file` runs with `header=False`, so a column
    name would be parsed as a SMILES and rejected as a bad token.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in pairs if line.strip()]
    if not lines:
        raise ReinventError("no warhead pairs to write")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_sampling_config(
    cfg: dict[str, Any], warheads_file: Path, output_csv: Path
) -> Path:
    """The sampling TOML. Returns where it was written."""
    settings = cfg["reinvent"]
    path = resolve(cfg, settings["sampling_config"])
    path.parent.mkdir(parents=True, exist_ok=True)

    model = settings["model_file"] or settings["prior_file"]
    body = "\n\n".join(
        [
            f'run_type = "sampling"\ndevice = {_toml_value(settings["device"])}',
            _toml_table(
                "parameters",
                {
                    "model_file": str(resolve(cfg, model)),
                    "smiles_file": str(warheads_file),
                    "output_file": str(output_csv),
                    # Per warhead pair, not per run.
                    "num_smiles": int(settings["num_smiles"]),
                    "unique_molecules": bool(settings["unique_molecules"]),
                    "randomize_smiles": bool(settings["randomize_smiles"]),
                    "isomeric_smiles": True,
                    "temperature": float(settings["temperature"]),
                },
            ),
        ]
    )
    path.write_text(body + "\n")
    return path


def write_transfer_learning_config(
    cfg: dict[str, Any], training_file: Path, validation_file: Path | None
) -> Path:
    """The fine-tuning TOML."""
    settings = cfg["reinvent"]
    tl = cfg["finetune"]
    path = resolve(cfg, tl["config_file"])
    path.parent.mkdir(parents=True, exist_ok=True)

    parameters: dict[str, Any] = {
        "input_model_file": str(resolve(cfg, settings["prior_file"])),
        "output_model_file": str(resolve(cfg, tl["output_model"])),
        "smiles_file": str(training_file),
        "num_epochs": int(tl["num_epochs"]),
        "batch_size": int(tl["batch_size"]),
        # REINVENT rejects anything below 100 outright.
        "sample_batch_size": max(100, int(tl["sample_batch_size"])),
        "save_every_n_epochs": int(tl["save_every_n_epochs"]),
        "num_refs": 0,
    }
    if validation_file is not None:
        parameters["validation_smiles_file"] = str(validation_file)

    body = "\n\n".join(
        [
            f'run_type = "transfer_learning"\ndevice = {_toml_value(settings["device"])}',
            _toml_table("parameters", parameters),
            _toml_table(
                "scheduler",
                {
                    "lr": float(tl["learning_rate"]),
                    "step": int(tl["lr_step"]),
                    "gamma": float(tl["lr_gamma"]),
                },
            ),
        ]
    )
    path.write_text(body + "\n")
    return path


# ---------------------------------------------------------------- the running


def environment_python(executable: str | None) -> Path | None:
    """The interpreter beside REINVENT's console script, if there is one.

    Console scripts live in the same `bin` directory as the interpreter of
    the environment that installed them, so this finds the right Python
    without having to be told which environment REINVENT is in.
    """
    if not executable:
        return None
    for name in ("python", "python3"):
        candidate = Path(executable).resolve().parent / name
        if candidate.is_file():
            return candidate
    return None


def build_command(
    cfg: dict[str, Any],
    config_path: Path,
    *,
    runtime: Runtime | None = None,
    log_path: Path | None = None,
) -> list[str]:
    """The argv for one REINVENT run.

    Uses the resource shim when `sandbox_safe` is set and the environment's
    interpreter is reachable; falls back to the console script otherwise, so
    a layout this cannot work out still runs, just without the protection.
    """
    settings = cfg["reinvent"]
    runtime = runtime or probe(cfg)

    command: list[str]
    interpreter = (
        environment_python(runtime.executable)
        if settings.get("sandbox_safe", True)
        else None
    )
    if interpreter is not None:
        command = [str(interpreter), "-c", RESOURCE_SHIM, str(config_path)]
    else:
        command = [str(runtime.executable), str(config_path)]

    if log_path is not None:
        command += ["--log-filename", str(log_path)]
    return command


def run_config(
    cfg: dict[str, Any], config_path: Path, *, log_path: Path | None = None
) -> subprocess.CompletedProcess:
    """Run one REINVENT config and return the finished process.

    REINVENT imposes no wall-clock limit of its own, so the timeout here is
    the only thing that stops a wedged run.
    """
    settings = cfg["reinvent"]
    runtime = probe(cfg)
    if runtime.problems:
        raise ReinventError("; ".join(runtime.problems))

    command = build_command(cfg, config_path, runtime=runtime, log_path=log_path)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Custom scoring components are found through sys.path, not a dedicated
    # variable: REINVENT walks the `reinvent_plugins.components` namespace
    # package, so our plugin directory has to be importable.
    plugin_dir = resolve(cfg, cfg["scoring_plugin"]["plugin_root"])
    package_root = Path(__file__).resolve().parent.parent
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(plugin_dir), str(package_root), existing) if p
    )

    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=float(settings["timeout_s"]),
            cwd=str(resolve(cfg, settings["working_dir"])),
        )
    except FileNotFoundError as exc:
        raise ReinventError(f"could not execute {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ReinventError(
            f"REINVENT did not finish within {settings['timeout_s']}s"
        ) from exc


# --------------------------------------------------------------- the output


@dataclass
class Sample:
    """One generated molecule, as REINVENT reports it."""

    smiles: str
    state: int
    warheads: str
    linker: str
    nll: float | None

    @property
    def valid(self) -> bool:
        return self.state == STATE_VALID

    @property
    def duplicate(self) -> bool:
        return self.state == STATE_DUPLICATE


def read_samples(path: str | Path) -> list[Sample]:
    """Parse the sampling CSV.

    REINVENT writes it with a naive `",".join`, so a field containing a comma
    would corrupt the row. SMILES do not normally contain commas, but a
    malformed row is skipped rather than raising, because losing one sample
    should not lose the run.
    """
    source = Path(path)
    if not source.is_file():
        raise ReinventError(f"REINVENT wrote no output at {source}")

    samples: list[Sample] = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in SAMPLE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ReinventError(
                f"{source} is missing expected columns {missing}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            try:
                state = int(row["SMILES_state"])
            except (TypeError, ValueError):
                continue
            nll_raw = row.get("NLL")
            try:
                nll = float(nll_raw) if nll_raw not in (None, "") else None
            except ValueError:
                nll = None
            samples.append(
                Sample(
                    smiles=(row.get("SMILES") or "").strip(),
                    state=state,
                    warheads=(row.get("Warheads") or "").strip(),
                    linker=(row.get("Linker") or "").strip(),
                    nll=nll,
                )
            )
    return samples


def summarise_samples(samples: list[Sample]) -> dict[str, Any]:
    """Validity and duplication, which is how a sampling run is judged."""
    total = len(samples)
    valid = sum(1 for s in samples if s.valid)
    duplicate = sum(1 for s in samples if s.duplicate)
    invalid = total - valid - duplicate
    return {
        "total": total,
        "valid": valid,
        "duplicate": duplicate,
        "invalid": invalid,
        "validity_percent": round(valid / total * 100, 2) if total else 0.0,
        "unique_valid": len({s.smiles for s in samples if s.valid}),
    }
