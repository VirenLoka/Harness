"""Selecting binding pockets from the CrossDocked test split.

`split_by_name.pt` pairs a pocket PDB with the reference ligand that was
crystallized in it, as relative paths inside `crossdocked_pocket10/`. Both
files come from the Pocket2Mol release; DiffSBDD's own repository ships only
the two `example/` pockets, so the split has to be downloaded separately.

The file is a torch archive whose payload is nothing but strings in tuples and
lists. `torch.load` would unpickle it with no restrictions at all, which is
more trust than a multi-gigabyte third-party download deserves, so the archive
is opened directly and only string containers are allowed to be constructed.
"""

from __future__ import annotations

import io
import pickle
import random
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POCKET_SUBDIR = "crossdocked_pocket10"
SPLIT_FILE = "split_by_name.pt"

INDEX_FILE = "index.pkl"
ARCHIVE = "crossdocked_pocket10.tar.gz"

DOWNLOAD_HINT = (
    f"Unpack {ARCHIVE} so that {POCKET_SUBDIR}/ sits inside "
    f"dataset.pockets.crossdocked_root, and put the small {SPLIT_FILE} beside "
    "it. Both come from the Pocket2Mol data release: "
    "https://github.com/pengxingang/Pocket2Mol/tree/main/data"
)

# Everything the payload legitimately needs to rebuild itself.
_ALLOWED_CLASSES = {
    ("builtins", "list"),
    ("builtins", "dict"),
    ("builtins", "tuple"),
    ("builtins", "str"),
    ("collections", "OrderedDict"),
}


class CrossDockedError(Exception):
    """A problem with the CrossDocked files, phrased so it can be acted on."""


class _StringsOnlyUnpickler(pickle.Unpickler):
    """Refuses to build anything but the containers the split actually uses."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_CLASSES:
            return super().find_class(module, name)
        raise CrossDockedError(
            f"{SPLIT_FILE} tried to construct {module}.{name}, which a split of "
            "file names has no reason to contain. Refusing to unpickle it."
        )

    def persistent_load(self, pid: Any) -> Any:
        # Tensor storages arrive this way. The split has none.
        raise CrossDockedError(
            f"{SPLIT_FILE} contains tensor data, so it is not the plain name "
            "split this reader expects."
        )


def _load_zip_archive(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        payloads = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not payloads:
            raise CrossDockedError(f"no data.pkl inside {path}")
        raw = archive.read(payloads[0])
    return _StringsOnlyUnpickler(io.BytesIO(raw)).load()


def _load_legacy_archive(path: Path) -> Any:
    """Pre-1.6 torch wrote magic, protocol and sys_info ahead of the payload."""
    with path.open("rb") as handle:
        unpickler = _StringsOnlyUnpickler(handle)
        for _ in range(3):
            unpickler.load()
        return unpickler.load()


def load_split(path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Read `split_by_name.pt` into {split: [(receptor, ligand), ...]}."""
    path = Path(path)
    if not path.is_file():
        raise CrossDockedError(f"{SPLIT_FILE} not found at {path}. {DOWNLOAD_HINT}")

    readers = (
        (_load_zip_archive, zipfile.is_zipfile(path)),
        (_load_legacy_archive, True),
    )
    failures: list[str] = []
    payload: Any = None
    for reader, applicable in readers:
        if not applicable:
            continue
        try:
            payload = reader(path)
            break
        except CrossDockedError:
            raise
        except Exception as exc:  # noqa: BLE001 - try the next container format
            failures.append(f"{reader.__name__}: {type(exc).__name__}: {exc}")
    else:
        raise CrossDockedError(
            f"could not read {path} in any known torch format:\n  "
            + "\n  ".join(failures)
        )

    if not isinstance(payload, dict):
        raise CrossDockedError(f"{path} holds a {type(payload).__name__}, not a dict")

    splits: dict[str, list[tuple[str, str]]] = {}
    for name, entries in payload.items():
        pairs: list[tuple[str, str]] = []
        for entry in entries or ():
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            receptor, ligand = str(entry[0]), str(entry[1])
            pairs.append((receptor, ligand))
        splits[str(name)] = pairs
    return splits


def load_index(path: str | Path) -> list[tuple[str, str]]:
    """Read `crossdocked_pocket10/index.pkl` into (receptor, ligand) pairs.

    Rows are (pocket_pdb, ligand_sdf, full_receptor_pdb, rmsd); only the first
    two are used, and the full receptor is not in the archive anyway.

    This is the *whole* CrossDocked index, not the published split. It ships
    inside the archive, so it unblocks a run before `split_by_name.pt` has been
    fetched, but pockets taken from it are not a held-out test set.
    """
    path = Path(path)
    if not path.is_file():
        raise CrossDockedError(f"{INDEX_FILE} not found at {path}. {DOWNLOAD_HINT}")
    with path.open("rb") as handle:
        payload = _StringsOnlyUnpickler(handle).load()
    if not isinstance(payload, (list, tuple)):
        raise CrossDockedError(f"{path} holds a {type(payload).__name__}, not a list")

    pairs: list[tuple[str, str]] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        receptor, ligand = row[0], row[1]
        if isinstance(receptor, str) and isinstance(ligand, str):
            pairs.append((receptor, ligand))
    if not pairs:
        raise CrossDockedError(f"{path} yielded no (receptor, ligand) pairs")
    return pairs


def verify_root(root: str | Path) -> list[str]:
    """Problems with a CrossDocked directory, phrased as things to go and do.

    `moltrain.tasks probe` calls this so a mis-unpacked archive is caught
    before a long GPU run rather than part way through it.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return [f"crossdocked_root is not a directory: {root_path}. {DOWNLOAD_HINT}"]

    problems: list[str] = []
    structures = root_path / POCKET_SUBDIR
    archive = root_path / ARCHIVE
    if not structures.is_dir():
        if archive.is_file():
            problems.append(
                f"{archive} is still packed. Run: tar -xzf {archive} -C {root_path}"
            )
        else:
            problems.append(f"{structures} does not exist. {DOWNLOAD_HINT}")
    elif (structures / POCKET_SUBDIR).is_dir():
        problems.append(
            f"{structures} holds another {POCKET_SUBDIR}/, so the archive was "
            f"unpacked one level too deep; point crossdocked_root at {structures}"
        )
    elif not any(entry.is_dir() for entry in structures.iterdir()):
        problems.append(f"{structures} has no pocket subdirectories in it")

    split = root_path / SPLIT_FILE
    if not split.is_file():
        message = (
            f"{split} is missing; it is a small separate file in the Pocket2Mol "
            "release and is not inside the archive"
        )
        if (structures / INDEX_FILE).is_file():
            message += (
                ". Until you have it, set dataset.pockets.split to `index` to "
                "use the archive's own index instead, which is not the held-out "
                "test set"
            )
        problems.append(message)
    return problems


# --------------------------------------------------------------- pocket naming

# CrossDocked ligand files are named like
# `1a0q_A_rec_1a0q_mtx_lig_tt_min_0.sdf`, where `mtx` is the HET code.
_LIGAND_CODE = re.compile(r"_([A-Za-z0-9]{2,3})_lig")
_PDB_ID = re.compile(r"^([0-9][A-Za-z0-9]{3})")


def _ligand_code(ligand_name: str) -> str | None:
    match = _LIGAND_CODE.search(Path(ligand_name).name)
    return match.group(1).upper() if match else None


def _pdb_id(receptor_name: str) -> str | None:
    match = _PDB_ID.match(Path(receptor_name).name)
    return match.group(1).upper() if match else None


def _family(receptor_name: str) -> str | None:
    """The parent directory encodes the protein, e.g. `ABL1_HUMAN_25_300_0`."""
    parent = Path(receptor_name).parent.name
    if not parent or parent in (".", "/"):
        return None
    trimmed = re.sub(r"(_\d+)+$", "", parent)
    return trimmed or parent


@dataclass
class Pocket:
    """One binding site, with the files needed to dock into it."""

    name: str
    pdb_file: str
    ref_ligand_file: str | None = None
    target: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pdb_file": self.pdb_file,
            "ref_ligand_file": self.ref_ligand_file,
            "target": self.target,
            "metadata": self.metadata,
        }


def pocket_from_entry(root: Path, index: int, receptor: str, ligand: str) -> Pocket:
    """Turn one split entry into a Pocket with a label a prompt can show."""
    structures = root / POCKET_SUBDIR
    pdb_id = _pdb_id(receptor)
    code = _ligand_code(ligand)
    family = _family(receptor)

    stem = pdb_id or Path(receptor).stem.replace("_", "-")
    name = f"{stem.lower()}-{index:03d}"
    label = pdb_id or Path(receptor).stem
    if family:
        label = f"{label} ({family})"
    if code:
        label = f"{label}, reference ligand {code}"

    return Pocket(
        name=name,
        pdb_file=str(structures / receptor),
        ref_ligand_file=str(structures / ligand),
        target=label,
        metadata={
            "source": "crossdocked",
            "split_index": index,
            "pdb_id": pdb_id,
            "ligand_code": code,
            "family": family,
            "receptor_entry": receptor,
            "ligand_entry": ligand,
        },
    )


def select_pockets(cfg: dict[str, Any]) -> list[Pocket]:
    """The pockets this dataset will be built from, in a stable order."""
    pockets_cfg = cfg["dataset"]["pockets"]
    source = pockets_cfg["source"]

    if source == "explicit":
        pockets = [
            Pocket(
                name=str(entry["name"]),
                pdb_file=str(entry["pdb_file"]),
                ref_ligand_file=entry.get("ref_ligand") or entry.get("ref_ligand_file"),
                target=str(entry.get("target", entry["name"])),
                metadata={"source": "explicit"},
            )
            for entry in pockets_cfg["explicit"]
        ]
        if not pockets:
            raise CrossDockedError(
                "dataset.pockets.source is `explicit` but dataset.pockets.explicit "
                "is empty; add entries of {name, pdb_file, ref_ligand}"
            )
    elif source == "crossdocked":
        root = pockets_cfg["crossdocked_root"]
        if not root:
            raise CrossDockedError(
                f"dataset.pockets.crossdocked_root is unset. {DOWNLOAD_HINT}"
            )
        root_path = Path(root).expanduser()
        split_name = pockets_cfg["split"]
        if split_name == "index":
            entries = load_index(root_path / POCKET_SUBDIR / INDEX_FILE)
        else:
            splits = load_split(root_path / SPLIT_FILE)
            if split_name not in splits:
                raise CrossDockedError(
                    f"split `{split_name}` not in {SPLIT_FILE}; it has {sorted(splits)}"
                )
            entries = splits[split_name]
        pockets = [
            pocket_from_entry(root_path, index, receptor, ligand)
            for index, (receptor, ligand) in enumerate(entries)
        ]
    else:
        raise CrossDockedError(
            f"unknown dataset.pockets.source: {source} (use crossdocked or explicit)"
        )

    seed = pockets_cfg.get("shuffle_seed")
    if seed is not None:
        random.Random(int(seed)).shuffle(pockets)

    # `offset` then `limit` cut a deterministic window out of the shuffled
    # split, so a training set and a held-out set are disjoint by
    # construction. Changing the seed instead would reshuffle the same pockets
    # and hand back an overlapping subset, leaking training pockets into the
    # evaluation.
    offset = int(pockets_cfg.get("offset") or 0)
    if offset < 0:
        raise CrossDockedError("dataset.pockets.offset cannot be negative")
    if offset >= len(pockets):
        raise CrossDockedError(
            f"dataset.pockets.offset is {offset} but the {pockets_cfg['split']} "
            f"split has only {len(pockets)} pockets"
        )
    pockets = pockets[offset:]

    limit = pockets_cfg.get("limit")
    if limit is not None:
        pockets = pockets[: int(limit)]
    return pockets


def missing_files(pockets: list[Pocket]) -> list[str]:
    """Paths a pocket needs that are not on disk, for a pre-flight report."""
    absent: list[str] = []
    for pocket in pockets:
        if not Path(pocket.pdb_file).is_file():
            absent.append(pocket.pdb_file)
        if pocket.ref_ligand_file and not Path(pocket.ref_ligand_file).is_file():
            absent.append(pocket.ref_ligand_file)
    return absent
