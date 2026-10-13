"""Measured properties of a molecule, cached and parallelized.

Reward computation is the expensive half of RL here: 2D properties are
microseconds, docking is seconds. Both are cached on disk keyed by canonical
SMILES (and receptor plus search settings for docking), because a policy
revisits the same molecules constantly, both within an episode and across the
group of rollouts GRPO samples for one task.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import resolve


@dataclass
class Properties:
    """Everything the reward function reads about one molecule."""

    smiles: str
    valid: bool
    sa_score: float | None = None
    qed: float | None = None
    exact_mw: float | None = None
    heavy_atoms: int | None = None
    clogp: float | None = None
    tpsa: float | None = None
    binding_affinity: float | None = None
    ligand_efficiency: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlCache:
    """Append-only cache. Survives restarts and is safe to inspect by eye."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written line from a killed run
                self._entries[record["key"]] = record["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = value
            with self.path.open("a") as handle:
                handle.write(json.dumps({"key": key, "value": value}) + "\n")

    def __len__(self) -> int:
        return len(self._entries)


class Oracle:
    """Evaluates molecules, honoring the configured docking policy."""

    def __init__(self, cfg: dict[str, Any], toolkit):
        self.cfg = cfg
        self.toolkit = toolkit
        self.server = toolkit.server
        docking = cfg["reward"]["docking"]
        self.mode = docking["mode"]
        self.engine = docking["engine"]
        self.exhaustiveness = int(docking["exhaustiveness"])
        self.num_modes = int(docking["num_modes"])
        self.timeout_s = float(docking["timeout_s"])
        self.semaphore = asyncio.Semaphore(int(docking["workers"]))
        self.property_cache = JsonlCache(
            resolve(cfg, cfg["reward"]["property_cache_path"])
        )
        self.docking_cache = JsonlCache(resolve(cfg, docking["cache_path"]))
        self.stats = {
            "property_hits": 0,
            "property_misses": 0,
            "dock_hits": 0,
            "dock_misses": 0,
        }

    # ------------------------------------------------------------- properties

    def canonical(self, smiles: str) -> str | None:
        result = self.server.validate_smiles(smiles)
        return result["canonical_smiles"] if result["valid"] else None

    def properties_2d(self, smiles: str) -> Properties:
        """Cheap, always-computed descriptors."""
        canonical = self.canonical(smiles)
        if canonical is None:
            return Properties(smiles=smiles, valid=False, error="SMILES did not parse")

        cached = self.property_cache.get(canonical)
        if cached is not None:
            self.stats["property_hits"] += 1
            return Properties(**cached)

        self.stats["property_misses"] += 1
        metrics = self.server.get_molecule_metrics(smiles=canonical)
        # get_sbdd_metrics also yields cLogP and TPSA; the score passed here is
        # a placeholder because only the 2D fields are read back.
        descriptors = self.server.get_sbdd_metrics(smiles=canonical, docking_score=-1.0)
        properties = Properties(
            smiles=canonical,
            valid=True,
            sa_score=metrics["sa_score"],
            qed=metrics["qed"],
            exact_mw=metrics["exact_mw"],
            heavy_atoms=metrics["num_heavy_atoms"],
            clogp=descriptors["cLogP"],
            tpsa=descriptors["TPSA"],
        )
        self.property_cache.put(canonical, properties.as_dict())
        return properties

    # ---------------------------------------------------------------- docking

    def _dock_key(self, smiles: str, receptor_file: str) -> str:
        return (
            f"{smiles}|{Path(receptor_file).name}|{self.engine}|{self.exhaustiveness}"
        )

    async def dock(
        self, smiles: str, receptor_file: str, label: str = "moltrain"
    ) -> float | None:
        """Binding affinity in kcal/mol, or None when docking fails."""
        canonical = self.canonical(smiles)
        if canonical is None or not receptor_file:
            return None

        key = self._dock_key(canonical, receptor_file)
        cached = self.docking_cache.get(key)
        if cached is not None:
            self.stats["dock_hits"] += 1
            return cached.get("binding_affinity")

        self.stats["dock_misses"] += 1
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    self.server.run_docking_simulation(
                        smiles=canonical,
                        receptor_file=receptor_file,
                        engine=self.engine,
                        exhaustiveness=self.exhaustiveness,
                        num_modes=self.num_modes,
                        label=label,
                    ),
                    timeout=self.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - a failed dock is a missing value
                self.docking_cache.put(
                    key, {"binding_affinity": None, "error": str(exc)[:200]}
                )
                return None
        affinity = float(result["binding_affinity"])
        self.docking_cache.put(key, {"binding_affinity": affinity})
        return affinity

    # ------------------------------------------------------------- evaluation

    async def evaluate(
        self,
        smiles: str,
        receptor_file: str | None,
        *,
        is_final: bool,
        label: str = "moltrain",
    ) -> Properties:
        """Full evaluation, docking only when the configured policy says to."""
        properties = self.properties_2d(smiles)
        if not properties.valid:
            return properties

        should_dock = receptor_file is not None and (
            self.mode == "every_candidate" or (self.mode == "final_only" and is_final)
        )
        if should_dock:
            affinity = await self.dock(properties.smiles, receptor_file, label=label)
            if affinity is not None:
                properties.binding_affinity = affinity
                if properties.heavy_atoms:
                    properties.ligand_efficiency = round(
                        affinity / properties.heavy_atoms, 4
                    )
        return properties

    async def evaluate_many(
        self, smiles_list: list[str], receptor_file: str | None, *, is_final: bool
    ) -> list[Properties]:
        """Evaluate a batch concurrently; docking workers bound the parallelism."""
        return list(
            await asyncio.gather(
                *(
                    self.evaluate(s, receptor_file, is_final=is_final)
                    for s in smiles_list
                )
            )
        )
