"""Measured linker properties, cached and parallelized.

`moltrain.oracle.Oracle` is built around a pocket. Its expensive half is
docking, its cheap half is a handful of 2D descriptors, and its cache key
carries a receptor, an engine and an exhaustiveness. None of that survives
the move to PROTACs: there is no docking here at all, so every measurement is
milliseconds of RDKit, and the thing a key has to carry instead is the
fragment pair.

**Why the fragments are in the key.** Every number this oracle exists to
report -- `span`, `linker_heavy_atoms`, `linker_smiles`, `fragments_intact`,
the reactive-group scan -- is defined by subtracting the two fragments from
the molecule. The same SMILES measured against a different fragment pair is a
different measurement, and often a wildly different one: change which region
counts as the warhead and the linker is a different set of atoms with a
different span. A cache keyed on canonical SMILES alone would serve the first
task's linker metrics to the second task and never look wrong, because every
value it returned would be a real measurement of something.

The measurement itself is `protactrain.properties.measure`, which is
`protacbench.grade.measure` -- the same function the hand-run grading loop
scores replies with. Nothing is reimplemented here; this module is the cache
and the thread pool around it.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from moltrain.logs import get_logger

# Same append-only, inspectable-by-eye storage moltrain uses. Only the key
# changes, so the class does not need to.
from moltrain.oracle import JsonlCache

from .config import resolve
from .properties import LinkerProperties, measure

LOG = get_logger("oracle")

__all__ = ["JsonlCache", "LinkerOracle", "cache_key"]

# One worker per rollout in a GRPO group (grpo.num_generations is 8 in
# configs/default.yaml), so a whole group's submissions measure in one wave.
# RDKit descriptor work releases the GIL in its C++ layer, so threads are the
# right shape here and processes would cost more in pickling than they save.
_DEFAULT_WORKERS = 8

# Prefix on the synthesis-score entries so they share one cache file with the
# property entries without colliding with them.
_SYNTHESIS = "synthesis"


def cache_key(canonical: str, fragment_smiles: Sequence[str], kind: str = "") -> str:
    """A key that cannot confuse two tasks that share a molecule.

    JSON rather than a delimiter join: fragment SMILES contain almost every
    punctuation character, and a separator one of them happens to contain
    would silently merge two keys into one.

    Fragment order is preserved, not sorted. `repair.match_fragments` is
    greedy and first-fit, so where the two fragments overlap -- two IMiD-based
    ligands, say -- the order they are offered in can decide which placement
    wins, and therefore which atoms are the linker.
    """
    parts: list[Any] = [kind, canonical, list(fragment_smiles)]
    return json.dumps(parts, separators=(",", ":"))


class LinkerOracle:
    """Measures proposed PROTACs against a task's fragments, with a disk cache.

    A policy revisits the same molecules constantly -- within one episode as
    it re-checks its own edit, and across the group of rollouts GRPO samples
    for one task, where several rollouts converge on the same answer. The
    cache is what keeps that from being paid for every time.
    """

    def __init__(self, cfg: dict[str, Any], cache_path: str | Path | None = None):
        self.cfg = cfg
        path = cache_path or cfg["reward"]["property_cache_path"]
        self.cache = JsonlCache(resolve(cfg, path))
        self.workers = int(cfg["reward"].get("property_workers") or _DEFAULT_WORKERS)
        self.stats = {
            "property_hits": 0,
            "property_misses": 0,
            "synthesis_hits": 0,
            "synthesis_misses": 0,
        }

    # ------------------------------------------------------------- properties

    def canonical(self, smiles: str) -> str | None:
        """Canonical SMILES, or None when RDKit cannot read the string.

        Imported lazily, and wrapped, because this is on the measurement path:
        a policy that emits a malformed SMILES must get a cache miss and a
        failed measurement, never an exception into the training step.
        """
        try:
            from rdkit import Chem, rdBase

            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles)
            return None if mol is None else Chem.MolToSmiles(mol)
        except Exception:  # noqa: BLE001 - an unreadable SMILES is a value, not a crash
            return None

    def evaluate(
        self, smiles: str, fragment_smiles: Sequence[str]
    ) -> LinkerProperties:
        """Measure one proposed PROTAC against the two fragments it must keep."""
        fragments = list(fragment_smiles)
        canonical = self.canonical(smiles)
        if canonical is None:
            # Not cached: an unparseable string is cheap to reject and there
            # is no canonical form to key it by.
            return LinkerProperties(
                smiles=smiles,
                measurable=False,
                error="RDKit could not read the SMILES",
            )

        key = cache_key(canonical, fragments)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["property_hits"] += 1
            return LinkerProperties(**cached)

        self.stats["property_misses"] += 1
        props = measure(canonical, fragments)
        # Failed measurements are cached too. A rollout group that all submit
        # the same broken molecule should pay the fragment match once.
        self.cache.put(key, props.as_dict())
        return props

    async def evaluate_many(
        self, smiles_list: Sequence[str], fragment_smiles: Sequence[str]
    ) -> list[LinkerProperties]:
        """Measure a batch concurrently, in input order.

        `asyncio.to_thread` runs on the loop's default executor, which is a
        thread pool; the semaphore is what bounds it to `workers` rather than
        letting a 64-rollout batch open 64 threads.
        """
        limit = asyncio.Semaphore(self.workers)

        async def one(smiles: str) -> LinkerProperties:
            async with limit:
                return await asyncio.to_thread(self.evaluate, smiles, fragment_smiles)

        return list(await asyncio.gather(*(one(s) for s in smiles_list)))

    def evaluate_many_sync(
        self, smiles_list: Sequence[str], fragment_smiles: Sequence[str]
    ) -> list[LinkerProperties]:
        """`evaluate_many` for callers outside an event loop, such as evaluation."""
        items = list(smiles_list)
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(
                pool.map(lambda s: self.evaluate(s, fragment_smiles), items)
            )

    # -------------------------------------------------------- synthesis score

    def synthesis_score(
        self, smiles: str, fragment_smiles: Sequence[str]
    ) -> float | None:
        """`protacbench.score.score_sample` as a dense 0-100 signal, or None.

        The threshold terms are mostly step-shaped: a molecule either got
        inside its band or it did not, and with the two fragments frozen most
        descriptors move little enough that a whole group of rollouts can
        score identically and produce no advantage at all. This is the term
        that still separates them -- it moves with motif familiarity, span,
        polarity, flexibility and liabilities continuously.

        None means "not scored", which the reward treats as a missing term
        rather than a zero: a molecule that failed the fragment match has
        already been floored on admissibility, and scoring its synthesis
        route would be scoring the wrong compound.
        """
        fragments = list(fragment_smiles)
        canonical = self.canonical(smiles)
        if canonical is None:
            return None

        key = cache_key(canonical, fragments, kind=_SYNTHESIS)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats["synthesis_hits"] += 1
            return cached.get("total")

        self.stats["synthesis_misses"] += 1
        total = self._score(canonical, fragments)
        self.cache.put(key, {"total": total})
        return total

    def _score(self, canonical: str, fragments: list[str]) -> float | None:
        """The uncached body. Never raises; a failure is a missing score."""
        props = self.evaluate(canonical, fragments)
        if not props.measurable or not props.fragments_intact:
            return None

        try:
            from rdkit import Chem, rdBase

            from protacbench.assess import molecule_properties
            from protacbench.repair import match_fragments
            from protacbench.score import score_sample
            from protacbench.structures import linker_view

            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(canonical)
            if mol is None:
                return None

            matches = match_fragments(mol, fragments)
            if matches is None:
                return None
            claimed = {index for match in matches for index in match}
            indices = [i for i in range(mol.GetNumAtoms()) if i not in claimed]
            if not indices:
                return None

            view = linker_view(mol, indices)
            # score_sample reads `reactive_groups` and `liable_groups` as
            # semicolon-joined strings, the shape assess.py writes into a
            # corpus row. They come off the measurement rather than being
            # re-scanned, so the tool, the reward and the score all agree on
            # which groups are in the linker.
            properties = dict(molecule_properties(mol))
            properties["reactive_groups"] = ";".join(props.reactive_groups)
            properties["liable_groups"] = ";".join(props.liability_groups)
            linker = {
                "path_length_bonds": view.path_length_bonds,
                "heavy_atoms": view.n_heavy_atoms,
                "attachments": view.n_attachments,
                "rotatable_bonds": view.rotatable_bonds,
                "heteroatom_fraction": view.heteroatom_fraction,
                "in_ring_fraction": view.in_ring_fraction,
                "exit_vector_distance": view.exit_vector_distance,
            }
            score = score_sample(mol, properties, linker, indices)
        except Exception as exc:  # noqa: BLE001 - a measurement must never raise
            LOG.debug(
                "synthesis score failed for %s: %s: %s",
                canonical,
                type(exc).__name__,
                str(exc)[:200],
            )
            return None
        return None if score.total is None else float(score.total)
