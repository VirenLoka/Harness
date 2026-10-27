"""ADC linker scoring as a REINVENT component.

This exposes `adclinker.score` to REINVENT so the same function that ranks
candidates downstream is the one REINVENT optimises against. Writing the
reward twice is how the two quietly drift apart.

**Discovery.** REINVENT walks `reinvent_plugins.components` as a *native
namespace package* and imports every module whose name starts with `comp_`.
Two consequences, both easy to get wrong:

- There must be **no `__init__.py`** in `reinvent_plugins/` or
  `reinvent_plugins/components/`. `reinvent/scoring/importer.py` raises
  `RuntimeError("No valid component directories found")` if the package turns
  out to be a regular one.
- There is no plugin-directory environment variable. Discovery is plain
  `sys.path`, so the directory *containing* `reinvent_plugins` goes on
  `PYTHONPATH`. `adclinker.reinvent.run_config` does that.

Import errors here are swallowed by REINVENT and logged as
`Component ADCLinker could not be imported`, so if the component silently
fails to appear that log line is where to look.

**Registration name.** The registry key is the class name lowercased, and
TOML lookup strips dashes and underscores, so `ADCLinker`, `adclinker` and
`adc_linker` all resolve to this component.

**Failures are NaN, not zero.** REINVENT's `ComponentResults` documents this,
and the distinction matters here: a molecule that fails a gate scores a
genuine **0.0** -- it was scored and it is unusable -- while a molecule RDKit
cannot read is **NaN**, meaning no opinion. Collapsing the two would let a
parse failure read as a design verdict.
"""

from __future__ import annotations

__all__ = ["ADCLinker"]

import os
import sys
from typing import List, Optional

import numpy as np
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults


def _import_adclinker():
    """Import the scoring package, adding the repository root if needed.

    REINVENT runs in its own environment, so `adclinker` is reachable only
    because its parent directory is on PYTHONPATH. Falling back to the
    location of this file keeps the component working when it is invoked by
    hand.
    """
    try:
        from adclinker import reference, score  # noqa: F401

        return reference, score
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        # .../adclinker/plugins/reinvent_plugins/components -> repo root
        root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        if root not in sys.path:
            sys.path.insert(0, root)
        from adclinker import reference, score  # noqa: F401

        return reference, score


@add_tag("__parameters")
@dataclass
class Parameters:
    """One entry per endpoint, as REINVENT collects them.

    `dataset` is the curated reagent file the score is calibrated against.
    `payload_smiles` is the drug that must survive intact; omit it and the
    payload gate is skipped, which is only right when scoring linkers in
    isolation rather than assembled conjugates.
    """

    # The element types are optional, not just the lists. REINVENT's
    # `collect_params` pads every endpoint to the same length with `None`
    # when a key is absent from one of them, so `List[str]` rejects a
    # perfectly ordinary two-endpoint config where only one names a payload.
    dataset: List[str]
    payload_smiles: Optional[List[Optional[str]]] = None
    handle_smarts: Optional[List[Optional[str]]] = None
    require_cleavable: Optional[List[Optional[bool]]] = None
    min_heavy_atoms: Optional[List[Optional[int]]] = None


@add_tag("__component")
class ADCLinker:
    def __init__(self, params: Parameters):
        reference, score = _import_adclinker()
        self._score = score

        self.number_of_endpoints = len(params.dataset)
        self.contexts = []

        for index in range(self.number_of_endpoints):
            dataset = params.dataset[index]
            profile = reference.load_profile(dataset, cache=None)
            fingerprints = reference.fingerprint_reference(dataset)
            self.contexts.append(
                score.Context(
                    profile=profile,
                    reference_fingerprints=fingerprints,
                    payload_smiles=_at(params.payload_smiles, index),
                    handle_smarts=_at(params.handle_smarts, index),
                    require_cleavable=_at(params.require_cleavable, index, True),
                    min_heavy_atoms=_at(params.min_heavy_atoms, index, 12),
                )
            )

    def __call__(self, smilies: List[str]) -> ComponentResults:
        from rdkit import Chem, rdBase

        all_scores = []
        for context in self.contexts:
            endpoint = []
            for smiles in smilies:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    # No opinion, as opposed to a bad opinion.
                    endpoint.append(np.nan)
                    continue
                try:
                    result = self._score.score_molecule(mol, context)
                except Exception:  # noqa: BLE001 - one molecule is not the run
                    endpoint.append(np.nan)
                    continue
                if not result.accepted or result.total is None:
                    endpoint.append(0.0)
                else:
                    endpoint.append(result.total / 100.0)
            all_scores.append(np.array(endpoint, dtype=float))

        return ComponentResults(all_scores)


def _at(values, index, default=None):
    if not values or index >= len(values) or values[index] is None:
        return default
    return values[index]
