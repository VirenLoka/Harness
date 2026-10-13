"""A scripted expert policy, for cold-starting SFT before any model is trained.

The expert is an ordinary `GenerateFn`: it takes a transcript and returns the
next assistant turn, so it runs through `run_episode` like any policy and its
tool calls are executed by the real toolkit. That matters more than it sounds.
Every number in a trace the expert produces came back from a tool, so SFT never
teaches the model to invent measurements -- the one failure mode that would
make the whole pipeline worthless.

What the expert knows is the *protocol*, not chemistry: measure, transform one
thing at a time, verify, submit. Its move library is a greedy hill-climb over
simplifications, each one characterised by measurement (see `MOVES`), and it is
deliberately simple. GRPO is what is supposed to find the chemistry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from . import atem
from .episode import Task
from .oracle import Properties
from .rewards import score_thresholds

_TOOL_OUTPUT = re.compile(
    r'<tool_output name="(?P<name>[^"]+)">\s*(?P<body>.*?)\s*</tool_output>',
    re.DOTALL,
)


@dataclass(frozen=True)
class Move:
    """One simplification, with the SA change measured when it was vetted."""

    name: str
    smarts: str
    rationale: str
    measured_delta_sa: float


# Ordered by measured benefit, so the greedy search spends its budget on the
# moves most likely to pay. Each delta was measured on a probe molecule
# carrying that group; it is an indication of size, not a promise.
#
# Three candidates were dropped after measurement rather than kept on
# plausibility: stripping a stereocentre (`[C@:1]>>[C:1]`) is a no-op because
# RDKit's reaction machinery carries the chirality tag through, and both
# cyclopropane ring-opening templates fragmented the molecule, raising SA by
# more than 4.
MOVES: tuple[Move, ...] = (
    Move(
        "remove_cyclopropane",
        "[C:1]1([CH2][CH2]1)>>[C:1]",
        "a spiro or fused cyclopropane is the single most expensive feature in "
        "a synthetic route; removing it also relieves ring strain",
        -1.10,
    ),
    Move(
        "remove_hydroxyl",
        "[C:1]([OH])>>[C:1][H]",
        "an aliphatic hydroxyl often sits on a stereocentre that has to be set "
        "deliberately; check the pocket still binds without the hydrogen bond",
        -0.57,
    ),
    Move(
        "tbutyl_to_methyl",
        "[C:1][CH0]([CH3])([CH3])[CH3]>>[C:1][CH3]",
        "trading tert-butyl for methyl cuts bulk and lipophilicity",
        -0.30,
    ),
    Move(
        "remove_gem_dimethyl",
        "[C:1]([CH3])([CH3])>>[C:1]([CH3])[H]",
        "a gem-dimethyl centre is a quaternary carbon to build",
        -0.25,
    ),
    Move(
        "sulfonamide_to_amide",
        "[c:1]-[SX4](=O)(=O)[N:2]>>[c:1]-C(=O)[N:2]",
        "an aryl amide is easier to make than an aryl sulfonamide and keeps the "
        "hydrogen-bond donor",
        -0.18,
    ),
    Move(
        "thioether_to_ether",
        "[C:1]-[SX2]-[C:2]>>[C:1]-O-[C:2]",
        "sulfur to oxygen removes an oxidation liability",
        -0.16,
    ),
    Move(
        "remove_cf3",
        "[C,c:1]-[CX4](F)(F)F>>[C,c:1][H]",
        "trifluoromethyl is cheap to draw and expensive to install",
        -0.15,
    ),
    Move(
        "nitrile_to_amide",
        "[c:1]-[CH0]#[NH0]>>[c:1]-C(=O)[NH2]",
        "a primary amide keeps the polarity and loses the nitrile",
        -0.10,
    ),
    Move(
        "nitro_to_amine",
        "[c:1]-[N+](=O)[O-]>>[c:1]-[NH2]",
        "aromatic nitro groups are a toxicity flag; the aniline also raises QED",
        -0.08,
    ),
    Move(
        "demethylate_amide_n",
        "[C:1](=O)[N:2]([CH3])>>[C:1](=O)[N:2][H]",
        "an NH amide restores a hydrogen-bond donor the pocket may want",
        -0.08,
    ),
    Move(
        "dehalogenate_aryl",
        "[c:1]-[F,Cl,Br,I]>>[c:1][H]",
        "dropping a halogen simplifies the route and usually raises QED",
        -0.08,
    ),
    Move(
        "truncate_alkyl",
        "[CH2:1]-[CH2]-[CH2]-[CH3]>>[CH2:1]-[CH3]",
        "a long flexible chain costs entropy on binding",
        -0.04,
    ),
)


def last_tool_output(transcript: str) -> tuple[str, Any] | None:
    """The most recent tool result in the transcript, parsed when it is JSON."""
    matches = list(_TOOL_OUTPUT.finditer(transcript))
    if not matches:
        return None
    match = matches[-1]
    body = match.group("body")
    try:
        return match.group("name"), json.loads(body)
    except json.JSONDecodeError:
        return match.group("name"), body


def _measured(metrics: Any, smiles: str) -> Properties | None:
    """Properties out of whatever shape the tool returned.

    `get_molecule_metrics` and `dock_and_score` nest their numbers differently,
    so the fields are looked for in both places.
    """
    if not isinstance(metrics, dict):
        return None
    source = metrics
    for key in ("metrics", "properties", "molecule_metrics"):
        nested = metrics.get(key)
        if isinstance(nested, dict) and "sa_score" in nested:
            source = nested
            break
    if source.get("sa_score") is None:
        return None

    affinity = metrics.get("binding_affinity")
    if affinity is None:
        docking = metrics.get("docking")
        if isinstance(docking, dict):
            affinity = docking.get("binding_affinity")
    return Properties(
        smiles=smiles,
        valid=True,
        sa_score=source.get("sa_score"),
        qed=source.get("qed"),
        exact_mw=source.get("exact_mw"),
        heavy_atoms=source.get("num_heavy_atoms"),
        binding_affinity=None if affinity is None else float(affinity),
    )


class ScriptedExpert:
    """A greedy simplification policy that measures before it accepts anything.

    One instance per episode: it keeps its own state and reads the transcript
    only for the latest tool result, which is what the real policy would also
    be conditioning on.
    """

    def __init__(
        self,
        task: Task,
        thresholds: dict[str, dict[str, Any]],
        *,
        dock: bool = False,
        exhaustiveness: int = 4,
        max_moves: int | None = None,
    ):
        self.task = task
        self.thresholds = thresholds
        self.dock = dock and bool(task.receptor_file)
        self.exhaustiveness = exhaustiveness
        self.moves = list(MOVES[: max_moves or len(MOVES)])

        self.stage = "measure"
        self.current = task.smiles
        self.best = task.smiles
        self.start: Properties | None = None
        self.best_progress = 0.0
        self.pending: str | None = None  # product awaiting verification
        self.pending_move: Move | None = None
        self.move_index = 0
        self.accepted: list[str] = []

    def _progress(self, candidate: Properties) -> float:
        """Mean progress toward the reward's thresholds, as the reward sees it."""
        if self.start is None:
            return 0.0
        outcomes = [
            outcome
            for outcome in score_thresholds(self.start, candidate, self.thresholds)
            if outcome.delta is not None
        ]
        if not outcomes:
            return 0.0
        return sum(outcome.progress for outcome in outcomes) / len(outcomes)

    # ------------------------------------------------------------------ turns

    def __call__(self, transcript: str) -> str:
        if self.stage == "measure":
            return self._measure()
        if self.stage == "read_baseline":
            return self._read_baseline(transcript)
        if self.stage == "propose":
            return self._propose()
        if self.stage == "read_products":
            return self._read_products(transcript)
        if self.stage == "verify":
            return self._verify(transcript)
        if self.stage == "canonicalise":
            return self._canonicalise()
        return self._submit(transcript)

    def _measure(self) -> str:
        self.stage = "read_baseline"
        return atem.render_continuation(
            "Measure the starting molecule before changing anything, so every "
            "later comparison is against a number I actually have.",
            ("get_molecule_metrics", {"smiles": self.current}),
        )

    def _read_baseline(self, transcript: str) -> str:
        latest = last_tool_output(transcript)
        if latest is not None:
            self.start = _measured(latest[1], self.current)
        self.stage = "propose"
        return self._propose()

    def _propose(self) -> str:
        if self.move_index >= len(self.moves):
            return self._canonicalise()
        move = self.moves[self.move_index]
        self.pending_move = move
        self.stage = "read_products"
        return atem.render_continuation(
            f"Try {move.name.replace('_', ' ')}: {move.rationale}. "
            "One change at a time, so the effect is attributable.",
            (
                "apply_smarts_reaction",
                {"smiles": self.current, "reaction_smarts": move.smarts},
            ),
        )

    def _read_products(self, transcript: str) -> str:
        latest = last_tool_output(transcript)
        products: list[str] = []
        if latest is not None and isinstance(latest[1], dict):
            for entry in latest[1].get("products") or ():
                if isinstance(entry, str):
                    products.append(entry)
                elif isinstance(entry, dict) and entry.get("smiles"):
                    products.append(str(entry["smiles"]))

        if not products:
            # The group is not present. Move on rather than burning a turn.
            self.move_index += 1
            self.stage = "propose"
            return self._propose()

        self.pending = products[0]
        self.stage = "verify"
        if self.dock:
            return atem.render_continuation(
                "The transformation applied. Dock the product so I can see "
                "whether the simplification cost any binding.",
                (
                    "dock_and_score",
                    {
                        "smiles": self.pending,
                        "receptor_file": self.task.receptor_file,
                        "exhaustiveness": self.exhaustiveness,
                        "label": f"{self.task.task_id}-{self.pending_move.name}",
                    },
                ),
            )
        return atem.render_continuation(
            "The transformation applied. Measure the product and keep it only "
            "if it is actually better.",
            ("get_molecule_metrics", {"smiles": self.pending}),
        )

    def _verify(self, transcript: str) -> str:
        latest = last_tool_output(transcript)
        candidate = (
            _measured(latest[1], self.pending)
            if latest is not None and self.pending
            else None
        )

        # Accept only when overall progress toward the reward's thresholds goes
        # up. Judging on SA alone makes the expert trade QED away for it, which
        # is exactly the behaviour SFT would then teach.
        if candidate is not None and self.pending:
            progress = self._progress(candidate)
            if progress > self.best_progress:
                self.current = self.best = self.pending
                self.best_progress = progress
                self.accepted.append(
                    self.pending_move.name if self.pending_move else "?"
                )

        self.pending = None
        self.move_index += 1
        self.stage = "propose"
        return self._propose()

    def _canonicalise(self) -> str:
        self.stage = "submit"
        return atem.render_continuation(
            "Reaction products carry explicit hydrogens. Canonicalise the best "
            "molecule before submitting it.",
            ("validate_smiles", {"smiles": self.best}),
        )

    def _submit(self, transcript: str = "") -> str:
        latest = last_tool_output(transcript)
        if (
            latest is not None
            and latest[0] == "validate_smiles"
            and isinstance(latest[1], dict)
            and latest[1].get("canonical_smiles")
        ):
            self.best = str(latest[1]["canonical_smiles"])
        self.stage = "done"
        if self.accepted:
            rationale = (
                "Applied " + ", ".join(self.accepted.copy()) + "; each step was "
                "measured and kept only when it improved SA or QED."
            )
        else:
            rationale = (
                "No simplification in my library applied to this molecule "
                "without making a measured property worse, so the starting "
                "molecule is my answer."
            )
        return atem.render_continuation(
            "Out of moves worth trying. Submit the best measured molecule.",
            ("submit_molecule", {"smiles": self.best, "rationale": rationale}),
        )
