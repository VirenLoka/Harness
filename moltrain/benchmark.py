"""Measure what an episode costs under each docking policy.

GRPO samples a group of rollouts per task, so a step costs roughly
`num_generations x episode_time`. This runs a realistic scripted episode under
each mode and reports the wall clock, so the docking policy can be chosen from
measured numbers instead of a guess.

    python -m moltrain.benchmark --receptor ../molbench/receptors/3PTB_BEN.pdbqt
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from . import atem
from .config import load_config
from .episode import Task, run_episode
from .oracle import Oracle
from .toolkit import build_toolkit

# A realistic trajectory: measure the input, test a transformation, dock a
# candidate, then submit. This is what a competent policy does.
BENZAMIDINE = "c1ccc(cc1)C(=N)N"
CANDIDATE = "Cc1ccc(cc1)C(=N)N"


def realistic_policy(
    receptor_file: str | None, start: str = BENZAMIDINE, candidate: str = CANDIDATE
):
    """Scripted stand-in for the model, exercising the tools a policy would."""
    cfg_exhaustiveness = 4
    turns = [
        atem.render_reasoning("Measure the starting molecule before changing it.")
        + atem.render_tool_call("get_molecule_metrics", {"smiles": start}),
        atem.render_tool_call(
            "apply_smarts_reaction",
            {"smiles": start, "reaction_smarts": "[cH:1]>>[c:1]C"},
        ),
    ]
    if receptor_file:
        turns.append(
            atem.render_tool_call(
                "dock_and_score",
                {
                    "smiles": candidate,
                    "receptor_file": receptor_file,
                    "exhaustiveness": cfg_exhaustiveness,
                },
            )
        )
    turns.append(
        atem.render_tool_call(
            "submit_molecule", {"smiles": candidate, "rationale": "tested analog"}
        )
    )
    remaining = list(turns)

    def generate(_transcript: str) -> str:
        return remaining.pop(0) if remaining else atem.render_final("done")

    return generate


async def measure(
    cfg,
    mode: str,
    receptor: str | None,
    episodes: int,
    fresh_cache: bool,
    start: str = BENZAMIDINE,
    candidate: str = CANDIDATE,
):
    cfg["reward"]["docking"]["mode"] = mode
    if fresh_cache:
        # A cold cache is the honest worst case for the first step of training.
        for key in ("cache_path",):
            path = Path(cfg["reward"]["docking"][key])
            cfg["reward"]["docking"][key] = str(
                path.with_name(f"bench_{mode}_{path.name}")
            )
        cfg["reward"]["property_cache_path"] = f"cache/bench_{mode}_properties.jsonl"

    toolkit = build_toolkit(cfg)
    oracle = Oracle(cfg, toolkit)

    durations, docks, rewards = [], [], []
    for index in range(episodes):
        # A distinct task id per episode; the molecule is the same, so later
        # episodes show the cache effect that GRPO's repeated sampling gets.
        task = Task(
            task_id=f"bench_{mode}_{index}",
            smiles=start,
            target="trypsin" if receptor else "none",
            receptor_file=receptor,
        )
        started = time.monotonic()
        result = await run_episode(
            cfg, task, realistic_policy(receptor, start, candidate), toolkit, oracle
        )
        durations.append(time.monotonic() - started)
        docks.append(result.docks_run)
        rewards.append(result.reward)
    return {
        "mode": mode,
        "first_s": durations[0],
        "median_s": statistics.median(durations),
        "total_s": sum(durations),
        "docks_first": docks[0],
        "docks_total": sum(docks),
        "mean_reward": statistics.fmean(rewards),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--receptor", default=None, help="prepared .pdbqt; omit for 2D only"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", default=["off", "final_only", "every_candidate"]
    )
    parser.add_argument(
        "--generations", type=int, default=8, help="GRPO group size, for the estimate"
    )
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--start", default=BENZAMIDINE, help="starting molecule")
    parser.add_argument(
        "--candidate", default=CANDIDATE, help="analog the policy tests"
    )
    args = parser.parse_args()

    print(
        f"{'mode':16s} {'1st ep':>8s} {'median':>8s} {'docks(1st)':>11s} {'reward':>8s}  {'GRPO step':>10s}"
    )
    for mode in args.modes:
        cfg = load_config(args.config)
        stats = asyncio.run(
            measure(
                cfg,
                mode,
                args.receptor,
                args.episodes,
                not args.keep_cache,
                args.start,
                args.candidate,
            )
        )
        # One GRPO step runs `generations` rollouts of one task. Rollouts after
        # the first in a group hit the cache, so use the median for them.
        step_estimate = stats["first_s"] + (args.generations - 1) * stats["median_s"]
        print(
            f"{stats['mode']:16s} {stats['first_s']:8.1f} {stats['median_s']:8.1f} "
            f"{stats['docks_first']:11d} {stats['mean_reward']:8.2f}  {step_estimate:9.0f}s"
        )
    print(
        f"\nGRPO step = 1 cold rollout + {args.generations - 1} warm ones, "
        "serial. Parallel rollouts divide that by the worker count."
    )


if __name__ == "__main__":
    main()
