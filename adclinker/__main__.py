"""Commands: `probe`, `profile`, `warheads`, `prepare`, `finetune`, `generate`.

`probe`, `profile` and `warheads` run no model and need nothing installed
beyond RDKit. `prepare` cuts the reference reagents into LinkInvent training
triples. `finetune` and `generate` shell out to REINVENT, which needs its own
environment but not necessarily a GPU -- the LinkInvent prior is a small RNN
and samples on CPU.

The pipeline in order:

    probe      -> is REINVENT reachable, is the prior present
    profile    -> measure real ADC linkers; everything calibrates off this
    warheads   -> build the two pieces for this antibody/payload pair
    prepare    -> cut the reference set into (warheads, linker) triples
    finetune   -> teach the prior what ADC linkers look like   (optional)
    generate   -> sample, score, rank  -> candidates.csv       (the deliverable)

`generate` works against the stock prior, so `prepare` and `finetune` are
optional. Run generate first and look at the scores before deciding whether
fine-tuning is worth it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from moltrain import logs

from . import reinvent as rv
from . import reference, score, slice as slicing, warheads as wh
from .config import ConfigError, load_config, resolve

LOG = logs.get_logger("adclinker")


def _profile(cfg: dict[str, Any], refresh: bool = False) -> reference.Profile:
    return reference.load_profile(
        resolve(cfg, cfg["reference"]["dataset"]),
        cache=resolve(cfg, cfg["reference"]["cache"]),
        refresh=refresh,
    )


def _designs(cfg: dict[str, Any]) -> list[wh.WarheadPair]:
    """The warhead pairs to generate against.

    One per cassette when `design.all_cassettes` is set, because the cassette
    is the choice with the largest effect on the molecule and enumerating it
    costs only sampling time.
    """
    design = cfg["design"]
    cassettes = (
        [c for c in wh.CASSETTES if c != "none"]
        if design["all_cassettes"]
        else [design["cassette"]]
    )
    pairs: list[wh.WarheadPair] = []
    problems: list[str] = []
    for cassette in cassettes:
        try:
            pairs.append(
                wh.build_pair(
                    design["handle"],
                    cassette,
                    design["payload"],
                    design["architecture"],
                )
            )
        except wh.WarheadError as exc:
            problems.append(f"{cassette}: {exc}")
    if not pairs:
        raise ConfigError(
            "no warhead pair could be built:\n  " + "\n  ".join(problems)
        )
    for problem in problems:
        LOG.warning("skipping %s", problem)
    return pairs


# ------------------------------------------------------------------- commands


def _cmd_probe(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    print("reference data")
    dataset = resolve(cfg, cfg["reference"]["dataset"])
    print(f"  dataset: {dataset}")
    print(f"  present: {dataset.is_file()}")

    print("\nREINVENT")
    runtime = rv.probe(cfg)
    for key, value in runtime.as_dict().items():
        print(f"  {key}: {value}")

    print("\nscoring plugin")
    plugin = resolve(cfg, cfg["scoring_plugin"]["plugin_root"])
    component = plugin / "reinvent_plugins" / "components" / "comp_adclinker.py"
    print(f"  root: {plugin}")
    print(f"  component present: {component.is_file()}")
    # REINVENT walks a native namespace package; a stray __init__.py makes it
    # a regular package and the importer refuses the whole directory.
    strays = list(plugin.rglob("__init__.py"))
    print(f"  namespace package clean: {not strays}")
    for stray in strays:
        print(f"    ! remove {stray}")

    print("\ndesign")
    try:
        for pair in _designs(cfg):
            print(f"  {pair.handle} + {pair.cassette} + {pair.payload}")
    except ConfigError as exc:
        print(f"  ! {exc}")
        return 2

    print()
    if runtime.problems:
        for problem in runtime.problems:
            print(f"! {problem}")
        return 1
    print("ready")
    return 0


def _cmd_profile(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    profile = _profile(cfg, refresh=args.refresh)
    print(f"reference set: {profile.source}")
    print(
        f"  {profile.n_unique} unique molecules, {profile.n_profiled} profiled "
        f"(the rest carry protecting or activating groups), "
        f"{profile.n_cleavable} with a cleavage motif"
    )
    print(f"\n  {'property':18s} {'p10':>9} {'median':>9} {'p90':>9}")
    for name, stats in profile.properties.items():
        print(
            f"  {name:18s} {stats.p10:9.2f} {stats.median:9.2f} {stats.p90:9.2f}"
        )
    print(
        "\n  These windows are what score.py plateaus on. Median cLogP near "
        "0.7\n  and Fsp3 near 0.9 is the whole reason ADC thresholds differ "
        "from\n  PROTAC ones."
    )
    if args.json:
        print(json.dumps(profile.as_dict(), indent=2))
    return 0


def _cmd_warheads(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    pairs = _designs(cfg)
    path = resolve(cfg, cfg["reinvent"]["warheads_file"])
    rv.write_warheads_file(path, (p.as_linkinvent_input() for p in pairs))
    print(f"wrote {len(pairs)} warhead pair(s) to {path}\n")
    for pair in pairs:
        print(f"  {pair.handle} + {pair.cassette} + {pair.payload}")
        print(f"    architecture: {pair.architecture}")
        print(f"    {wh.CASSETTES[pair.cassette].note}")
        line = pair.as_linkinvent_input()
        print(f"    input: {line[:100]}{'...' if len(line) > 100 else ''}\n")
    return 0


def _cmd_prepare(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    settings = cfg["finetune"]
    dataset = resolve(cfg, cfg["reference"]["dataset"])
    smiles = reference.read_reference_smiles(
        dataset, cfg["reference"]["smiles_column"]
    )
    if args.limit:
        smiles = smiles[: args.limit]
    print(f"cutting {len(smiles)} reference reagents into training triples...")

    triples, report = slicing.build_triples(
        smiles,
        min_linker_atoms=int(settings["min_linker_atoms"]),
        max_linker_atoms=int(settings["max_linker_atoms"]),
        min_warhead_atoms=int(settings["min_warhead_atoms"]),
        max_triples_per_molecule=int(settings["max_pairs_per_molecule"]),
        seed=int(settings["seed"]),
    )
    written = slicing.write_split(
        triples,
        resolve(cfg, settings["training_file"]),
        resolve(cfg, settings["validation_file"]),
        validation_fraction=float(settings["validation_fraction"]),
        seed=int(settings["seed"]),
    )

    summary = report.as_dict()
    print(f"\n  {summary['triples']} triples from {summary['molecules_used']} molecules")
    print("  rejected cuts:")
    for reason, count in list(summary["rejected"].items())[:6]:
        print(f"    {count:>7}  {reason}")
    print()
    for key, value in written.items():
        print(f"  {key}: {value}")
    print(
        "\n  Split by source molecule, not by triple: cuts from one reagent "
        "are\n  near-duplicates, and splitting them across train and "
        "validation is\n  exactly the leakage that makes the Linker-GPT "
        "paper's numbers unreliable."
    )
    return 0


def _cmd_finetune(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    settings = cfg["finetune"]
    training = resolve(cfg, settings["training_file"])
    if not training.is_file():
        print(
            f"no training data at {training}. Run `python -m adclinker prepare` "
            "first.",
            file=sys.stderr,
        )
        return 1
    validation = resolve(cfg, settings["validation_file"])
    config_path = rv.write_transfer_learning_config(
        cfg, training, validation if validation.is_file() else None
    )
    print(f"wrote {config_path}")
    if args.dry_run:
        print("\n--- config ---")
        print(config_path.read_text())
        return 0

    log_path = resolve(cfg, cfg["run"]["log_dir"]) / "finetune.log"
    print(f"running REINVENT transfer learning (log: {log_path})")
    result = rv.run_config(cfg, config_path, log_path=log_path)
    if result.returncode != 0:
        print(f"REINVENT exited {result.returncode}", file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        return 1
    output = resolve(cfg, settings["output_model"])
    print(f"\nfine-tuned model: {output}")
    print(
        "  Point reinvent.model_file at it to sample from it instead of the "
        "stock prior."
    )
    return 0


def _cmd_generate(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Sample, score, rank. This is the deliverable for the LLM stage."""
    pairs = _designs(cfg)
    warheads_path = resolve(cfg, cfg["reinvent"]["warheads_file"])
    rv.write_warheads_file(
        warheads_path, (p.as_linkinvent_input() for p in pairs)
    )

    samples_csv = resolve(cfg, cfg["reinvent"]["samples_csv"])
    config_path = rv.write_sampling_config(cfg, warheads_path, samples_csv)
    per_pair = int(cfg["reinvent"]["num_smiles"])
    print(
        f"sampling {per_pair} molecules for each of {len(pairs)} warhead "
        f"pair(s) = {per_pair * len(pairs)} total"
    )
    if args.dry_run:
        print("\n--- config ---")
        print(config_path.read_text())
        print("--- warheads ---")
        print(warheads_path.read_text())
        return 0

    log_path = resolve(cfg, cfg["run"]["log_dir"]) / "generate.log"
    result = rv.run_config(cfg, config_path, log_path=log_path)
    if result.returncode != 0:
        print(f"REINVENT exited {result.returncode}", file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        return 1

    samples = rv.read_samples(samples_csv)
    stats = rv.summarise_samples(samples)
    print(
        f"\n  sampled {stats['total']}: {stats['valid']} valid "
        f"({stats['validity_percent']}%), {stats['duplicate']} duplicate, "
        f"{stats['invalid']} invalid"
    )

    rows = _score_samples(cfg, pairs, samples)
    paths = _write_report(cfg, rows)
    accepted = [r for r in rows if r.get("accepted")]
    print(f"  {len(accepted)} of {len(rows)} passed every gate")
    if accepted:
        best = accepted[0]
        print(f"  best score {best['score_total']}: {best['score_reason']}")

    gate_counts: dict[str, int] = {}
    for row in rows:
        for code in (row.get("gates_failed") or "").split(";"):
            if code:
                gate_counts[code] = gate_counts.get(code, 0) + 1
    if gate_counts:
        print("\n  why candidates were rejected:")
        for code, count in sorted(gate_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>7}  {code}: {score.GATES.get(code, '')}")

    print()
    for label, path in paths.items():
        print(f"{label:>16}: {path}")
    return 0


def _score_samples(
    cfg: dict[str, Any], pairs: list[wh.WarheadPair], samples: list[rv.Sample]
) -> list[dict[str, Any]]:
    """Score every valid sample against the design it came from."""
    profile = _profile(cfg)
    dataset = resolve(cfg, cfg["reference"]["dataset"])
    fingerprints = reference.fingerprint_reference(dataset)

    # The sampler echoes the warhead line back, so each sample can be mapped
    # to the cassette it was generated for.
    by_input = {p.as_linkinvent_input(): p for p in pairs}
    payload_smiles = wh.PAYLOADS[cfg["design"]["payload"]].smiles
    handle_smarts = _handle_smarts(cfg["design"]["handle"])

    context = score.Context(
        profile=profile,
        reference_fingerprints=fingerprints,
        payload_smiles=payload_smiles,
        handle_smarts=handle_smarts,
        require_cleavable=bool(cfg["score"]["require_cleavable"]),
        min_heavy_atoms=int(cfg["score"]["min_heavy_atoms"]),
    )

    rows: list[dict[str, Any]] = []
    rebuild_failures = 0
    for sample in samples:
        if sample.state == rv.STATE_INVALID or not sample.smiles:
            continue
        pair = by_input.get(sample.warheads)

        # Rebuild from the generated linker and the ORIGINAL warheads rather
        # than trusting REINVENT's assembled SMILES. The model was handed
        # flattened warheads -- its input vocabulary has no stereochemistry
        # tokens -- so its own assembly has every stereocentre in the payload
        # and the cassette unspecified. Re-joining here restores all of them
        # exactly, because the model never saw them.
        assembled = sample.smiles
        if pair is not None and sample.linker:
            try:
                assembled = wh.assemble(sample.linker, pair)
            except wh.WarheadError:
                rebuild_failures += 1
                continue

        result = score.score_smiles(assembled, context)
        row = {
            "smiles": assembled,
            "reinvent_smiles": sample.smiles,
            "linker": sample.linker,
            "nll": sample.nll,
            "duplicate": sample.duplicate,
            "handle": pair.handle if pair else cfg["design"]["handle"],
            "cassette": pair.cassette if pair else "",
            "payload": pair.payload if pair else cfg["design"]["payload"],
            "architecture": pair.architecture if pair else "",
        }
        row.update(result.as_row())
        rows.append(row)

    if rebuild_failures:
        LOG.warning(
            "%d generated linker(s) could not be rejoined to their warheads "
            "and were dropped",
            rebuild_failures,
        )
    return score.rank(rows)


def _handle_smarts(handle: str) -> str | None:
    """The reactive core of the chosen handle, for the one-handle gate."""
    cores = {
        "maleimide": "O=C1C=CC(=O)N1",
        "maleimidocaproyl": "O=C1C=CC(=O)N1",
        "nhs_ester": "O=C(ON1C(=O)CCC1=O)",
        "dbco": "C#Cc1ccccc1",
        "bromoacetyl": "[CX4](Br)C(=O)",
    }
    return cores.get(handle)


def _write_report(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Path]:
    settings = cfg["report"]
    out_dir = resolve(cfg, settings["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n = int(settings["top_n"])
    kept = rows[:top_n] if top_n > 0 else rows

    csv_path = out_dir / settings["candidates_csv"]
    if kept:
        leading = [
            "rank",
            "score_total",
            "accepted",
            "score_reason",
            "smiles",
            "linker",
            "cassette",
            "handle",
            "payload",
        ]
        seen: list[str] = []
        for row in kept:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = [k for k in leading if k in seen] + [
            k for k in seen if k not in leading
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
    else:
        csv_path.write_text("")

    accepted = [r for r in rows if r.get("accepted")]
    scores = [float(r["score_total"]) for r in accepted if r.get("score_total")]
    summary = {
        "scored": len(rows),
        "accepted": len(accepted),
        "written": len(kept),
        "unique_smiles": len({r["smiles"] for r in accepted}),
        "best_score": max(scores) if scores else None,
        "median_score": sorted(scores)[len(scores) // 2] if scores else None,
    }
    summary_path = out_dir / settings["summary_json"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {"candidates_csv": csv_path, "summary_json": summary_path}


COMMANDS = {
    "probe": _cmd_probe,
    "profile": _cmd_profile,
    "warheads": _cmd_warheads,
    "prepare": _cmd_prepare,
    "finetune": _cmd_finetune,
    "generate": _cmd_generate,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adclinker",
        description="Generate and rank ADC linker-payload candidates.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="redo cached work")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the config but run nothing"
    )
    parser.add_argument("--json", action="store_true", help="also dump JSON")
    logs.add_cli_flags(parser)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    logs.configure_from_args(cfg, args, stem=f"adclinker-{args.command}")

    try:
        return COMMANDS[args.command](cfg, args)
    except (
        ConfigError,
        rv.ReinventError,
        reference.ReferenceError,
        slicing.SliceError,
        wh.WarheadError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
