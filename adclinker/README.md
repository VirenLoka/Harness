# adclinker

Generates and ranks ADC linker-payload candidates with REINVENT's LinkInvent,
scored against a profile measured from real ADC linker reagents. Output is a
ranked CSV for the LLM optimisation stage.

```
antibody site + payload
      │
  warheads   handle + cleavage cassette  |  payload        (library)
      │
  generate   REINVENT LinkInvent designs the spacer        (GPU optional)
      │
  assemble   rejoin to the ORIGINAL stereochemistry
      │
  score      3 gates, 7 weighted terms, all data-calibrated
      │
      ▼
  candidates.csv  ──►  LLM optimisation
```

`DESIGN_REVIEW.md` is the mechanistic argument for why the pipeline has this
shape. Read it before changing the chemistry.

## Commands

```bash
python -m adclinker probe       # what this host can do; runs no model
python -m adclinker profile     # measure real linkers; the calibration
python -m adclinker warheads    # build the two pieces for this design
python -m adclinker prepare     # cut the reference set into training triples
python -m adclinker finetune    # transfer-learn the prior        (optional)
python -m adclinker generate    # sample, assemble, score, rank   (the output)
```

All take `--config`, `--limit`, `--refresh`, `--dry-run`, `--json`.

First run on a new host:

```bash
scripts/setup_models.sh     # clones REINVENT4 + Linker-GPT, recovers the prior
```

## Three things found by reading the code, each of which changes the result

### The prior has no stereochemistry tokens

LinkInvent's input vocabulary is **34 tokens** and contains neither `[C@H]`
nor `[C@@H]`. MMAE has ten stereocentres and valine-citrulline two, so a
faithful warhead is rejected outright at token validation.

The fix is not to give up the stereochemistry. `as_linkinvent_input()` flattens
the warheads before handing them over, and `assemble()` rejoins the generated
linker to the **original** warheads afterwards. The model only ever sees the
warheads as context for designing a spacer, so the stereocentres never pass
through it and cannot be corrupted by it. Verified: a generated candidate comes
back with all twelve centres intact.

This also means **REINVENT's own assembled SMILES is discarded**. It is built
from the flattened warheads, so every stereocentre in it is unspecified. Only
the `Linker` column is used.

### The prior is still in REINVENT's git history

REINVENT 4.8 deleted `priors/` and moved it to Zenodo, so a fresh clone has no
model to sample from. The blobs are still reachable:

```bash
git -C REINVENT4 show \
  1561377efd74e7954fb4a87f88a4b81a0eb6e208~1:priors/linkinvent.prior \
  > REINVENT4/priors/linkinvent.prior
```

90 MB, and it verifies against the `sha512.sum` that was deleted beside it.
`setup_models.sh` does this automatically.

### REINVENT dies on the way out in a restricted container

`Reinvent.py` calls `report_resource_usage()` **after** the run finishes, and
that calls `resource.getrusage()`. A sandboxed container — a Jupyter image,
most notably — refuses it with `PermissionError: Operation not permitted`.
REINVENT guards the *import* of `resource` against Windows but not the
*call*, so a complete set of samples ends up on disk behind a non-zero exit
code.

Two defences, because either alone leaves a gap:

- `reinvent.sandbox_safe` (default on) runs REINVENT through a shim that
  makes that one call tolerant and then hands straight to its own entry
  point. The interpreter is found beside the console script, so nothing has
  to be told which environment REINVENT lives in; if it cannot be found, the
  plain console script is used instead.
- `generate` reads the output before trusting the exit code. If REINVENT
  wrote usable samples and then failed, the run is used and the failure is
  reported rather than discarded.

### A bare maleimide cannot be acylated

The maleimide nitrogen is already an imide with two carbonyls; bonding a third
gives a triacylamine, which is not a stable compound. RDKit sanitises it
happily. This is exactly why clinical ADCs use maleimido**caproyl** — the six
carbons put the amide a safe distance from the imide nitrogen.

`build_pair` refuses the combination and says so. The junction guard also
catches anhydrides, acyl halides and gem-diols, because the libraries combine
freely and not every combination is a real molecule.

## The scoring function

Every window is measured from the ~1,700 real reagents in `reference.py`,
after deduplicating and dropping the 39% that carry Fmoc, Boc, NHS or PNP —
bench chemistry that is gone from the finished conjugate.

**Three gates.** Not tradeable, because averaging them into a weighted score
lets a good cLogP hide a fatal defect:

| Gate | Why |
| --- | --- |
| payload intact | the obvious cheat is to edit the drug; the result scores well and is a different compound |
| cleavable motif present | a linker that never releases its payload is not a worse candidate, it is not a candidate |
| exactly one handle | none cannot be conjugated, two crosslinks |

**Seven weighted terms**, polarity highest at 30:

| Term | Weight |
| --- | --- |
| polarity (cLogP) | 30 |
| saturation (Fsp3) | 15 |
| ADC-likeness (nearest-neighbour Tanimoto) | 15 |
| novelty | 10 |
| polar surface (TPSA) | 10 |
| size | 10 |
| synthesisability (SA) | 10 |

Polarity leads because linker-payload hydrophobicity drives aggregation and
accelerated clearance, and that is what kills ADCs in vivo. ADC-likeness is a
**floor, not a target** — maximising it returns MC-Val-Cit-PAB every time,
which is not a design — so it is paired with an explicit novelty term.

### Why this is not `protacbench/score.py` with different constants

The measured reference profile:

| | median |
| --- | --- |
| cLogP | **0.7** |
| Fsp3 | **0.88** |
| rotatable bonds | **25** |

An oral drug sits at cLogP 2–3 with Fsp3 around 0.4. ADC linkers are long,
saturated, floppy, polar chains. A PROTAC linker is a *geometric* problem —
span decides whether a ternary complex forms, and 25 rotatable bonds is a
failure. An ADC linker is a *chemical* problem — nothing holds the payload
anywhere. Scoring one with the other's thresholds would reject every real ADC
linker. The two scorers duplicate a few helpers rather than share them, so
tuning one cannot silently move the other.

## Using the scorer as REINVENT's reward

`plugins/reinvent_plugins/components/comp_adclinker.py` exposes the same
function to REINVENT, so the reward and the ranking criterion cannot drift.

REINVENT discovers components by walking `reinvent_plugins.components` as a
**native namespace package**, so there must be no `__init__.py` in
`plugins/reinvent_plugins/` or its `components/`. There is no plugin-directory
env var — discovery is plain `sys.path`, and `reinvent.run_config` puts the
directory on `PYTHONPATH`. Import failures are swallowed and logged as
`Component ADCLinker could not be imported`, which is where to look if it
silently fails to appear.

```toml
[[stage.scoring.component]]
[stage.scoring.component.ADCLinker]

[[stage.scoring.component.ADCLinker.endpoint]]
name = "ADC linker quality"
weight = 1.0
params.dataset = "/abs/path/Linker-GPT/data/linkers.csv"
params.payload_smiles = "..."
params.handle_smarts = "O=C1C=CC(=O)N1"
```

A gate failure scores a genuine **0.0** — it was scored and it is unusable.
An unreadable SMILES is **NaN**, meaning no opinion. Collapsing the two would
let a parse failure read as a design verdict.

## Fine-tuning

`prepare` cuts the reference reagents into `(warheads, linker)` triples, which
is the format LinkInvent transfer learning wants. This is the step where the
fine-tuning either does or does not make biological sense, so three
constraints apply:

- **One warhead must carry a conjugation handle.** A real reagent is
  `handle – spacer – trigger`, and the lesson we want is "given a handle on
  one side, produce the spacer real reagents use". Cutting `Fmoc-Val-Cit-PAB`
  into two random halves teaches peptide chemistry instead. The handle-bearing
  warhead is also placed first, so the model consistently sees the antibody
  side on the left — the same way inference presents it.
- **Cuts are single acyclic bonds.** A broken ring gives two fragments no
  chemistry reassembles.
- **The linker is size-bounded at both ends.**

The split is **by source molecule, and triples are deduplicated first**. Cuts
of one reagent are near-duplicates, and a catalogue is full of near-identical
reagents, so without both steps the same example lands on both sides. That is
precisely the leakage that makes Linker-GPT's own reported numbers unreliable
— 49% of its validation set is also in its training set.

On the real reference set this yields about **5,500 triples from 1,382
molecules**, spread across azide, alkyne, NHS ester, maleimide, DBCO,
haloacetyl and thiol chemistry.

Fine-tuning is optional, and worth deciding empirically: run `generate`
against the stock prior first and look at the scores. REINVENT's own
documentation notes that transfer learning for LinkInvent has limited effect
since only the generated part is affected.

## What this pipeline does not do

Stated plainly so it is not assumed:

- **It does not design the antibody attachment.** That is one covalent bond
  formed by a known reaction, picked from a short list by the conjugation
  site. No model should be inventing it.
- **It does not choose the payload attachment point.** For payloads in
  clinical use the tolerated position is published and is in `PAYLOADS`. For a
  novel payload `propose_attachment_points` suggests candidates, but attaching
  at the wrong position destroys potency, the molecule still scores perfectly,
  and nothing here would catch it. Keep a human on that decision.
- **It does not predict DAR, plasma stability, payload release efficiency,
  bystander killing, internalisation or therapeutic index.** Those need
  wet-lab work. This ranks candidates for synthesis, and nothing more.

## Running the tests

```bash
molbench/.venv/bin/python -m pytest adclinker/tests -q   # from the repo root
```

52 tests, no GPU, no REINVENT, no network. The end-to-end test replaces the
`reinvent` console script with one that writes the same CSV its sampler
writes, so config generation, the subprocess call, output parsing and the
stereochemistry-restoring re-assembly all run for real against scripted
output.
