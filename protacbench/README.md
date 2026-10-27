# protacbench

Assessment and triage of DiffPROTAC → Pro-GAT linker samples. It samples,
measures every molecule on ~30 parameters, sorts them into `ok` / `poor` /
`broken`, and writes a CSV saying **why** each rejected one failed, with the
full PROTAC and linker SMILES and a ready-to-run repair prompt.

```
Pro-GAT sampling (GPU) → XYZ → own Open Babel conversion → assessment
  → samples.csv  (everything, ranked, with verdicts and scores)
  → rejects.csv  (the failures, with reasons and a repair prompt)
  → repair_tasks.jsonl  (the same, as structured tasks)
       ↓
  LLM repair through the DeepSeek harness, with molbench mounted over MCP
       ↓  each answer verified against the original fragments
  → repaired.csv / results.jsonl / summary.json  (what was salvaged)
```

**No model pass runs without a GPU.** `runner.sample` refuses unless a CUDA
device is visible, like `moltrain.diffsbdd`. Everything downstream works on
files already on disk, so assessment runs anywhere.

## Commands

```bash
python -m protacbench probe    # what this host can do; runs no model pass
python -m protacbench plan     # the exact Pro-GAT argv, without running it
python -m protacbench sample   # Pro-GAT sampling + repair  (GPU)
python -m protacbench assess   # measure, rank, triage, write the CSVs
python -m protacbench repair   # LLM repair of the rejects  (model endpoint)
python -m protacbench prompts  # paste-ready tasks, one file per molecule
python -m protacbench grade    # score replies against the rubric they got
```

All of them take `--verbose`, `--quiet`, `--no-progress`, `--log-file`,
`--limit`.

`prompts` and `grade` are the two ends of a hand-run loop, for when you are
driving a model yourself rather than through the endpoint:

```bash
python -m protacbench grade --reply-dir replies/ --verbose
```

Name each reply `<target>-<sample>.txt`, the way `prompts` names its tasks, or
pass `--task`. The scorecard splits into **chemistry** — is the molecule a good
PROTAC — and **protocol** — did the reply say so truthfully, in the form it was
asked for. The two fail independently, and that is the point: a sound molecule
in a reply with no `REPAIRED:` line is a harness bug, while a parseable answer
whose cLogP climbed 1.1 is the model. `repair.verify` scores both of those the
same, so `grade` measures three things it cannot:

- **regression against the input**, not just the absolute limits. Every
  descriptor can get substantially worse while still passing its threshold,
  which is exactly what a repair that deletes a heteroatom to make a reactive
  group disappear looks like.
- **liabilities no tool on the server tests for**, such as an aminal ring.
  `check_protac_linker` returns `ok` on those, so a model that only satisfies
  the checker would otherwise score like one that did the chemistry.
- **whether the reply's numbers are true**, and whether it stated the ones it
  broke. A value matching the input rather than the answer is recorded as an
  echo, not a lie, since replies routinely quote the task back.

First time on a new host, fetch the models:

```bash
scripts/setup_models.sh
```

Pro-GAT and DiffLinker are not vendored here. Both publish their weights in
their own git history — Pro-GAT carries 244 MB of checkpoints and datasets as
ordinary tracked files — so the script clones them at a pinned commit and
gets code and weights together. That keeps all of it out of this repository
and its LFS quota, and it is why `.gitignore` excludes both directories.

## Four things in Pro-GAT that this works around

These were found by reading the code, and each one changes the output.

**The failure reason is destroyed.** `run_progat.py`'s `convert_xyz_to_sdf`
*deletes* the SDF whenever conversion, parsing or the connectivity check
fails, and `compute_repair.compute_metrics` then records only
`"{uuid}/{sample}"` with no reason. Four distinct causes — an impossible
valence, an Open Babel write failure, an RDKit parse failure, and a
disconnected molecule — all reach the log as the same thing: a missing file.
This package re-converts from the XYZ itself and keeps the evidence, so every
rejection carries a code and a sentence.

**Descriptors are computed on radicals.** Open Babel writes heavy-atom-only
SDFs, and molblock semantics tell RDKit the listed valences are complete, so
sanitising assigns radical electrons to every open valence. An ordinary amide
comes back as `O=C([N][C][C]O...)`. Measured on the test fixture, that reads
QED 0.5485 and a mass 24 hydrogens short, against a true 0.6029. Pro-GAT
reports SMILES and metrics from exactly this state.
`structures.restore_implicit_hydrogens` clears the radicals and lets implicit
hydrogens fill the valences, which recovers the molecule exactly. It does not
hide real errors: an atom with too many bonds still fails sanitisation, and
that failure is reported.

**Recovery rate is hardcoded to zero.** `compute_repair.py` sets
`recovery = 0.0` with a comment-free literal, even though `trainer.pred`
writes `true_.xyz` — the reference molecule — next to every sample. This
package reads it and reports a real recovery count.

**The two stages use different file layouts**, and assuming one finds nothing
of the other. Sampling writes a directory per PROTAC
(`{uuid}/{sample}_.xyz`); the repair model's `save_as_xyz` writes a flat
directory keyed by a composite name (`{uuid}-{linker_len}_{sample}_.xyz`).
Worse, `postprocess.py` copies repaired SDFs *back into* the sampling tree, so
after a full run you cannot tell generated from repaired. Reading XYZ and
converting independently keeps the two stages separate, which is what makes
the repair delta measurable.

## Linker size: why the default is 0

`run_progat.py` only installs its fixed-size `sample_fn` when `--linker_size`
is positive. At zero, `Trainer.sample_chain` falls back to each entry's own
`linker_mask.sum()`. Native linker sizes in `protacs_test.pt` run from 3 to 37
with a median of 13, so `--linker_size 9` forces all 400 PROTACs to a 9-atom
linker regardless of the geometry they were built around, which depresses
validity for a reason that has nothing to do with the model.

`progat.linker_size: 0` is therefore the default here. When it is zero the
assessment reads each target's native length back out of the dataset, because
every PROTAC then has a different linker and the extraction has to know which
atoms are which.

## How the linker is isolated

By index, not by substructure matching. Pro-GAT writes fragments first and the
linker as a trailing contiguous block — verified against `protacs_test.pt`,
where entry 0 has 46 atoms with the linker at indices 32–45 — and Open Babel
preserves input order for xyz→sdf, which the tests assert. So the linker is
exactly the last *n* atoms.

This also means the linker is recoverable from a **broken** sample. A
disconnected molecule still has its linker sitting in the trailing block, and
naming it (`Linker: CCOCCOCC`) is the single most useful thing the repair
agent can be handed — it says what to reattach instead of leaving the agent to
infer it.

## The parameters

**Hard gates** decide whether a molecule exists. Failing one is `broken`:

`obabel_missing`, `obabel_timeout`, `obabel_wrote_nothing`, `unreadable_sdf`,
`impossible_valence`, `sanitization_failed`, `disconnected`, `atom_clash`,
`linker_not_bridging`, `linker_size_mismatch`, `missing_xyz`

`atom_clash` is checked on the raw coordinates *before* bond perception,
because Open Babel infers bonds from distance — two atoms on top of each other
yield a confidently wrong molecule rather than an error.

`linker_not_bridging` fires when the linker touches a fragment at anything
other than two atoms. One attachment means it dangles; three means it branched
into a substituent.

**Soft gates** decide whether it is worth keeping. Failing one is `poor`:

`sa_too_high`, `qed_too_low`, `mw_out_of_range`, `clogp_too_high`,
`too_flexible`, `pains_alert`, `bad_linker_ring`, `macrocycle`,
`linker_too_short`, `linker_too_long`

Thresholds are deliberately wide, because PROTACs sit outside Lipinski space
by design: MW 400–1600, cLogP ≤ 9, up to 30 rotatable bonds.

**Measured and reported, never gating:** heavy atoms, MW and exact MW, SA,
QED, cLogP, TPSA, HBD, HBA, rotatable bonds, total/aromatic/aliphatic rings,
stereocentres, formal charge, Fsp3, PAINS descriptions.

**Linker-specific:** heavy atoms, attachment count, the attachment and
fragment-anchor atom indices, **bond span between the fragments**, exit-vector
distance in Ångströms, rotatable bonds, heteroatom fraction, in-ring fraction.

The bond span is worth singling out. It is not the atom count — a linker can
be long in atoms yet short across the fragments if it folds through a ring —
and it is the span, not the count, that governs whether the two warheads can
occupy their own binding sites at once.

**Geometry:** minimum interatomic distance, clashing pair count, centroid
spread.

## Ranking: the synthesis score

Triage answers "is this molecule real, and is it worth keeping". That is a
verdict, not an ordering, and a verdict cannot say which of four hundred `ok`
PROTACs to make first. `score.py` adds the ordering: six weighted terms, each
reporting its own number and a sentence, because a composite with no
breakdown is not reviewable.

| Term | Weight | What it reads |
| --- | --- | --- |
| synthesisability | 25 | RDKit SA score |
| motif familiarity | 25 | fraction of linker atoms covered by routine chemistry |
| exit-vector span | 20 | bonds between the two fragment anchors |
| polarity | 10 | cLogP |
| flexibility | 10 | rotatable bonds |
| liabilities | 10 | PAINS, strained rings, macrocycles, new linker stereocentres |

Three choices in there are opinions worth stating.

**QED is absent.** PROTACs violate Lipinski by construction, so QED measures
how unlike a PROTAC a molecule is. Rewarding it would rank the worst
candidates highest.

**SA carries less than motif familiarity.** SA is nearly flat across
generated linkers — a PEG chain and a C18 alkyl chain both score about 2,
because both really are easy to make — so on its own it does not separate a
good linker from a useless one. Motif coverage against a library of chemistry
a lab already runs (amide, PEG ether, piperazine, triazole, sulfonamide,
carbamate…) does, and it is coverage rather than a match count: one amide in
a twenty-atom exotic linker should not read as familiar.

**Only linker stereocentres count.** Fragment stereocentres come with the
warhead and the E3 ligand and are not the generator's doing. A new one in the
linker is, and each is another chiral synthesis or another separation.

Ranking is by **verdict first, then score**. The score says how hard a
molecule is to make, not whether it is a PROTAC, and the two come apart: a
three-bond linker is trivially synthesisable and cannot bridge two binding
sites. A `broken` sample gets no score at all rather than a zero — zero is a
position in the ranking and these molecules do not have one.

## The repair handoff

`rejects.csv` carries a `repair_prompt` column and `repair_tasks.jsonl` carries
the same thing as structured tasks. Each prompt names the failure, prescribes a
specific remedy for it, gives the full PROTAC and linker SMILES with the
measurements, and states the constraints:

- **do not change either fragment** — they are a warhead and an E3-ligase
  ligand, and an agent that "fixes" a disconnected PROTAC by deleting a
  fragment has destroyed the molecule
- exactly two attachment points, one per fragment
- keep the linker near its required length, which comes from ternary-complex
  geometry rather than preference
- measure before claiming; never state a number no tool returned

The prompts target the molbench tool surface (`validate_smiles`,
`get_molecule_metrics`, `apply_smarts_reaction`, `submit_molecule`), which is
the same surface `moltrain` trains against.

## The repair loop

`python -m protacbench repair` takes `repair_tasks.jsonl`, runs each task
through the DeepSeek harness with molbench mounted over MCP, and verifies
what comes back.

Four things about the harness shape this, all found by reading its shipped
code:

- **Headless is a profile, not a flag.** `dsh --profile headless` is the
  non-interactive mode, and its entire option set is `--json`, `--session-id`
  and `--help`.
- **Launcher flags come first.** `--profile` and `--patch` are the launcher's;
  the first token it does not recognise ends its parsing, so `--json` goes
  last.
- **The model is not selectable from the command line.** Provider and model
  come from a YAML row, so `harness.py` writes a cordis patch from our own
  config. That is also what makes the whole thing portable: nothing points at
  anyone's home directory, and the GPU box gets the same file with its own
  endpoint in it. The API key is never written into the patch — it carries
  `apiKeyEnv`, the *name* of the variable, and the value goes through the
  subprocess environment.
- **There is no run timeout anywhere in the harness.** Nothing will stop a
  wedged run, so `harness.timeout_s` is ours and it is not optional.

### Two bugs in the handoff this fixes

**`submit_molecule` does not exist.** The prompts asked the agent to call it
to deliver its answer. That tool belongs to `moltrain`'s own toolkit, which
synthesises it as a terminal action; molbench's MCP server never had it, so
every repair would have dead-ended on a missing tool. The answer now comes
back as a `REPAIRED: <smiles>` line in the final message, which the harness
hands over losslessly.

**Tool names arrive namespaced.** MCP tools reach the model as
`mcp__molbench__<name>`. The prompts named them bare, which sends the agent
looking for tools that are not mounted.

### Verifying the answer instead of believing it

An agent asked to repair a disconnected PROTAC has an easy way out: delete one
fragment. The result is a single connected molecule that sanitises, scores
well on every descriptor, and is not a PROTAC at all. Nothing in the chemistry
of the answer gives this away — the molecule is fine, it is just the wrong
molecule.

So `assess` records `fragment_smiles`, the warhead and the E3 ligand, before
any repair happens. Afterwards each fragment is matched back into whatever
came out. Both present and disjoint means the agent edited the linker; either
one missing means it did something else, and the answer is rejected whatever
its descriptors say. The linker is then whatever is left over, which is how a
repaired molecule gets measured on exactly the same footing as a generated
one — same gates, same score, same thresholds. Only then is "improved" a claim
worth making.

Outcomes, best to worst: `repaired`, `still_poor`, `still_broken`,
`fragments_lost`, `unparseable`, `no_answer`, `harness_failed`.
`fragments_lost` is counted separately in the summary because it is the
failure mode that would otherwise look like success.

### Pointing it at a model

```yaml
harness:
  model:
    base_url: http://127.0.0.1:8000/v1   # the gateway from serve.py
    name: muse-glimmer-30b
    api_key_env: VLLM_API_KEY
```

```bash
export VLLM_API_KEY=<your key>
python -m protacbench probe     # confirms Node, molbench, the key, the endpoint
python -m protacbench repair --limit 20
```

`permission_mode` defaults to `workspace-write`. Repair touches no files, so
the restrictive default is right; `danger-full-access` is the only mode under
which nothing can fail closed, and it switches the file sandbox off entirely,
so it stays opt-in.

## Running the tests

```bash
molbench/.venv/bin/python -m pytest protacbench/tests -q   # from the repo root
```

75 tests, no GPU, no Node, no model endpoint and no network. The repair
tests replace the harness with a script that emits the same NDJSON event
stream, so the subprocess driver, the answer contract and the verification
all run for real against a scripted answer. Samples are built to Pro-GAT's real layout and broken the
ways the diffusion model breaks them, then run through the real Open Babel and
RDKit path. They skip themselves when Open Babel is absent.
