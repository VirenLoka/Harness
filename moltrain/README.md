# moltrain

GRPO training for tool-using molecular optimization. The policy is handed a
molecule from a structure-based generative model (DiffSBDD) and has to improve
it with the molbench tools, measured against thresholds relative to that
molecule's own starting properties.

```
pocket (+ reference ligand) -> DiffSBDD -> starting molecule
      -> LLM optimization episode (tool calls) -> submitted lead
      -> reward from measured property deltas
```

**Training needs neither the DeepSeek Harness nor Node.** The tools are
imported straight out of `molbench/server.py` as Python functions, so the
surface the policy trains on is byte-identical to the one it is deployed
against, and the whole thing runs in a bare JupyterLab container.

## Status

| Stage | What it is | State |
| --- | --- | --- |
| 0 | Environment: tools, ATEM parsing, episodes, reward, caching, cost benchmark | **done, tested** |
| 1 | Task builder: CrossDocked pockets + DiffSBDD to `tasks.jsonl` | **done, tested** |
| 2 | SFT warmup: scripted expert traces, completion-only loss | **done**, trainer unrun |
| 3 | GRPO: batched rollouts, masked tokens, group advantages | **done**, trainer unrun |
| 4 | Held-out evaluation against controls | **done, tested** |

"Trainer unrun" is literal and it matters: everything except the two
`Trainer.train()` calls is exercised by the test suite. The TRL calls
themselves have never executed, because this machine has no GPU. Run
`python -m moltrain.sft check` and `python -m moltrain.grpo check` on the
H200 host first; they diagnose the TRL version against what this package
passes it and name anything that would be silently dropped.

## Running it on the H200 host

Nothing in this package runs a model pass on a machine without a GPU, by
design: `moltrain.diffsbdd` refuses to sample without a visible CUDA device
rather than grinding for hours and appearing to hang.

```bash
# 0. unpack the pockets (1.6 GB archive, about 10 GB unpacked)
mkdir -p /data/crossdocked
tar -xzf DiffSBDD/data/crossdocked_pocket10.tar.gz -C /data/crossdocked
#    and put split_by_name.pt beside crossdocked_pocket10/ -- see below

# 1. check the host before spending GPU time on it
python -m moltrain.tasks probe
python -m moltrain.sft check
python -m moltrain.grpo check

# 2. build the dataset
python -m moltrain.tasks plan      # the DiffSBDD argv, without running it
python -m moltrain.tasks sample    # GPU
python -m moltrain.tasks build     # Open Babel + Vina

# 3. warm up the format
python -m moltrain.traces collect --source scripted
python -m moltrain.sft train

# 4. measure one GRPO step before committing to five hundred
python -m moltrain.grpo rollout --tasks-per-step 1
python -m moltrain.grpo train

# 5. report against the controls
python -m moltrain.evaluate --tasks data/tasks_heldout.jsonl \
  --arms unchanged expert policy
```

Hold out pockets by building two task files: run `tasks build` with a
`dataset.pockets.limit` that covers the training pockets, then again with a
different `shuffle_seed` or an explicit list and a different `tasks_file`.

## Layout

| Path | What it is |
| --- | --- |
| `config.py`, `configs/default.yaml` | One YAML file drives every stage |
| `atem.py` | Parses and renders Muse Glimmer's ATEM channel format |
| `toolkit.py` | The trained tool surface; wraps molbench, never raises |
| `oracle.py` | Measured properties, cached on disk and docked in parallel |
| `rewards.py` | Relative thresholds with shaped progress |
| `episode.py` | The multi-turn loop, policy-agnostic |
| `crossdocked.py` | Reads the CrossDocked test split and names its pockets |
| `diffsbdd.py` | Drives DiffSBDD sampling as a subprocess |
| `tasks.py` | Builds `tasks.jsonl`: pocket + start molecule + baseline |
| `experts.py` | A scripted expert; its move library was vetted by measurement |
| `traces.py` | Collects SFT records, one per assistant turn |
| `policy.py` | vLLM generation: stop strings, batching, context trimming |
| `grpo_core.py` | Token masking and group advantages |
| `sft.py`, `grpo.py` | Stage 2 and 3 trainers, each with a `check` mode |
| `trl_compat.py` | Filters settings against the installed TRL's signature |
| `evaluate.py` | Held-out evaluation against do-nothing and expert controls |
| `benchmark.py` | Measures episode cost under each docking mode |
| `tests/` | 104 tests; the full loop runs with a scripted policy, no GPU |

## Building the dataset (Stage 1)

A task is one pocket, one starting molecule and that molecule's measured
baseline. Baselines are recorded once here because every reward threshold is
relative to them.

### What has to be on the GPU host

Sampling a diffusion model on CPU takes hours per pocket, so
`moltrain.diffsbdd` refuses to run without a visible CUDA device rather than
appearing to hang. Set `dataset.diffsbdd.allow_cpu: true` only for a
deliberate smoke test.

Two Python environments are involved, because DiffSBDD's dependency set
(torch, pytorch-lightning, torch-scatter, openbabel, wandb) conflicts with
molbench's. Run `moltrain` from molbench's environment and point
`dataset.diffsbdd.python` at DiffSBDD's:

```yaml
dataset:
  diffsbdd:
    python: /opt/conda/envs/diffsbdd/bin/python
    checkpoint: checkpoints/crossdocked_fullatom_cond.ckpt
```

### Getting the CrossDocked pockets

`DiffSBDD/data/crossdocked_pocket10.tar.gz` (1.6 GB) holds 183,468
pocket/ligand pairs under a single `./crossdocked_pocket10/` directory, plus
its own `index.pkl`. Unpack it on the GPU host:

```bash
mkdir -p /data/crossdocked
tar -xzf DiffSBDD/data/crossdocked_pocket10.tar.gz -C /data/crossdocked
```

That gives `/data/crossdocked/crossdocked_pocket10/`, so:

```yaml
dataset:
  pockets:
    crossdocked_root: /data/crossdocked
    split: test
    limit: 25        # null takes all 100 test pockets
```

**The archive does not contain `split_by_name.pt`.** That is the file naming
the 100 published test pockets, and it is a separate small download in the
[Pocket2Mol data release](https://github.com/pengxingang/Pocket2Mol/tree/main/data).
Put it next to `crossdocked_pocket10/`. DiffSBDD's own `process_crossdock.py`
needs the same file, so it is worth having either way.

Without it, `split: index` reads the archive's bundled `index.pkl` instead.
That works today and resolves to real structures, but it is the full 184,057-row
index rather than the held-out test set, so pockets drawn from it may be ones
the DiffSBDD checkpoint was trained on. `probe` says so explicitly when the
split file is absent.

Run `python -m moltrain.tasks probe` after unpacking: it checks the layout and
names the fix for a still-packed archive, an unpack one level too deep, or a
missing split file.

Receptors in this dataset are the 10 Angstrom pocket shell
(`..._pocket10.pdb`), not whole proteins — the full `_rec.pdb` files are not in
the archive. The default `dataset.box.padding` of 5 Angstroms keeps the docking
box inside that shell; raising it much past 10 would put box volume where there
is no protein.

### The commands

```bash
python -m moltrain.tasks probe    # what this host can do; runs no model pass
python -m moltrain.tasks pockets  # which pockets were selected, and are present
python -m moltrain.tasks plan     # the exact DiffSBDD argv, without running it
python -m moltrain.tasks sample   # generate ligands  (GPU)
python -m moltrain.tasks build    # SDF -> tasks.jsonl (Open Babel + Vina)
```

`sample` and `build` both skip work already on disk, so a long run resumes
after an interruption and a failing pocket is recorded rather than aborting
the other 99. `build` writes `tasks_report.json` next to `tasks.jsonl` with
per-pocket provenance: how many molecules DiffSBDD produced, how many survived
each filter, and which pockets failed and why.

### Starting molecules without a GPU

`dataset.source` picks where episodes begin:

| value | start molecule | needs |
| --- | --- | --- |
| `diffsbdd` | sampled for the pocket | GPU |
| `reference_ligand` | the crystal ligand already bound there | nothing |
| `smiles_file` | a flat SMILES list | nothing |

`reference_ligand` is the useful one for a dry run on the real pockets: it
exercises receptor preparation, boxes, filters and baselines end to end, and
gives a sanity reference for what "optimizing a known good binder" scores.

### Docking boxes

The box wraps the reference ligand plus `dataset.box.padding`, clamped per
axis to `[min_size, max_size]`. Sizing to the ligand rather than using a fixed
cube keeps the search volume near the pocket the affinity was measured in —
binding affinity is the threshold most sensitive to box size. Pocket PDBs are
staged under the pocket's own unique name before preparation, because
CrossDocked reuses file stems like `1a0q_A_rec.pdb` across directories and
molbench names receptors after the stem.

## The episode

The policy sees its goals, the starting molecule and its measured properties,
then works through tool calls. It ends an episode by calling
`submit_molecule`, which is far more reliable than parsing a SMILES out of
prose and gives the reward an unambiguous answer to score.

Tools available during training:

`validate_smiles`, `get_molecule_metrics`, `apply_smarts_reaction`,
`generate_3d_conformer`, `run_docking_simulation`, `get_sbdd_metrics`,
`dock_and_score`, `submit_molecule`

`dock_and_score` is the harness plugin's TypeScript composite, reimplemented
in Python here so the trained surface matches the deployed one. Edit
`episode.tools` to change the surface.

## Reward

Thresholds are relative to each episode's own starting molecule, so an easy
pocket and a hard one are scored on the same footing. They come in two kinds,
and the distinction is the whole design:

```yaml
thresholds:
  sa_score:         {direction: decrease, goal_delta: 0.5}   # a goal
  qed:              {direction: increase, goal_delta: 0.05}  # a goal
  binding_affinity: {direction: decrease, goal_delta: 0.0, tolerance: 0.5}
```

A **goal** (`goal_delta > 0`) asks for improvement and pays *shaped* progress
in [0, 1] rather than a step function, so moving SA from 4.9 to 4.6 against a
0.5 goal still produces gradient.

A **constraint** (`goal_delta <= 0`, "do not get worse") is a gate. It earns
nothing when satisfied and is charged by how far past its allowance it went,
in tolerance units, capped at `constraint_violation_cap` so one ruinous
rollout cannot swamp its group's advantage.

Smaller terms cover validity, novelty, tool-call success and a per-call
penalty. Failing to submit scores worse than submitting something mediocre,
so giving up is never safe.

### Three incentive bugs the evaluation harness caught

These are recorded because each one would have quietly ruined a training run,
and each was found by running the controls rather than by reading the code.

**A flat per-call penalty made measuring irrational.** At 0.02 per call, the
expert's sixteen honest calls cost 0.32 and erased the 0.116 of threshold
progress they bought, so the do-nothing control outscored it 0.380 to 0.296.
The reward's optimum was to submit blind. Now only calls beyond
`turn_penalty_free_calls` are charged, at 0.005: measuring is the task, not
waste.

**Submitting blind collected the tool-success term.** A single
`submit_molecule` call has a 100% success rate. The term is now withheld
unless the policy called at least one measurement tool.

**Constraints paid full marks for inaction.** An unchanged molecule satisfies
"maintain binding affinity" perfectly, so scoring it as progress gave the
untouched caffeine submission 0.708 -- more than an episode that actually
improved something. Constraints no longer pay progress at all.

Measured spread after the fixes, scripted expert, docking off:

| molecule | reward | what happened |
| --- | --- | --- |
| stereo-dense chain | 1.274 | SA and QED both moved |
| nitroaryl | 1.051 | nitro to aniline; QED up 0.11 |
| CF3 spiro | 0.960 | cyclopropane removed, SA 3.66 to 2.64 |
| gem-dimethyl | 0.724 | modest SA gain |
| aspirin | 0.370 | nothing in the library applies; filtered from SFT |

## SFT warmup (Stage 2)

GRPO cannot learn from a rollout the ATEM parser rejects -- it scores badly for
the wrong reason. A short warmup on traces known to parse and known to have
scored well gets the format out of the way so GRPO's gradient goes into
decisions rather than syntax.

```bash
python -m moltrain.traces collect --source scripted   # no GPU needed
python -m moltrain.sft check
python -m moltrain.sft train
```

### Tool output never appears in a completion

Each record is one assistant turn: the transcript before it as `prompt`, the
turn itself as `completion`. Every tool result therefore lands in some later
record's prompt and never in a completion, so TRL's completion-only loss masks
them with no custom collator.

This is not a tidiness point. Training on the whole transcript would teach the
model to *write* tool results, which is how a tool-using model learns to state
measurements it never made. `sft.py` refuses to train if any completion
contains a `<tool_output>`, and the test suite asserts the invariant on
freshly collected traces.

### The expert, and what it is for

`experts.py` is a greedy hill-climb over twelve SMARTS simplifications. It
runs through `run_episode` like any policy, so its tool calls are executed for
real and every number in its traces came back from a tool.

The library was built by measurement, not plausibility. Each move's SA effect
was measured on a probe carrying that group, and the moves are tried in order
of measured benefit. Three candidates were dropped after measuring them:
stripping a stereocentre (`[C@:1]>>[C:1]`) is a no-op, because RDKit's reaction
machinery carries the chirality tag through, and both cyclopropane
ring-opening templates fragmented the molecule and raised SA by over 4.

The expert accepts a move only when progress toward the configured thresholds
goes up. Judging on SA alone made it trade QED away -- which SFT would then
have taught.

It teaches protocol, not chemistry: measure, change one thing, verify, submit.
GRPO is what is supposed to find the chemistry.

## GRPO (Stage 3)

```bash
python -m moltrain.grpo check                      # diagnose TRL and the tasks
python -m moltrain.grpo rollout --tasks-per-step 1 # measure a real step, no training
python -m moltrain.grpo train
```

`rollout` is worth running first. It executes one step's worth of rollouts with
the real model and reports cost, truncations and the degenerate-group rate,
without touching the optimiser.

### The mask

A rollout is one token sequence alternating between what the model wrote and
what the environment appended, and only the model's tokens may carry gradient.
`grpo_core.build_completion` derives the mask from exact string arithmetic over
the recorded turns: because a transcript only ever grows, the environment's
contribution between two turns is the slice between them, and that is checked
rather than assumed.

On a measured expert episode, **42% of completion tokens are
environment-authored**. Without the mask that is 42% of the training signal
teaching the model to produce tool output.

`grpo check` greps the installed `GRPOTrainer` for `completion_mask` and fails
the check if it is absent, because a TRL that ignores the column would train on
those tokens silently.

### Degenerate groups

GRPO's baseline is the group mean, so a group whose rollouts all score the same
yields zero advantage and teaches nothing. `grpo_core.degenerate_groups` counts
them and the rollout report prints the count.

This is not hypothetical: the scripted expert is deterministic, so **every one
of its groups is degenerate and GRPO on it is exactly a no-op**. A test asserts
that, to keep anyone from wiring the expert into Stage 3. The same counter
catches a temperature set too low, and tasks that are uniformly trivial or
uniformly hopeless.

### Batched generation

The episode loop asks for one turn at a time, so eight concurrent rollouts
would decode eight sequences in series. `policy.BatchingGenerator` collects the
concurrent requests into a single engine batch, which is what makes a group
cost roughly one rollout's worth of decode. A group of eight becomes one
`generate` call, which the tests assert.

### Context window

`policy.trim_transcript` fits an over-long transcript by dropping its oldest
tool exchanges, keeping the system turn, the goals and the starting molecule,
and telling the model that history was dropped rather than leaving a silent
hole. Observed prompts run about 3,200 tokens against a 24,576 budget, so this
is a guard rather than a routine path.

## Held-out evaluation (Stage 4)

```bash
python -m moltrain.evaluate --tasks data/tasks_heldout.jsonl \
  --arms unchanged expert policy
```

Training reward is not a result: it is measured at exhaustiveness 4, with a
warm cache, on tasks the policy has seen. Evaluation re-docks at
exhaustiveness 16 into a separate cache, so a reported number is never one
measured at the cheaper training setting.

Two controls run beside the policy, because "mean reward 0.7" means nothing on
its own:

| arm | what it does | why |
| --- | --- | --- |
| `unchanged` | submits its input untouched | the floor a policy must clear |
| `expert` | the scripted move library | a policy that cannot beat twelve hand-written SMARTS patterns has not learned chemistry |
| `policy` | the trained model | needs the GPU; the other two do not |

Reported per arm: mean and median reward, per-threshold pass rates and mean
deltas, invalid and unchanged submissions, tool-failure rate, malformed-call
count and seconds per episode.

On the two example pockets, with the corrected reward: `unchanged` 0.200,
`expert` 0.480.

## Docking cost

Docking dominates the reward's cost, so `reward.docking.mode` chooses how much
of it runs: `off`, `final_only`, or `every_candidate`. Measure before choosing:

```bash
python -m moltrain.benchmark --receptor ../molbench/receptors/1IEP_STI.pdbqt \
  --start "<starting SMILES>" --candidate "<analog SMILES>"
```

Measured on an M2 with 8 workers, imatinib-sized ligand, exhaustiveness 4:

| mode | first episode | warm episode | GRPO step (8 rollouts, serial) |
| --- | --- | --- | --- |
| `off` | 8.8 s | 8.8 s | ~70 s |
| `final_only` | 29.0 s | 10.4 s | ~102 s |
| `every_candidate` | 28.5 s | 11.0 s | ~106 s |

`every_candidate` costs about 4% more than `final_only`, because GRPO samples
a group of rollouts for the same task and the disk cache absorbs the repeats.
Re-measure on the training box: more CPU cores mean more docking workers.

Caches live in `cache/` keyed by canonical SMILES (plus receptor and search
settings for docking) and survive restarts. Delete them to force re-evaluation.

## Running the tests

```bash
cd /Users/virenloka/Boot/harness && molbench/.venv/bin/python -m pytest moltrain/tests -q
```

104 tests. The CrossDocked split tests build their own archive fixture, so
they need neither torch nor the multi-gigabyte download. The expert's tool
calls run for real, so the tests also catch an RDKit upgrade breaking the
move library.

Episodes run against a scripted policy, so the environment is fully exercised
with no model and no GPU. Docking paths use the engines in
`molbench/.conda-dock` when present.

## Training shape (one H200)

A 30B policy cannot be full fine-tuned on one H200 -- Adam states alone would
need roughly 360 GB -- so training is LoRA on a frozen quantized base, with
vLLM colocated in-process for rollouts (`grpo.vllm_mode: colocate`). The
reference policy for the KL term comes free by disabling the adapter.

To validate the loop end to end before spending 30B hours on it, point
`model.name` at a 1-4B model. Nothing else in the config changes.
