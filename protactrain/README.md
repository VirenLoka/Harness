# protactrain

GRPO training for tool-using PROTAC linker optimisation. The policy is handed
a PROTAC whose linker a generative model got wrong, told which fragments it
may not touch, and has to repair the linker with the molbench tools --
measured against bands calibrated from a real 1024-sample run and the 128
published PROTACs behind it.

It is the sibling of `moltrain`, which does the same for DiffSBDD pocket
tasks. The generic machinery (ATEM parsing, the completion mask, TRL
compatibility, batched generation, logging) is **imported** from moltrain and
the entire measurement layer is **imported** from protacbench. What lives here
is the task, the reward and the training shape.

```
Pro-GAT samples -> protacbench assess -> repair_tasks.jsonl   (2,229 rejects)
      -> protactrain.tasks build     -> data/tasks.jsonl      (+ optimisation class)
      -> protactrain.traces collect  -> scripted expert traces
      -> protactrain.sft    train    -> format warmup         (2x H200, DDP)
      -> protactrain.grpo   train    -> the policy            (2x H200, DDP)
      -> protactrain.evaluate        -> four arms, stratified by class
```

**Pro-GAT is strictly upstream and is never a tool.** It produced the broken
molecules; the RL policy *is* the repair step. The rollout path is pure-CPU
RDKit, so a training step runs no second model and the only GPU work in it is
decoding.

## Status

| Stage | What it is | State |
| --- | --- | --- |
| 0 | Environment: toolkit, episode, properties, reward, protocol, oracle cache | **done**, no GPU needed |
| 1 | Task builder: `repair_tasks.jsonl` -> `tasks.jsonl` with optimisation classes | **done**, no GPU needed |
| 2 | SFT warmup: scripted expert traces, completion-only loss | **done**, trainer unrun |
| 3 | GRPO: batched rollouts, masked tokens, group advantages, DDP | **done**, trainer unrun |
| 4 | Four-arm held-out evaluation | **done**, `unchanged` arm only |
| -- | This environment, the launcher, the remote setup script | **written, never run on a GPU host** |

Read the right-hand column literally. **Nothing in this package has executed
on a GPU.** Not one forward pass, not one vLLM engine, not one `torchrun`.
The Mac this was written on has no CUDA device, and every GPU path in here
refuses to start without one rather than grinding for hours and appearing to
hang -- the same refusal `moltrain.diffsbdd` and `protacbench.runner.sample`
make.

So the first thing to do on the H200 box is not `train`:

```bash
protactrain/scripts/setup_remote.sh
python -m protactrain.sft check
python -m protactrain.grpo check
```

Those three run no model pass. `check` diagnoses the installed TRL against
what this package actually passes it and names anything that would be
silently dropped -- which matters more here than in moltrain, because the DDP
and vLLM settings below are the parts most likely to differ between TRL
releases.

## Running it on the H200 box

The whole sequence, in order. Only the last four lines need a GPU.

```bash
# 0. environment, once. Creates the conda env if absent, checks both GPUs,
#    the sibling packages, the libraries and the dataset, and reports every
#    problem rather than stopping at the first one.
protactrain/scripts/setup_remote.sh

# 1. what this host can do, and what the corpus looks like. No model pass.
python -m protactrain.tasks probe
python -m protactrain.tasks classes      # the class split, before building

# 2. build the dataset from protacbench's rejects
python -m protactrain.tasks build

# 3. diagnose TRL before spending GPU hours on it
python -m protactrain.sft check
python -m protactrain.grpo check

# 4. calibrate the expert, then warm up the format. Neither needs a model.
#    `calibrate` is what sets sft.min_reward, which ships as null: moltrain's
#    0.45 was measured against its own expert's spread and does not transfer
#    to a reward with different bands, holds and protocol terms. Copy the
#    number it prints into configs/default.yaml before collecting.
python -m protactrain.experts calibrate --per-class 8
python -m protactrain.traces collect --source scripted

# 5. train                                                      (GPU)
protactrain/scripts/launch_ddp.sh sft
python -m protactrain.grpo rollout --tasks-per-step 1
protactrain/scripts/launch_ddp.sh grpo

# 6. report against the controls                                (GPU)
python -m protactrain.evaluate --arms unchanged one_shot policy policy_best_of_3
```

`grpo rollout --tasks-per-step 1` is worth the five minutes. It executes one
step's worth of real rollouts and reports cost, truncations and the
degenerate-group rate without touching the optimiser, so a group size or a
temperature that produces no gradient is found before step 500 rather than
after it.

Read three numbers off that report before going further:

| number | want | what a bad reading means |
| --- | --- | --- |
| `degenerate groups` | 0 / N | every rollout in a group scored the same, so the advantages are zero and the step teaches nothing. Raise the temperature or the group size. |
| `grounding_mean` | non-zero | a dead-flat `0.000` means the claim audit has been disconnected again -- `protocol.score_grounding` must read `claim.verdict`, not `claim.status`. Grounding carries half the protocol weight; at zero the reward is outcome-only and will not teach the model to stop stating measurements it never made. `protactrain/tests/test_protocol.py` has the regression test. |
| `false_claims` | falling | the habit this package exists to suppress. |

The launcher echoes the exact `torchrun` command it is about to run, and
`--dry-run` prints it and exits:

```bash
protactrain/scripts/launch_ddp.sh grpo --dry-run
WORLD_SIZE=1 protactrain/scripts/launch_ddp.sh grpo    # single-card smoke test
protactrain/scripts/launch_ddp.sh grpo configs/mine.yaml --verbose
```

Everything after the stage and the optional config path is passed through to
the stage's own CLI, so `--verbose`, `--quiet`, `--no-progress` and
`--log-file` work there as they do everywhere else in this repo.

### Why the environment is its own

```bash
conda env create -f protactrain/environment.yml   # or let setup_remote.sh do it
```

It cannot be merged into an existing one. DiffSBDD pins `pytorch=2.0.1
*cuda11.8*`, Pro-GAT pins `torch==2.8.0` with `torch-scatter` built against
`pt28cu128`, and REINVENT4 pins `torch==2.12.0` against the cu126 wheel index:
three mutually exclusive torch builds, none of them the one vLLM 0.29 wants.
protactrain imports none of those three, so it gets a clean environment with
loose pins instead of the exact-build linux-64 exports the other three ship
(those re-solve on no machine but the one that wrote them).

If the container already has a vLLM built for its own CUDA and torch, keep it.
`setup_remote.sh` never upgrades an existing environment, for exactly that
reason.

### Thread caps

A JupyterLab container exposes every core on the host while capping how many
threads one process may create, and OpenBLAS, OMP and MKL each size their pool
from the visible core count -- so the first BLAS-backed import dies with
`blas_thread_init: pthread_create failed for thread 63 of 64`. Importing
`protactrain` caps all five variables to `PROTACTRAIN_NUM_THREADS` (default 8)
before any submodule can import numpy, and `launch_ddp.sh` sets them again
before `torchrun` starts, because by the time the package is imported the
launcher has already loaded torch. Two ranks on one host double the demand,
which is what makes this worth doing twice.

## The dataset

`../repair_tasks.jsonl`, written by `python -m protacbench assess`: 2,229
rejected samples across 346 targets, all of stage `generated`, verdicts `poor`
(1,960) and `broken` (269). Every row already carries `fragment_smiles`, the
`linker{}` geometry and the full `properties{}` block -- which is to say the
measured baseline a relative reward needs, recorded before any repair
happened.

`tasks build` adds the one thing it does not carry: the **optimisation class**
from OPTIMISATION.md -- `reactive`, `structural`, `chemistry` or `qed_only`.
Evaluation stratifies by it, because the three real classes fail in different
ways and will not have the same yield: a reactive-group swap is one local
edit, while lowering SA on a tangled linker is a redesign. Expected yields are
60-80%, 30-50% and 10-30% respectively. A near-zero `chemistry` yield is a
finding, not a bug: those molecules need resampling, not harder optimisation.

The task ids lose their `-generated` suffix on the way in. GRPO rewrites ids
to `{id}@{generation}` and traces to `{id}#{attempt}`, and a third suffix
makes group membership ambiguous -- which silently destroys advantages rather
than failing.

## Why there is no docking

A PROTAC binds two proteins at once. Its activity is decided by the ternary
complex -- target, degrader and E3 ligase, with cooperativity between the two
protein surfaces -- and docking one flexible ligand into one rigid receptor
models none of that. Three separate reasons, any one of which is sufficient:

- **It measures the wrong thing.** A score for the warhead in its own pocket
  is a number about the part of the molecule the policy is *forbidden to
  touch*, and whose affinity is already known from the literature.
- **It is past where the search is reliable.** These molecules carry ~26
  rotatable bonds. Vina's conformational search is not trustworthy there, so
  the variance between two runs of the same molecule would swamp the signal
  between two different linkers.
- **It would dominate the cost.** In moltrain, docking is the reward's entire
  expense. Here the reward is RDKit descriptors and a substructure match:
  milliseconds, on CPU, inside the training step.

`protacbench/OPTIMISATION.md:161` argues this at length and the tool list in
`configs/default.yaml` reflects it -- no `run_docking_simulation`, no
`dock_and_score`, no `get_sbdd_metrics`. `environment.yml` therefore ships no
Open Babel, no Vina and no smina either.

## Why QED is absent from the reward

Not forgotten -- excluded, and the number says why. The 128 published PROTACs
this corpus is calibrated against have a **median QED of 0.064**, and 43 of
them (34%) fail the shipped `min_qed: 0.05` gate. Raising the bar makes it
worse: 0.10 would reject 73% of real PROTACs.

QED scores oral-small-molecule likeness out of MW, logP, HBD/HBA, PSA,
rotatable bonds and aromatic rings. On a ~940 Da bifunctional degrader it is
measuring how *unlike* a PROTAC the molecule is, so rewarding it would rank
the worst candidates highest. protacbench's own `score.py` excludes it from
the synthesis score for the same reason.

The consequence is a control set that costs nothing to build: **952 of the
2,229 tasks have `qed_too_low` as their only recorded failure**, so under this
reward they have genuinely nothing to fix. They are kept (`include_controls:
true`) because what a policy does with a molecule that needs no work is worth
measuring. An agent that edits one anyway has learned to edit, not to repair.

## The reward

### Bands, not thresholds

Every chemistry target is a band with a `low`, a `high` and a `tolerance`:

```yaml
span:        {low: 8,   high: 13,  tolerance: 4}     # out to grade.py's 17
sa_score:    {low: null, high: 4.9, tolerance: 1.1}  # out to MAX_SA 6.0
clogp:       {low: 3.5, high: 5.6, tolerance: 1.9}   # out to MAX_CLOGP 7.5
```

`low`/`high` are OPTIMISATION.md's p25-p75 bands, measured from the clean
passing samples of the real run. `tolerance` is the slack out to the hard
limit that `protacbench.grade` enforces. One-sided targets leave the other
side null.

A band rather than a direction, because "lower cLogP is better" is false
here. A PROTAC with cLogP 2.0 is as wrong as one with 8.0, and a
monotone reward would walk the policy straight out the bottom of the
distribution while every step looked like progress.

### Goal versus hold: which band a value started in decides what the term is

This is the whole design, and it is why the reward is relative to each
episode's own starting molecule rather than absolute.

| at the start | the term becomes | what it pays |
| --- | --- | --- |
| **outside** its band | a **goal** | shaped progress in [0, 1] for the distance closed |
| **inside** its band | a **hold** | nothing; charged only if the episode pushes it out |

Shaped rather than a step function, so moving span from 19 to 15 against a
band ending at 13 still produces gradient instead of scoring the same as not
moving at all.

A hold earning *nothing* is the part that is easy to get wrong. moltrain's
evaluation harness caught the alternative: constraints that paid full marks
for inaction scored an untouched caffeine submission at 0.708, more than an
episode that actually improved something. Here, with both fragments frozen,
most descriptors of most tasks barely move -- so a reward that paid for
satisfied targets would pay handsomely for doing nothing, on nearly every
task. Nothing is ever paid for leaving a satisfied target alone.

Separately, `regression` charges descriptor drift **regardless of band**, from
`grade.REGRESSION_TOLERANCE`. A molecule can sit inside every band and still
have been walked the wrong way the entire episode.

### Floors, not weighted terms

A weighted term can be outweighed. These cannot:

- either fragment missing or altered
- attachment count other than two
- a disqualifying reactive group inside the linker
- the submission is the input, unchanged

The first is the one that earns its keep. An agent asked to fix a disconnected
PROTAC has an easy out: delete a fragment. The result parses, is one molecule,
and improves SA, cLogP, TPSA, rotatable bonds and MW **all at once**. No
descriptor catches it, because the molecule is fine -- it is just not a
PROTAC. OPTIMISATION.md calls this the test that matters most, and
protacbench counts `fragments_lost` separately in its summary precisely
because it is the failure that looks like success.

Recalling the reference molecule is a **flag, not a floor**. 94 of the 1,024
original samples already recover the published answer exactly, so these
molecules are in the literature and a 30B model may well have memorised them.
That is worth knowing separately and must not count as a repair -- but it is
also not cheating in the way deleting a warhead is, so it is recorded rather
than punished to -1.

### The protocol half, and why it has to exist

`weights.grounding: 0.5`, `weights.discipline: 0.3`, `weights.verification:
0.15`.

**An outcome-only reward teaches tool use only by accident.** If a molecule
can be improved by guessing, then guessing scores exactly as well as
measuring, and GRPO will find that out -- it is a search over whatever the
reward actually rewards, not over what the reward was meant to mean. The model
that comes out the other side would be good at proposing SMILES and no better
at knowing when to call a tool, which is the opposite of the objective.

protacbench already built the instrument and already argues the point: its
`grade` scorecard splits **chemistry** (is the molecule good) from
**protocol** (did the reply say so truthfully, in the form it was asked for)
and fails the two independently. `protocol.py` turns that second half into
reward.

**Grounding** is the load-bearing term and the only one that can go negative
on its own: +1 when every number the reply states came back from a tool, -1
when none did. `grade.audit_claims` labels each asserted number `ok`, `echo`
(it matches the *input* -- quoting the task back, not lying), `false` or
`unmeasurable`. A `false` claim is a tool-using model stating a measurement it
never made, and an outcome reward is completely blind to it: the molecule is
unaffected by what was said about it. At 0.5 the term can overturn a mediocre
chemistry result, which is the intent.

**Discipline** scores measure-first, change-one-thing, re-measure as three
separate checks over the call log rather than one bundled pass/fail, so the
gradient can tell "never measured" from "measured once at the start and then
edited blind".

**Verification** is the cheapest of the three: did the episode call
`check_protac_linker` -- the tool that decides acceptance -- before submitting.

And because all three ask for *more* tool calls, the per-call price has to be
right. moltrain measured what happens when it is not: a flat 0.02 per call
made its expert's sixteen honest calls cost 0.32, erasing the 0.116 of
progress they bought, and the reward's optimum became submitting blind. Here
the first `turn_penalty_free_calls: 10` are free and the rest cost 0.005.
Measuring is the task, not waste.

## The training shape: two H200s, two engines

`grpo.ddp.world_size: 2`, `launcher: torchrun`, `use_trl_vllm: false`,
`engine_per_rank: true`.

### The four-engine trap

moltrain sets TRL's `use_vllm: True` *and* starts its own `vllm.LLM` inside
`rollout_func` (`moltrain/grpo.py:144`, `:206`). One process therefore holds
two engines. On one card with one process that is merely wasteful. **Under DDP
it is four engines on two GPUs, and neither card fits them.**

TRL's `rollout_func` is the documented place to do your own generation, so the
moment one is supplied TRL's internal engine is dead weight. Turning it off
leaves exactly one engine per rank, which is the shape that fits:

```yaml
grpo:
  use_trl_vllm: false              # rollout_func does its own generation
  vllm_gpu_memory_utilization: 0.35
  engine_per_rank: true            # one engine per process, rank-gated
```

`engine_per_rank` also gates the asyncio loop. The episode loop is async and
moltrain calls `asyncio.run()` inside the training step; that is not rank-safe
as written, so both the engine and the loop are created per process and keyed
on the local rank.

### Why LoRA

A 30B policy cannot be full fine-tuned on an H200 -- Adam states alone are
roughly 360 GB -- so training is LoRA (r=32, alpha=64) on a frozen quantized
base, with vLLM colocated in-process for rollouts. The reference policy for
the KL term comes free by disabling the adapter.

To validate the loop end to end before spending 30B hours on it, point
`model.name` at a 1-4B model. Nothing else in the config changes.

### Degenerate groups

GRPO's baseline is the group mean, so a group whose eight rollouts all score
the same yields zero advantage and teaches nothing. The rollout report counts
them (`moltrain.grpo_core.degenerate_groups`). Watch this number on the
`qed_only` control tasks in particular: a task with genuinely nothing to fix
is the most likely thing in this corpus to produce eight identical rollouts,
and a corpus half made of those would quietly halve the gradient.

### The mask

A rollout is one token sequence alternating between what the model wrote and
what the environment appended, and only the model's tokens may carry gradient.
`moltrain.grpo_core.build_completion` derives the mask by exact string
arithmetic over the recorded turns. On a measured moltrain expert episode,
**42% of completion tokens are environment-authored**; without the mask that
is 42% of the training signal teaching the model to write tool output. `grpo
check` greps the installed `GRPOTrainer` for `completion_mask` and fails if it
is absent, because a TRL that ignores the column would train on those tokens
silently.

## SFT warmup

GRPO cannot learn from a rollout the ATEM parser rejects -- it scores badly
for the wrong reason. A short warmup on traces known to parse and known to
have scored well gets the format out of the way so GRPO's gradient goes into
decisions rather than syntax.

Each record is one assistant turn: the transcript before it as `prompt`, the
turn itself as `completion`. Every tool result therefore lands in some later
record's prompt and never in a completion, so TRL's completion-only loss masks
them with no custom collator. This is not tidiness. Training on the whole
transcript teaches the model to *write* tool results, which is precisely how a
tool-using model learns to state measurements it never made -- the habit the
grounding term exists to punish. `sft.py` refuses to train if any completion
contains a tool output.

The expert teaches protocol, not chemistry: measure, change one thing, verify,
submit. GRPO is what is supposed to find the chemistry.

## Evaluation

OPTIMISATION.md's four arms, stratified by optimisation class, at least 50
molecules per class:

| arm | what it is | what it tells you |
| --- | --- | --- |
| `unchanged` | returns the input untouched | the floor. No GPU needed, and it is a real floor here -- with the fragments frozen, most descriptors barely move |
| `one_shot` | the prompt, no tools, one reply | how much comes from chemistry knowledge alone |
| `policy` | full prompt, tools, iterate | the real number |
| `policy_best_of_3` | best of three independent runs | the ceiling, and how much of it is sampling luck |

How to read them, from OPTIMISATION.md: `policy` close to `unchanged` means
the prompt is not working -- check whether the agent is measuring at all
before blaming the model. `policy_best_of_3` far above `policy` means the
model can solve these but unreliably, so sample several times rather than
reworking the prompt. A high pass rate with a high fragments-lost rate means
the agent is cheating and the floors are earning their keep.

Evaluation uses its own property cache (`separate_cache: true`), so a reported
number is never one that came out of the training cache.

## Layout

| Path | What it is |
| --- | --- |
| `config.py`, `configs/default.yaml` | One YAML file drives every stage |
| `properties.py` | `measure(smiles, fragment_smiles) -> LinkerProperties`; never raises |
| `oracle.py` | The same, cached on disk and keyed on the fragment pair |
| `rewards.py` | Bands, goals versus holds, regressions, the floors |
| `protocol.py` | Grounding, measurement discipline, verification |
| `toolkit.py` | The trained tool surface; wraps molbench, never raises |
| `episode.py` | The multi-turn loop, policy-agnostic |
| `tasks.py` | `repair_tasks.jsonl` -> `tasks.jsonl`, plus the class split |
| `traces.py` | Collects SFT records, one per assistant turn |
| `sft.py`, `grpo.py` | Stages 2 and 3, each with a `check` mode |
| `evaluate.py` | The four arms, stratified |
| `environment.yml` | The training environment; no docking stack |
| `scripts/launch_ddp.sh` | torchrun across the two H200s; `--dry-run` prints the command |
| `scripts/setup_remote.sh` | First-run setup and host report; runs no model pass |

Generic machinery is imported, not copied: `moltrain.atem`,
`moltrain.grpo_core`, `moltrain.trl_compat`, `moltrain.policy`,
`moltrain.hostcheck`, `moltrain.logs`, `moltrain.progress`. Measurement is
imported from `protacbench.grade`, `.structures`, `.score`, `.repair` and
`.assess`. Tools are imported out of `molbench/server.py` as plain Python
functions -- no MCP, no Node -- so the surface the policy trains on is the one
it is deployed against.

The same split moltrain keeps: library modules **log**, CLIs **print**. Every
command takes `--verbose`, `--quiet`, `--no-progress` and `--log-file`; the
console shows INFO and the run's log file always records DEBUG, so an
overnight job leaves a full trace even though the console stayed readable.
