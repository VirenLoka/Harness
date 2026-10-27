# Driving an LLM to optimise PROTAC linkers

Everything here is calibrated against one real 1024-sample run and the 128
real, published PROTACs that run was built from. No threshold below is a
number anyone chose; each one is measured.

## Before anything else: turn off the QED gate

The 128 reference molecules in `reference_smiles` are real PROTACs from the
literature. Scored against the shipped threshold:

| | reference PROTACs (n=128) |
| --- | --- |
| MW median | 940 |
| cLogP median | 5.0 |
| QED median | **0.064** |
| **fail `min_qed: 0.05`** | **43 of 128 — 34%** |

A gate that rejects a third of the ground truth is not measuring quality. QED
scores oral-small-molecule likeness from MW, logP, HBD/HBA, PSA, rotatable
bonds and aromatic rings. A PROTAC is ~940 Da and bifunctional by
construction, so QED is measuring *how unlike an oral drug it is* — which is
the wrong question. Raising the bar makes it worse: 0.10 would reject 73% of
real PROTACs, 0.15 would reject 88%.

```yaml
thresholds:
  min_qed: 0.0      # report it, never gate on it
```

That alone moves **206 samples** out of `poor` with no chemistry done. Asking
an LLM to "improve drug-likeness" on them would make them *worse* PROTACs.

## What to send, and what to leave alone

After the reactive-group fix, the 1024 samples fall into five classes:

| Class | n | Send to the LLM? |
| --- | --- | --- |
| `structural` — disconnected, bad valence, non-bridging | 81 | **Yes.** 69 still have a salvaged linker and both fragments. |
| `reactive` — aldehyde, imine, Michael acceptor in the linker | 267 | **Yes.** The highest-value class: one functional-group swap each. |
| `chemistry` — bad ring, PAINS, SA, cLogP, flexibility | 143 | **Yes**, hardest and lowest expected yield. |
| `qed_only` — QED is the sole complaint | 206 | **No.** Fix the threshold, not the molecule. |
| `clean` — passes everything | 327 | **No.** Nothing to repair. Rank and pick. |

**491 molecules** are worth an optimisation run. The other 533 are either
already fine or fine once the gate is corrected.

## Thresholds the prompt should carry

Measured from the clean passing samples of the real run, with the reference
PROTACs as the sanity check.

| Property | Target | Hard limit | Source |
| --- | --- | --- | --- |
| Linker span | 8–13 bonds | 7–17 | p25–p75 of clean samples |
| Linker heavy atoms | 8–14 | 5–25 | p25–p75 of clean samples |
| SA score | ≤ 4.9 | ≤ 6.0 | p75 of clean samples |
| cLogP | 3.5–5.6 | ≤ 9.0 | p25–p75; ARV-471 is ~6.8 |
| TPSA | 170–217 | ≥ 120 | p25–p75 of clean samples |
| Rotatable bonds | 13–18 | ≤ 30 | p25–p75 of clean samples |
| MW | 780–940 | 400–1600 | p25–p75; reference median 940 |
| QED | — | **no gate** | 34% of real PROTACs fail 0.05 |
| Reactive groups in linker | none | none | hard rejection |

**Target** is what to aim for; **hard limit** is what makes a molecule
unusable. The prompt should quote both, because an agent given only a hard
limit optimises to the edge of it.

## The system prompt

Give this once, as the agent's standing instruction.

```
You are a medicinal chemist optimising the linker of a PROTAC.

A PROTAC has three parts: a warhead that binds a target protein, an
E3-ubiquitin-ligase ligand, and a linker joining them. It works by holding
both proteins together in a ternary complex, so the linker's length and
composition decide whether it functions at all.

THE TWO FRAGMENTS ARE NOT YOURS TO CHANGE. The warhead and the E3 ligand are
the reason the molecule exists; their structures came from medicinal
chemistry programmes, not from a generator. You may only edit the linker
between them. A molecule that scores better because you deleted or altered a
fragment is a different compound and will be rejected automatically.

You are NOT making this molecule drug-like in the oral small-molecule sense.
PROTACs are ~900 Da and violate Lipinski by construction. Do not optimise
QED. Do not reduce molecular weight toward typical drug ranges. Doing so
makes the molecule worse at its job.

WHAT A GOOD LINKER LOOKS LIKE, measured from 128 published PROTACs and from
the passing samples of this run:
  - spans 8 to 13 bonds between the two fragments (never below 7 or above 17)
  - 8 to 14 heavy atoms
  - built from chemistry a lab runs on Monday: amide, PEG ether, short alkyl,
    piperazine, piperidine, triazole, sulfonamide, carbamate
  - SA score at or below 4.9
  - cLogP between 3.5 and 5.6
  - TPSA at or above 170
  - contains NO reactive or unstable group: no aldehyde, ketone in-chain,
    imine, Michael acceptor, epoxide, aziridine, acyl halide, anhydride,
    vinyl ether, enol, allene, peroxide or nitroso. These react with protein
    lysines and cysteines in vivo or do not survive as drawn.

TOOLS. You have:
  mcp__molbench__validate_smiles(smiles)
  mcp__molbench__get_molecule_metrics(smiles)
  mcp__molbench__apply_smarts_reaction(smiles, reaction_smarts)
  mcp__molbench__generate_3d_conformer(smiles)

MEASUREMENT DISCIPLINE. Never state a number you did not get from a tool. If
you believe a change lowered cLogP, call get_molecule_metrics and read it.
Make ONE change at a time and re-measure after each. An unmeasured claim is
treated as false.

HOW TO WORK:
  1. validate_smiles on the input, to see how it parses.
  2. get_molecule_metrics, to confirm the starting values.
  3. Identify the smallest edit to the linker that addresses the stated
     problem. Prefer replacing a group over rebuilding the linker.
  4. Apply it, re-measure, check it did not break something else.
  5. Repeat until the problem is fixed or you have tried three distinct
     approaches.
  6. validate_smiles on your final answer before reporting it.

FINISH your reply with exactly one line, and nothing after it:

REPAIRED: <the full repaired PROTAC SMILES>

If you cannot repair it without changing a fragment, finish with:

REPAIRED: none

and say above the line what you tried and why it failed. A truthful failure
is worth more than a molecule that scores well by cheating.
```

## The per-task prompt

One per molecule. Everything in `{braces}` comes from the CSV row.

```
Repair the linker of PROTAC {target_id}, sample {sample_id}.

WHAT IS WRONG
{for each failure_code: "- {code}: {description}. {remedy}"}
{failure_detail}

THE MOLECULE
  Full PROTAC: {protac_smiles}
  Linker only: {linker_smiles}
  Fragment 1:  {fragment_smiles[0]}    <- must survive unchanged
  Fragment 2:  {fragment_smiles[1]}    <- must survive unchanged

AS MEASURED
  MW {prop_mw}            SA {prop_sa_score}       cLogP {prop_clogp}
  TPSA {prop_tpsa}        rotatable {prop_rotatable_bonds}
  linker: {linker_heavy_atoms} heavy atoms spanning
          {linker_path_length_bonds} bonds between the fragments
{if reactive_groups: "  reactive groups in the linker: {reactive_groups}"}

WHAT TO AIM FOR
  linker span 8-13 bonds (currently {linker_path_length_bonds})
  SA at or below 4.9 (currently {prop_sa_score})
  cLogP 3.5 to 5.6 (currently {prop_clogp})
  no reactive groups

The linker must keep roughly its current length: {linker_size} heavy atoms.
That length comes from the geometry of the ternary complex, not from
preference. Shortening it to improve a descriptor breaks the molecule.
```

### Per-failure remedies

Select by failure code and paste into the WHAT IS WRONG block. These are
specific on purpose — generic advice produces generic edits.

| Code | Remedy to give |
| --- | --- |
| `reactive_linker_group` | Replace the reactive group with a stable equivalent: aldehyde or in-chain ketone → amide or primary alcohol; Michael acceptor → the saturated chain; imine → the corresponding amine; epoxide → diol or ether. Do not delete the atoms — keep the linker's length. |
| `disconnected` | Join the pieces into one molecule. The linker must bond to both fragments, one bond at each end. |
| `sanitization_failed` | Fix the valence or aromaticity violation. Use validate_smiles after each edit. |
| `linker_not_bridging` | Rebuild so the linker has exactly two attachment points, one per fragment. Touching one fragment twice makes it a substituent, not a linker. |
| `bad_linker_ring` | Replace the strained or oversized ring with an open chain or a five- or six-membered ring. |
| `pains_alert` | Remove the interference substructure from the linker. Do not alter either fragment, even if the alert matches there. |
| `sa_too_high` | Simplify: prefer amide, ether and short alkyl over unusual rings and dense substitution. |
| `clogp_too_high` | Make the linker more polar. Ether oxygens or an amide in place of alkyl is the usual move. |
| `too_flexible` | Replace part of a flexible chain with a para-substituted ring or a short rigid spacer. |
| `macrocycle` | Break the macrocycle. The linker should connect the fragments without closing a ring through them. |

## Acceptance tests

Run every returned molecule through all of these. The first four are
rejections, not deductions — a molecule failing one is not a worse
candidate, it is not a candidate.

**1. It parses, and it is one molecule.** `MolFromSmiles` returns non-None
and `GetMolFrags` returns one fragment.

**2. Both fragments are present as substructures.** Match each
`fragment_smiles` back into the result; both must match, and the two matches
must not overlap. *This is the test that matters most.* An agent asked to fix
a disconnected PROTAC can always delete a fragment — the result parses,
sanitises, scores beautifully and is not a PROTAC. No descriptor catches it.

**3. The linker attaches at exactly two atoms**, one per fragment.

**4. No disqualifying reactive group inside the linker.** Checking the whole
molecule is wrong: a warhead may legitimately contain chemistry a linker may
not.

**5. Linker span is 7–17 bonds.** Below 7 the two warheads cannot occupy
their binding sites at once, whatever the descriptors say.

**6. SA has not worsened by more than 0.5**, and is at or below 6.0.

**7. cLogP ≤ 9.0 and TPSA ≥ 120.**

**8. It is not the input unchanged.** Canonicalise both and compare.

**9. It is not the reference molecule.** If the result equals
`reference_smiles`, the model recalled the published answer rather than
solving the problem. That is worth knowing separately and must not count as a
repair.

Test 9 matters more than it looks: 94 of the 1024 generated samples already
recover the reference exactly, so these molecules are in the literature and a
large model may well have memorised them.

## Measuring how far this can actually go

A single optimisation run tells you a number with nothing to compare it to.
Four arms, same molecules, same acceptance tests:

| Arm | What it is | What it tells you |
| --- | --- | --- |
| **0. Unchanged** | return the input untouched | the floor. Any arm not beating this is doing nothing. |
| **1. One-shot** | prompt, no tools, one reply | how much comes from chemistry knowledge alone |
| **2. Tool-using** | full prompt, tools, iterate | the real number |
| **3. Tool-using, three attempts** | best of three independent runs | the ceiling, and how much is sampling luck |

Stratify by class — `structural`, `reactive` and `chemistry` fail in
different ways and will not have the same yield. Use at least 50 molecules
per class per arm, drawn randomly, so a difference of a few percent means
something.

### What to record per attempt

- which acceptance tests passed, individually — not just pass/fail
- the score before and after
- **fragments-lost rate**, separately and prominently. It is the failure that
  looks like success.
- reactive-group-introduced rate: did the repair add a new one?
- tool calls made, and how many were measurements rather than edits. An agent
  that edits without measuring is guessing.
- wall-clock and token cost per molecule

### How to read it

- **Arm 2 ≈ Arm 0** → the prompt is not working. Look at whether the agent is
  measuring at all before blaming the model.
- **Arm 3 ≫ Arm 2** → the model can solve these but does so unreliably.
  Sample several times and keep the best rather than reworking the prompt.
- **High pass rate, high fragments-lost** → the agent is cheating and the
  acceptance tests are earning their keep. Strengthen the constraint wording.
- **`reactive` class salvages well, `chemistry` does not** → expected. A
  functional-group swap is a local edit; lowering SA on a tangled linker is a
  redesign.

### Expected yields

Stated so a result can disappoint you. From the failure distribution:

- `reactive` (267): **high**, perhaps 60–80%. Each needs one group swap, the
  linker length is already right, and the edit is local.
- `structural` (81): **moderate**, perhaps 30–50%. 69 have a salvaged linker
  and both fragments, so the agent is told exactly what to reattach.
- `chemistry` (143): **low**, perhaps 10–30%. SA and cLogP are whole-molecule
  properties largely set by the fragments, and the linker is a small part of
  the molecule. Some of these are not fixable by linker edits at all.

If `chemistry` comes back near zero, that is a finding, not a bug. It means
those molecules need a different linker rather than a repaired one, and the
right response is to resample, not to optimise harder.

## What not to ask for

- **Do not ask for improved QED.** It is the wrong objective and 34% of real
  PROTACs fail the current gate.
- **Do not ask for reduced molecular weight.** ~900 Da is correct here.
- **Do not ask the agent to pick between candidates.** Score them yourself;
  the ranking is cheap, reproducible and not a judgement call.
- **Do not let a repaired molecule into the candidate set without test 2.**
  Everything else is a quality gradient; that one is an identity check.
