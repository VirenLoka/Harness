# Repairing ADC linkers with an LLM agent

Everything here is measured. Where a number appears it came from one of two
sets, and the text says which:

- **1,692 reference reagents** — the curated Linker-GPT catalogue,
  deduplicated from 3,679 rows to 2,764 unique structures, then filtered to
  those carrying no Fmoc, Boc, tBu, NHS or p-nitrophenyl group. Those are
  activating or protecting groups that are gone from the installed linker, so
  measuring across them describes a bottle on a shelf.
- **16 approved ADCs** — the drug-linker of every FDA- or EMA-approved
  antibody-drug conjugate that could be structurally confirmed, in
  `tests/data/approved_adcs.json`. Ten of the sixteen were assembled by two
  independent searches and agreed on canonical SMILES in every case.

The second set is the one that matters. Any threshold can be argued for; only
a threshold that accepts all sixteen marketed drugs has been shown not to be
nonsense.

## The thing that was wrong, and why it is worth reading

The scorer measured the whole conjugate against windows built from
payload-free reagents. MMAE alone is MW 718 with cLogP 4, so every candidate
landed outside every window whatever its spacer. Four terms — polarity,
polar surface, size, synthesisability, 60% of the weight — returned the same
value for everything.

What that cost, on the same four spacers:

| spacer | before | after |
|---|---|---|
| PEG2, the real MC-PEG2-Val-Cit-PAB-MMAE spacer | 43.2 | **88.0** |
| brentuximab vedotin as marketed | 54.8 | **87.3** |
| PEG8 | 37.4 | 85.2 |
| C12 greasy alkyl | 42.9 | **57.5** |

The scorer preferred a greasy chain to PEG. Adding PEG is the standard fix
for the failure mode the scorer's own docstring named. It was not a weighting
mistake; it was measuring the wrong molecule.

The lesson generalises: **the payload is a constant across every candidate for
one design.** Leaving it in adds the same offset to every bulk descriptor
while destroying the calibration. Gates run on the conjugate, scores run on
the linker.

## What is actually a threshold

**One hard ceiling.**

```
linker cLogP ≤ 3.14
```

Two independent derivations agree: it is the 90th percentile of the 1,692
reagents, and it sits just above 2.87, the most lipophilic linker among the
16 approved ADCs (Mylotarg and Besponsa's hydrazone-disulfide). It rejects
none of them.

It is one-sided, and that is evidence rather than preference. Hydrophobicity
is monotone for an ADC linker — the linker is discarded on release and never
crosses a membrane — so there is no point at which being more polar starts to
hurt. Enhertu's linker measures **−1.71**, below the reference band entirely,
and carries eight payloads per antibody because of it.

**Four advisory bands**, the 10th–90th percentile of the reagent set. Worth
noticing, never a failure. Each is outside for 2 to 3 of the 16 approved
ADCs, which is the tolerable rate for something that only prints a note.

| property | band | median |
|---|---|---|
| TPSA | 66 – 285 | 139 |
| heavy atoms | 16 – 85 | 36 |
| molecular mass | 242 – 1237 | 526 |
| rotatable bonds | 9 – 73 | 25 |

**Two measured and never judged.** Their bands reject most of the clinical
set, so they are reported as raw numbers and compared against nothing:

| property | band | rejects |
|---|---|---|
| Fsp3 | 0.50 – 1.00 | **10 of 16** — Adcetris, Enhertu, Polivy, Padcev, Tivdak |
| HBD | 0 – 4 | **9 of 16** |

The reason is that the reference set is not the same population as a finished
drug-linker. It is full of bare PEG and alkyl reagents, which is why its
median Fsp3 is 0.88. An installed cleavable linker carries a p-aminobenzyl
aromatic ring and a dipeptide, which puts a real one near 0.48. A band
measured on fragments flags the standard of care as abnormal.

## The gates, which are not tradeable

A candidate failing one of these is not a worse candidate. It is not a
candidate, and averaging it into a weighted score would let a good cLogP hide
it.

| gate | what it catches |
|---|---|
| `payload_lost` | the drug is gone; this is a different compound |
| `payload_altered` | a stereocentre inverted. MMAE has ten, and potency depends on all of them |
| `payload_internal` | linker on both sides of the payload, so cutting it cannot release anything |
| `trigger_lost` | nothing gets cut, so the payload is never released |
| `no_release_chemistry` | a trigger with nothing between it and the payload that frees the drug |
| `handle_lost` | cannot be attached to an antibody |
| `handle_duplicated` | a second handle crosslinks two antibodies into an aggregate |
| `bad_junction` | chemistry nobody would make, usually a ketone where an amide was intended |
| `reactive_group` | reactive chemistry in the spacer, outside the handle |

Five candidate gates were tried and **rejected a marketed drug**. Each was
wrong, and each failed in a way that looked completely reasonable from inside
the code:

- A trigger search over the whole molecule read **DM1's own ansa-ring
  carbamate** as a cleavage trigger. Now scoped to exclude the payload — but
  still allowed to span the junction, because in a disulfide-linked ADC the
  disulfide *is* the junction and Elahere's DM4 contributes one of the two
  sulfurs.
- A reactive-group check exempting only the conjugation handle blamed **DM1's
  epoxide** and **calicheamicin's enediyne** on the linker, rejecting
  Kadcyla, Elahere, Mylotarg and Besponsa. The payload is exempt too: its
  chemistry is given, not designed.
- "An amide never releases" rejected **Decnupaz**, whose Ala-Ala dipeptide is
  bonded straight to its payload and is cut there. A peptide trigger that
  reaches the junction releases by being cut itself.
- Two-sided cLogP and TPSA windows marked down **Enhertu** (−1.71) and
  **Trodelvy** (TPSA 340) for the polarity that makes them work.

## Release is two things, and this is the failure nothing else sees

A **trigger** is what gets cut: Val-Cit, Val-Ala, Ala-Ala, GGFG, a hindered
disulfide, a glucuronide, a carbonate. A **release chemistry** is what then
lets the payload leave as itself, usually a p-aminobenzyl carbamate that
undergoes 1,6-elimination once the peptide in front of it is gone.

A cassette bonded straight to the payload by a plain amide has a trigger and
no spacer. It is cut, and the payload leaves still wearing an amino acid. It
looks perfect in every descriptor, and `check_payload_release` is the only
thing here that sees it.

`releases_free_drug: false` is not automatically a bug. Trastuzumab emtansine
is approved and releases Lys-MCC-DM1 rather than DM1; belantamab mafodotin
releases Cys-mc-MMAF. Both are deliberate, and both work only because those
payloads stay potent as that adduct. The set of payloads that do is small, so
this should be a decision somebody made rather than something discovered in
vivo.

What the tool identifies across the 16 approved ADCs, all correctly:

| release chemistry | drugs |
|---|---|
| `pab_carbamate` | Adcetris, Polivy, Padcev, Tivdak, Aidixi, Zynlonta, Emrelis |
| `aminomethylene` | Enhertu, Datroway |
| `disulfide` | Elahere, Mylotarg, Besponsa |
| `ester` (CL2A carbonate) | Trodelvy |
| direct peptide at the junction | Decnupaz |
| none — adduct only | Kadcyla, Blenrep |

## The junction trap

maleimidocaproyl ends in an acyl group meant to become an amide against the
cassette's valine. Hand it a spacer that starts with a **carbon** and it
becomes a dialkyl ketone instead — a reactive carbonyl mid-linker that
condenses with lysine in vivo, and not a compound anyone would make.

```
*CCOCCOCC*     ->  ...CH2CH2-C(=O)-CH2CH2-...     a ketone        WRONG
*NCCOCCOCC*    ->  ...CH2CH2-C(=O)-NH-CH2...      an amide        right
```

RDKit sanitises the first without complaint and every descriptor looks fine.
The guard that would have caught it only ever ran when the two warheads were
built, never on the assembled molecule, so every carbon-starting spacer the
generator produced passed silently.

## Running the repair loop

```bash
python -m adclinker generate          # sample, score, rank -> candidates.csv
python -m adclinker prompts           # -> runs/prompts/
```

`prompts` writes a system prompt, one task file per candidate, a JSONL for
batch driving, and a manifest. Repair classes, in the order they are checked:

| class | meaning |
|---|---|
| `payload` | the drug is gone, epimerised, or buried mid-chain |
| `release` | no trigger, or a trigger with nothing that frees the drug |
| `handle` | zero or two conjugation handles |
| `structural` | disconnected, unparseable, or reactive chemistry in the spacer |
| `chemistry` | any other gate |
| `greasy` | passes every gate but breaches the cLogP ceiling |
| `clean` | nothing to repair — excluded by default |

`clean` is excluded deliberately. Handing an agent a molecule with nothing
stated to fix invites it to change one that was already fine.

Withheld from every prompt: `score_total`, `score_reason`, `rank`,
`reinvent_smiles`, `nll`. The score columns go because a composite is gameable
in a way raw descriptors are not. `reinvent_smiles` goes because it is the
generator's own assembly, built from flattened warheads — every stereocentre
in the payload is unspecified in it, and an agent that copies it back in
fails the one gate that matters.

## What not to ask for

**Do not ask for drug-likeness.** A drug-linker is 1,000–1,500 Da and the
whole conjugate is an antibody. It is never swallowed. Optimising QED or
pushing mass toward oral ranges makes it worse, and the same mistake in the
PROTAC pipeline produced repairs whose notes read *"shortened linker to
improve QED"* — against an explicit instruction not to, because a stale
per-task prompt contradicted the standing one.

**Do not dock anything.** Docking scores one small molecule into one rigid
pocket. The payload is held by an antibody and released inside a cell;
nothing about this molecule's behaviour is a binding pose, and a drug-linker
has around 25 rotatable bonds, far past where that search stays reliable.

**Do not let it design a cassette.** The protease appears in no model here, so
a generated substrate cannot be checked against anything. Swap for a known
cassette or keep the one you have.

**Do not let it touch the handle.** Which chemistry it is follows from the
antibody's conjugation site. Reactivity is not something to reason about from
a structure.

## Where the scale sits

The 16 approved ADCs score **60.6 to 95.7**, with fourteen between 81.7 and
95.7. A greasy generated spacer scores **57.5**, below all of them. Mylotarg
and Besponsa are the two low ones at 60.6, on a linker cLogP of 2.87 — the
oldest chemistry in the set, and Mylotarg was withdrawn and reintroduced at a
lower dose.

Treat anything below about 60 as not worth making, and do not read much into
differences of a few points above 85.
