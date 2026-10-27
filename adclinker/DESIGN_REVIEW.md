# ADC linker design: does the proposed pipeline make mechanistic sense?

A review of the Stage 2 pipeline before any of it gets built, because two of
its five steps describe something other than ADC conjugation, and building
them as written would produce molecules that cannot work as ADCs for reasons
no amount of computational validation would surface.

The short version: **steps 1, 4 and 5 are sound. Steps 2 and 3 are not ADC
conjugation as literally written, but there are two real technologies they
closely resemble, and under either reading the pipeline becomes defensible.**

---

## 1. What the proposed pipeline says

1. Choose a target antigen, an effective antibody, and a payload
2. Choose a particular **antibody pocket for the payload conjugation**
3. Generate a small **anchor molecule for that pocket** with DiffSBDD
4. Generate a **linker between the payload and the anchor** with DiffLinker,
   "that can bind to the target"
5. Sanity checks, plus computational validation routes

## 2. How ADC conjugation actually works

Every approved ADC attaches its linker-payload to the antibody through
**covalent bond formation to a specific amino acid or glycan**, not through a
small molecule occupying a pocket:

| Chemistry | Site | Example |
| --- | --- | --- |
| NHS ester / SMCC | surface lysines (stochastic, ~80 available on IgG1) | trastuzumab emtansine |
| Maleimide | cysteines from reduced interchain disulfides | brentuximab vedotin, trastuzumab deruxtecan |
| Maleimide | engineered cysteine (THIOMAB-style) | site-specific, DAR 2 |
| Transglutaminase | Q295 after deglycosylation | site-specific |
| Glycan remodelling | N297 glycan, trimmed then clicked | site-specific |
| Sortase / formylglycine tag | engineered C-terminal or internal motif | site-specific |

The bond is permanent on the timescale that matters. An ADC circulates for
days to weeks, and the linker-payload has to stay attached the whole time;
premature release is the main driver of off-target toxicity.

### Why "a payload sitting in an antibody pocket" cannot be the attachment

A non-covalent anchor-in-pocket complex fails on four counts at once:

- **Dissociation.** To survive a week in circulation at the low nanomolar
  concentrations an ADC reaches, you would need a sub-picomolar Kd with an
  extremely slow off-rate. Small molecules essentially never achieve this
  against an antibody surface.
- **Dilution.** Unlike an intracellular target, there is no reservoir keeping
  the anchor bound. Every dissociation event is permanent loss, and the freed
  payload is a systemically circulating cytotoxin.
- **Uncontrolled DAR.** Drug-to-antibody ratio would be an equilibrium
  property varying with concentration, not a manufacturable specification.
- **Antibody surfaces are poor small-molecule targets.** They are flat and
  polar by evolutionary design. This is precisely why the best-known
  Fc-binding reagents are peptides rather than small molecules.

So step 2 + step 3, read literally, describe a non-covalent drug carrier, not
an ADC.

## 3. Two readings that make steps 2-3 real

### Reading A: the anchor is a *directing group*, not the attachment

There is an established technology shaped almost exactly like the proposal:
an **Fc-affinity reagent that binds the antibody non-covalently and thereby
positions a reactive warhead next to one specific lysine**, which it then
acylates. The non-covalent recognition event chooses the site; a covalent bond
is what persists. Ajinomoto's AJICAP chemistry works this way, using an
Fc-binding peptide.

Under this reading the pipeline becomes coherent and genuinely novel:

- step 2 picks an **Fc surface patch** adjacent to a target lysine
- step 3 designs an **anchor that binds that patch** — exactly a
  pocket-conditioned generation problem, which is what DiffSBDD does
- the anchor carries a reactive group; after acyl transfer the anchor is
  *released and washed away*, leaving payload-linker on a defined lysine

This is the reading I would recommend. It keeps the generative models doing
what they are good at while the final product is still a covalent,
manufacturable ADC. It also has a clean computational success criterion:
anchor affinity for the Fc patch, and the distance from the anchor's warhead
to the target lysine's ε-amine.

### Reading B: the Fab nucleotide-binding site

IgG molecules have a **conserved nucleotide-binding site** in the variable
region, between the heavy and light chain domains, which binds small
indole-containing molecules. It has been exploited for site-specific
conjugation via UV crosslinking. This is a genuine antibody pocket that a
genuine small molecule genuinely binds, so DiffSBDD has something real to
work with.

The caveat is serious: that site is **close to the CDRs**. Hanging a payload
off it risks occluding the paratope and killing antigen binding — the one
property the antibody exists for. Published work indicates the site is
distinct enough from the paratope to be usable, but for any new anchor this
has to be checked explicitly (see §6).

### What does not work

If the intent really is a non-covalent payload carrier with no covalent step
anywhere, the construct is not an ADC and the failure modes in §2 apply. I
would want to say so before building it rather than after.

## 4. Step 4 needs restating

> "generating a linker between the payload and the anchor molecule **that can
> bind to the target**"

Two problems:

**The linker must not bind anything.** Target binding is the antibody's job.
A linker that binds the target antigen, or anything else in plasma, is a
liability: it creates an off-antibody pharmacology, promotes aggregation, and
competes with the antibody's own binding. The design goal for a linker is the
opposite of affinity — it should be **inert in circulation and invisible**.

**What the linker must actually do** is a different list, and none of it is
expressible as a 3D shape-fitting objective:

1. **Stay attached in plasma.** Thiosuccinimides formed from maleimides can
   undergo retro-Michael exchange onto serum albumin, and the rate depends
   strongly on the local charge environment of the conjugation site. This is
   a chemistry and site choice, not a geometry choice.
2. **Release the payload in the right compartment, or not at all.** Cleavable
   linkers carry a specific recognition motif — a dipeptide such as
   valine-citrulline or a tetrapeptide such as Gly-Gly-Phe-Gly for lysosomal
   cathepsins, a pH-labile carbonate or hydrazone, or a
   glutathione-reducible disulfide. Non-cleavable linkers rely on complete
   proteolysis of the antibody in the lysosome, which only works for payloads
   that remain active as an amino-acid adduct.
   **A 3D generative model has no representation of enzymatic cleavability.**
   A generated linker will not contain a cleavable motif unless it is
   constrained to.
3. **Keep the payload's pharmacophore intact.** Most payloads tolerate one
   attachment point. Novel attachment chemistry usually destroys potency, and
   this is not something a geometry model knows.
4. **Not be greasy.** Linker-payload hydrophobicity drives aggregation and
   accelerated clearance; hydrophobic-interaction retention tracks with
   clearance rate, which is why high-DAR constructs use polyethylene glycol
   or other polar spacers. A model optimising 3D fit will tend to produce
   exactly the wrong thing here, because burying hydrophobic surface looks
   favourable to it.

So DiffLinker is a reasonable tool for the *connectivity and 3D feasibility*
part of step 4 — joining payload and anchor is a legitimate two-fragment
linking problem — but it must be wrapped in constraints, not trusted on its
own. Concretely: generate with DiffLinker, then **require** a cleavable motif
by construction (splice a Val-Cit-PAB or GGFG cassette in and let DiffLinker
design only the spacer segments), and filter hard on polarity.

One more trap: DiffLinker's pocket-conditioned checkpoints design linkers that
**fit a pocket**. Conditioning linker generation on the antibody pocket would
teach it to lay the linker along the antibody surface, which is close to the
opposite of what you want — a linker should project the payload away from the
antibody in a shielded, compact conformation, then be reachable by a protease
once internalised. Use the unconditioned or GEOM checkpoints for the
payload-anchor linking step.

## 5. What the pipeline is missing entirely

| Missing | Why it matters | Can it be computed? |
| --- | --- | --- |
| Drug-to-antibody ratio and its distribution | the primary determinant of therapeutic index | No — it is a process property |
| Antigen internalisation rate | a non-internalising antigen cannot deliver a payload at all; this constrains antigen choice in step 1 | Partly, from literature |
| Lysosomal trafficking and payload escape | the payload must reach the cytosol after release | No |
| Bystander effect intent | membrane-permeable released payload kills neighbours; cell-impermeable does not. This is a *design choice* that dictates linker cleavability | No |
| Free-payload potency of the actual released species | the released species is often payload-plus-linker-remnant, not clean payload | Weakly |
| Immunogenicity of the new chemistry | a peptide linker creates new potential epitopes | Partly |

Step 1 as written ("choose an effective antibody and payload") hides the
single most consequential decision in ADC design: **the antigen must
internalise and traffic to the lysosome**. A high-affinity antibody against a
non-internalising antigen makes a useless ADC. If step 1 is going to be
automated at all, internalisation should be an explicit filter.

## 6. Computational validation that would actually mean something

Ordered by how much signal you get per unit of effort.

### Tier 1 — cheap, catches most garbage

- **Chemical validity and synthesisability.** RDKit sanitisation, synthetic
  accessibility, and a retrosynthesis check on the linker in isolation. This
  is the same machinery already built in `molbench`.
- **Cleavable-motif presence.** Substructure match against a curated library
  of ADC-validated cleavage cassettes. A linker with no motif and no
  non-cleavable justification is rejected, not scored.
- **Polarity and aggregation proxies.** cLogP and topological polar surface
  area of the whole linker-payload, plus the count of rotatable bonds. Flag
  anything greasier than the known-good reference set.
- **Attachment-chemistry sanity.** Is there exactly one reactive handle? Is
  it one of the chemistries that works? Is the payload's attachment point the
  one that is known to be tolerated?

### Tier 2 — structural, needs the antibody model

- **Conjugation-site accessibility.** Solvent-accessible surface area of the
  target residue on the antibody structure. A buried lysine does not react.
- **Paratope occlusion.** Distance from the conjugation site to the CDR loops,
  and whether a modelled linker-payload overlaps the antigen-binding surface.
  This is the check that would catch Reading B's main risk.
- **FcRn and effector-site preservation.** If conjugating in the Fc, check
  distance to the FcRn-binding interface, since disrupting it shortens
  half-life.
- **Anchor affinity, for Reading A.** Dock the generated anchor against the Fc
  patch. This is exactly what `molbench`'s `dock_and_score` already does, so
  the existing tooling transfers directly.
- **Warhead-to-lysine geometry, for Reading A.** With the anchor posed, measure
  the distance and approach angle from its reactive carbon to the target
  lysine's ε-nitrogen. This is the single most diagnostic number in the whole
  design, and it is cheap once the pose exists.

### Tier 3 — dynamics, expensive but informative

- **Linker conformational ensemble.** Generate conformers or run short
  molecular dynamics on the linker-payload and look at the end-to-end
  distance distribution and radius of gyration. A linker that collapses the
  payload back onto the antibody surface is an aggregation risk.
- **Conjugate-level hydrophobic patch analysis.** Spatial aggregation
  propensity on the modelled conjugate, to see whether the payload creates a
  new exposed hydrophobic patch.
- **Protease compatibility of the cleavage site.** Dock or simulate the
  scissile bond in a cathepsin B active site and check that the P1-P4
  residues are accommodated and the carbonyl is positioned for attack. This
  tests cleavability far more honestly than motif matching alone.
- **Thiosuccinimide stability, if maleimide chemistry.** Local electrostatics
  around the conjugation site predict retro-Michael susceptibility.

### What cannot be validated computationally

Worth stating plainly so it does not get promised: plasma stability, DAR
distribution, actual payload release efficiency, bystander killing,
internalisation rate, and therapeutic index. These need wet-lab work. The
honest framing for any computational ADC pipeline is that it **ranks
candidates for synthesis**, and nothing more.

## 7. Recommendation

Build the pipeline under **Reading A**, with these changes:

1. **Step 1** adds an explicit internalisation filter on the antigen, and
   records the payload's known-tolerated attachment point as a hard
   constraint rather than a free variable.
2. **Step 2** becomes "choose a conjugation residue and the surface patch
   adjacent to it", with the patch defined on a real antibody structure.
3. **Step 3** is unchanged in mechanism — DiffSBDD against that patch — but
   the generated anchor is scored on *patch affinity plus warhead-to-lysine
   geometry*, and it is understood to be a reagent that leaves, not part of
   the product.
4. **Step 4** splices a mandatory cleavage cassette into the linker and uses
   DiffLinker only for the spacer segments, with unconditioned checkpoints,
   and filters on polarity rather than rewarding 3D burial.
5. **Step 5** runs Tier 1 on everything, Tier 2 on survivors, Tier 3 on the
   handful that are going to be made.

Under Reading B the same structure applies, with step 2 fixed to the Fab
nucleotide-binding site and paratope occlusion promoted from Tier 2 to a
gating check.

The parts of this that are genuinely novel and worth doing are the
anchor-directed site selection and the geometric warhead-to-lysine criterion.
The part that needs the most guarding is linker generation, because an
unconstrained 3D model will confidently produce linkers that are chemically
fine and mechanistically useless.
