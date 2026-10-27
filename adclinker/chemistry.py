"""ADC chemistry tables, shared with the molbench tool that enforces them.

**These are a deliberate copy of the tables in `molbench/server.py`, and
`molbench/tests/test_server.py` fails if the two ever differ.**

The copy exists because the two packages cannot import each other. molbench
is a standalone MCP server with its own dependency set, and making it import
an ADC pipeline would break running it on its own; adclinker cannot import
molbench either, because molbench is a flat directory whose server imports
its siblings by name. So the tables are written twice and a test holds them
together.

They have to agree, which is why the test exists rather than a comment asking
politely. The scorer here decides what becomes a repair task; the molbench
tool decides whether a repair is accepted. If those two drift, an agent is
optimising against a target that moves between the task it was given and the
check it is graded by -- and the failure looks like a bad model rather than
two lists that fell out of step.
"""

from __future__ import annotations

# (p10, median, p90) across 1,692 payload-free reference reagents.
ADC_LINKER_WINDOWS = {'clogp': (-0.81, 0.68, 3.14),
 'tpsa': (66.4, 139.3, 284.8),
 'heavy_atoms': (16.0, 36.0, 85.0),
 'mw': (242.3, 526.1, 1236.5),
 'rotatable_bonds': (9.0, 25.0, 73.0),
 'fsp3': (0.5, 0.88, 1.0),
 'hbd': (0.0, 1.0, 4.0),
 'hba': (5.0, 10.0, 26.0)}

# The one monotone limit rather than a band, and the only hard threshold
# here. Two independent derivations agree on it: it is the 90th percentile of
# the 1,692 reference reagents, and it sits just above 2.87, the most
# lipophilic linker among the 16 approved ADCs (Mylotarg and Besponsa's
# hydrazone-disulfide). It rejects none of them.
#
# One-sided on purpose. Hydrophobicity is monotone -- a greasier drug-linker
# aggregates more and clears faster, and there is no point at which being
# more polar starts to hurt, because the linker is discarded on release and
# never crosses a membrane. Enhertu's linker measures -1.71, below the
# reference band entirely, and is approved.
LINKER_CLOGP_CEILING = 3.14

# Which of the windows above are worth saying anything about, and which are
# measured but never judged.
#
# This split is not a matter of taste. Each band was checked against the
# linkers of 16 approved ADCs, and two of them reject most of that set:
#
#   fsp3 [0.50-1.00]  rejects 10 of 16, Adcetris, Enhertu, Polivy, Padcev
#                     and Tivdak among them
#   hbd  [0-4]        rejects 9 of 16
#
# The reason is that the reference set is not the same population as a
# finished drug-linker. It is full of bare PEG and alkyl reagents, which is
# why its median Fsp3 is 0.88 -- but an installed cleavable linker carries a
# p-aminobenzyl aromatic ring and a dipeptide, which puts a real one near
# 0.48. Judging a complete linker by a band measured on fragments flags the
# clinical standard of care as abnormal.
#
# The advisory four are wrong for 2 to 3 of 16, which is a tolerable rate for
# something that only ever prints a note. The rest are reported as raw
# numbers and compared against nothing.
JUDGED_PROPERTIES = frozenset({"clogp"})
ADVISORY_PROPERTIES = frozenset({"tpsa", "heavy_atoms", "mw", "rotatable_bonds"})

# name -> (SMARTS for the reactive core, what it conjugates to)
ADC_HANDLES = {'maleimide': ('O=C1C=CC(=O)N1', 'a free cysteine thiol, reversibly'),
 'bromoacetyl': ('[CX4;H2](Br)C(=O)', 'a free cysteine thiol, irreversibly'),
 'iodoacetyl': ('[CX4;H2](I)C(=O)', 'a free cysteine thiol, irreversibly'),
 'nhs_ester': ('O=C(ON1C(=O)CCC1=O)', 'a solvent-exposed lysine amine'),
 'carboxylic_acid': ('[CX3](=[OX1])[OX2H1]',
                     'a solvent-exposed lysine amine, after activation to an '
                     'NHS ester at conjugation time. This is how a '
                     'lysine-targeting drug-linker is normally drawn and '
                     'sold -- as the free acid -- so it counts as a handle '
                     'even though it is not itself reactive'),
 'dbco': ('N1C(=O)c2ccccc2C#Cc2ccccc21',
          'an engineered azide, strain-promoted'),
 'bcn': ('C1CC2CCC1C#C2', 'an engineered azide, strain-promoted'),
 'tetrazine': ('c1nnc(nn1)', 'a trans-cyclooctene tag'),
 'azide': ('[NX2]=[NX2+]=[NX1-]', 'a strained alkyne')}

# What gets cut, as (name, SMARTS).
ADC_TRIGGERS = (('val_cit', '[CX4]([CX4]([CH3])[CH3])C(=O)N[CX4](CCCNC(N)=O)C(=O)'),
 ('citrulline', '[CX4](CCCNC(N)=O)C(=O)'),
 ('val_ala', '[CX4]([CX4]([CH3])[CH3])C(=O)N[CX4]([CH3])C(=O)'),
 ('ala_ala', '[NX3][CX4]([CH3])C(=O)N[CX4]([CH3])C(=O)'),
 ('phe_lys', '[CX4](Cc1ccccc1)C(=O)N[CX4](CCCC[NX3])C(=O)'),
 ('ggfg', 'NCC(=O)NCC(=O)N[CX4](Cc1ccccc1)C(=O)NCC(=O)'),
 ('disulfide', '[#6][SX2][SX2][#6]'),
 ('hydrazone', '[CX3]=[NX2][NX3]'),
 ('glucuronide', 'OC1OC(C(O)C(O)C1O)C(=O)O'),
 ('carbonate', '[OX2][CX3](=[OX1])[OX2]'))

# (name, SMARTS, releases_free_drug, what it means).
RELEASE_CHEMISTRIES = (('pab_carbamate',
  '[#7;!$([#7][#7])]C(=O)O[CH2]c1ccc([#7])cc1',
  True,
  'self-immolative p-aminobenzyl carbamate: once the trigger is cut the '
  'spacer eliminates and the payload leaves as a free amine'),
 ('pab_ether',
  '[OX2;!$(O=*)][CH2]c1ccc([#7])cc1',
  True,
  'self-immolative p-aminobenzyl ether, releasing a free alcohol payload'),
 ('aminomethylene',
  '[NX3;$(NC=O)][CH2][OX2]',
  True,
  'self-immolative aminomethylene spacer, which collapses after the peptide '
  'is cleaved and releases unmodified payload'),
 ('disulfide',
  '[#6][SX2][SX2][#6]',
  True,
  'reduced by intracellular glutathione, releasing a thiol payload'),
 ('hydrazone',
  '[CX3]=[NX2][NX3]',
  True,
  'hydrolyses in the acidic endosome, releasing a ketone or aldehyde '
  'payload; known to leak in circulation'),
 ('glucuronide',
  'OC1OC(C(O)C(O)C1O)C(=O)O',
  True,
  'cut by lysosomal beta-glucuronidase; very polar, which helps at high '
  'drug-to-antibody ratio'),
 ('ester',
  '[CX3](=[OX1])[OX2][#6]',
  True,
  'hydrolysable, but by non-specific serum esterases rather than a lysosomal '
  'trigger, so release is not tumour-selective'),
 ('amide',
  '[CX3](=[OX1])[NX3]',
  False,
  'a stable amide: it does not hydrolyse on its own, so the payload is freed '
  'only when the antibody itself is proteolysed, and it leaves carrying the '
  'linker remnant'),
 ('thioether',
  '[#6][SX2;!$(SS)][#6]',
  False,
  'a stable thioether; the payload comes off only as an adduct, when the '
  'antibody is degraded'),
 ('ether',
  '[#6][OX2;!$(OC=O)][#6]',
  False,
  'a stable ether; the payload cannot be released from it'))

# (name, SMARTS, why it is wrong).
BAD_JUNCTIONS = (('dialkyl_ketone',
  '[CX4][CX3](=[OX1])[CX4]',
  'a ketone in the linker condenses with lysine amines in vivo; an acyl '
  'handle joined to a carbon-starting spacer makes one where an amide was '
  'intended. Start the spacer with N or O instead'),
 ('triacylamine',
  '[NX3]([CX3]=[OX1])([CX3]=[OX1])[CX3]=[OX1]',
  'three acyl groups on one nitrogen is not a stable compound; this is what '
  'acylating a bare maleimide nitrogen produces'),
 ('anhydride',
  '[CX3](=[OX1])[OX2][CX3]=[OX1]',
  'an anhydride hydrolyses in plasma within minutes'),
 ('gem_diol',
  '[CX4]([OX2H])[OX2H]',
  'a gem-diol is the hydrate of a ketone, not an isolable group'),
 ('acyl_halide',
  '[CX3](=[OX1])[F,Cl,Br,I]',
  'an acyl halide does not survive water'),
 ('peroxide', '[OX2][OX2]', 'a peroxide is not stable enough to administer'),
 ('aldehyde', '[CX3H1](=O)[#6]', 'an aldehyde cross-links protein amines'),
 ('alpha_ketoamide',
  '[NX3][CX3](=[OX1])[CX3](=[OX1])[#6]',
  'an alpha-ketoamide is electrophilic and hydrolytically fragile. It '
  'appears when a spacer is built onto a carbamate carbonyl, which is what '
  'the spacer_after_cassette architecture hands a generator'),
 ('free_thiol',
  '[SX2H]',
  "a free thiol in the linker caps the molecule's own maleimide, scrambles "
  "any disulfide in the cassette, and gives albumin's Cys34 somewhere to "
  'attach. Cap it as a thioether or remove it'),
 ('unhindered_disulfide',
  '[CX4;H2,H3][SX2][SX2][CX4;H2,H3]',
  'a disulfide with no substitution on either alpha carbon is reduced by '
  'circulating glutathione long before the conjugate reaches a cell. The '
  'gem-dimethyl hindrance in every clinical disulfide linker exists for '
  'exactly this reason'),
 ('aldehyde_derived_hydrazone',
  '[CX3;H1]=[NX2][NX3]',
  'an aldehyde-derived hydrazone hydrolyses far too readily to survive '
  'circulation. The acid-labile ADC trigger is a ketone-derived '
  "acylhydrazone, as in gemtuzumab ozogamicin's AcBut linker"))

