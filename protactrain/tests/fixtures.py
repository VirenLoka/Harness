"""Real PROTACs from the corpus, and the specific ways they go wrong.

Every molecule here is a verbatim row from `repair_tasks.jsonl`, not a toy.
That is not fussiness. The reward's floors all hang on matching two fragments
back out of a ~60-heavy-atom bifunctional molecule, and `match_fragments`
behaves nothing like it does on a two-ring test case: a toy "PROTAC" either
matches trivially or fails to match at all, so a test built on one passes
whatever the floor logic says. The degenerate cases are also real -- the
three-attachment molecule below is one of the 86 rows in the corpus whose
linker touches its fragments at other than two atoms, not something written
to make a branch fire.

Where a test needs a molecule the corpus does not contain -- a linker with an
aldehyde spliced in -- it is derived from one of these by a documented edit to
the linker only, with the fragments untouched, and the resulting measurement
is asserted in the test rather than trusted.
"""

from __future__ import annotations

# --- 102-0: a VHL / thiohydantoin degrader, admissible as it stands.
#
# Its only recorded failure is `qed_too_low`, which this package deliberately
# does not score, so it is one of the 952 control tasks. Measured by
# protacbench.grade.measure: span 10, linker 10 heavy atoms, cLogP 6.67,
# SA 4.94, TPSA 217.2. Against the default bands that makes span and
# linker_heavy_atoms holds (both inside), and cLogP a live goal (6.67 against
# a 3.5-5.6 band) -- which is exactly the mix the goal-versus-hold split
# exists to tell apart.
TASK_102_0 = "102-0"
PROTAC = (
    "Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CCCCCNC(=O)CCNC(=O)"
    "c2ccc(N3C(=S)N(c4ccc(C#N)c(C(F)(F)F)c4)C(=O)C3(C)C)cc2F)C(C)(C)C)cc1"
)

# The warhead (thiohydantoin) and the E3 ligand (VHL). Frozen: the policy may
# edit only what lies between them.
FRAGMENTS = [
    "CC1(C)C(=O)N(c2ccc(C#N)c(C(F)(F)F)c2)C(=S)N1c1ccc(C(N)=O)c(F)c1",
    "Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC=O)C(C)(C)C)cc1",
]

# --- 102-1: the same two fragments bridged by PEG3 instead of the diamide.
#
# Used wherever a test needs a *different* admissible molecule over the same
# fragment pair -- a plausible repair of 102-0 rather than an arbitrary edit.
# Measured: span 11, cLogP 5.65, SA 5.02, TPSA 215.8.
PROTAC_PEG = (
    "Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)COCCOCCOCCNC(=O)"
    "c2ccc(N3C(=S)N(c4ccc(C#N)c(C(F)(F)F)c4)C(=O)C3(C)C)cc2F)C(C)(C)C)cc1"
)

# --- the warhead deleted, the VHL half and the linker kept.
#
# This is "the failure that looks like success": every descriptor improves at
# once because a third of the molecule is gone. Only matching the fragments
# back catches it, which is why it is a floor and not a weighted term.
PROTAC_FRAGMENT_DELETED = (
    "Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CCCCCNC(=O)CCN)"
    "C(C)(C)C)cc1"
)

# --- an aldehyde spliced into 102-0's linker chain.
#
# Derived, not from the corpus: all three linker aldehydes in the corpus sit
# on disconnected molecules, because in a generated sample an aldehyde needs a
# loose end. The substitution is inside the linker and leaves both fragments
# and the span untouched; the test asserts the measurement rather than
# assuming it.
PROTAC_REACTIVE = PROTAC.replace(
    "NC(=O)CCCCCNC(=O)CCNC(=O)", "NC(=O)CCCC(C=O)CNC(=O)CCNC(=O)"
)

# --- 1112-6: a real corpus molecule whose linker attaches at three atoms.
#
# Verdict `broken`, failure code `linker_not_bridging`. It parses, it is one
# piece and both fragments match, so nothing short of counting the attachment
# bonds rejects it -- which is the point of testing the floor on this rather
# than on something that fails earlier.
TASK_1112_6 = "1112-6"
PROTAC_THREE_ATTACHMENTS = (
    "CC(C)c1c2c(c(-c3ccc(F)cc3)n1CCC(O)CC(O)CC(=O)O)-c1ccc(cc1)CCC"
    "(CCCCCCNc1cccc3c1C(=O)N(C1CCC(=O)NC1=O)C3=O)OCCOCCCCCc1cccc(c1)NC2=O"
)
FRAGMENTS_1112 = [
    "CC(C)c1c(c(c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O)-c1ccccc1)"
    "C(=O)Nc1ccccc1",
    "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O",
]
