"""The two pieces handed to the generator, and the libraries they come from.

LinkInvent takes two warheads and designs what joins them. For an ADC those
warheads are the antibody-side handle and the payload, and what gets designed
is the spacer. Everything in this module exists to build that pair correctly,
because the generator's output is only as good as what it is asked to bridge.

Three deliberate decisions.

**The handle is not a design choice.** It follows from the conjugation site:
a cysteine takes a maleimide, a surface lysine takes an activated ester, an
engineered azide takes a strained alkyne. There are a handful that work, they
work because of decades of wet-lab validation, and a generative model has no
way to invent a new one responsibly -- reactivity is not a shape.

**The cleavage cassette is supplied, never generated.** Val-Cit is cut by
cathepsin B; that is a property of the protease, which appears nowhere in any
model here. A generator asked to invent cleavability will not, so the cassette
goes inside a warhead where it is structurally guaranteed, and the model is
left to do the part it can actually do.

**Which warhead carries the cassette decides where the spacer lands**, and
both options are real architectures. See `ARCHITECTURES`.

Attachment points are atom-mapped dummies: `[*:1]` always points back toward
the antibody, `[*:2]` toward the payload. Keeping the direction in the map
number is what lets pieces be joined without tracking orientation separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------- the handles

# Keyed by the conjugation chemistry the site demands. `[*:1]` is the open
# valence the spacer attaches to.
@dataclass(frozen=True)
class Handle:
    name: str
    smiles: str
    site: str  # the residue/feature this chemistry targets
    note: str


HANDLES: dict[str, Handle] = {
    "maleimide": Handle(
        "maleimide",
        "O=C1C=CC(=O)N1[*:1]",
        "cysteine",
        "Michael addition to a free thiol from a reduced interchain disulfide "
        "or an engineered cysteine. The thiosuccinimide can undergo "
        "retro-Michael exchange onto serum albumin, and the rate depends on "
        "the local charge environment of the site.",
    ),
    "maleimidocaproyl": Handle(
        "maleimidocaproyl",
        "O=C1C=CC(=O)N1CCCCCC(=O)[*:1]",
        "cysteine",
        "Maleimide with the classical six-carbon spacer already attached, as "
        "in brentuximab vedotin. Use when you want the spacer fixed rather "
        "than designed.",
    ),
    "nhs_ester": Handle(
        "nhs_ester",
        "O=C(ON1C(=O)CCC1=O)[*:1]",
        "lysine",
        "Acylates a solvent-exposed lysine epsilon-amine. Stochastic across "
        "the ~80 lysines of an IgG1 unless the site is directed.",
    ),
    "dbco": Handle(
        "dbco",
        "O=C([*:1])CCN1C(=O)c2ccccc2C#Cc2ccccc21",
        "azide",
        "Strain-promoted cycloaddition onto an azide installed by glycan "
        "remodelling or an engineered tag. Bioorthogonal and site-specific.",
    ),
    "bromoacetyl": Handle(
        "bromoacetyl",
        "BrCC(=O)[*:1]",
        "cysteine",
        "Alkylates a thiol irreversibly, forming a thioether that cannot "
        "undergo retro-Michael. Slower to react than maleimide.",
    ),
}

# -------------------------------------------------------------- the cassettes

# `[*:1]` faces the antibody, `[*:2]` faces the payload.
@dataclass(frozen=True)
class Cassette:
    name: str
    smiles: str
    cleavable: bool
    trigger: str
    note: str


CASSETTES: dict[str, Cassette] = {
    "val_cit_pab": Cassette(
        "val_cit_pab",
        "[*:1]C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=O)C(=O)"
        "Nc1ccc(COC(=O)[*:2])cc1",
        True,
        "cathepsin B",
        "The brentuximab vedotin cassette. Valine-citrulline is cut in the "
        "lysosome and the p-aminobenzyl carbamate then self-immolates, "
        "releasing an unmodified amine payload.",
    ),
    "val_ala_pab": Cassette(
        "val_ala_pab",
        "[*:1]C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](C)C(=O)"
        "Nc1ccc(COC(=O)[*:2])cc1",
        True,
        "cathepsin B",
        "Valine-alanine in place of valine-citrulline. More stable in plasma "
        "and less hydrophilic, at some cost in cleavage rate.",
    ),
    "ggfg": Cassette(
        "ggfg",
        "[*:1]C(=O)CNC(=O)CNC(=O)[C@H](Cc1ccccc1)NC(=O)CN[*:2]",
        True,
        "cathepsin",
        "The trastuzumab deruxtecan tetrapeptide. Cleaved in the lysosome and "
        "tolerant of high drug-to-antibody ratios because it is very polar.",
    ),
    "disulfide": Cassette(
        "disulfide",
        "[*:1]C(C)(C)SS[*:2]",
        True,
        "glutathione",
        "Reduced by intracellular glutathione. The gem-dimethyl hindrance "
        "slows reduction in plasma; without it the linker is too labile.",
    ),
    "hydrazone": Cassette(
        "hydrazone",
        "[*:1]C(C)=NN[*:2]",
        True,
        "endosomal pH",
        "Hydrolyses in the acidic endosome. Known to be leaky in circulation "
        "-- included for comparison rather than as a recommendation.",
    ),
    "none": Cassette(
        "none",
        "[*:1][*:2]",
        False,
        "none",
        "Non-cleavable. The antibody is proteolysed entirely in the lysosome "
        "and the released species is payload plus a linker remnant and a "
        "lysine. Only works for payloads that stay active as that adduct, as "
        "in trastuzumab emtansine.",
    ),
}

# ---------------------------------------------------------------- the payloads

# Attachment points are published for every payload in clinical use. Getting
# this wrong destroys potency, and nothing downstream would catch it, so the
# known ones are a table rather than a guess.
@dataclass(frozen=True)
class Payload:
    name: str
    smiles: str
    attachment: str
    note: str


PAYLOADS: dict[str, Payload] = {
    "mmae": Payload(
        "mmae",
        "[*:1]N(C)[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H]"
        "([C@@H](C)CC)[C@@H](OC)CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)"
        "N[C@H](C)[C@@H](O)c1ccccc1",
        "N-terminal secondary amine",
        "Monomethyl auristatin E, a tubulin inhibitor. Membrane-permeable "
        "once released, so it gives a bystander effect.",
    ),
    "mmaf": Payload(
        "mmaf",
        "[*:1]N(C)[C@@H](C(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H]"
        "([C@@H](C)CC)[C@@H](OC)CC(=O)N1CCC[C@H]1[C@H](OC)[C@@H](C)C(=O)"
        "N[C@H](C)[C@@H](c1ccccc1)C(=O)O",
        "N-terminal secondary amine",
        "Auristatin F. The terminal carboxylate makes it cell-impermeable, so "
        "there is no bystander killing.",
    ),
    "dxd": Payload(
        "dxd",
        "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c-1nc1cc(F)c(C)c3c1c2"
        "[C@@H](NC(=O)CO[*:1])CC3",
        "the glycolamide hydroxyl",
        "The real DXd: exatecan bearing the glycolamide that trastuzumab "
        "deruxtecan releases. The linker attaches at the glycolamide oxygen "
        "through an aminomethylene acetal, which collapses once the GGFG "
        "tetrapeptide in front of it is cut.\n\n"
        "This entry replaces a placeholder that was neither DXd nor a "
        "camptothecin: MW 330 against the real 493, four rings against six, "
        "and no alpha-hydroxy-delta-lactone at all. That lactone IS the "
        "pharmacophore -- it is what traps the topoisomerase I cleavage "
        "complex -- so the placeholder would have had every design built "
        "around a molecule that cannot kill anything. Taken from the "
        "verified structure of Enhertu in tests/data/approved_adcs.json.",
    ),
    "exatecan": Payload(
        "exatecan",
        "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c-1nc1cc(F)c(C)c3c1c2"
        "[C@@H]([*:1])CC3",
        "the C1 primary amine",
        "Exatecan itself, attached through its amine rather than through a "
        "glycolamide. Use when you want the spacer to carry the whole "
        "release chemistry instead of inheriting deruxtecan's.",
    ),
}

# Functional groups that can plausibly carry a linker on a novel payload,
# ranked by how routinely they are used. Only ever a suggestion.
ATTACHMENT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("primary_aliphatic_amine", "[NX3;H2;!$(NC=O);!R]"),
    ("secondary_aliphatic_amine", "[NX3;H1;!$(NC=O);!R]"),
    ("thiol", "[SX2;H1]"),
    ("primary_alcohol", "[OX2;H1][CX4;H2]"),
    ("carboxylic_acid", "[CX3](=O)[OX2;H1]"),
    ("phenol", "[OX2;H1]c"),
)

# ------------------------------------------------------------ the architecture

ARCHITECTURES: dict[str, str] = {
    "spacer_before_cassette": (
        "Warhead 1 is the bare handle; warhead 2 is cassette plus payload. "
        "The generated spacer therefore sits between the handle and the "
        "cleavage cassette, which is where the PEG sits in MC-PEG2-Val-Cit-"
        "PAB-MMAE and most other clinical cleavable ADCs. This is the default."
    ),
    "spacer_after_cassette": (
        "Warhead 1 is handle plus cassette; warhead 2 is the bare payload. "
        "The generated spacer sits between the self-immolative group and the "
        "payload. A real but less common architecture, and worth running "
        "alongside the default because it explores different chemistry."
    ),
}


# Junctions that RDKit will happily sanitise and a chemist would reject.
# Checked after every join, because the libraries combine freely and not every
# combination is a real molecule.
#
# The one that actually bites: a bare maleimide's nitrogen is already an imide
# with two carbonyls, so bonding it straight to an acyl gives a triacylamine,
# which is not a stable compound. That is precisely why clinical ADCs use
# maleimido*caproyl* -- the six carbons put the amide a safe distance from the
# imide nitrogen.
IMPLAUSIBLE_JUNCTIONS: tuple[tuple[str, str], ...] = (
    (
        "triacylamine",
        "[NX3]([CX3]=[OX1])([CX3]=[OX1])[CX3]=[OX1]",

    ),
    ("anhydride", "[CX3](=[OX1])[OX2][CX3]=[OX1]"),
    ("acyl_halide", "[CX3](=[OX1])[F,Cl,Br,I]"),
    ("gem_diol", "[CX4]([OX2H])[OX2H]"),
)


class WarheadError(Exception):
    """A warhead could not be built from the pieces given."""


def implausible_junctions(smiles: str) -> list[str]:
    """Names of any chemically unreasonable groups the join created."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return ["unparseable"]
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - reported by the caller's own sanitise
        return []
    hits = []
    for name, smarts in IMPLAUSIBLE_JUNCTIONS:
        query = Chem.MolFromSmarts(smarts)
        if query is not None and mol.HasSubstructMatch(query):
            hits.append(name)
    return hits


# -------------------------------------------------------------- the assembling


def _dummy_with_map(mol: Any, map_number: int) -> int | None:
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == map_number:
            return atom.GetIdx()
    return None


def join(left: str, right: str, *, left_map: int = 1, right_map: int = 1) -> str:
    """Bond `left`'s dummy to `right`'s dummy and drop both.

    The neighbours of the two dummy atoms are joined by a single bond. Any
    dummy not named by `left_map` / `right_map` survives, which is how a
    cassette keeps its far attachment point while its near one is consumed.
    """
    from rdkit import Chem

    a = Chem.MolFromSmiles(left, sanitize=False)
    b = Chem.MolFromSmiles(right, sanitize=False)
    if a is None or b is None:
        raise WarheadError(f"could not parse: {left!r} or {right!r}")

    combined = Chem.RWMol(Chem.CombineMols(a, b))
    offset = a.GetNumAtoms()

    a_dummy = _dummy_with_map(a, left_map)
    b_dummy = _dummy_with_map(b, right_map)
    if a_dummy is None:
        raise WarheadError(f"{left!r} has no [*:{left_map}] attachment point")
    if b_dummy is None:
        raise WarheadError(f"{right!r} has no [*:{right_map}] attachment point")
    b_dummy += offset

    def sole_neighbour(index: int) -> int:
        neighbours = combined.GetAtomWithIdx(index).GetNeighbors()
        if len(neighbours) != 1:
            raise WarheadError(
                f"attachment point at index {index} has "
                f"{len(neighbours)} neighbours; it must have exactly one"
            )
        return neighbours[0].GetIdx()

    a_anchor = sole_neighbour(a_dummy)
    b_anchor = sole_neighbour(b_dummy)

    # A direct join -- the `none` cassette, which is two dummies bonded to
    # each other -- would ask for a bond between the two anchors, which is
    # exactly right, but the dummies must go first or the indices shift.
    combined.AddBond(a_anchor, b_anchor, Chem.BondType.SINGLE)
    for index in sorted([a_dummy, b_dummy], reverse=True):
        combined.RemoveAtom(index)

    product = combined.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Exception as exc:  # noqa: BLE001 - a bad join is a result, not a crash
        raise WarheadError(f"joining produced an invalid molecule: {exc}") from exc
    return Chem.MolToSmiles(product)


def strip_atom_maps(smiles: str) -> str:
    """`[*:1]` becomes `*`, which is the token LinkInvent's input vocabulary has."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise WarheadError(f"could not parse {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def flatten(smiles: str) -> str:
    """Drop atom maps and stereochemistry: what LinkInvent's vocabulary has."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise WarheadError(f"could not parse {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def assemble(linker: str, pair: WarheadPair) -> str:
    """Rebuild the full molecule from a generated linker and the real warheads.

    REINVENT reports an assembled SMILES of its own, but it is assembled from
    the *flattened* warheads it was given, so every stereocentre in the
    payload and the cassette comes back unspecified. Joining the generated
    linker to the originals instead keeps all of it exactly as supplied --
    the model never saw the stereochemistry, so it cannot have changed it.

    The linker arrives with two unmapped attachment points. They are numbered
    by order of appearance, which is the order LinkInvent itself uses.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(linker, sanitize=False)
    if mol is None:
        raise WarheadError(f"could not parse generated linker {linker!r}")

    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise WarheadError(
            f"a linker needs exactly two attachment points, {linker!r} has "
            f"{len(dummies)}"
        )
    for number, atom in enumerate(dummies, start=1):
        atom.SetAtomMapNum(number)
    mapped = Chem.MolToSmiles(mol)

    joined = join(pair.first, mapped, left_map=1, right_map=1)
    return join(joined, pair.second, left_map=2, right_map=1)


def renumber_dummy(smiles: str, old: int, new: int) -> str:
    """Relabel one attachment point, so a surviving `[*:2]` can become `[*:1]`."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise WarheadError(f"could not parse {smiles!r}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == old:
            atom.SetAtomMapNum(new)
    return Chem.MolToSmiles(mol)


@dataclass
class WarheadPair:
    """What goes to the generator, and what it means."""

    first: str  # antibody side
    second: str  # payload side
    handle: str
    cassette: str
    payload: str
    architecture: str

    def as_linkinvent_input(self) -> str:
        """The two warheads as LinkInvent can actually read.

        Two things are stripped, for two different reasons.

        **Atom maps**, because LinkInvent's input vocabulary tokenises `*` as
        a single character and has no `[*:1]` token at all.

        **Stereochemistry**, because the prior's input vocabulary is 34 tokens
        and contains neither `[C@H]` nor `[C@@H]`. MMAE has ten stereocentres
        and Val-Cit two, so a faithful warhead is rejected at token
        validation. This is not a loss: the model only ever sees the warheads
        as context for designing the spacer, and `assemble()` rebuilds the
        final molecule from the generated linker and the *original*
        stereochemically correct warheads. The stereocentres never pass
        through the model at all, so they cannot be corrupted by it.
        """
        return f"{flatten(self.first)}|{flatten(self.second)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "warhead_1": self.first,
            "warhead_2": self.second,
            "handle": self.handle,
            "cassette": self.cassette,
            "payload": self.payload,
            "architecture": self.architecture,
        }


def build_pair(
    handle: str,
    cassette: str,
    payload: str,
    architecture: str = "spacer_before_cassette",
) -> WarheadPair:
    """Assemble the two warheads for one design.

    Both returned warheads carry exactly one `[*:1]`, which is what the
    generator attaches its spacer to.
    """
    if handle not in HANDLES:
        raise WarheadError(f"unknown handle {handle!r}; have {sorted(HANDLES)}")
    if cassette not in CASSETTES:
        raise WarheadError(f"unknown cassette {cassette!r}; have {sorted(CASSETTES)}")
    if payload not in PAYLOADS:
        raise WarheadError(f"unknown payload {payload!r}; have {sorted(PAYLOADS)}")
    if architecture not in ARCHITECTURES:
        raise WarheadError(
            f"unknown architecture {architecture!r}; have {sorted(ARCHITECTURES)}"
        )

    handle_smiles = HANDLES[handle].smiles
    cassette_smiles = CASSETTES[cassette].smiles
    payload_smiles = PAYLOADS[payload].smiles

    if architecture == "spacer_before_cassette":
        # cassette[*:2] -> payload[*:1]; the cassette's [*:1] stays open.
        second = join(cassette_smiles, payload_smiles, left_map=2, right_map=1)
        first = handle_smiles
    else:
        # handle[*:1] -> cassette[*:1]; the cassette's [*:2] stays open and is
        # relabelled so both warheads present a [*:1] to the generator.
        first = renumber_dummy(
            join(handle_smiles, cassette_smiles, left_map=1, right_map=1), 2, 1
        )
        second = payload_smiles

    for which, smiles in (("warhead 1", first), ("warhead 2", second)):
        problems = implausible_junctions(smiles)
        if problems:
            hint = ""
            if "triacylamine" in problems and handle == "maleimide":
                hint = (
                    "\n  A bare maleimide nitrogen is already an imide, so "
                    "acylating it is not a real compound. Use the "
                    "`maleimidocaproyl` handle, which carries the six-carbon "
                    "spacer that clinical ADCs use for exactly this reason, "
                    "or the `spacer_before_cassette` architecture, where the "
                    "generator supplies the spacer itself."
                )
            raise WarheadError(
                f"{which} would contain {', '.join(problems)}: "
                f"{handle} + {cassette} does not join directly.{hint}"
            )

    return WarheadPair(
        first=first,
        second=second,
        handle=handle,
        cassette=cassette,
        payload=payload,
        architecture=architecture,
    )


def propose_attachment_points(smiles: str) -> list[dict[str, Any]]:
    """Candidate positions to hang a linker on an unknown payload.

    A suggestion for a human, never an answer. Attaching at the wrong position
    destroys potency, the molecule still scores perfectly, and no check in
    this pipeline would notice. For any payload in clinical use the tolerated
    position is published; use `PAYLOADS`.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise WarheadError(f"could not parse payload SMILES {smiles!r}")

    found: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for name, smarts in ATTACHMENT_CANDIDATES:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        for match in mol.GetSubstructMatches(query):
            index = match[0]
            if index in claimed:
                continue
            claimed.add(index)
            atom = mol.GetAtomWithIdx(index)
            found.append(
                {
                    "atom_index": index,
                    "group": name,
                    "symbol": atom.GetSymbol(),
                    "in_ring": atom.IsInRing(),
                }
            )
    return found
