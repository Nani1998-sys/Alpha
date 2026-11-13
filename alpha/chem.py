"""Chemistry layer: turn a chemical structure into molecular descriptors.

The key feature of Alpha is that you register a new drug *just by giving its
chemical structure* (a SMILES string). This module parses that structure with
RDKit and computes the physicochemical descriptors used throughout drug
discovery, plus standard drug-likeness rule sets (Lipinski's Rule of 5,
Veber). It also produces clearly-labelled *heuristic* developability flags
(these are rules of thumb for teaching/triage, not validated predictions).

Full descriptors require RDKit. If RDKit is not installed, a lightweight
fallback parser still returns a molecular formula, molecular weight and atom
counts so the platform remains usable.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - import guard
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


class StructureError(ValueError):
    """Raised when a chemical structure cannot be parsed."""


# Standard atomic weights (IUPAC, as used by RDKit) for the fallback parser.
_ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.812, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305,
    "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.067, "Cl": 35.453, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.39, "Ga": 69.723,
    "Ge": 72.61, "As": 74.922, "Se": 78.96, "Br": 79.904, "Kr": 83.8, "Rb": 85.468,
    "Sr": 87.62, "Zr": 91.224, "Mo": 95.94, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.412, "In": 114.818, "Sn": 118.711,
    "Sb": 121.76, "Te": 127.6, "I": 126.904, "Xe": 131.29, "Cs": 132.905,
    "Ba": 137.328, "La": 138.906, "Gd": 157.25, "Pt": 195.078, "Au": 196.967,
    "Hg": 200.59, "Tl": 204.383, "Pb": 207.2, "Bi": 208.98, "*": 0.0,
}
# Normal valences of the SMILES "organic subset" (atoms written without
# brackets get implicit hydrogens up to the lowest valence that fits).
_ORGANIC_VALENCE = {
    "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6),
    "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,), "*": (),
}
_AROMATIC_SYMBOLS = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S",
                     "se": "Se", "as": "As", "te": "Te"}
_BOND_ORDER = {"-": 1.0, "=": 2.0, "#": 3.0, "$": 4.0, ":": 1.5, "/": 1.0, "\\": 1.0}
_BRACKET_RE = re.compile(
    r"^(\d+)?"                                   # isotope
    r"(\*|[A-Z][a-z]?|se|as|te|[bcnops])"        # element (lowercase = aromatic)
    r"(@@?(?:TH[12]|AL[12]|SP[123]|TB\d{1,2}|OH\d{1,2})?)?"  # chirality
    r"(H\d*)?"                                   # hydrogen count
    r"([+-]\d+|\++|-+)?"                         # charge
    r"(:\d+)?$"                                  # atom class
)


def have_rdkit() -> bool:
    return _HAVE_RDKIT


# ---------------------------------------------------------------------------
# RDKit-backed analysis
# ---------------------------------------------------------------------------
def _analyze_rdkit(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructureError(f"RDKit could not parse SMILES: {smiles!r}")

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    molar_refractivity = Crippen.MolMR(mol)

    desc: Dict[str, Any] = {
        "canonical_smiles": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "h_bond_donors": hbd,
        "h_bond_acceptors": hba,
        "rotatable_bonds": rotatable,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "fraction_csp3": round(fraction_csp3, 3),
        "molar_refractivity": round(molar_refractivity, 2),
        "qed": round(QED.qed(mol), 3),
        "inchikey": Chem.MolToInchiKey(mol),
        "backend": "rdkit",
    }
    desc["drug_likeness"] = _drug_likeness(mw, logp, hbd, hba, tpsa, rotatable)
    desc["heuristics"] = _developability_heuristics(mw, logp, tpsa, hbd, hba, rotatable)
    return desc


# ---------------------------------------------------------------------------
# Fallback parser (no RDKit): a validating SMILES reader
# ---------------------------------------------------------------------------
def parse_smiles(smiles: str) -> Dict[str, Any]:
    """Parse SMILES into atoms (with implicit hydrogens) and bonds.

    Validates syntax (unknown characters, unbalanced branches, unclosed ring
    bonds, dangling bonds) and assigns implicit hydrogens to organic-subset
    atoms using the OpenSMILES valence rules. Aromatic bonds count 1.5.
    Raises StructureError on invalid input. Pure Python, no dependencies.
    """
    s = smiles.strip()
    if not s:
        raise StructureError("Empty SMILES string")
    atoms: List[Dict[str, Any]] = []
    bonds: List[tuple] = []          # (i, j, order)
    branch_stack: List[int] = []
    rings: Dict[str, tuple] = {}     # ring label -> (atom index, bond symbol)
    prev: Optional[int] = None
    pending: Optional[str] = None    # bond symbol waiting for its next atom
    i, n = 0, len(s)

    def fail(msg: str):
        raise StructureError(f"Invalid SMILES {smiles!r}: {msg} (at position {i + 1})")

    def add_bond(a: int, b: int, sym: Optional[str]):
        if a == b:
            fail("an atom cannot bond to itself")
        if any({a, b} == {x[0], x[1]} for x in bonds):
            fail("duplicate bond between the same two atoms")
        implicit_arom = sym is None and atoms[a]["aromatic"] and atoms[b]["aromatic"]
        if sym is None:
            order = 1.5 if implicit_arom else 1.0
        else:
            order = _BOND_ORDER[sym]
        bonds.append([a, b, order, implicit_arom])

    def add_atom(atom: Dict[str, Any]):
        nonlocal prev, pending
        atoms.append(atom)
        idx = len(atoms) - 1
        if prev is not None:
            add_bond(prev, idx, pending)
        elif pending is not None:
            fail("bond symbol with no atom before it")
        prev, pending = idx, None

    while i < n:
        ch = s[i]
        if ch == "[":
            j = s.find("]", i)
            if j < 0:
                fail("unclosed '['")
            m = _BRACKET_RE.match(s[i + 1:j])
            if not m:
                fail(f"malformed bracket atom [{s[i + 1:j]}]")
            sym = m.group(2)
            aromatic = sym in _AROMATIC_SYMBOLS
            el = _AROMATIC_SYMBOLS.get(sym, sym)
            if el not in _ATOMIC_WEIGHT:
                fail(f"unknown element {sym!r}")
            hgrp = m.group(4)
            hcount = 0 if not hgrp else (int(hgrp[1:]) if len(hgrp) > 1 else 1)
            isotope = int(m.group(1)) if m.group(1) else None
            chg = m.group(5) or ""
            if chg in ("", None):
                charge = 0
            elif chg[1:].isdigit():
                charge = int(chg[1:]) * (1 if chg[0] == "+" else -1)
            else:
                charge = len(chg) * (1 if chg[0] == "+" else -1)
            add_atom({"el": el, "aromatic": aromatic, "bracket": True,
                      "h": hcount, "charge": charge, "isotope": isotope})
            i = j + 1
        elif s.startswith(("Cl", "Br"), i):
            add_atom({"el": s[i:i + 2], "aromatic": False, "bracket": False, "charge": 0})
            i += 2
        elif ch in "BCNOPSFI*":
            add_atom({"el": ch, "aromatic": False, "bracket": False, "charge": 0})
            i += 1
        elif ch in "bcnops":
            add_atom({"el": ch.upper(), "aromatic": True, "bracket": False, "charge": 0})
            i += 1
        elif ch in _BOND_ORDER:
            if pending is not None:
                fail("two bond symbols in a row")
            if prev is None:
                fail("bond symbol with no atom before it")
            pending = ch
            i += 1
        elif ch == "(":
            if prev is None:
                fail("branch '(' with no atom before it")
            if i + 1 < n and s[i + 1] == ")":
                fail("empty branch '()'")
            branch_stack.append(prev)
            i += 1
        elif ch == ")":
            if not branch_stack:
                fail("unbalanced ')'")
            if pending is not None:
                fail("bond symbol at the end of a branch")
            prev = branch_stack.pop()
            i += 1
        elif ch.isdigit() or ch == "%":
            if prev is None:
                fail("ring-closure digit with no atom before it")
            if ch == "%":
                label = s[i + 1:i + 3]
                if len(label) != 2 or not label.isdigit():
                    fail("'%' must be followed by two digits")
                i += 3
            else:
                label = ch
                i += 1
            if label in rings:
                other, sym0 = rings.pop(label)
                sym = pending or sym0
                if pending and sym0 and pending != sym0:
                    fail(f"conflicting bond orders on ring closure {label}")
                add_bond(other, prev, sym)
            else:
                rings[label] = (prev, pending)
            pending = None
        elif ch == ".":
            if pending is not None:
                fail("bond symbol before '.'")
            if prev is None:
                fail("'.' with no atom before it")
            prev = None
            i += 1
        else:
            fail(f"unexpected character {ch!r}")

    if pending is not None:
        fail("SMILES ends with a bond symbol")
    if branch_stack:
        fail("unclosed branch '('")
    if rings:
        fail(f"unclosed ring bond(s): {', '.join(sorted(rings))}")
    if not atoms:
        fail("no atoms found")

    # Ring bonds: a bond lies on a ring iff its two ends stay connected
    # without it. An unwritten bond between aromatic atoms is aromatic only
    # inside a ring; between two separate rings (biphenyl) it is single.
    def connected_without(skip: int) -> bool:
        parent = list(range(len(atoms)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for k, bnd in enumerate(bonds):
            if k != skip:
                parent[find(bnd[0])] = find(bnd[1])
        return find(bonds[skip][0]) == find(bonds[skip][1])

    in_ring = [connected_without(k) for k in range(len(bonds))]
    for k, bnd in enumerate(bonds):
        if bnd[3] and not in_ring[k]:
            bnd[2] = 1.0

    # Implicit hydrogens for organic-subset atoms. Aliphatic atoms: the lowest
    # normal valence >= the bond-order sum (OpenSMILES). Aromatic atoms (whose
    # aromatic bonds count 1.5): only the lowest valence applies, so e.g.
    # thiophene sulfur and pyridine nitrogen correctly get no hydrogen.
    valence_sum = [0.0] * len(atoms)
    ring_bonds = [0] * len(atoms)
    for k, (a, b, order, _) in enumerate(bonds):
        valence_sum[a] += order
        valence_sum[b] += order
        if in_ring[k]:
            ring_bonds[a] += 1
            ring_bonds[b] += 1
    for k, atom in enumerate(atoms):
        if atom["aromatic"] and ring_bonds[k] < 2:
            fail(f"aromatic atom '{atom['el'].lower()}' is not in a ring")
        if atom["bracket"]:
            continue
        allowed = _ORGANIC_VALENCE.get(atom["el"], ())
        used = valence_sum[k]
        if atom["aromatic"]:
            target = allowed[0] if allowed else None
        else:
            target = next((v for v in allowed if v >= used - 1e-9), None)
        atom["h"] = int(max(0, math.floor(target - used + 1e-9))) if target else 0
    return {"atoms": atoms, "bonds": [tuple(b[:3]) for b in bonds]}


def _formula(counts: Dict[str, int], charge: int) -> str:
    """Molecular formula in RDKit's convention (C, then H, then alphabetical;
    H first when there is no carbon), so both backends print the same."""
    def part(el):
        c = counts[el]
        return el + (str(c) if c > 1 else "")
    head = [e for e in ("C", "H") if e in counts]
    order = head + sorted(e for e in counts if e not in ("C", "H", "*"))
    f = "".join(part(e) for e in order)
    if charge:
        f += ("+" if charge > 0 else "-") + (str(abs(charge)) if abs(charge) > 1 else "")
    return f


# Isotope masses for hydrogen; other isotopes use their mass number.
_ISOTOPE_MASS = {("H", 1): 1.008, ("H", 2): 2.014, ("H", 3): 3.016}


def _atom_mass(atom: Dict[str, Any]) -> float:
    iso = atom.get("isotope")
    if iso:
        return _ISOTOPE_MASS.get((atom["el"], iso), float(iso))
    return _ATOMIC_WEIGHT[atom["el"]]


def _analyze_fallback(smiles: str) -> Dict[str, Any]:
    mol = parse_smiles(smiles)
    counts: Dict[str, int] = {}
    mw = 0.0
    for atom in mol["atoms"]:
        counts[atom["el"]] = counts.get(atom["el"], 0) + 1
        mw += _atom_mass(atom)
        if atom["h"]:
            counts["H"] = counts.get("H", 0) + atom["h"]
            mw += atom["h"] * _ATOMIC_WEIGHT["H"]
    charge = sum(a["charge"] for a in mol["atoms"])
    n_atoms, n_bonds = len(mol["atoms"]), len(mol["bonds"])
    # Connected components via union-find, for the ring count.
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in mol["bonds"]:
        parent[find(a)] = find(b)
    components = len({find(k) for k in range(n_atoms)})
    return {
        "canonical_smiles": smiles.strip(),
        "formula": _formula(counts, charge),
        "mol_weight": round(mw, 2),
        "heavy_atoms": sum(1 for a in mol["atoms"] if a["el"] not in ("H", "*")),
        "rings": n_bonds - n_atoms + components,
        "formal_charge": charge,
        "atom_counts": counts,
        "backend": "fallback",
        "warning": ("RDKit not installed: formula, MW, atom and ring counts only. "
                    "Install RDKit for logP, TPSA, QED and drug-likeness."),
    }


# ---------------------------------------------------------------------------
# Rule sets
# ---------------------------------------------------------------------------
def _drug_likeness(mw, logp, hbd, hba, tpsa, rotatable) -> Dict[str, Any]:
    """Lipinski Rule of 5 and Veber rules for oral drug-likeness."""
    lipinski_flags = {
        "mw_<=500": mw <= 500,
        "logp_<=5": logp <= 5,
        "hbd_<=5": hbd <= 5,
        "hba_<=10": hba <= 10,
    }
    n_violations = sum(1 for ok in lipinski_flags.values() if not ok)
    veber_flags = {
        "rotatable_bonds_<=10": rotatable <= 10,
        "tpsa_<=140": tpsa <= 140,
    }
    return {
        "lipinski": {
            "violations": n_violations,
            "pass": n_violations <= 1,  # Ro5 allows up to 1 violation
            "flags": lipinski_flags,
        },
        "veber": {
            "pass": all(veber_flags.values()),
            "flags": veber_flags,
        },
    }


def _developability_heuristics(mw, logp, tpsa, hbd, hba, rotatable) -> Dict[str, str]:
    """Rough, teaching-oriented flags. NOT validated predictions."""
    h: Dict[str, str] = {}

    # Oral absorption rough triage (Lipinski/Veber space).
    if mw <= 500 and -0.5 <= logp <= 5 and tpsa <= 140 and hbd <= 5:
        h["oral_absorption"] = "favorable (within Lipinski/Veber space)"
    else:
        h["oral_absorption"] = "possible liability (outside Lipinski/Veber space)"

    # CNS penetration rough rule of thumb (TPSA / logP window).
    if tpsa < 90 and 1 <= logp <= 4 and mw < 450 and hbd <= 3:
        h["cns_penetration"] = "plausibly CNS-penetrant (rule of thumb)"
    else:
        h["cns_penetration"] = "likely low CNS penetration (rule of thumb)"

    # Lead-/fragment-likeness note.
    if mw < 300:
        h["size_class"] = "fragment / small lead"
    elif mw <= 500:
        h["size_class"] = "lead-/drug-like"
    else:
        h["size_class"] = "beyond Rule-of-5 (large molecule)"

    h["disclaimer"] = (
        "Heuristics are rules of thumb for triage/education only; they do NOT "
        "predict real ADME. Provide measured PK/PD parameters for simulation."
    )
    return h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_structure(smiles: str) -> Dict[str, Any]:
    """Parse a chemical structure (SMILES) and return descriptors.

    Uses RDKit when available (full descriptor set + drug-likeness), otherwise
    a lightweight fallback (formula / MW / atom counts).
    """
    if not smiles or not smiles.strip():
        raise StructureError("Empty SMILES string")
    smiles = smiles.strip()
    if _HAVE_RDKIT:
        return _analyze_rdkit(smiles)
    return _analyze_fallback(smiles)


def inchikey_for(smiles: str) -> Optional[str]:
    """Return the InChIKey for a structure (RDKit only), else None."""
    if not _HAVE_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None
