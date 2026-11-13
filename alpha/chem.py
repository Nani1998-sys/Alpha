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

import re
from typing import Any, Dict, Optional

try:  # pragma: no cover - import guard
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


class StructureError(ValueError):
    """Raised when a chemical structure cannot be parsed."""


# Standard atomic weights for the common organic elements (fallback parser).
_ATOMIC_WEIGHT = {
    "H": 1.008, "B": 10.811, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904,
    "I": 126.904, "Na": 22.990, "K": 39.098, "Si": 28.085, "Se": 78.971,
}


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
# Fallback parser (no RDKit): formula, MW and atom counts only
# ---------------------------------------------------------------------------
_ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
_TOKEN_RE = re.compile(
    r"Cl|Br|[cnops]|[BCNOFPSI]|\[[^\]]+\]|[0-9]|[=#\-\+\(\)\\/@\.%]|H"
)


def _analyze_fallback(smiles: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    # Count explicit atoms in brackets and bare organic-subset atoms.
    bracket_re = re.compile(r"\[([0-9]*)([A-Z][a-z]?)([^\]]*)\]")
    stripped = smiles
    for m in bracket_re.finditer(smiles):
        el = m.group(2)
        counts[el] = counts.get(el, 0) + 1
    stripped = bracket_re.sub("", smiles)
    # Bare atoms: two-letter first, then single.
    for el in ("Cl", "Br"):
        n = stripped.count(el)
        if n:
            counts[el] = counts.get(el, 0) + n
            stripped = stripped.replace(el, "")
    aromatic = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P"}
    for ch in stripped:
        if ch.isalpha():
            el = aromatic.get(ch, ch.upper() if ch.islower() else ch)
            if el in _ATOMIC_WEIGHT:
                counts[el] = counts.get(el, 0) + 1
    if not counts:
        raise StructureError(f"Could not parse any atoms from SMILES: {smiles!r}")
    mw = sum(_ATOMIC_WEIGHT.get(el, 0.0) * n for el, n in counts.items())
    formula = "".join(f"{el}{counts[el] if counts[el] > 1 else ''}"
                      for el in sorted(counts))
    return {
        "canonical_smiles": smiles,
        "formula": formula,
        "mol_weight": round(mw, 2),
        "heavy_atoms": sum(n for el, n in counts.items() if el != "H"),
        "atom_counts": counts,
        "backend": "fallback",
        "warning": "RDKit not installed: only formula/MW/atom counts available.",
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
