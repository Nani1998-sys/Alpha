"""The built-in SMILES reader used when RDKit is not installed.

It previously ignored hydrogens (aspirin came out as C9O4, MW 172) and
accepted text such as "hello world" as a molecule.
"""
import json
import os

import pytest

from alpha import chem

CASES = json.load(open(os.path.join(os.path.dirname(__file__), "data", "smiles_cases.json"),
                       encoding="utf-8"))


def test_fallback_aspirin_without_rdkit():
    d = chem._analyze_fallback("CC(=O)Oc1ccccc1C(=O)O")
    assert d["formula"] == "C9H8O4"
    assert d["mol_weight"] == pytest.approx(180.16, abs=0.02)
    assert d["rings"] == 1


@pytest.mark.parametrize("smiles", CASES["invalid"])
def test_fallback_rejects_invalid_smiles(smiles):
    with pytest.raises(chem.StructureError):
        chem._analyze_fallback(smiles)


@pytest.mark.skipif(not chem.have_rdkit(), reason="needs RDKit as the reference")
@pytest.mark.parametrize("name", sorted(CASES["valid"]))
def test_fallback_matches_rdkit(name):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    smiles = CASES["valid"][name]
    mol = Chem.MolFromSmiles(smiles)
    d = chem._analyze_fallback(smiles)
    assert d["formula"] == rdMolDescriptors.CalcMolFormula(mol)
    assert d["mol_weight"] == pytest.approx(Descriptors.MolWt(mol), abs=0.05)
    assert d["rings"] == rdMolDescriptors.CalcNumRings(mol)
