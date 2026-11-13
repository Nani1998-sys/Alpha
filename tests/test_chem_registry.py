"""Chemistry + registry tests."""
import os

import pytest

from alpha import chem
from alpha.models import PDParameters, PKParameters
from alpha.registry import DrugRegistry


def test_analyze_aspirin_formula_and_mw():
    d = chem.analyze_structure("CC(=O)Oc1ccccc1C(=O)O")
    assert d["formula"] == "C9H8O4"
    assert d["mol_weight"] == pytest.approx(180.16, abs=0.1)
    if d.get("backend") == "rdkit":
        assert d["h_bond_donors"] == 1
        assert d["drug_likeness"]["lipinski"]["pass"] is True


def test_analyze_bad_smiles_raises():
    with pytest.raises(chem.StructureError):
        chem.analyze_structure("this-is-not-valid-%%%")


def test_registry_add_by_structure_and_roundtrip(tmp_path):
    db = os.path.join(tmp_path, "drugs.json")
    reg = DrugRegistry(db)
    reg.add_drug(
        "Ibuprofen",
        smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        pk=PKParameters(CL=3.5, Vc=10.0, ka=1.5, F=0.9),
        pd=PDParameters(model="emax", E0=0, Emax=100, EC50=10, effect_units="%"),
    )
    reg.save()

    reg2 = DrugRegistry(db)
    assert reg2.has("ibuprofen")  # case-insensitive
    drug = reg2.get("Ibuprofen")
    assert drug.pk.CL == 3.5
    assert drug.pd.model == "emax"
    if chem.have_rdkit():
        assert drug.descriptors["formula"] == "C13H18O2"
        assert drug.inchikey is not None


def test_registry_rejects_duplicate_without_overwrite(tmp_path):
    db = os.path.join(tmp_path, "drugs.json")
    reg = DrugRegistry(db)
    reg.add_drug("X", smiles="CCO", pk=PKParameters(CL=1, Vc=1))
    with pytest.raises(ValueError):
        reg.add_drug("X", smiles="CCO", pk=PKParameters(CL=1, Vc=1))
    # overwrite works
    reg.add_drug("X", smiles="CCO", pk=PKParameters(CL=2, Vc=2), overwrite=True)
    assert reg.get("X").pk.CL == 2
