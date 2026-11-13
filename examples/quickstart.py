"""Quickstart: register a drug from its chemical structure, then simulate it.

Run:  python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha import (
    DosingRegimen,
    DrugRegistry,
    PDParameters,
    PKParameters,
    simulate_drug,
    text_report,
)

# 1. Open (or create) a drug database.
reg = DrugRegistry("my_drugs.json")

# 2. Insert a NEW drug molecule just by giving its chemical structure.
#    Descriptors (MW, logP, TPSA, Lipinski, QED, InChIKey) are computed
#    automatically from the SMILES. Attach its PK and PD parameters.
reg.add_drug(
    name="Demo-Molecule",
    smiles="CC(=O)Oc1ccccc1C(=O)O",         # <- chemical structure (aspirin)
    pk=PKParameters(CL=39.0, Vc=10.0, ka=2.0, F=0.68),
    pd=PDParameters(model="emax", E0=0, Emax=100, EC50=5.0,
                    effect_units="% effect"),
    notes="Example drug inserted by structure.",
    overwrite=True,
)
reg.save()

drug = reg.get("Demo-Molecule")
print("Computed descriptors from the structure:")
for k in ("formula", "mol_weight", "logp", "tpsa", "qed", "inchikey"):
    print(f"  {k:12}: {drug.descriptors.get(k)}")
print()

# 3. Test the drug under a dosing regimen.
regimen = DosingRegimen(dose=325, route="oral", interval=6, n_doses=8)
result = simulate_drug(drug, regimen, t_end=72)

# 4. Inspect the results.
print(text_report(drug, result))

# 5. (Optional) export for plotting / further analysis.
result.to_csv("demo_timecourse.csv")
print("\nWrote demo_timecourse.csv")
