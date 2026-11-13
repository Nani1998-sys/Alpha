"""Populate alpha/data/drugs.json with a handful of well-known drugs.

PK/PD values here are *illustrative, literature-approximate* figures for
demonstration and teaching. They are NOT a clinical reference. Units:
clearance L/h, volumes L, rate constants 1/h, concentrations mg/L, time h.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha.models import PDParameters, PKParameters  # noqa: E402
from alpha.registry import DEFAULT_DB, DrugRegistry  # noqa: E402

SEED = [
    dict(
        name="Indinavir",
        smiles="CC(C)(C)NC(=O)[C@@H]1CN(Cc2cccnc2)CCN1C[C@@H](O)C[C@@H](Cc1ccccc1)"
               "C(=O)N[C@H]1c2ccccc2C[C@H]1O",
        pk=PKParameters(CL=50.0, Vc=130.0, ka=2.0, F=0.65),
        pd=PDParameters(model="imax", E0=100, Imax=99, IC50=3.4e-4,
                        effect_units="% protease activity"),
        target_name="HIV-1 protease",
        target_pdb="1HSG",
        target_ligand="MK1",
        notes="HIV protease inhibitor. Co-crystallised in PDB 1HSG, so it can be "
              "redocked into its own target. Measured Ki ~0.56 nM.",
    ),
    dict(
        name="Caffeine",
        smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        pk=PKParameters(CL=6.0, Vc=36.0, ka=4.0, F=0.99),
        pd=PDParameters(model="emax", E0=0, Emax=10, EC50=5.0,
                        effect_units="alertness (0-10)"),
        notes="CNS stimulant. Illustrative PK (~5 h half-life).",
    ),
    dict(
        name="Ibuprofen",
        target_name="Cyclooxygenase-2 (COX-2)",
        target_pdb="4PH9",
        target_ligand="IBP",
        smiles="CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        pk=PKParameters(CL=3.5, Vc=10.0, ka=1.5, F=0.9),
        pd=PDParameters(model="sigmoid_emax", E0=0, Emax=100, EC50=10.0, hill=1.5,
                        effect_units="% analgesia"),
        notes="NSAID. Short half-life (~2 h). Co-crystallised in PDB 4PH9 (ligand IBP); "
              "COX-2 needs its heme cofactor, so dock with --keep-waters HEM.",
    ),
    dict(
        name="Acetaminophen",
        smiles="CC(=O)Nc1ccc(O)cc1",
        pk=PKParameters(CL=21.0, Vc=63.0, ka=3.0, F=0.88),
        pd=PDParameters(model="emax", E0=0, Emax=100, EC50=8.0,
                        effect_units="% antipyresis"),
        notes="Analgesic/antipyretic (paracetamol).",
    ),
    dict(
        name="Theophylline",
        target_name="Phosphodiesterase / adenosine receptor",
        smiles="Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        pk=PKParameters(CL=2.8, Vc=35.0, ka=3.5, F=1.0),
        pd=PDParameters(model="emax", E0=0, Emax=100, EC50=10.0,
                        effect_units="% bronchodilation"),
        notes="Narrow therapeutic index; target ~10-20 mg/L.",
    ),
    dict(
        name="Metformin",
        target_name="AMPK pathway (indirect)",
        smiles="CN(C)C(=N)NC(=N)N",
        pk=PKParameters(CL=30.0, Vc=90.0, ka=1.0, F=0.55),
        pd=PDParameters(model="emax", E0=0, Emax=40, EC50=1.0,
                        effect_units="% HGP suppression"),
        notes="Renally cleared antihyperglycemic.",
    ),
    dict(
        name="Warfarin",
        target_name="Vitamin K epoxide reductase (VKORC1)",
        smiles="CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O",
        pk=PKParameters(CL=0.2, Vc=8.0, ka=1.5, F=0.95),
        pd=PDParameters(model="imax", E0=100, Imax=70, IC50=1.5,
                        effect_units="% clotting activity", ke0=0.02),
        notes="Anticoagulant; very long half-life, delayed effect (ke0).",
    ),
    dict(
        name="Propofol",
        smiles="CC(C)c1cccc(C(C)C)c1O",
        pk=PKParameters(CL=108.0, Vc=20.0, Q=40.0, Vp=120.0),
        pd=PDParameters(model="sigmoid_emax", E0=0, Emax=100, EC50=3.0, hill=3,
                        effect_units="% hypnosis", ke0=0.5),
        notes="IV anesthetic; two-compartment; rapid onset/offset.",
    ),
    dict(
        name="Gentamicin",
        smiles="CNC1CCC(N)C(OC2OC(CN)CCC2N)C1OC1OC(C(C)(O)CO)C(O)C1NC",
        pk=PKParameters(CL=5.4, Vc=18.0),
        pd=PDParameters(model="emax", E0=0, Emax=100, EC50=2.0,
                        effect_units="% bactericidal"),
        notes="Aminoglycoside (IV); concentration-dependent killing.",
    ),
]


def main():
    # Start fresh.
    if os.path.exists(DEFAULT_DB):
        os.remove(DEFAULT_DB)
    reg = DrugRegistry(DEFAULT_DB)
    for entry in SEED:
        reg.add_drug(overwrite=True, **entry)
        print(f"  seeded {entry['name']}")
    reg.save()
    print(f"\nWrote {len(reg)} drugs to {DEFAULT_DB}")


if __name__ == "__main__":
    main()
