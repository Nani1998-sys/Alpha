"""Alpha - a computational platform for testing drugs in silico.

Register a drug from its chemical structure, attach pharmacokinetic (PK) and
pharmacodynamic (PD) parameters, and simulate its concentration- and
effect-time course under any dosing regimen.

Quick start
-----------
>>> from alpha import DrugRegistry, PKParameters, PDParameters, DosingRegimen, simulate_drug
>>> reg = DrugRegistry(path="my_drugs.json")
>>> reg.add_drug(
...     "MyDrug",
...     smiles="CC(=O)Oc1ccccc1C(=O)O",
...     pk=PKParameters(CL=10, Vc=30, ka=1.2, F=0.7),
...     pd=PDParameters(model="emax", E0=0, Emax=100, EC50=2.0, effect_units="%"),
... )
>>> result = simulate_drug(reg.get("MyDrug"),
...                        DosingRegimen(dose=100, route="oral",
...                                      interval=12, n_doses=5))
>>> result.metrics["Cmax"]
"""
from __future__ import annotations

from .chem import analyze_structure, have_rdkit
from .models import Drug, DosingRegimen, PDParameters, PKParameters
from .pd import effect_from_concentration, simulate_pd
from .pk import PKResult, simulate_pk
from .registry import DrugRegistry
from .report import plot_simulation, text_report
from .simulate import SimulationResult, compute_nca, simulate_drug

__version__ = "0.1.0"

__all__ = [
    "Drug",
    "DosingRegimen",
    "PKParameters",
    "PDParameters",
    "PKResult",
    "SimulationResult",
    "DrugRegistry",
    "analyze_structure",
    "have_rdkit",
    "simulate_pk",
    "simulate_pd",
    "effect_from_concentration",
    "simulate_drug",
    "compute_nca",
    "plot_simulation",
    "text_report",
    "__version__",
]
