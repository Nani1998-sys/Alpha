"""End-to-end: structure -> registry -> simulate -> metrics."""
import os

from alpha.models import DosingRegimen, PDParameters, PKParameters
from alpha.registry import DrugRegistry
from alpha.simulate import simulate_drug


def test_full_workflow(tmp_path):
    db = os.path.join(tmp_path, "drugs.json")
    reg = DrugRegistry(db)
    reg.add_drug(
        "Caffeine",
        smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        pk=PKParameters(CL=6.0, Vc=36.0, ka=4.0, F=1.0),
        pd=PDParameters(model="emax", E0=0, Emax=10, EC50=5.0,
                        effect_units="alertness score"),
    )
    reg.save()

    drug = reg.get("Caffeine")
    regimen = DosingRegimen(dose=200, route="oral", interval=6, n_doses=4)
    result = simulate_drug(drug, regimen, t_end=48)

    m = result.metrics
    assert m["Cmax"] > 0
    assert m["Tmax"] > 0
    assert "terminal_half_life" in m
    assert "accumulation_ratio" in m  # multiple dosing
    assert result.effect is not None
    assert result.effect.max() <= 10.0 + 1e-6  # bounded by Emax + E0
    rows = result.to_rows()
    assert len(rows) > 100
    assert "effect" in rows[0]
