# Alpha — in silico drug testing (PK/PD)

**Alpha** is a computational pharmacology platform. You register a drug **just by
giving its chemical structure**, attach its **pharmacokinetic (PK)** and
**pharmacodynamic (PD)** parameters, and simulate how the drug behaves in the
body under any dosing regimen — concentration over time, effect over time, and
the standard exposure metrics (Cmax, Tmax, AUC, half-life, accumulation).

It is built on well-established, transparent pharmacometric models (compartmental
PK ODEs + Emax/Hill PD) and on [RDKit](https://www.rdkit.org/) for cheminformatics.

> ⚠️ **Scope & disclaimer.** Alpha is a **modelling and simulation / teaching
> tool**. It simulates PK/PD from parameters *you provide or fit from data*. It
> does **not** predict clinical safety or efficacy, and the structure-based
> "heuristics" it reports (e.g. oral-absorption / CNS flags) are rules of thumb,
> **not** validated ADME predictions. Nothing here is medical advice or a
> substitute for laboratory, preclinical, or clinical testing. The bundled drug
> parameters are *illustrative, literature-approximate* figures for demos only.

---

## What it does

| Capability | How |
|---|---|
| **Insert a new drug by chemical structure** | Give a SMILES string; Alpha parses it with RDKit and auto-computes MW, logP, TPSA, H-bond donors/acceptors, rotatable bonds, QED, InChIKey, and Lipinski/Veber drug-likeness. Invalid structures are rejected with the reason and position. |
| **Provide PK** | Clearance + volume parameterisation: 1- or 2-compartment; IV bolus, IV infusion, or first-order oral absorption with bioavailability (F) and lag time. |
| **Provide PD** | Emax, sigmoidal Emax (Hill), linear, log-linear, inhibitory Imax, and an optional effect-compartment (`ke0`) for delayed effect / hysteresis. |
| **Test dosing regimens** | Single or multiple dosing; get the full concentration- and effect-time course. |
| **Get exposure metrics** | Cmax, Tmax, AUC(last)/AUC(∞), terminal half-life, accumulation ratio, average steady-state concentration. |
| **Store a library of drugs** | A simple JSON database you can grow, query, and version-control. |
| **Visualise** | matplotlib plots, CSV/JSON export, and an interactive web explorer. |

---

## Install

```bash
pip install -r requirements.txt     # numpy, rdkit, matplotlib
# or, as a package:
pip install -e .[all]
```

NumPy is the only hard requirement. Without RDKit, Alpha still reads structures
with its built-in SMILES parser (formula, molecular weight, atom and ring
counts, validated against RDKit). Without matplotlib, everything works except
`--plot`.

## Quick start (Python)

```python
from alpha import DrugRegistry, PKParameters, PDParameters, DosingRegimen, simulate_drug

reg = DrugRegistry("my_drugs.json")

# Insert a new molecule *by its chemical structure*:
reg.add_drug(
    "MyDrug",
    smiles="CC(=O)Oc1ccccc1C(=O)O",                     # chemical structure
    pk=PKParameters(CL=39, Vc=10, ka=2.0, F=0.68),      # pharmacokinetics
    pd=PDParameters(model="emax", E0=0, Emax=100, EC50=5.0, effect_units="%"),  # pharmacodynamics
)
reg.save()

# Test it under a dosing regimen:
result = simulate_drug(reg.get("MyDrug"),
                       DosingRegimen(dose=325, route="oral", interval=6, n_doses=8))

print(result.metrics["Cmax"], result.metrics["terminal_half_life"])
```

See [`examples/quickstart.py`](examples/quickstart.py).

## Quick start (command line)

```bash
# Analyse a chemical structure (descriptors + drug-likeness):
python -m alpha describe --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Insert a new drug molecule by structure, with PK + PD:
python -m alpha add --name Losartan \
  --smiles "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1" \
  --cl 50 --vc 34 --ka 1.1 --f 0.33 \
  --pd sigmoid_emax --e0 120 --emax 30 --ec50 0.2 --hill 1.2 --effect-units "mmHg SBP drop"

# List / inspect the library:
python -m alpha list
python -m alpha show --name Losartan

# Test a dosing regimen (writes a plot, CSV and JSON):
python -m alpha simulate --name Theophylline --dose 300 --route oral \
  --interval 8 --ndoses 6 --tend 72 --plot theo.png --csv theo.csv
```

A seed library of 8 well-known drugs ships in `alpha/data/drugs.json`
(rebuild with `python scripts/build_seed_db.py`). To keep your own library
elsewhere, pass `--db my_drugs.json` (before or after the command) or set the
`ALPHA_DB` environment variable.

## The models (what's under the hood)

**Pharmacokinetics** — amounts are integrated as ODEs and concentration is
`C = A_central / Vc`:

```
dA_depot/dt      = -ka·A_depot
dA_central/dt    =  ka·A_depot + R_infusion(t) − k10·A_central − k12·A_central + k21·A_peripheral
dA_peripheral/dt =  k12·A_central − k21·A_peripheral
```

with `k10 = CL/Vc`, `k12 = Q/Vc`, `k21 = Q/Vp`. Between doses the system is
linear with constant input, so it is solved **exactly** with a matrix
exponential, `[y(t+h); 1] = expm(M·h)·[y(t); 1]`, segment by segment between
dose and infusion events. This is accurate to machine precision, stable for any
rate constants (very fast absorption or elimination, very long simulations),
and integrates AUC exactly. The test suite checks it against the closed-form
solutions for IV bolus, oral (with lag time), infusion, repeat dosing and the
two-compartment bi-exponential. The web explorer uses the identical method and
gives the same numbers.

**Exposure metrics.** Cmax and Tmax come from the simulated curve (at an IV
bolus both the trough and the peak are recorded). `AUC_last` is exact;
`AUC_inf = F·total dose / CL` (mass balance, exact for linear PK).
`model_half_life` is ln 2 / β, the terminal half-life (ln 2 / k10 for one
compartment); two-compartment drugs also report `distribution_half_life`
(ln 2 / α). `terminal_half_life` is the curve-based estimate, the way
non-compartmental analysis reads it from sampled data. The accumulation ratio
uses the slowest process: β, or ka when absorption is slower (flip-flop).

**Pharmacodynamics** — effect from (effect-site) concentration:

```
Emax:          E = E0 + Emax·C / (EC50 + C)
Sigmoid Emax:  E = E0 + Emax·C^h / (EC50^h + C^h)
Imax:          E = E0 − Imax·C / (IC50 + C)
Effect compartment:  dCe/dt = ke0·(C − Ce)   (drives E, giving hysteresis)
```

## Units

Default, internally-consistent set: clearance **L/h**, volumes **L**, rate
constants **1/h**, dose **mg**, time **h** → concentration **mg/L**. Any
consistent system works.

## Project layout

```
alpha/
  chem.py       structure → descriptors (RDKit) + drug-likeness
  models.py     Drug, PKParameters, PDParameters, DosingRegimen
  pk.py         compartmental PK ODE engine
  pd.py         PD models + effect compartment
  simulate.py   orchestration + NCA metrics
  registry.py   JSON drug database (add by structure)
  report.py     text reports + matplotlib plots
  cli.py        command-line interface
  data/         seed drug library
examples/       quickstart script
scripts/        seed-database builder
tests/          pytest suite (incl. analytical validation)
web/            interactive PK/PD explorer (open in a browser)
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

101 tests: analytical validation of the PK engine, PD models, the SMILES parser
(checked against RDKit on 47 real drugs, plus 17 malformed inputs it must
reject), the drug registry and the command line.

## License

MIT.
