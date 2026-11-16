# Alpha — in silico drug testing (PK/PD)

**Alpha** is a computational pharmacology platform that runs the chain from
structure to dose:

> **chemical structure → 3D protein target → docked pose → predicted potency → PK/PD time course**

You register a drug **just by giving its chemical structure**, optionally **dock
it into a real protein structure in 3D**, attach its **pharmacokinetic (PK)** and
**pharmacodynamic (PD)** parameters, and simulate how it behaves in the body
under any dosing regimen — concentration over time, effect over time, and the
standard exposure metrics (Cmax, Tmax, AUC, half-life, accumulation).

It is built on well-established, transparent tools: compartmental PK solved
exactly with a matrix exponential, Emax/Hill PD, [RDKit](https://www.rdkit.org/)
for cheminformatics, [AutoDock Vina](https://vina.scripps.edu/) for docking,
structures from the [RCSB PDB](https://www.rcsb.org/) and
[AlphaFold DB](https://alphafold.ebi.ac.uk/), and
[3Dmol.js](https://3dmol.csb.pitt.edu/) for interactive 3D.

> ⚠️ **Scope & disclaimer.** Alpha is a **modelling and simulation / teaching
> tool**. It simulates PK/PD from parameters *you provide or fit from data*. It
> does **not** predict clinical safety or efficacy, and the structure-based
> "heuristics" it reports (e.g. oral-absorption / CNS flags) are rules of thumb,
> **not** validated ADME predictions. Docking predicts *binding geometry*, and
> its scores rank poses rather than measuring affinity: Vina's error is roughly
> 2–3 kcal/mol, over two orders of magnitude in Kd, and the top-scored pose is
> not always the correct one. Nothing here is medical advice or a substitute for
> laboratory, preclinical, or clinical testing. The bundled drug parameters are
> *illustrative, literature-approximate* figures for demos only.

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
| **Fetch a 3D protein target** | By PDB ID from the RCSB PDB (experimental), or by UniProt accession from AlphaFold DB (predicted, with pLDDT confidence). Cached on disk, so it works offline afterwards. |
| **Dock the molecule into it** | AutoDock Vina, with the ligand prepared from SMILES by RDKit + Meeko and the search box placed from a co-crystallised ligand. Reports every pose, its score, an estimated Kd, and the residues it contacts. |
| **See it in 3D** | Self-contained interactive pages: protein cartoon, binding-site residues, and each docked pose. |
| **Turn a score into potency** | Convert a docking score to an EC50/IC50 estimate and feed it straight into the PD model — clearly labelled as order-of-magnitude. |
| **Visualise** | matplotlib plots, CSV/JSON export, and an interactive web explorer with a live 3D docking panel. |

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

## Protein structures, docking and 3D

```bash
# Find a target
python -m alpha search-pdb "HIV-1 protease indinavir"

# Inspect a structure (experimental, from the RCSB PDB)
python -m alpha fetch --pdb 1HSG
python -m alpha fetch --pdb 1HSG --waters          # waters bridging ligand and protein
python -m alpha fetch --pdb 1HSG --view3d hiv.html # interactive 3D

# Or a predicted structure from AlphaFold, by UniProt accession
python -m alpha fetch --uniprot P00533 --view3d egfr.html

# Dock. Indinavir ships with its target stored, so this needs no coordinates:
python -m alpha dock --name Indinavir --keep-residues B:308 \
  --exhaustiveness 16 --view3d docked.html --sdf poses.sdf

# Any molecule into any target, with the box taken from a bound ligand:
python -m alpha dock --smiles "CC(C)Cc1ccc(C(C)C(=O)O)cc1" --ligand-name Ibuprofen \
  --pdb 4PH9 --ref-ligand IBP --keep-waters HEM --view3d cox2.html
```

From Python:

```python
from alpha import fetch_pdb, dock, view_docking, save_html, pd_from_docking

target = fetch_pdb("1HSG")                       # HIV-1 protease + indinavir
water  = target.bridging_waters(target.get_ligand("MK1"))[0]["residue"]
result = dock(target, INDINAVIR_SMILES, reference_ligand="MK1",
              keep_residues=[water], exhaustiveness=16)

print(result.best_affinity, result.best.kd_text())          # -10.63 kcal/mol, 16.1 nM
print([c.label for c in result.best.contacts][:4])          # ASP25B, GLY48A, ASP29A, ASP25A
save_html(view_docking(result, target), "docked.html")

pd, note = pd_from_docking(result, model="imax", E0=100, Emax=99,
                           molecular_weight=613.8)          # -> PD parameters
```

**How the docking works.** The molecule is built in 3D from SMILES (RDKit
ETKDGv3 + MMFF94), typed and given its rotatable bonds by Meeko, and written as
PDBQT. The receptor is stripped of solvent and other ligands — optionally
keeping a named structural water or cofactor — protonated at a chosen pH and
charged by Open Babel. The search box is centred on a co-crystallised ligand and
padded. AutoDock Vina then searches, and each returned pose is rebuilt as a real
molecule so its contacting residues can be listed.

**How good is it?** `tests/test_docking.py` redocks indinavir into its own
crystal structure (1HSG) and asserts that a pose within 2 Å of the
crystallographic geometry comes back. It does — Vina reproduces the pose to
**0.8 Å**. But it ranked that pose *second*, not first, and the estimated Kd
(16 nM) is about 30× weaker than the measured Ki (0.56 nM). That is normal, and
it is why the tool shows you every pose with its contacts instead of a single
answer.

**AlphaFold structures are predictions.** Every residue carries a pLDDT
confidence score, which Alpha reports and the 3D viewer colours with AlphaFold's
own scheme. Regions below 70 are unreliable and below 50 are often intrinsically
disordered; docking into them is not meaningful.

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
  structure.py  fetch/parse protein structures (RCSB PDB, AlphaFold DB)
  docking.py    AutoDock Vina docking + score-to-potency conversion
  viewer3d.py   interactive 3D viewers (3Dmol.js)
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

145 tests: analytical validation of the PK engine, PD models, the SMILES parser
(checked against RDKit on 47 real drugs, plus 17 malformed inputs it must
reject), structure parsing and pocket detection, docking maths and the 3D
viewers, the drug registry and the command line.

AUC(0–∞) is computed by mass balance from the drug actually delivered inside
the simulated window, so doses scheduled after `t_end` and the unfinished part
of an infusion do not inflate it.

The redocking validation actually runs AutoDock Vina and takes about a minute,
so it is marked `slow` and excluded by default:

```bash
python -m pytest -q                 # fast tests
python -m pytest -q -m slow         # the real redocking run
```

## License

MIT.
