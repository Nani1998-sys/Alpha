"""The full chain: protein target -> docking -> potency -> PK/PD time course.

Docks indinavir back into its own crystal structure (HIV-1 protease, PDB 1HSG),
turns the score into a potency estimate, and simulates the effect of a dosing
regimen with it. Writes an interactive 3D page of the result.

Run:  python examples/docking_workflow.py
Needs: rdkit, meeko, and the AutoDock Vina and Open Babel programs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpha import (  # noqa: E402
    DosingRegimen,
    DrugRegistry,
    dock,
    docking_available,
    fetch_pdb,
    pd_from_docking,
    save_html,
    simulate_drug,
    text_report,
    view_docking,
)

have = docking_available()
if not all(have.values()):
    missing = ", ".join(k for k, v in have.items() if not v)
    sys.exit(f"This example needs the docking stack; missing: {missing}.\n"
             "See requirements.txt for how to install it.")

# 1. The drug, with its protein target already recorded in the library.
drug = DrugRegistry().get("Indinavir")
print(f"{drug.name}: {drug.descriptors['formula']}, MW {drug.descriptors['mol_weight']}")
print(f"  target: {drug.target_name}  (PDB {drug.target_pdb}, ligand {drug.target_ligand})")

# 2. Fetch the 3D structure of that target (cached after the first run).
target = fetch_pdb(drug.target_pdb)
print(f"\n{target.identifier}: {target.title[:60]}")
print(f"  {target.method}, {target.resolution} A, {target.residue_count()} residues")

# 3. HIV-1 protease has a conserved water bridging the ligand and the enzyme
#    flaps. Keeping it makes docking markedly better.
reference = target.get_ligand(drug.target_ligand)
bridging = target.bridging_waters(reference)
water = bridging[0]["residue"] if bridging else None
print(f"  bridging water: {water} ({bridging[0]['distance_to_ligand']} A from the ligand)")

# 4. Dock.
print("\nDocking (this takes a minute or two) ...")
result = dock(target, drug.smiles, reference_ligand=drug.target_ligand,
              ligand_name=drug.name, exhaustiveness=16, n_poses=6, seed=42,
              keep_residues=[water] if water else ())
print(f"\n{'rank':>4} {'affinity':>10} {'est. Kd':>10}   contacts")
for pose in result.poses:
    print(f"{pose.rank:>4} {pose.affinity:>7.2f} kcal {pose.kd_text():>10}   "
          + ", ".join(c.label for c in pose.contacts[:6]))
print("\nScores rank poses; they are not measured affinities, and the top-scored")
print("pose is not always the right one. Inspect them all.")

# 5. Interactive 3D.
page = save_html(view_docking(result, target), "docked.html")
print(f"\n3D viewer -> {page}")

# 6. Turn the score into a potency estimate and simulate with it.
mw = drug.descriptors["mol_weight"]
pd, note = pd_from_docking(result, model="imax", E0=100, Emax=99,
                           molecular_weight=mw,
                           effect_units="% protease activity")
print(f"\n{note}")
measured = drug.pd.IC50
print(f"  docking-derived IC50: {pd.IC50:.4g} mg/L")
print(f"  measured IC50:        {measured:.4g} mg/L  "
      f"({pd.IC50 / measured:.0f}x difference)")

drug.pd = pd
result_sim = simulate_drug(drug, DosingRegimen(dose=800, route="oral",
                                               interval=8, n_doses=3), t_end=32)
print("\n" + text_report(drug, result_sim))
