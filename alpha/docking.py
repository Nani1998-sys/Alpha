"""Molecular docking: predict how a drug binds to a protein.

This wraps **AutoDock Vina** (Trott & Olson, J. Comput. Chem. 2010), the
standard open-source docking program, rather than reimplementing one. The
pipeline is:

1. Build a 3D conformer of the drug from its SMILES with RDKit (ETKDGv3
   distance geometry, then MMFF94 energy minimisation).
2. Convert it to PDBQT with Meeko, which assigns AutoDock atom types and
   finds the rotatable bonds the search will sample.
3. Prepare the receptor: strip solvent and other ligands (optionally keeping
   named structural waters or cofactors), add hydrogens at a chosen pH and
   assign charges with Open Babel.
4. Place a search box around the binding site, usually taken from a ligand
   already bound in the crystal structure.
5. Run Vina, then rebuild each predicted pose as a real molecule and list the
   residues it contacts.

**What docking can and cannot tell you.** Docking samples plausible binding
geometries and scores them with an empirical function. In a redocking test
against HIV-1 protease (see ``tests/test_dock.py``) Vina finds the
crystallographic pose to within 0.8 A, but ranks it *second* - the top-scored
pose is a different one. That is typical: treat the pose list as a set of
hypotheses to inspect, not a single answer, and treat scores as a rough
ranking. Vina's own documentation puts the scoring error around
2-3 kcal/mol, which is more than two orders of magnitude in affinity, so a
predicted Kd is an order-of-magnitude estimate at best and never a substitute
for a measured one.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .structure import Ligand, ProteinStructure

# Gas constant (kcal/mol/K) and standard temperature, for converting a
# binding free energy into a dissociation constant.
R_KCAL = 1.987204259e-3
T_STANDARD = 298.15


class DockingError(RuntimeError):
    """Raised when docking cannot be set up or run."""


# ---------------------------------------------------------------------------
# External programs
# ---------------------------------------------------------------------------
def vina_executable() -> Optional[str]:
    return shutil.which("vina")


def obabel_executable() -> Optional[str]:
    return shutil.which("obabel")


def docking_available() -> Dict[str, bool]:
    """Report which pieces of the docking stack are installed."""
    import importlib.util

    def installed(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return {
        "vina": vina_executable() is not None,
        "openbabel": obabel_executable() is not None,
        "rdkit": installed("rdkit"),
        "meeko": installed("meeko"),
    }


def require_docking_stack() -> None:
    have = docking_available()
    missing = [k for k, v in have.items() if not v]
    if not missing:
        return
    hints = {
        "vina": "AutoDock Vina: `conda install -c conda-forge vina`, "
                "`apt install autodock-vina`, or download from "
                "https://github.com/ccsb-scripps/AutoDock-Vina/releases",
        "openbabel": "Open Babel: `conda install -c conda-forge openbabel` "
                     "or `apt install openbabel`",
        "rdkit": "RDKit: `pip install rdkit`",
        "meeko": "Meeko: `pip install meeko gemmi`",
    }
    raise DockingError(
        "Docking needs these, which are not installed: "
        + ", ".join(missing) + ".\n  "
        + "\n  ".join(hints[m] for m in missing))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class Contact:
    chain: str
    resseq: int
    resname: str
    min_distance: float

    @property
    def label(self) -> str:
        return f"{self.resname}{self.resseq}{self.chain}"


@dataclass
class DockingPose:
    """One predicted binding geometry."""

    rank: int
    affinity: float                  # Vina score, kcal/mol (more negative = tighter)
    rmsd_lb: float                   # RMSD to the top pose, lower bound
    rmsd_ub: float
    pdb_block: str = ""              # the pose as PDB text
    sdf_block: str = ""              # the pose as SDF (keeps bond orders)
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contacts: List[Contact] = field(default_factory=list)

    @property
    def estimated_kd_molar(self) -> float:
        """Kd implied by the score, via dG = RT ln Kd. Order-of-magnitude only."""
        return math.exp(self.affinity / (R_KCAL * T_STANDARD))

    def kd_text(self) -> str:
        kd = self.estimated_kd_molar
        for unit, scale in (("M", 1), ("mM", 1e-3), ("uM", 1e-6), ("nM", 1e-9), ("pM", 1e-12)):
            if kd >= scale:
                return f"{kd / scale:.3g} {unit}"
        return f"{kd / 1e-15:.3g} fM"


@dataclass
class DockingResult:
    ligand_name: str
    smiles: str
    receptor_id: str
    center: Tuple[float, float, float]
    box_size: Tuple[float, float, float]
    exhaustiveness: int
    seed: int
    poses: List[DockingPose] = field(default_factory=list)
    receptor_pdb: str = ""
    kept_waters: List[str] = field(default_factory=list)
    reference_ligand: Optional[str] = None

    @property
    def best(self) -> DockingPose:
        if not self.poses:
            raise DockingError("Docking produced no poses")
        return self.poses[0]

    @property
    def best_affinity(self) -> float:
        return self.best.affinity

    def summary(self) -> Dict[str, object]:
        return {
            "ligand": self.ligand_name,
            "receptor": self.receptor_id,
            "best_affinity_kcal_per_mol": round(self.best_affinity, 2),
            "estimated_kd": self.best.kd_text(),
            "n_poses": len(self.poses),
            "box_center": tuple(round(c, 2) for c in self.center),
            "box_size": self.box_size,
            "contact_residues": [c.label for c in self.best.contacts],
        }


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def ligand_3d_from_smiles(smiles: str, seed: int = 0xF00D,
                          max_iters: int = 2000, n_confs: int = 1):
    """Build an energy-minimised 3D conformer of a molecule from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DockingError(f"Could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if n_confs == 1:
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise DockingError(
                    f"Could not generate 3D coordinates for {smiles!r}. "
                    "The structure may be over-constrained or invalid.")
    else:
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if mol.GetNumConformers() == 0:
            raise DockingError(f"Could not generate conformers for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
    except Exception:
        AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
    return mol


def ligand_to_pdbqt(mol) -> str:
    """Convert a 3D RDKit molecule to PDBQT with Meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise DockingError("Meeko could not prepare this ligand")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise DockingError(f"Meeko could not write PDBQT: {err}")
    return pdbqt


def receptor_to_pdbqt(structure: ProteinStructure, keep_waters: Sequence[str] = (),
                      keep_resnames: Sequence[str] = (),
                      keep_residues: Sequence[str] = (), ph: float = 7.4,
                      workdir: Optional[str] = None) -> Tuple[str, str]:
    """Prepare a rigid receptor PDBQT. Returns (pdbqt_text, cleaned_pdb_text)."""
    if obabel_executable() is None:
        raise DockingError(
            "Open Babel ('obabel') is needed to prepare the receptor. "
            "Install it with `conda install -c conda-forge openbabel` or "
            "`apt install openbabel`.")
    pdb_text = structure.to_pdb(
        polymer_only=True,
        keep_resnames=list(keep_resnames) + list(keep_waters),
        keep_residues=list(keep_residues))
    tmp = workdir or tempfile.mkdtemp(prefix="alpha_dock_")
    rec_pdb = os.path.join(tmp, "receptor.pdb")
    rec_pdbqt = os.path.join(tmp, "receptor.pdbqt")
    with open(rec_pdb, "w", encoding="utf-8") as fh:
        fh.write(pdb_text)
    proc = subprocess.run(
        [obabel_executable(), rec_pdb, "-O", rec_pdbqt, "-xr", "-p", str(ph)],
        capture_output=True, text=True)
    if not os.path.exists(rec_pdbqt) or os.path.getsize(rec_pdbqt) == 0:
        raise DockingError(f"Open Babel failed to prepare the receptor: {proc.stderr[:400]}")
    return open(rec_pdbqt, encoding="utf-8").read(), pdb_text


def box_from_ligand(ligand: Ligand, padding: float = 5.0,
                    min_size: float = 16.0, max_size: float = 30.0
                    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Search box centred on a bound ligand, padded to allow it to move."""
    center = ligand.centroid()
    size = tuple(min(max(e + 2 * padding, min_size), max_size) for e in ligand.extent())
    return center, size


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------
def _parse_vina_log_scores(pdbqt_text: str) -> List[Tuple[float, float, float]]:
    out = []
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK VINA RESULT"):
            parts = line.split(":")[1].split()
            out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def dock(
    structure: ProteinStructure,
    smiles: str,
    center: Optional[Sequence[float]] = None,
    box_size: Optional[Sequence[float]] = None,
    reference_ligand: Optional[str] = None,
    ligand_name: str = "ligand",
    exhaustiveness: int = 8,
    n_poses: int = 9,
    seed: int = 42,
    cpu: Optional[int] = None,
    keep_waters: Sequence[str] = (),
    keep_residues: Sequence[str] = (),
    ph: float = 7.4,
    contact_cutoff: float = 4.0,
    padding: float = 5.0,
) -> DockingResult:
    """Dock a molecule (given as SMILES) into a protein structure.

    The search box comes from ``center``/``box_size`` if given, otherwise
    from ``reference_ligand`` (the residue name of a ligand already bound in
    the structure, e.g. ``"MK1"``), otherwise from the largest bound ligand.

    ``exhaustiveness`` trades speed for thoroughness; 8 is Vina's default,
    16-32 is more reliable and proportionally slower.
    """
    require_docking_stack()

    # --- binding site ---
    ref = None
    if center is None or box_size is None:
        if reference_ligand:
            ref = structure.get_ligand(reference_ligand)
        else:
            candidates = structure.ligands()
            if not candidates:
                raise DockingError(
                    f"{structure.identifier} has no bound ligand to locate the "
                    "binding site, so you must pass center= and box_size= "
                    "explicitly. (Hint: list bound ligands with "
                    "ProteinStructure.ligands().)")
            ref = candidates[0]
        auto_center, auto_size = box_from_ligand(ref, padding=padding)
        center = center if center is not None else auto_center
        box_size = box_size if box_size is not None else auto_size
    center = tuple(float(c) for c in center)
    box_size = tuple(float(s) for s in box_size)

    workdir = tempfile.mkdtemp(prefix="alpha_dock_")
    try:
        # --- prepare both partners ---
        mol = ligand_3d_from_smiles(smiles, seed=seed)
        lig_pdbqt = ligand_to_pdbqt(mol)
        lig_path = os.path.join(workdir, "ligand.pdbqt")
        with open(lig_path, "w", encoding="utf-8") as fh:
            fh.write(lig_pdbqt)
        rec_pdbqt, rec_pdb = receptor_to_pdbqt(
            structure, keep_waters=keep_waters, keep_residues=keep_residues,
            ph=ph, workdir=workdir)
        rec_path = os.path.join(workdir, "receptor.pdbqt")
        with open(rec_path, "w", encoding="utf-8") as fh:
            fh.write(rec_pdbqt)

        # --- run vina ---
        out_path = os.path.join(workdir, "poses.pdbqt")
        cmd = [
            vina_executable(),
            "--receptor", rec_path, "--ligand", lig_path,
            "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{box_size[0]:.2f}", "--size_y", f"{box_size[1]:.2f}",
            "--size_z", f"{box_size[2]:.2f}",
            "--exhaustiveness", str(int(exhaustiveness)),
            "--num_modes", str(int(n_poses)),
            "--seed", str(int(seed)), "--out", out_path,
        ]
        if cpu:
            cmd += ["--cpu", str(int(cpu))]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path):
            raise DockingError(
                "AutoDock Vina did not produce any poses.\n"
                f"  command: {' '.join(cmd[:3])} ...\n"
                f"  stderr: {proc.stderr.strip()[:500]}\n"
                f"  stdout: {proc.stdout.strip()[-300:]}")
        poses_text = open(out_path, encoding="utf-8").read()

        # --- rebuild poses as real molecules ---
        result = DockingResult(
            ligand_name=ligand_name, smiles=smiles,
            receptor_id=structure.identifier, center=center, box_size=box_size,
            exhaustiveness=int(exhaustiveness), seed=int(seed),
            receptor_pdb=rec_pdb,
            kept_waters=list(keep_waters) + list(keep_residues),
            reference_ligand=ref.label if ref else reference_ligand,
        )
        scores = _parse_vina_log_scores(poses_text)
        mols = _poses_to_mols(out_path)
        for i, (aff, lb, ub) in enumerate(scores):
            pose = DockingPose(rank=i + 1, affinity=aff, rmsd_lb=lb, rmsd_ub=ub)
            if mols is not None and i < mols.GetNumConformers():
                pose.pdb_block, pose.sdf_block, pose.centroid = _conformer_blocks(mols, i)
                pose.contacts = _contacts(structure, mols, i, contact_cutoff)
            result.poses.append(pose)
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _poses_to_mols(pdbqt_path: str):
    """Rebuild docked poses as an RDKit molecule with one conformer per pose."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem

        pm = PDBQTMolecule.from_file(pdbqt_path, skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(pm)
        return Chem.RemoveHs(mols[0]) if mols and mols[0] is not None else None
    except Exception:
        return None


def _conformer_blocks(mol, conf_id: int):
    from rdkit import Chem

    single = Chem.Mol(mol)
    single.RemoveAllConformers()
    single.AddConformer(mol.GetConformer(conf_id), assignId=True)
    conf = single.GetConformer()
    n = single.GetNumAtoms()
    cx = sum(conf.GetAtomPosition(i).x for i in range(n)) / n
    cy = sum(conf.GetAtomPosition(i).y for i in range(n)) / n
    cz = sum(conf.GetAtomPosition(i).z for i in range(n)) / n
    return Chem.MolToPDBBlock(single), Chem.MolToMolBlock(single), (cx, cy, cz)


def _contacts(structure: ProteinStructure, mol, conf_id: int,
              cutoff: float) -> List[Contact]:
    conf = mol.GetConformer(conf_id)
    pts = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z) for i in range(mol.GetNumAtoms())]
    c2 = cutoff * cutoff
    best: Dict[Tuple[str, int, str], float] = {}
    for a in structure.atoms:
        if a.hetatm or a.element == "H":
            continue
        d2 = min((a.x - x) ** 2 + (a.y - y) ** 2 + (a.z - z) ** 2 for x, y, z in pts)
        if d2 <= c2:
            key = (a.chain, a.resseq, a.resname)
            if key not in best or d2 < best[key]:
                best[key] = d2
    out = [Contact(chain=c, resseq=s, resname=n, min_distance=round(d ** 0.5, 2))
           for (c, s, n), d in best.items()]
    return sorted(out, key=lambda c: c.min_distance)


# ---------------------------------------------------------------------------
# Linking docking to pharmacodynamics
# ---------------------------------------------------------------------------
def pd_from_docking(result: DockingResult, model: str = "emax",
                    E0: float = 0.0, Emax: float = 100.0,
                    molecular_weight: Optional[float] = None,
                    effect_units: str = "% of maximum effect"):
    """Seed a PD model with the potency implied by a docking score.

    The Vina score is converted to a dissociation constant via
    ``dG = RT ln Kd`` and used as the EC50/IC50. **This is a crude starting
    point, not a measurement**: docking scores carry errors of a few
    kcal/mol, which is orders of magnitude in Kd. Use it to explore
    "if this compound had roughly this potency, what would the time course
    look like?" and replace it with measured data as soon as you have it.

    If ``molecular_weight`` is given the potency is returned in mg/L, to match
    the concentration units the PK engine uses; otherwise it is in molar.
    """
    from .models import PDParameters

    kd_molar = result.best.estimated_kd_molar
    potency = kd_molar * molecular_weight * 1000.0 if molecular_weight else kd_molar
    note = (f"EC50/IC50 estimated from a Vina score of "
            f"{result.best_affinity:.2f} kcal/mol (Kd ~ {result.best.kd_text()}); "
            "order-of-magnitude only")
    if model in ("imax", "sigmoid_imax"):
        return PDParameters(model=model, E0=E0, Imax=Emax, IC50=potency,
                            effect_units=effect_units), note
    return PDParameters(model=model, E0=E0, Emax=Emax, EC50=potency,
                        effect_units=effect_units), note
