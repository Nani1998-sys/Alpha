"""Docking: box placement, score conversion, PD seeding, and a real
redocking validation against a crystal structure.

The fast tests need no external programs. The redocking test actually runs
AutoDock Vina and is marked ``slow``; run it with::

    python -m pytest -m slow tests/test_docking.py
"""
import os

import pytest

from alpha.docking import (
    DockingError,
    DockingPose,
    box_from_ligand,
    docking_available,
    pd_from_docking,
    require_docking_stack,
)
from alpha.structure import parse_pdb

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "1hsg_site.pdb")
INDINAVIR = ("CC(C)(C)NC(=O)[C@@H]1CN(Cc2cccnc2)CCN1C[C@@H](O)C[C@@H](Cc1ccccc1)"
             "C(=O)N[C@H]1c2ccccc2C[C@H]1O")

HAVE = docking_available()
needs_stack = pytest.mark.skipif(
    not all(HAVE.values()),
    reason=f"docking stack incomplete: {HAVE}")


@pytest.fixture(scope="module")
def site():
    return parse_pdb(open(FIXTURE, encoding="utf-8").read(), identifier="1HSG-site")


# --- box placement ---------------------------------------------------------
def test_box_is_centred_on_the_ligand_and_padded(site):
    lig = site.get_ligand("MK1")
    center, size = box_from_ligand(lig, padding=5.0)
    assert center == pytest.approx(lig.centroid())
    for dim, extent in zip(size, lig.extent()):
        assert dim >= extent, "box must at least contain the ligand"
        assert dim <= 30.0, "box is capped so the search stays tractable"


def test_box_respects_minimum_size(site):
    """A tiny ligand still gets a box big enough to search."""
    lig = site.get_ligand("MK1")
    lig.atoms = lig.atoms[:2]                      # shrink it artificially
    _, size = box_from_ligand(lig, padding=0.0, min_size=16.0)
    assert all(s >= 16.0 for s in size)


# --- score -> affinity -----------------------------------------------------
@pytest.mark.parametrize("affinity,expected_kd_molar", [
    (-10.0, 4.5e-8),      # ~45 nM
    (-7.0, 7.1e-6),       # ~7 uM
    (-12.6, 5.7e-10),     # ~0.6 nM
])
def test_score_converts_to_kd(affinity, expected_kd_molar):
    pose = DockingPose(rank=1, affinity=affinity, rmsd_lb=0, rmsd_ub=0)
    assert pose.estimated_kd_molar == pytest.approx(expected_kd_molar, rel=0.05)


def test_kd_text_picks_sensible_units():
    assert DockingPose(1, -10.0, 0, 0).kd_text().endswith("nM")
    assert DockingPose(1, -5.0, 0, 0).kd_text().endswith("uM")
    assert DockingPose(1, -2.0, 0, 0).kd_text().endswith("mM")


def test_tighter_score_means_smaller_kd():
    assert (DockingPose(1, -11.0, 0, 0).estimated_kd_molar
            < DockingPose(1, -8.0, 0, 0).estimated_kd_molar)


# --- PD seeding ------------------------------------------------------------
def _fake_result(affinity=-10.0):
    from alpha.docking import DockingResult

    r = DockingResult(ligand_name="X", smiles="CCO", receptor_id="TEST",
                      center=(0, 0, 0), box_size=(20, 20, 20),
                      exhaustiveness=8, seed=1)
    r.poses.append(DockingPose(rank=1, affinity=affinity, rmsd_lb=0, rmsd_ub=0))
    return r


def test_pd_from_docking_sets_potency_in_molar():
    pd, note = pd_from_docking(_fake_result(-10.0), model="emax")
    assert pd.model == "emax"
    assert pd.EC50 == pytest.approx(4.5e-8, rel=0.05)
    assert "order-of-magnitude" in note
    pd.validate()


def test_pd_from_docking_converts_to_mg_per_litre():
    """The PK engine works in mg/L, so a molecular weight converts the units."""
    mw = 613.8
    pd, _ = pd_from_docking(_fake_result(-10.0), model="emax", molecular_weight=mw)
    assert pd.EC50 == pytest.approx(4.5e-8 * mw * 1000, rel=0.05)


def test_pd_from_docking_inhibitory_model():
    pd, _ = pd_from_docking(_fake_result(-9.0), model="imax", E0=100, Emax=80)
    assert pd.model == "imax" and pd.IC50 is not None and pd.EC50 is None
    assert pd.E0 == 100 and pd.Imax == 80
    pd.validate()


def test_missing_stack_error_names_what_to_install(monkeypatch):
    monkeypatch.setattr("alpha.docking.vina_executable", lambda: None)
    monkeypatch.setattr("alpha.docking.obabel_executable", lambda: None)
    with pytest.raises(DockingError) as exc:
        require_docking_stack()
    msg = str(exc.value)
    assert "vina" in msg.lower() and "openbabel" in msg.lower()
    assert "install" in msg.lower() or "conda" in msg.lower()


# --- ligand 3D generation --------------------------------------------------
@pytest.mark.skipif(not HAVE["rdkit"], reason="needs RDKit")
def test_ligand_3d_from_smiles_produces_real_coordinates():
    from alpha.docking import ligand_3d_from_smiles

    mol = ligand_3d_from_smiles("CC(=O)Oc1ccccc1C(=O)O", seed=7)
    assert mol.GetNumConformers() == 1
    conf = mol.GetConformer()
    zs = {round(conf.GetAtomPosition(i).z, 3) for i in range(mol.GetNumAtoms())}
    assert len(zs) > 1, "coordinates must be genuinely 3D, not flat"
    # a C-C single bond is ~1.5 A; check the geometry is physically sane
    from rdkit.Chem import rdMolTransforms
    bond = mol.GetBondWithIdx(0)
    d = rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    assert 1.0 < d < 2.0


@pytest.mark.skipif(not HAVE["rdkit"], reason="needs RDKit")
def test_bad_smiles_rejected():
    from alpha.docking import ligand_3d_from_smiles

    with pytest.raises(DockingError):
        ligand_3d_from_smiles("not a molecule (((")


# --- the real thing --------------------------------------------------------
@pytest.mark.slow
@needs_stack
def test_redocking_recovers_the_crystal_pose(site):
    """Redock indinavir into HIV-1 protease and compare with the crystal pose.

    Docking is judged by whether it reproduces a known binding mode. Vina
    finds this one to well under 2 A, but does not always rank it first, so
    the check is that a near-native pose appears among the top poses - which
    is also how the result should be read in practice.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign
    from alpha.docking import dock
    RDLogger.DisableLog("rdApp.*")

    result = dock(site, INDINAVIR, reference_ligand="MK1", ligand_name="Indinavir",
                  exhaustiveness=8, n_poses=9, seed=42, cpu=4)

    assert result.poses, "docking produced no poses"
    assert -14 < result.best_affinity < -6, (
        f"implausible score {result.best_affinity} kcal/mol for a known "
        "nanomolar inhibitor")
    # every pose should sit in the pocket, touching the catalytic aspartates
    labels = {c.label for c in result.best.contacts}
    assert any(l.startswith("ASP25") for l in labels), (
        f"best pose does not contact the catalytic Asp25: {sorted(labels)}")

    template = Chem.MolFromSmiles(INDINAVIR)
    xtal = AllChem.AssignBondOrdersFromTemplate(
        template, Chem.MolFromPDBFile(FIXTURE, removeHs=True, sanitize=False,
                                      flavor=0, proximityBonding=True))
    # fall back to the ligand-only block if the whole-file read is ambiguous
    if xtal.GetNumAtoms() != template.GetNumAtoms():
        lig_pdb = site.get_ligand("MK1").to_pdb()
        xtal = AllChem.AssignBondOrdersFromTemplate(
            template, Chem.MolFromPDBBlock(lig_pdb, removeHs=True, sanitize=False))

    rmsds = []
    for pose in result.poses:
        if not pose.sdf_block:
            continue
        m = Chem.RemoveHs(Chem.MolFromMolBlock(pose.sdf_block, sanitize=False))
        try:
            rmsds.append(rdMolAlign.CalcRMS(m, xtal))
        except RuntimeError:
            continue
    assert rmsds, "could not compare any pose with the crystal structure"
    assert min(rmsds) < 2.0, (
        f"no pose reproduced the crystal binding mode; best RMSD "
        f"{min(rmsds):.2f} A (all: {[round(r, 2) for r in rmsds]})")
