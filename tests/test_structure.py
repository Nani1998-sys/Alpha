"""Protein structure parsing and geometry.

These run offline against ``tests/data/1hsg_site.pdb``: the real binding-site
region of PDB entry 1HSG (HIV-1 protease with the inhibitor indinavir, residue
name MK1), trimmed to residues within 8 A of the ligand.
"""
import os

import pytest

from alpha.structure import (
    ProteinStructure,
    StructureError,
    fetch_pdb,
    parse_pdb,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "1hsg_site.pdb")


@pytest.fixture(scope="module")
def site() -> ProteinStructure:
    return parse_pdb(open(FIXTURE, encoding="utf-8").read(), identifier="1HSG-site")


def test_parses_atoms_and_metadata(site):
    assert len(site.atoms) > 400
    assert site.resolution == 1.9 or site.resolution == 2.0  # REMARK 2 value
    assert "IMMUNODEFICIENCY" in site.title.upper()
    assert set(site.chains) == {"A", "B"}


def test_separates_polymer_from_ligands(site):
    names = {lig.resname for lig in site.ligands()}
    assert names == {"MK1"}, "waters and additives should be excluded by default"
    assert len(site.waters()) == 2
    assert {l.resname for l in site.ligands(include_solvent=True)} == {"MK1", "HOH"}


def test_ligand_geometry(site):
    mk1 = site.get_ligand("MK1")
    assert mk1.n_heavy_atoms == 45
    cx, cy, cz = mk1.centroid()
    assert (round(cx, 1), round(cy, 1), round(cz, 1)) == (13.1, 22.5, 5.6)
    assert all(4.0 < e < 20.0 for e in mk1.extent())
    assert mk1.to_pdb().count("HETATM") == 45


def test_missing_ligand_error_lists_what_is_there(site):
    with pytest.raises(StructureError) as exc:
        site.get_ligand("ZZZ")
    assert "MK1" in str(exc.value)


def test_pocket_residues_include_catalytic_aspartates(site):
    """HIV-1 protease cleaves via Asp25 from each half of the dimer; any
    correct pocket detection must find both."""
    pocket = site.pocket_residues(site.get_ligand("MK1"), radius=4.5)
    found = {(r["chain"], r["resseq"], r["resname"]) for r in pocket}
    assert ("A", 25, "ASP") in found
    assert ("B", 25, "ASP") in found
    assert ("A", 50, "ILE") in found      # flap residue
    assert all(r["min_distance"] <= 4.5 for r in pocket)
    assert not any(r["resname"] == "HOH" for r in pocket)


def test_residues_near_point(site):
    center = site.get_ligand("MK1").centroid()
    near = site.residues_near(center, radius=6.0)
    far = site.residues_near(center, radius=10.0)
    assert 0 < len(near) < len(far)
    assert near == sorted(near, key=lambda r: r["min_distance"])


def test_sequence_is_protein_only(site):
    seq = site.sequence("A")
    assert seq and set(seq) <= set("ACDEFGHIKLMNPQRSTVWYXUO")
    assert len(seq) == site.residue_count("A")


def test_to_pdb_filters(site):
    full = site.to_pdb()
    polymer = site.to_pdb(polymer_only=True)
    with_water = site.to_pdb(polymer_only=True, keep_resnames=["HOH"])
    assert "MK1" in full and "MK1" not in polymer
    assert "HOH" not in polymer and "HOH" in with_water
    assert polymer.rstrip().endswith("END")


def test_alternate_locations_and_models_are_collapsed():
    pdb = (
        "ATOM      1  N  AALA A   1      11.000  11.000  11.000  0.50 20.00           N\n"
        "ATOM      2  N  BALA A   1      12.000  12.000  12.000  0.50 20.00           N\n"
        "MODEL        2\n"
        "ATOM      3  N   GLY A   2      99.000  99.000  99.000  1.00 20.00           N\n"
        "ENDMDL\n"
    )
    s = parse_pdb(pdb)
    assert len(s.atoms) == 1, "keep only altloc A and only model 1"
    assert s.atoms[0].altloc == "A"


def test_empty_structure_raises():
    with pytest.raises(StructureError):
        parse_pdb("HEADER    nothing here\nEND\n")


def test_plddt_summary_only_for_predictions(site):
    assert site.plddt_summary() is None          # experimental structure
    pdb = "".join(
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
        f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{95.0 - i:6.2f}           C\n"
        for i in range(1, 11))
    pred = parse_pdb(pdb, identifier="AF-TEST", source="AlphaFold")
    assert pred.is_predicted
    s = pred.plddt_summary()
    assert s["mean_plddt"] == pytest.approx(89.5, abs=0.1)
    assert 0 <= s["very_high_>90"] <= 1
    assert sum(s[k] for k in ("very_high_>90", "confident_70_90",
                              "low_50_70", "very_low_<50")) == pytest.approx(1.0, abs=0.01)


def test_bad_pdb_id_is_rejected_before_any_download():
    with pytest.raises(StructureError) as exc:
        fetch_pdb("P00533", cache_dir=None)      # a UniProt accession, not a PDB ID
    assert "fetch_alphafold" in str(exc.value)


def test_cache_is_used_without_network(tmp_path):
    """A cached structure must load with no network access at all."""
    cached = os.path.join(tmp_path, "1ABC.pdb")
    with open(cached, "w", encoding="utf-8") as fh:
        fh.write(open(FIXTURE, encoding="utf-8").read())
    s = fetch_pdb("1ABC", cache_dir=str(tmp_path), with_metadata=False)
    assert s.identifier == "1ABC"
    assert len(s.atoms) > 400
