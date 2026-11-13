"""The generated 3D viewer pages: structure, data embedding and safety."""
import json
import os
import re

import pytest

from alpha.docking import DockingPose, DockingResult
from alpha.structure import parse_pdb
from alpha.viewer3d import CDN_3DMOL, save_html, view_docking, view_structure

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "1hsg_site.pdb")


@pytest.fixture(scope="module")
def site():
    return parse_pdb(open(FIXTURE, encoding="utf-8").read(), identifier="1HSG-site")


def _embedded_data(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m, "the page must embed its coordinates as a JSON literal"
    return json.loads(m.group(1))


def test_structure_page_is_self_contained(site):
    html = view_structure(site)
    assert html.startswith("<!doctype html>")
    assert html.count("<script") == 2          # the library, and the page's own
    assert CDN_3DMOL in html
    data = _embedded_data(html)
    assert "ATOM" in data["receptor"]
    assert data["entities"], "the bound ligand should be rendered"
    assert data["pocket"], "pocket residues should be marked"
    assert {"chain", "resseq"} == set(data["pocket"][0])


def test_structure_page_reports_the_metadata(site):
    html = view_structure(site)
    assert "1HSG-site" in html
    assert "X-Ray" in html or "Resolution" in html
    assert "Cartoon" in html and "Spin" in html


def test_predicted_structures_get_confidence_colouring_and_a_warning():
    pdb = "".join(
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
        f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{95.0 - i * 5:6.2f}           C\n"
        for i in range(1, 11))
    pred = parse_pdb(pdb, identifier="AF-TEST", source="AlphaFold")
    html = view_structure(pred)
    data = _embedded_data(html)
    assert data["colorByPlddt"] is True
    assert len(data["plddtBands"]) == 4
    assert "predicted" in html.lower()
    assert "pLDDT" in html
    # experimental structures must not claim confidence colouring
    exp = parse_pdb(pdb, identifier="X", source="RCSB")
    assert _embedded_data(view_structure(exp))["colorByPlddt"] is False


def _docking_result(site, n=3):
    lig = site.get_ligand("MK1")
    r = DockingResult(ligand_name="Indinavir", smiles="CCO",
                      receptor_id=site.identifier, center=lig.centroid(),
                      box_size=(20, 20, 20), exhaustiveness=8, seed=42,
                      receptor_pdb=site.to_pdb(polymer_only=True),
                      reference_ligand=lig.label)
    for i in range(n):
        r.poses.append(DockingPose(
            rank=i + 1, affinity=-10.0 + i * 0.4, rmsd_lb=i * 1.1, rmsd_ub=i * 1.4,
            pdb_block=lig.to_pdb(), centroid=lig.centroid()))
    return r


def test_docking_page_shows_every_pose(site):
    html = view_docking(_docking_result(site, n=4))
    data = _embedded_data(html)
    assert len(data["entities"]) == 4
    assert data["entityMode"] == "one", "poses are shown one at a time"
    assert html.count("<option") == 4
    assert "-10.00 kcal/mol" in html
    assert "Indinavir" in html and "1HSG-site" in html


def test_docking_page_states_the_limitations(site):
    html = view_docking(_docking_result(site)).lower()
    assert "not measured affinities" in html
    assert "kcal/mol" in html
    assert "not always the correct one" in html, (
        "the page must warn that the top pose can be wrong")


def test_docking_page_needs_a_receptor(site):
    r = _docking_result(site)
    r.receptor_pdb = ""
    with pytest.raises(ValueError, match="receptor"):
        view_docking(r)


def test_html_is_escaped_against_injection(site):
    r = _docking_result(site)
    r.ligand_name = '<img src=x onerror="alert(1)">'
    html = view_docking(r)
    # the angle brackets are neutralised, so no element is ever created; the
    # attribute text surviving as inert plain text is fine
    assert "<img" not in html
    assert "&lt;img" in html
    assert 'onerror="alert(1)"' not in html


def test_json_payload_cannot_break_out_of_the_script_tag(site):
    r = _docking_result(site)
    r.poses[0].pdb_block = "</script><script>alert(1)</script>"
    html = view_docking(r)
    # only a literal "</script" can end the inline script, and it must be escaped
    assert html.count("</script") == 2, "only the two real closing tags"
    assert "<\\/script>" in html


def test_local_js_option(tmp_path, site):
    js = os.path.join(tmp_path, "3Dmol-min.js")
    open(js, "w").write("// stub")
    html = view_structure(site, local_js=js)
    assert js in html and CDN_3DMOL not in html
    with pytest.raises(FileNotFoundError):
        view_structure(site, local_js=os.path.join(tmp_path, "missing.js"))


def test_save_html_returns_absolute_path(tmp_path, site):
    out = save_html(view_structure(site), os.path.join(tmp_path, "v.html"))
    assert os.path.isabs(out) and os.path.getsize(out) > 10_000
    assert open(out, encoding="utf-8").read().startswith("<!doctype html>")
