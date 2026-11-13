"""Interactive 3D viewers for protein structures and docking results.

Produces a self-contained HTML page that renders a structure with
`3Dmol.js <https://3dmol.csb.pitt.edu/>`_. The coordinates are embedded in
the page, so once written the file needs no server and no further downloads
apart from the viewer library itself (which can also be bundled locally with
``local_js=``).

Two entry points:

* :func:`view_structure` - a protein on its own: cartoon, surface, bound
  ligands, and for AlphaFold models the official pLDDT confidence colouring.
* :func:`view_docking` - a docking result: the receptor, the pocket residues
  it binds through, and every predicted pose, switchable, with its score.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .structure import ProteinStructure

CDN_3DMOL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"

# AlphaFold's published pLDDT colour scheme.
PLDDT_BANDS = [
    (90, 100, "#0053D6", "Very high (pLDDT > 90)"),
    (70, 90, "#65CBF3", "Confident (70-90)"),
    (50, 70, "#FFDB13", "Low (50-70)"),
    (0, 50, "#FF7D45", "Very low (< 50)"),
]


def _js(value) -> str:
    """Embed a Python value as a JavaScript literal, safely."""
    return json.dumps(value).replace("</", "<\\/")


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__JS_SRC__"></script>
<style>
  :root {
    --bg: #f5f7fa; --surface: #ffffff; --ink: #0f1b2d; --muted: #5a6b84;
    --line: #dce3ec; --accent: #1f6feb; --shadow: 0 1px 2px rgba(15,27,45,.06), 0 8px 24px rgba(15,27,45,.07);
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0a111e; --surface:#111b2e; --ink:#e8eef9; --muted:#93a4c0; --line:#22314c;
            --accent:#4f93ff; --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35); color-scheme: dark; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 "Inter",system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:18px 16px 28px; }
  header h1 { font-size:1.3rem; margin:0 0 3px; letter-spacing:-.01em; }
  header p { margin:0; color:var(--muted); font-size:.88rem; }
  .grid { display:grid; grid-template-columns:1fr 290px; gap:16px; margin-top:16px; align-items:start; }
  @media (max-width:820px){ .grid{ grid-template-columns:1fr; } }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:13px; box-shadow:var(--shadow); padding:14px; }
  .card h2 { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 10px; font-weight:600; }
  #viewport { position:relative; width:100%; height:580px; border-radius:11px; overflow:hidden; border:1px solid var(--line); background:var(--surface); }
  @media (max-width:820px){ #viewport{ height:420px; } }
  .row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:9px; }
  button, select { font:inherit; font-size:.82rem; color:var(--ink); background:var(--surface);
    border:1px solid var(--line); border-radius:8px; padding:6px 11px; cursor:pointer; }
  button:hover, select:hover { border-color:var(--accent); }
  button[aria-pressed="true"] { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:600; }
  select { width:100%; cursor:pointer; }
  label.ck { display:flex; align-items:center; gap:7px; font-size:.84rem; margin-bottom:6px; cursor:pointer; }
  table { width:100%; border-collapse:collapse; font-size:.82rem; }
  td { padding:3px 0; } td:last-child { text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
  .k { color:var(--muted); }
  .chips { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { font-size:.75rem; border:1px solid var(--line); border-radius:7px; padding:3px 7px; font-variant-numeric:tabular-nums; }
  .swatch { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
  .note { font-size:.78rem; color:var(--muted); margin-top:10px; line-height:1.45; }
  .big { font-size:1.45rem; font-weight:600; font-variant-numeric:tabular-nums; }
  .err { padding:14px; color:var(--muted); font-size:.85rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>__HEADING__</h1>
    <p>__SUBTITLE__</p>
  </header>
  <div class="grid">
    <div class="card">
      <div class="row" id="styleRow"></div>
      <div id="viewport"><div class="err" id="fallback" hidden>
        The 3D viewer library could not be loaded. If you are offline, regenerate this
        page with a local copy of 3Dmol.js (<code>local_js=</code>).</div></div>
      <div class="note" id="hint">Drag to rotate · scroll to zoom · shift-drag to pan · click an atom for its label</div>
    </div>
    <div>__SIDEBAR__</div>
  </div>
</div>
<script>
"use strict";
const DATA = __DATA__;

function boot() {
  if (typeof $3Dmol === "undefined") { document.getElementById("fallback").hidden = false; return; }
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const viewer = $3Dmol.createViewer(document.getElementById("viewport"),
      { backgroundColor: dark ? "#111b2e" : "#ffffff" });

  // model 0 = receptor; then one model per extra entity (ligands, poses)
  viewer.addModel(DATA.receptor, "pdb");
  const extraStart = 1;
  DATA.entities.forEach(e => viewer.addModel(e.data, e.format));

  const state = { protein: DATA.defaultProteinStyle, pocket: true, entity: 0, spin: false };

  function proteinStyle() {
    const s = {};
    if (state.protein === "cartoon") s.cartoon = DATA.colorByPlddt
        ? { colorfunc: a => plddtColor(a.b) } : { color: "spectrum" };
    else if (state.protein === "trace") s.cartoon = { style: "trace", color: "spectrum" };
    else if (state.protein === "sticks") s.stick = { radius: 0.12, colorscheme: "default" };
    return s;
  }
  function plddtColor(b) {
    for (const [lo, hi, col] of DATA.plddtBands) if (b >= lo && b <= hi) return col;
    return "#888888";
  }
  function render() {
    viewer.setStyle({}, {});
    if (state.protein !== "none") viewer.setStyle({ model: 0 }, proteinStyle());
    // pocket residues as sticks
    if (state.pocket && DATA.pocket.length) {
      viewer.addStyle({ model: 0, or: DATA.pocket.map(r => ({ chain: r.chain, resi: r.resseq })) },
                      { stick: { radius: 0.14, colorscheme: dark ? "cyanCarbon" : "grayCarbon" } });
    }
    // the selected entity (a docked pose or a bound ligand)
    DATA.entities.forEach((e, i) => {
      const show = DATA.entityMode === "all" || i === state.entity;
      viewer.setStyle({ model: extraStart + i },
        show ? { stick: { radius: 0.22, colorscheme: e.scheme || "greenCarbon" },
                 sphere: { scale: 0.22 } } : {});
    });
    viewer.render();
  }
  function fit() {
    // Frame the ligand together with the residues it touches, so the pocket
    // is readable without losing the surrounding fold.
    if (DATA.entities.length) {
      // Frame the ligand, then pull back so the pocket walls and the
      // surrounding fold stay visible around it.
      viewer.zoomTo({ model: extraStart + state.entity });
      viewer.zoom(0.42);
    } else {
      viewer.zoomTo();
    }
    viewer.render();
  }

  // style buttons
  const row = document.getElementById("styleRow");
  [["cartoon", "Cartoon"], ["trace", "Trace"], ["sticks", "Sticks"], ["none", "Hide protein"]]
    .forEach(([v, label]) => {
      const b = document.createElement("button");
      b.textContent = label; b.setAttribute("aria-pressed", String(v === state.protein));
      b.onclick = () => { state.protein = v; [...row.children].forEach(c => c.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true"); render(); };
      row.appendChild(b);
    });
  const spin = document.createElement("button");
  spin.textContent = "Spin"; spin.setAttribute("aria-pressed", "false");
  spin.onclick = () => { state.spin = !state.spin; spin.setAttribute("aria-pressed", String(state.spin)); viewer.spin(state.spin ? "y" : false); };
  row.appendChild(spin);
  const reset = document.createElement("button");
  reset.textContent = "Recentre"; reset.onclick = fit; row.appendChild(reset);

  const pocketBox = document.getElementById("pocketToggle");
  if (pocketBox) pocketBox.onchange = () => { state.pocket = pocketBox.checked; render(); };
  const sel = document.getElementById("entitySelect");
  if (sel) sel.onchange = () => { state.entity = +sel.value; updateInfo(); render(); fit(); };

  function updateInfo() {
    const info = document.getElementById("entityInfo");
    if (!info) return;
    const e = DATA.entities[state.entity];
    info.innerHTML = e && e.info ? e.info : "";
  }

  viewer.setClickable({}, true, (atom) => {
    viewer.addLabel(`${atom.resn || ""}${atom.resi || ""} ${atom.atom || ""}`.trim(),
      { position: atom, backgroundColor: dark ? "#1f2d48" : "#0f1b2d", backgroundOpacity: 0.85,
        fontColor: "#fff", fontSize: 11 });
    viewer.render();
  });

  updateInfo(); render(); fit();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
</script>
</body>
</html>
"""


def _render(title: str, heading: str, subtitle: str, sidebar: str,
            data: Dict, js_src: str) -> str:
    return (_PAGE
            .replace("__TITLE__", _esc(title))
            .replace("__HEADING__", heading)
            .replace("__SUBTITLE__", subtitle)
            .replace("__SIDEBAR__", sidebar)
            .replace("__JS_SRC__", js_src)
            .replace("__DATA__", _js(data)))


def _js_source(local_js: Optional[str]) -> str:
    if not local_js:
        return CDN_3DMOL
    if not os.path.exists(local_js):
        raise FileNotFoundError(f"local 3Dmol.js not found: {local_js}")
    return local_js


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Structure viewer
# ---------------------------------------------------------------------------
def view_structure(structure: ProteinStructure,
                   show_ligands: bool = True,
                   highlight_ligand: Optional[str] = None,
                   pocket_radius: float = 4.5,
                   color_by_plddt: Optional[bool] = None,
                   local_js: Optional[str] = None) -> str:
    """Render a protein structure in 3D. Returns a complete HTML page."""
    if color_by_plddt is None:
        color_by_plddt = structure.is_predicted

    entities, pocket = [], []
    ligs = structure.ligands() if show_ligands else []
    if highlight_ligand:
        ligs = [structure.get_ligand(highlight_ligand)]
    for lig in ligs[:6]:
        entities.append({
            "data": lig.to_pdb(), "format": "pdb", "scheme": "greenCarbon",
            "label": lig.label,
            "info": f"<div class='big'>{_esc(lig.resname)}</div>"
                    f"<table><tr><td class='k'>Chain</td><td>{_esc(lig.chain)}</td></tr>"
                    f"<tr><td class='k'>Residue</td><td>{lig.resseq}</td></tr>"
                    f"<tr><td class='k'>Heavy atoms</td><td>{lig.n_heavy_atoms}</td></tr></table>",
        })
    if ligs:
        pocket = [{"chain": r["chain"], "resseq": r["resseq"]}
                  for r in structure.pocket_residues(ligs[0], pocket_radius)]

    # sidebar
    rows = [("Source", structure.source), ("Chains", ", ".join(structure.chains)),
            ("Residues", structure.residue_count()), ("Atoms", len(structure.polymer_atoms))]
    if structure.method:
        rows.insert(1, ("Method", structure.method.title()))
    if structure.resolution:
        rows.insert(2, ("Resolution", f"{structure.resolution} Å"))
    if structure.uniprot:
        rows.append(("UniProt", structure.uniprot))
    table = "".join(f"<tr><td class='k'>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)

    plddt_block = ""
    summary = structure.plddt_summary()
    if color_by_plddt and summary:
        bands = "".join(
            f"<div class='chip'><span class='swatch' style='background:{c}'></span>{_esc(lbl)}</div>"
            for _, _, c, lbl in PLDDT_BANDS)
        plddt_block = (
            "<div class='card' style='margin-top:14px'><h2>Model confidence (pLDDT)</h2>"
            f"<div class='big'>{summary['mean_plddt']}</div>"
            "<div class='k' style='font-size:.78rem;margin-bottom:9px'>mean pLDDT</div>"
            f"<div class='chips'>{bands}</div>"
            "<p class='note'>This is a <strong>predicted</strong> structure, not an "
            "experimental measurement. Regions below 70 are unreliable and below 50 are "
            "often intrinsically disordered. Docking into low-confidence regions is "
            "not meaningful.</p></div>")

    lig_select = ""
    if len(entities) > 1:
        opts = "".join(f"<option value='{i}'>{_esc(e['label'])}</option>"
                       for i, e in enumerate(entities))
        lig_select = f"<h2>Bound ligand</h2><select id='entitySelect'>{opts}</select>"

    sidebar = (
        f"<div class='card'><h2>Structure</h2><table>{table}</table>"
        + ("<label class='ck' style='margin-top:10px'><input type='checkbox' "
            "id='pocketToggle' checked> Show pocket residues</label>" if pocket else "")
        + "</div>"
        + (f"<div class='card' style='margin-top:14px'>{lig_select}<div id='entityInfo'></div></div>" if entities else "")
        + plddt_block)

    data = {
        "receptor": structure.to_pdb(polymer_only=True),
        "entities": entities,
        "pocket": pocket,
        "entityMode": "all" if len(entities) <= 1 else "one",
        "defaultProteinStyle": "cartoon",
        "colorByPlddt": bool(color_by_plddt and summary),
        "plddtBands": [[lo, hi, c] for lo, hi, c, _ in PLDDT_BANDS],
    }
    title = f"{structure.identifier} in 3D"
    subtitle = _esc(structure.title or "Protein structure") + \
        (" · predicted by AlphaFold" if structure.is_predicted else "")
    return _render(title, _esc(structure.identifier), subtitle, sidebar, data,
                   _js_source(local_js))


# ---------------------------------------------------------------------------
# Docking viewer
# ---------------------------------------------------------------------------
def view_docking(result, structure: Optional[ProteinStructure] = None,
                 local_js: Optional[str] = None) -> str:
    """Render a docking result: receptor, pocket and every predicted pose."""
    receptor_pdb = result.receptor_pdb or (structure.to_pdb(polymer_only=True)
                                           if structure else "")
    if not receptor_pdb:
        raise ValueError("No receptor coordinates; pass the structure= used for docking")

    entities = []
    for pose in result.poses:
        contacts = ", ".join(c.label for c in pose.contacts[:10]) or "none within cutoff"
        entities.append({
            "data": pose.sdf_block or pose.pdb_block,
            "format": "sdf" if pose.sdf_block else "pdb",
            "scheme": "magentaCarbon" if pose.rank == 1 else "greenCarbon",
            "label": f"Pose {pose.rank} — {pose.affinity:.2f} kcal/mol",
            "info": (f"<div class='big'>{pose.affinity:.2f}<span style='font-size:.8rem;color:var(--muted)'> kcal/mol</span></div>"
                     f"<table><tr><td class='k'>Rank</td><td>{pose.rank} of {len(result.poses)}</td></tr>"
                     f"<tr><td class='k'>Estimated K<sub>d</sub></td><td>{_esc(pose.kd_text())}</td></tr>"
                     f"<tr><td class='k'>RMSD to pose 1</td><td>{pose.rmsd_lb:.2f} Å</td></tr>"
                     f"<tr><td class='k'>Contacts</td><td>{len(pose.contacts)} residues</td></tr></table>"
                     f"<p class='note'><strong>Contacting residues:</strong> {_esc(contacts)}</p>"),
        })

    pocket = [{"chain": c.chain, "resseq": c.resseq} for c in result.best.contacts]
    opts = "".join(f"<option value='{i}'>{_esc(e['label'])}</option>"
                   for i, e in enumerate(entities))

    rows = [("Receptor", result.receptor_id), ("Ligand", result.ligand_name),
            ("Poses", len(result.poses)), ("Exhaustiveness", result.exhaustiveness)]
    if result.reference_ligand:
        rows.append(("Box from", result.reference_ligand))
    if result.kept_waters:
        rows.append(("Waters kept", ", ".join(result.kept_waters)))
    table = "".join(f"<tr><td class='k'>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)

    sidebar = (
        f"<div class='card'><h2>Docking run</h2><table>{table}</table>"
        "<label class='ck' style='margin-top:10px'><input type='checkbox' id='pocketToggle' checked>"
        " Show contacting residues</label></div>"
        f"<div class='card' style='margin-top:14px'><h2>Predicted pose</h2>"
        f"<select id='entitySelect'>{opts}</select>"
        "<div id='entityInfo' style='margin-top:11px'></div></div>"
        "<div class='card' style='margin-top:14px'><h2>How to read this</h2>"
        "<p class='note'>Scores rank poses; they are not measured affinities. AutoDock "
        "Vina's scoring error is roughly 2-3 kcal/mol, which is more than two orders of "
        "magnitude in K<sub>d</sub>, so the estimate above is an order of magnitude at "
        "best. The top-scored pose is also not always the correct one &mdash; in a "
        "redocking test on HIV-1 protease the crystal pose came second. Look through "
        "all the poses and judge them on the contacts they make.</p></div>")

    data = {
        "receptor": receptor_pdb,
        "entities": entities,
        "pocket": pocket,
        "entityMode": "one",
        "defaultProteinStyle": "cartoon",
        "colorByPlddt": False,
        "plddtBands": [[lo, hi, c] for lo, hi, c, _ in PLDDT_BANDS],
    }
    heading = f"{_esc(result.ligand_name)} docked into {_esc(result.receptor_id)}"
    subtitle = (f"Best score {result.best_affinity:.2f} kcal/mol "
                f"(estimated K<sub>d</sub> {_esc(result.best.kd_text())}) · "
                f"{len(result.poses)} predicted poses")
    return _render(f"{result.ligand_name} docked into {result.receptor_id}",
                   heading, subtitle, sidebar, data, _js_source(local_js))


def save_html(html: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(path)
