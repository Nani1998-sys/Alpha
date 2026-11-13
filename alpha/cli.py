"""Command-line interface for Alpha.

Examples
--------
# Analyze a chemical structure (no registration):
python -m alpha describe --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Register a new drug by its structure + PK/PD:
python -m alpha add --name Aspirin --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --cl 39 --vc 10 --ka 2.0 --f 0.68 \
    --pd emax --e0 0 --emax 100 --ec50 5 --effect-units "% COX inhibition"

# List and inspect:
python -m alpha list
python -m alpha show --name Aspirin

# Simulate a dosing regimen:
python -m alpha simulate --name Aspirin --dose 325 --route oral \
    --interval 6 --ndoses 8 --tend 72 --plot aspirin.png --csv aspirin.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import chem
from .models import DosingRegimen, PDParameters, PKParameters
from .registry import DEFAULT_DB, DrugRegistry
from .structure import DEFAULT_CACHE as DEFAULT_STRUCT_CACHE
from .report import plot_simulation, text_report
from .simulate import simulate_drug


def _build_pk(args) -> Optional[PKParameters]:
    if args.cl is None and args.vc is None:
        return None
    if args.cl is None or args.vc is None:
        raise SystemExit("PK requires at least --cl and --vc")
    return PKParameters(
        CL=args.cl, Vc=args.vc, ka=args.ka, F=args.f, tlag=args.tlag,
        Q=args.q, Vp=args.vp,
    )


def _build_pd(args) -> Optional[PDParameters]:
    if not args.pd or args.pd == "none":
        return None
    return PDParameters(
        model=args.pd, E0=args.e0, Emax=args.emax, EC50=args.ec50,
        Imax=args.imax, IC50=args.ic50, hill=args.hill, slope=args.slope,
        ke0=args.ke0, effect_units=args.effect_units or "",
    )


def _add_pk_pd_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("pharmacokinetics (PK)")
    g.add_argument("--cl", type=float, help="clearance (L/h)")
    g.add_argument("--vc", type=float, help="central volume (L)")
    g.add_argument("--ka", type=float, help="absorption rate constant (1/h, oral)")
    g.add_argument("--f", type=float, default=1.0, help="bioavailability 0-1 (oral)")
    g.add_argument("--tlag", type=float, default=0.0, help="absorption lag (h)")
    g.add_argument("--q", type=float, help="inter-compartmental clearance (L/h)")
    g.add_argument("--vp", type=float, help="peripheral volume (L)")
    g = p.add_argument_group("pharmacodynamics (PD)")
    g.add_argument("--pd", choices=["none", "linear", "loglinear", "emax",
                                    "sigmoid_emax", "imax", "sigmoid_imax"],
                   help="PD model")
    g.add_argument("--e0", type=float, default=0.0, help="baseline effect")
    g.add_argument("--emax", type=float, help="max effect")
    g.add_argument("--ec50", type=float, help="conc. for half-max effect")
    g.add_argument("--imax", type=float, help="max inhibition")
    g.add_argument("--ic50", type=float, help="conc. for half-max inhibition")
    g.add_argument("--hill", type=float, default=1.0, help="Hill coefficient")
    g.add_argument("--slope", type=float, help="slope (linear/loglinear)")
    g.add_argument("--ke0", type=float, help="effect-compartment rate (1/h)")
    g.add_argument("--effect-units", type=str, help="label for effect axis")


def cmd_describe(args) -> int:
    desc = chem.analyze_structure(args.smiles)
    print(json.dumps(desc, indent=2))
    return 0


def cmd_add(args) -> int:
    reg = DrugRegistry(args.db)
    pk = _build_pk(args)
    pd = _build_pd(args)
    drug = reg.add_drug(
        args.name, smiles=args.smiles, pk=pk, pd=pd,
        notes=args.notes or "", overwrite=args.overwrite,
        target_name=args.target_name or "", target_pdb=args.target_pdb or "",
        target_ligand=args.target_ligand or "",
    )
    reg.save()
    print(f"Added {drug.name!r} to {reg.path}")
    if drug.descriptors:
        d = drug.descriptors
        print(f"  Formula {d.get('formula')}, MW {d.get('mol_weight')}, "
              f"logP {d.get('logp')}, QED {d.get('qed')}")
    if drug.pk:
        print(f"  PK: CL={drug.pk.CL} Vc={drug.pk.Vc} "
              f"({drug.pk.n_compartments}-compartment)")
    if drug.pd:
        print(f"  PD: {drug.pd.model}")
    return 0


def cmd_list(args) -> int:
    reg = DrugRegistry(args.db)
    if not len(reg):
        print("(registry is empty)")
        return 0
    for d in reg.all():
        pk = "PK" if d.pk else "--"
        pd = d.pd.model if d.pd else "--"
        mw = d.descriptors.get("mol_weight", "?") if d.descriptors else "?"
        tgt = f"  target={d.target_pdb}" if d.target_pdb else ""
        print(f"  {d.name:<20} MW={mw:<8} [{pk}] PD={pd}{tgt}")
    return 0


def cmd_show(args) -> int:
    reg = DrugRegistry(args.db)
    drug = reg.get(args.name)
    print(json.dumps(drug.to_dict(), indent=2, default=str))
    return 0


def _fmt_struct(s) -> str:
    bits = [f"{s.identifier}  ({s.source})"]
    if s.title:
        bits.append(f"  {s.title[:72]}")
    meta = []
    if s.method:
        meta.append(s.method.title())
    if s.resolution:
        meta.append(f"{s.resolution} A")
    meta.append(f"{len(s.chains)} chain(s): {', '.join(s.chains)}")
    meta.append(f"{s.residue_count()} residues")
    bits.append("  " + "  |  ".join(meta))
    ligs = s.ligands()
    bits.append("  bound ligands: " + (", ".join(
        f"{l.label} ({l.n_heavy_atoms} heavy atoms)" for l in ligs) or "none"))
    summary = s.plddt_summary()
    if summary:
        bits.append(f"  mean pLDDT {summary['mean_plddt']} "
                    f"({summary['very_low_<50'] * 100:.0f}% very low confidence) "
                    "- this is a PREDICTED structure")
    return "\n".join(bits)


def _load_structure(args):
    from .structure import fetch_alphafold, fetch_pdb

    if getattr(args, "uniprot", None):
        return fetch_alphafold(args.uniprot, cache_dir=args.cache)
    if getattr(args, "pdb", None):
        return fetch_pdb(args.pdb, cache_dir=args.cache)
    raise SystemExit("give a structure with --pdb (e.g. 1HSG) or --uniprot (e.g. P00533)")


def cmd_fetch(args) -> int:
    s = _load_structure(args)
    print(_fmt_struct(s))
    if args.waters:
        ligs = s.ligands()
        if ligs:
            bridging = s.bridging_waters(ligs[0])
            print(f"\n  waters bridging {ligs[0].label} and the protein:")
            for w in bridging[:8] or [{"residue": "(none found)",
                                       "distance_to_ligand": "", "protein_contacts": ""}]:
                print(f"    {w['residue']:>8}  {w['distance_to_ligand']} A to ligand, "
                      f"{w['protein_contacts']} protein contact(s)")
            if bridging:
                print(f"    keep one when docking:  --keep-residues {bridging[0]['residue']}")
    if args.sequence:
        for ch in s.chains:
            print(f"\n>{s.identifier}_{ch}\n{s.sequence(ch)}")
    if args.view3d:
        from .viewer3d import save_html, view_structure
        path = save_html(view_structure(s, local_js=args.local_js), args.view3d)
        print(f"\n3D viewer -> {path}")
    return 0


def cmd_search_pdb(args) -> int:
    from .structure import search_pdb

    hits = search_pdb(args.query, limit=args.limit)
    if not hits:
        print("no matching structures")
        return 0
    print(f"{len(hits)} match(es) in the RCSB PDB for {args.query!r}:")
    for h in hits:
        print(f"  {h['pdb_id']}   score {h['score']}")
    print("\nInspect one with:  python -m alpha fetch --pdb <ID>")
    return 0


def cmd_dock(args) -> int:
    from .docking import dock, pd_from_docking
    from .registry import DrugRegistry

    smiles, name = args.smiles, args.ligand_name
    drug = None
    if args.name:
        drug = DrugRegistry(args.db).get(args.name)
        if not drug.smiles:
            raise SystemExit(f"{drug.name!r} has no chemical structure stored; "
                             "re-add it with --smiles, or pass --smiles here")
        smiles, name = drug.smiles, drug.name
    if not smiles:
        raise SystemExit("give the molecule with --name (from the registry) or --smiles")

    # fall back to the drug's stored target
    if drug is not None and not (args.pdb or args.uniprot) and drug.target_pdb:
        args.pdb = drug.target_pdb
        print(f"Using {drug.name}'s stored target: {drug.target_pdb}"
              + (f" ({drug.target_name})" if drug.target_name else ""))
    if drug is not None and not args.ref_ligand and drug.target_ligand:
        args.ref_ligand = drug.target_ligand
    structure = _load_structure(args)
    center = tuple(args.center) if args.center else None
    size = tuple(args.size) if args.size else None
    print(f"Docking {name} into {structure.identifier} "
          f"(exhaustiveness {args.exhaustiveness}) ...")
    result = dock(
        structure, smiles, center=center, box_size=size,
        reference_ligand=args.ref_ligand, ligand_name=name,
        exhaustiveness=args.exhaustiveness, n_poses=args.n_poses,
        seed=args.seed, cpu=args.cpu, keep_waters=args.keep_waters or (),
        keep_residues=args.keep_residues or (),
    )
    print(f"\nBox centre {tuple(round(c, 2) for c in result.center)}  "
          f"size {tuple(round(s, 1) for s in result.box_size)}"
          + (f"  (from {result.reference_ligand})" if result.reference_ligand else ""))
    print(f"\n{'rank':>4} {'affinity':>12} {'est. Kd':>10}   contacting residues")
    for p in result.poses:
        print(f"{p.rank:>4} {p.affinity:>8.2f} kcal {p.kd_text():>10}   "
              + ", ".join(c.label for c in p.contacts[:8]))
    print("\nScores rank poses; they are not measured affinities (Vina's error is "
          "~2-3 kcal/mol).\nThe top-scored pose is not always the right one - "
          "look through them all.")

    if args.view3d:
        from .viewer3d import save_html, view_docking
        path = save_html(view_docking(result, structure, local_js=args.local_js), args.view3d)
        print(f"\n3D viewer -> {path}")
    if args.sdf:
        with open(args.sdf, "w", encoding="utf-8") as fh:
            for p in result.poses:
                if p.sdf_block:
                    fh.write(p.sdf_block.rstrip() + f"\n>  <vina_affinity>\n{p.affinity}\n\n$$$$\n")
        print(f"Poses (SDF) -> {args.sdf}")
    if args.json:
        import json as _json
        with open(args.json, "w", encoding="utf-8") as fh:
            _json.dump({"summary": result.summary(),
                        "poses": [{"rank": p.rank, "affinity": p.affinity,
                                   "estimated_kd": p.kd_text(),
                                   "contacts": [c.label for c in p.contacts]}
                                  for p in result.poses]}, fh, indent=2)
        print(f"Results (JSON) -> {args.json}")

    if args.set_pd:
        if drug is None:
            raise SystemExit("--set-pd needs --name, so the result can be stored on a drug")
        mw = (drug.descriptors or {}).get("mol_weight")
        pd, note = pd_from_docking(result, model=args.pd_model, E0=args.e0,
                                   Emax=args.emax, molecular_weight=mw)
        reg = DrugRegistry(args.db)
        reg.set_pd(drug.name, pd)
        reg.save()
        unit = "mg/L" if mw else "M"
        potency = pd.IC50 if pd.IC50 is not None else pd.EC50
        print(f"\nStored PD on {drug.name!r}: {pd.model}, "
              f"{'IC50' if pd.IC50 is not None else 'EC50'} = {potency:.4g} {unit}")
        print(f"  {note}")
        print(f"  Simulate it with:  python -m alpha simulate --name {drug.name} --dose <mg>")
    return 0


def cmd_simulate(args) -> int:
    reg = DrugRegistry(args.db)
    drug = reg.get(args.name)
    regimen = DosingRegimen(
        dose=args.dose, route=args.route, interval=args.interval,
        n_doses=args.ndoses, infusion_duration=args.infusion_duration,
    )
    result = simulate_drug(drug, regimen, t_end=args.tend)
    print(text_report(drug, result))
    if args.csv:
        result.to_csv(args.csv)
        print(f"\nWrote time course -> {args.csv}")
    if args.plot:
        plot_simulation(result, path=args.plot)
        print(f"Wrote plot -> {args.plot}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"drug": drug.name, "metrics": result.metrics,
                       "rows": result.to_rows()}, fh, indent=2)
        print(f"Wrote JSON -> {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpha",
        description="Alpha - in silico drug testing (PK/PD) from chemical structure",
    )
    db_help = "path to the drug database JSON (default: bundled library, or $ALPHA_DB)"
    p.add_argument("--db", default=DEFAULT_DB, help=db_help)
    # Also accept --db after the subcommand (`alpha add --db my.json ...`).
    # SUPPRESS keeps the top-level value when it is not given here.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS, help=db_help)
    sub = p.add_subparsers(dest="command", required=True)

    pd_ = sub.add_parser("describe", help="compute descriptors for a SMILES")
    pd_.add_argument("--smiles", required=True)
    pd_.set_defaults(func=cmd_describe)

    pa = sub.add_parser("add", parents=[common], help="register a new drug by its structure")
    pa.add_argument("--name", required=True)
    pa.add_argument("--smiles", help="chemical structure (SMILES)")
    pa.add_argument("--notes", help="free-text notes")
    pa.add_argument("--target-name", help="the protein this drug acts on")
    pa.add_argument("--target-pdb", help="PDB ID of that protein, used by `dock`")
    pa.add_argument("--target-ligand", help="residue name of a co-crystallised "
                                            "ligand marking the binding site")
    pa.add_argument("--overwrite", action="store_true")
    _add_pk_pd_args(pa)
    pa.set_defaults(func=cmd_add)

    pl = sub.add_parser("list", parents=[common], help="list drugs in the registry")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", parents=[common], help="show a drug's full record")
    ps.add_argument("--name", required=True)
    ps.set_defaults(func=cmd_show)

    psim = sub.add_parser("simulate", parents=[common], help="simulate a dosing regimen")
    psim.add_argument("--name", required=True)
    psim.add_argument("--dose", type=float, required=True, help="dose amount (mg)")
    psim.add_argument("--route", default="oral",
                      choices=["iv_bolus", "iv_infusion", "oral"])
    psim.add_argument("--interval", type=float, help="dosing interval tau (h)")
    psim.add_argument("--ndoses", type=int, default=1, help="number of doses")
    psim.add_argument("--infusion-duration", type=float,
                      help="infusion duration (h, for iv_infusion)")
    psim.add_argument("--tend", type=float, help="simulation end time (h)")
    psim.add_argument("--plot", help="write a PNG plot to this path")
    psim.add_argument("--csv", help="write the time course to CSV")
    psim.add_argument("--json", help="write metrics + time course to JSON")
    psim.set_defaults(func=cmd_simulate)

    # ---- structures & docking -------------------------------------------
    struct_opts = argparse.ArgumentParser(add_help=False)
    struct_opts.add_argument("--pdb", help="RCSB PDB ID, e.g. 1HSG (experimental structure)")
    struct_opts.add_argument("--uniprot", help="UniProt accession, e.g. P00533 "
                                               "(AlphaFold predicted structure)")
    struct_opts.add_argument("--cache", default=DEFAULT_STRUCT_CACHE,
                             help="directory for downloaded structures")
    struct_opts.add_argument("--local-js", help="path to a local 3Dmol.js for offline viewers")

    pf = sub.add_parser("fetch", parents=[struct_opts],
                        help="download and describe a protein structure")
    pf.add_argument("--sequence", action="store_true", help="also print the sequence(s)")
    pf.add_argument("--waters", action="store_true",
                    help="list structural waters bridging the ligand and the protein")
    pf.add_argument("--view3d", metavar="OUT.html", help="write an interactive 3D viewer")
    pf.set_defaults(func=cmd_fetch)

    psp = sub.add_parser("search-pdb", help="full-text search of the RCSB PDB")
    psp.add_argument("query")
    psp.add_argument("--limit", type=int, default=10)
    psp.set_defaults(func=cmd_search_pdb)

    pdk = sub.add_parser("dock", parents=[common, struct_opts],
                         help="dock a molecule into a protein structure")
    pdk.add_argument("--name", help="drug from the registry (uses its stored structure)")
    pdk.add_argument("--smiles", help="molecule to dock, as SMILES")
    pdk.add_argument("--ligand-name", default="ligand", help="label for the molecule")
    pdk.add_argument("--ref-ligand", help="residue name of a bound ligand that marks "
                                          "the site, e.g. MK1 (default: the largest one)")
    pdk.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                     help="search box centre, if there is no bound ligand")
    pdk.add_argument("--size", type=float, nargs=3, metavar=("X", "Y", "Z"),
                     help="search box size in angstroms")
    pdk.add_argument("--exhaustiveness", type=int, default=8,
                     help="search effort (default 8; 16-32 is more reliable and slower)")
    pdk.add_argument("--n-poses", type=int, default=9, help="how many poses to report")
    pdk.add_argument("--seed", type=int, default=42, help="random seed (reproducibility)")
    pdk.add_argument("--cpu", type=int, help="CPU cores to use")
    pdk.add_argument("--keep-waters", nargs="*", metavar="RESNAME",
                     help="keep EVERY group of this name, e.g. HEM for a cofactor")
    pdk.add_argument("--keep-residues", nargs="*", metavar="CHAIN:RESSEQ",
                     help="keep individual groups, e.g. B:301 for one structural "
                          "water (see 'fetch --waters')")
    pdk.add_argument("--view3d", metavar="OUT.html", help="write an interactive 3D viewer")
    pdk.add_argument("--sdf", metavar="OUT.sdf", help="write the poses as SDF")
    pdk.add_argument("--json", metavar="OUT.json", help="write the results as JSON")
    pdk.add_argument("--set-pd", action="store_true",
                     help="estimate EC50/IC50 from the score and store it on the drug")
    pdk.add_argument("--pd-model", default="emax",
                     choices=["emax", "sigmoid_emax", "imax", "sigmoid_imax"])
    pdk.add_argument("--e0", type=float, default=0.0, help="PD baseline for --set-pd")
    pdk.add_argument("--emax", type=float, default=100.0, help="PD maximum for --set-pd")
    pdk.set_defaults(func=cmd_dock)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # surface a clean message to the terminal
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
