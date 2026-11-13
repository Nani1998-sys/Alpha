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
        print(f"  {d.name:<20} MW={mw:<8} [{pk}] PD={pd}")
    return 0


def cmd_show(args) -> int:
    reg = DrugRegistry(args.db)
    drug = reg.get(args.name)
    print(json.dumps(drug.to_dict(), indent=2, default=str))
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
