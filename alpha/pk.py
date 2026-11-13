"""Pharmacokinetics (PK): what the body does to the drug.

A general compartmental PK engine supporting:

* one- and two-compartment disposition,
* IV bolus, IV (zero-order) infusion and first-order oral absorption,
* bioavailability (F), absorption lag (tlag),
* single or multiple dosing.

The disposition is integrated as a system of ODEs on drug *amounts*:

    A_depot      - un-absorbed drug at the absorption site (oral only)
    A_central    - drug in the central (measured) compartment
    A_peripheral - drug in the peripheral tissue compartment (2-comp only)

    dA_depot/dt      = -ka * A_depot
    dA_central/dt    =  ka*A_depot + R_inf(t) - k10*A_central
                        - k12*A_central + k21*A_peripheral
    dA_peripheral/dt =  k12*A_central - k21*A_peripheral

Plasma concentration is  C = A_central / Vc.

Integration is piecewise between dose/infusion breakpoints so bolus doses are
applied exactly (as instantaneous jumps in amount) and infusion rates are
constant within each segment. SciPy's ``solve_ivp`` (LSODA) is used when
available; a built-in RK4 integrator is the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .models import DosingRegimen, PKParameters

try:  # pragma: no cover - import guard
    from scipy.integrate import solve_ivp

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class PKResult:
    """Time course of a PK simulation."""

    t: np.ndarray                 # time grid
    conc: np.ndarray              # plasma concentration (central), C = A_central / Vc
    amounts: np.ndarray           # shape (3, N): depot, central, peripheral amounts
    dose_times: List[float]
    pk: PKParameters
    regimen: DosingRegimen


def _rhs(t, y, k10, k12, k21, ka, inf_rate):
    a_dep, a_cen, a_per = y
    d_dep = -ka * a_dep
    d_cen = ka * a_dep + inf_rate - k10 * a_cen - k12 * a_cen + k21 * a_per
    d_per = k12 * a_cen - k21 * a_per
    return (d_dep, d_cen, d_per)


def _rk4_segment(y0, t0, t1, args, teval):
    """Fixed-step RK4 fallback integrator over [t0, t1] sampled at teval."""
    out = np.empty((3, len(teval)))
    y = np.array(y0, dtype=float)
    out[:, 0] = y
    for i in range(1, len(teval)):
        h = teval[i] - teval[i - 1]
        tc = teval[i - 1]
        # sub-step for accuracy on stiff-ish systems
        n_sub = max(1, int(np.ceil(h / 0.02)))
        hs = h / n_sub
        for _ in range(n_sub):
            k1 = np.array(_rhs(tc, y, *args))
            k2 = np.array(_rhs(tc + hs / 2, y + hs / 2 * k1, *args))
            k3 = np.array(_rhs(tc + hs / 2, y + hs / 2 * k2, *args))
            k4 = np.array(_rhs(tc + hs, y + hs * k3, *args))
            y = y + hs / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tc += hs
        out[:, i] = y
    return out


def simulate_pk(
    pk: PKParameters,
    regimen: DosingRegimen,
    t_end: Optional[float] = None,
    n_points: int = 2000,
) -> PKResult:
    """Simulate the plasma concentration-time course.

    Parameters
    ----------
    pk        : PK parameters (CL, Vc, optionally ka/F/tlag and Q/Vp).
    regimen   : dosing schedule.
    t_end     : end of simulation (h). Defaults to a sensible multiple of the
                dosing schedule / terminal half-life.
    n_points  : approximate number of output time points.
    """
    pk.validate()
    regimen.validate()
    if regimen.route == "oral" and pk.ka is None:
        raise ValueError("Oral dosing requires an absorption rate constant ka")

    mc = pk.micro_constants()
    k10, k12, k21 = mc["k10"], mc["k12"], mc["k21"]
    ka = pk.ka or 0.0

    dose_times = regimen.dose_times()

    if t_end is None:
        thalf = np.log(2) / k10
        last_dose = dose_times[-1]
        if regimen.route == "iv_infusion" and regimen.infusion_duration:
            last_dose += regimen.infusion_duration
        t_end = max(last_dose + 5 * thalf, last_dose * 1.2, 24.0)

    # Build dosing events.
    bolus_events: List[Tuple[float, int, float]] = []  # (time, comp_index, amount)
    infusion_windows: List[Tuple[float, float, float]] = []  # (start, stop, rate)
    for dt in dose_times:
        if regimen.route == "oral":
            bolus_events.append((dt + pk.tlag, 0, pk.F * regimen.dose))  # -> depot
        elif regimen.route == "iv_bolus":
            bolus_events.append((dt, 1, regimen.dose))                   # -> central
        elif regimen.route == "iv_infusion":
            dur = regimen.infusion_duration
            infusion_windows.append((dt, dt + dur, regimen.dose / dur))

    # Breakpoints where dynamics change.
    bps = {0.0, float(t_end)}
    for te, _, _ in bolus_events:
        if te <= t_end:
            bps.add(te)
    for s, e, _ in infusion_windows:
        if s <= t_end:
            bps.add(s)
        if e <= t_end:
            bps.add(e)
    breakpoints = sorted(b for b in bps if 0.0 <= b <= t_end)

    def inf_rate_at(tmid: float) -> float:
        return sum(r for s, e, r in infusion_windows if s <= tmid < e)

    y = np.zeros(3)
    t_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for i in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[i], breakpoints[i + 1]
        # Apply instantaneous bolus/depot additions scheduled at t0.
        for te, comp, amt in bolus_events:
            if abs(te - t0) < 1e-9:
                y[comp] += amt
        rate = inf_rate_at((t0 + t1) / 2.0)
        span = t1 - t0
        npts = max(3, int(round(n_points * span / t_end)) + 2)
        teval = np.linspace(t0, t1, npts)
        args = (k10, k12, k21, ka, rate)
        if _HAVE_SCIPY:
            sol = solve_ivp(
                _rhs, (t0, t1), y, t_eval=teval, args=args,
                method="LSODA", rtol=1e-8, atol=1e-10,
            )
            seg = sol.y
            y = sol.y[:, -1].copy()
        else:
            seg = _rk4_segment(y, t0, t1, args, teval)
            y = seg[:, -1].copy()
        # Stitch, dropping the duplicated boundary point after the first segment.
        if t_all:
            t_all.append(teval[1:])
            y_all.append(seg[:, 1:])
        else:
            t_all.append(teval)
            y_all.append(seg)

    # Apply any bolus scheduled exactly at t_end (no dynamics after, but record).
    for te, comp, amt in bolus_events:
        if abs(te - t_end) < 1e-9:
            y[comp] += amt

    t = np.concatenate(t_all) if t_all else np.array([0.0, t_end])
    amounts = np.concatenate(y_all, axis=1) if y_all else np.zeros((3, 2))
    # Clamp tiny negative values from floating-point noise (amounts are >= 0).
    amounts = np.clip(amounts, 0.0, None)
    conc = amounts[1] / pk.Vc

    return PKResult(
        t=t, conc=conc, amounts=amounts,
        dose_times=dose_times, pk=pk, regimen=regimen,
    )
