"""Pharmacokinetics (PK): what the body does to the drug.

A general compartmental PK engine supporting:

* one- and two-compartment disposition,
* IV bolus, IV (zero-order) infusion and first-order oral absorption,
* bioavailability (F), absorption lag (tlag),
* single or multiple dosing.

The disposition is a system of linear ODEs on drug *amounts*:

    A_depot      - un-absorbed drug at the absorption site (oral only)
    A_central    - drug in the central (measured) compartment
    A_peripheral - drug in the peripheral tissue compartment (2-comp only)

    dA_depot/dt      = -ka * A_depot
    dA_central/dt    =  ka*A_depot + R_inf(t) - k10*A_central
                        - k12*A_central + k21*A_peripheral
    dA_peripheral/dt =  k12*A_central - k21*A_peripheral

Plasma concentration is  C = A_central / Vc.

Solution method
---------------
Between dosing events the input R_inf is constant, so the system is
``y' = K*y + b`` with constant K and b. Its solution is *exact*:

    [y(t+h); 1] = expm(M*h) [y(t); 1],     M = [[K, b], [0, 0]]

We split the time axis at every dose / infusion start / infusion stop, apply
bolus doses as instantaneous jumps, and propagate each segment with one
matrix exponential. This is exact to machine precision, unconditionally
stable for any rate constants (no stiffness problems), needs only NumPy, and
is fast. The browser explorer (web/index.html) uses the identical method, so
both give the same numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .models import DosingRegimen, PKParameters

# [6/6] Pade coefficients for exp(x).
_PADE6 = (1.0, 1.0 / 2, 5.0 / 44, 1.0 / 66, 1.0 / 792, 1.0 / 15840, 1.0 / 665280)


def _expm(m: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a [6/6] Pade
    approximant (accurate to ~1e-16 after scaling ||X||_1 <= 0.5)."""
    norm = float(np.abs(m).sum(axis=0).max())
    s = int(math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = m / (2.0 ** s)
    c = _PADE6
    ident = np.eye(m.shape[0])
    x2 = x @ x
    x4 = x2 @ x2
    x6 = x4 @ x2
    u = x @ (c[1] * ident + c[3] * x2 + c[5] * x4)
    v = c[0] * ident + c[2] * x2 + c[4] * x4 + c[6] * x6
    e = np.linalg.solve(v - u, v + u)
    for _ in range(s):
        e = e @ e
    return e


def _system_matrix(
    k10: float, k12: float, k21: float, ka: float, rate: float, vc: float
) -> np.ndarray:
    """Augmented 5x5 matrix for state [dep, cen, per, auc, 1].

    The ``auc`` row integrates C = A_central / Vc, so the area under the
    curve is exact (not a trapezoid estimate on the output grid).
    """
    return np.array(
        [
            [-ka, 0.0, 0.0, 0.0, 0.0],
            [ka, -(k10 + k12), k21, 0.0, rate],
            [0.0, k12, -k21, 0.0, 0.0],
            [0.0, 1.0 / vc, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@dataclass
class PKResult:
    """Time course of a PK simulation.

    At an IV-bolus dose time the concentration jumps, so that time appears
    twice in ``t``: first with the pre-dose (trough) value, then with the
    post-dose (peak) value.
    """

    t: np.ndarray                 # time grid
    conc: np.ndarray              # plasma concentration (central), C = A_central / Vc
    amounts: np.ndarray           # shape (3, N): depot, central, peripheral amounts
    auc: np.ndarray               # exact cumulative AUC from 0 to t
    dose_times: List[float]
    pk: PKParameters
    regimen: DosingRegimen


def default_t_end(pk: PKParameters, regimen: DosingRegimen) -> float:
    """A horizon long enough to see the drug wash out (~5 half-lives of the
    slowest process after the last dose)."""
    slowest = pk.disposition_rates()["terminal"]
    if regimen.route == "oral" and pk.ka:
        slowest = min(slowest, pk.ka)  # flip-flop kinetics: absorption-limited
    last = regimen.dose_times()[-1]
    if regimen.route == "oral":
        last += pk.tlag
    if regimen.route == "iv_infusion" and regimen.infusion_duration:
        last += regimen.infusion_duration
    return max(last + 5.0 * math.log(2) / slowest, last * 1.2, 24.0)


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
    t_end     : end of simulation (h). Defaults to ~5 terminal half-lives
                after the last dose.
    n_points  : approximate number of output time points.
    """
    pk.validate()
    regimen.validate()
    if regimen.route == "oral" and pk.ka is None:
        raise ValueError("Oral dosing requires an absorption rate constant ka")
    if t_end is None:
        t_end = default_t_end(pk, regimen)
    if t_end <= 0:
        raise ValueError("t_end must be > 0")
    t_end = float(t_end)

    mc = pk.micro_constants()
    k10, k12, k21 = mc["k10"], mc["k12"], mc["k21"]
    ka = pk.ka or 0.0
    dose_times = regimen.dose_times()

    # Dosing events.
    bolus_events: List[Tuple[float, int, float]] = []         # (time, compartment, amount)
    infusion_windows: List[Tuple[float, float, float]] = []   # (start, stop, rate)
    for dt in dose_times:
        if regimen.route == "oral":
            bolus_events.append((dt + pk.tlag, 0, pk.F * regimen.dose))  # -> depot
        elif regimen.route == "iv_bolus":
            bolus_events.append((dt, 1, regimen.dose))                   # -> central
        else:  # iv_infusion
            dur = regimen.infusion_duration
            infusion_windows.append((dt, dt + dur, regimen.dose / dur))

    # Breakpoints where the dynamics change.
    bps = {0.0, t_end}
    bps.update(te for te, _, _ in bolus_events if te <= t_end)
    for s, e, _ in infusion_windows:
        bps.update(x for x in (s, e) if x <= t_end)
    breakpoints = sorted(b for b in bps if 0.0 <= b <= t_end)

    def inf_rate_at(tmid: float) -> float:
        return sum(r for s, e, r in infusion_windows if s <= tmid < e)

    aug = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # dep, cen, per, auc, 1
    t_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for i in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[i], breakpoints[i + 1]
        jump = False
        for te, comp, amt in bolus_events:           # instantaneous doses at t0
            if abs(te - t0) < 1e-9:
                aug[comp] += amt
                jump = jump or comp == 1             # central bolus: C jumps
        npts = max(3, int(round(n_points * (t1 - t0) / t_end)) + 2)
        teval = np.linspace(t0, t1, npts)
        h = (t1 - t0) / (npts - 1)
        rate = inf_rate_at((t0 + t1) / 2.0)
        phi = _expm(_system_matrix(k10, k12, k21, ka, rate, pk.Vc) * h)
        phi[4] = (0.0, 0.0, 0.0, 0.0, 1.0)  # constant-input row is exactly e5
        seg = np.empty((4, npts))
        seg[:, 0] = aug[:4]
        for j in range(1, npts):
            aug = phi @ aug
            seg[:, j] = aug[:4]
        # Stitch segments. The boundary point is shared with the previous
        # segment, so drop it -- unless a bolus made C jump there, in which
        # case keep it so both the trough and the peak are recorded.
        if t_parts and not jump:
            teval, seg = teval[1:], seg[:, 1:]
        t_parts.append(teval)
        y_parts.append(seg)

    t = np.concatenate(t_parts)
    states = np.concatenate(y_parts, axis=1)
    # Clamp round-off below zero (amounts are physically >= 0).
    amounts = np.clip(states[:3], 0.0, None)
    conc = amounts[1] / pk.Vc

    return PKResult(
        t=t, conc=conc, amounts=amounts, auc=states[3],
        dose_times=dose_times, pk=pk, regimen=regimen,
    )
