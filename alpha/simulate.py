"""Orchestration: run a full PK (+ optional PD) simulation for a drug and
compute summary exposure metrics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .models import Drug, DosingRegimen
from .pd import simulate_pd
from .pk import PKResult, simulate_pk


def _sig(x: float, digits: int = 6) -> float:
    """Round to significant figures (not decimal places), so that small
    concentrations such as 0.000335 mg/L keep their precision."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


@dataclass
class SimulationResult:
    t: np.ndarray
    conc: np.ndarray                      # plasma concentration
    effect: Optional[np.ndarray] = None   # effect (if PD provided)
    effect_driver: Optional[np.ndarray] = None  # Ce or C used to drive PD
    metrics: Dict[str, Any] = field(default_factory=dict)
    dose_times: List[float] = field(default_factory=list)
    drug_name: str = ""
    conc_units: str = "mg/L"
    time_units: str = "h"
    effect_units: str = ""

    def to_rows(self) -> List[Dict[str, float]]:
        rows = []
        for i in range(len(self.t)):
            row = {"time": float(self.t[i]), "concentration": float(self.conc[i])}
            if self.effect is not None:
                row["effect"] = float(self.effect[i])
                row["effect_driver"] = float(self.effect_driver[i])
            rows.append(row)
        return rows

    def to_csv(self, path: str) -> None:
        import csv

        rows = self.to_rows()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _terminal_kel(t: np.ndarray, conc: np.ndarray) -> Optional[float]:
    """Estimate the terminal elimination rate constant by log-linear
    regression over the terminal (post-peak, declining) phase, the way
    non-compartmental analysis does it from sampled data."""
    if len(t) < 3:
        return None
    imax = int(np.argmax(conc))
    tail_t, tail_c = t[imax:], conc[imax:]
    mask = tail_c > 0
    tail_t, tail_c = tail_t[mask], tail_c[mask]
    if len(tail_t) < 3:
        return None
    start = int(len(tail_t) * 0.6)          # last ~40% of the declining tail
    xt, xc = tail_t[start:], tail_c[start:]
    if len(xt) < 2 or np.ptp(xt) == 0:
        return None
    slope, _ = np.polyfit(xt, np.log(xc), 1)
    return -slope if slope < 0 else None


def compute_nca(pk_result: PKResult) -> Dict[str, Any]:
    """Exposure metrics for a simulated concentration-time course.

    Cmax/Tmax and AUC are read from the exact simulation. AUC_inf uses the
    mass-balance identity AUC(0-inf) = F*Dose_total / CL, exact for linear
    PK. ``terminal_half_life`` is the data-driven (NCA-style) estimate from
    the terminal slope; ``model_half_life`` is the exact value from the
    parameters (ln2/beta for two compartments, ln2/k10 for one).
    """
    t, c = pk_result.t, pk_result.conc
    pk, reg = pk_result.pk, pk_result.regimen
    imax = int(np.argmax(c))

    metrics: Dict[str, Any] = {
        "Cmax": _sig(c[imax]),
        "Tmax": _sig(t[imax]),
        "AUC_last": _sig(pk_result.auc[-1]),
        "C_last": _sig(c[-1]),
    }
    f_sys = pk.F if reg.route == "oral" else 1.0
    metrics["AUC_inf"] = _sig(f_sys * reg.dose * reg.n_doses / pk.CL)

    kel = _terminal_kel(t, c)
    if kel:
        metrics["terminal_kel"] = _sig(kel)
        metrics["terminal_half_life"] = _sig(math.log(2) / kel)

    # Exact, parameter-based constants.
    rates = pk.disposition_rates()
    metrics["CL"] = pk.CL
    metrics["Vc"] = pk.Vc
    metrics["k10"] = _sig(pk.micro_constants()["k10"])
    metrics["model_half_life"] = _sig(math.log(2) / rates["terminal"])
    if rates["distribution"] is not None:
        metrics["distribution_half_life"] = _sig(math.log(2) / rates["distribution"])
    if reg.route == "oral" and pk.ka:
        metrics["absorption_half_life"] = _sig(math.log(2) / pk.ka)

    # Accumulation / steady state for multiple dosing. Accumulation is set
    # by the slowest process: terminal disposition, or absorption if slower
    # (flip-flop kinetics).
    if reg.n_doses > 1 and reg.interval:
        lam = rates["terminal"]
        if reg.route == "oral" and pk.ka:
            lam = min(lam, pk.ka)
        metrics["accumulation_ratio"] = _sig(1.0 / (1.0 - math.exp(-lam * reg.interval)))
        metrics["avg_css"] = _sig(f_sys * reg.dose / (pk.CL * reg.interval))
    return metrics


def simulate_drug(
    drug: Drug,
    regimen: DosingRegimen,
    t_end: Optional[float] = None,
    n_points: int = 2000,
    conc_units: str = "mg/L",
    time_units: str = "h",
) -> SimulationResult:
    """Run PK (+ PD if the drug has PD parameters) and compute metrics."""
    if drug.pk is None:
        raise ValueError(
            f"Drug {drug.name!r} has no PK parameters. Provide pharmacokinetics "
            "before simulating."
        )
    pk_result = simulate_pk(drug.pk, regimen, t_end=t_end, n_points=n_points)
    metrics = compute_nca(pk_result)

    effect = driver = None
    effect_units = ""
    if drug.pd is not None and drug.pd.model != "none":
        driver, effect = simulate_pd(pk_result.t, pk_result.conc, drug.pd)
        effect_units = drug.pd.effect_units
        metrics["Emax_observed"] = _sig(np.max(effect))
        metrics["Emin_observed"] = _sig(np.min(effect))
        # Largest departure from baseline: the peak for stimulatory models,
        # the deepest inhibition for inhibitory ones.
        i_peak = int(np.argmax(np.abs(effect - drug.pd.E0)))
        metrics["time_of_peak_effect"] = _sig(pk_result.t[i_peak])

    return SimulationResult(
        t=pk_result.t,
        conc=pk_result.conc,
        effect=effect,
        effect_driver=driver,
        metrics=metrics,
        dose_times=pk_result.dose_times,
        drug_name=drug.name,
        conc_units=conc_units,
        time_units=time_units,
        effect_units=effect_units,
    )
