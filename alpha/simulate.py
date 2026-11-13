"""Orchestration: run a full PK (+ optional PD) simulation for a drug and
compute summary exposure metrics (non-compartmental analysis, NCA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .models import Drug, DosingRegimen, PDParameters, PKParameters

# NumPy renamed trapz -> trapezoid in 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz
from .pd import simulate_pd
from .pk import PKResult, simulate_pk


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
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _terminal_kel(t: np.ndarray, conc: np.ndarray) -> Optional[float]:
    """Estimate the terminal elimination rate constant by log-linear
    regression over the terminal (post-peak, declining) phase."""
    if len(t) < 3:
        return None
    imax = int(np.argmax(conc))
    tail_t = t[imax:]
    tail_c = conc[imax:]
    mask = tail_c > 0
    tail_t, tail_c = tail_t[mask], tail_c[mask]
    if len(tail_t) < 3:
        return None
    # Use the last ~40% of the declining tail for the terminal slope.
    n = len(tail_t)
    start = max(0, int(n * 0.6))
    xt, xc = tail_t[start:], tail_c[start:]
    if len(xt) < 2 or np.ptp(xt) == 0:
        return None
    slope, _ = np.polyfit(xt, np.log(xc), 1)
    kel = -slope
    return kel if kel > 0 else None


def compute_nca(pk_result: PKResult) -> Dict[str, Any]:
    """Non-compartmental analysis metrics from a concentration-time course."""
    t, c = pk_result.t, pk_result.conc
    imax = int(np.argmax(c))
    cmax = float(c[imax])
    tmax = float(t[imax])
    auc_last = float(_trapz(c, t))

    kel = _terminal_kel(t, c)
    metrics: Dict[str, Any] = {
        "Cmax": round(cmax, 4),
        "Tmax": round(tmax, 3),
        "AUC_last": round(auc_last, 4),
        "C_last": round(float(c[-1]), 4),
    }
    if kel:
        thalf = np.log(2) / kel
        auc_inf = auc_last + c[-1] / kel
        metrics.update(
            {
                "terminal_kel": round(kel, 5),
                "terminal_half_life": round(thalf, 3),
                "AUC_inf": round(auc_inf, 4),
            }
        )

    # Model-based constants (exact, from parameters).
    pk = pk_result.pk
    mc = pk.micro_constants()
    metrics["CL"] = pk.CL
    metrics["Vc"] = pk.Vc
    metrics["k10"] = round(mc["k10"], 5)
    metrics["model_half_life"] = round(np.log(2) / mc["k10"], 3)

    # Accumulation / steady-state hints for multiple dosing.
    reg = pk_result.regimen
    if reg.n_doses > 1 and reg.interval:
        tau = reg.interval
        acc = 1.0 / (1.0 - np.exp(-mc["k10"] * tau))
        metrics["accumulation_ratio"] = round(acc, 3)
        metrics["avg_css"] = round(
            (reg.dose * (pk.F if reg.route == "oral" else 1.0)) / (pk.CL * tau), 4
        )
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
        metrics["Emax_observed"] = round(float(np.max(effect)), 4)
        metrics["Emin_observed"] = round(float(np.min(effect)), 4)
        metrics["time_of_peak_effect"] = round(
            float(pk_result.t[int(np.argmax(effect))]), 3
        )

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
