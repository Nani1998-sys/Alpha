"""Regression tests for bugs found in the error-testing pass.

Each test pins down one defect that was reproduced, then fixed. The PK
checks compare against closed-form analytical solutions.
"""
import math

import numpy as np
import pytest

from alpha.models import Drug, DosingRegimen, PDParameters, PKParameters
from alpha.pk import simulate_pk
from alpha.simulate import compute_nca, simulate_drug


def rel_err(sim, exact):
    return np.max(np.abs(sim - exact)) / np.max(np.abs(exact))


# --- Bug: metrics rounded to 4 decimal places lost low concentrations -------
def test_low_concentration_metrics_keep_precision():
    """Digoxin-like exposure (~3e-4 mg/L) used to report Cmax=0.0003 (-10%)."""
    pk = PKParameters(CL=7, Vc=500, ka=1.5, F=0.7)
    r = simulate_pk(pk, DosingRegimen(dose=0.25), t_end=72)
    m = compute_nca(r)
    assert m["Cmax"] == pytest.approx(r.conc.max(), rel=1e-5)
    assert m["C_last"] == pytest.approx(r.conc[-1], rel=1e-5)


def test_tiny_dose_metrics_not_rounded_to_zero():
    r = simulate_pk(PKParameters(CL=3, Vc=30, ka=1), DosingRegimen(dose=1e-6), t_end=48)
    assert compute_nca(r)["Cmax"] > 0


# --- Bug: fixed-step integrator blew up on fast (stiff) kinetics ------------
@pytest.mark.parametrize("t_end", [5, 1000, 5000])
def test_fast_elimination_is_stable_and_exact(t_end):
    """k = 200/h. The old RK4 fallback (and the web engine) overflowed."""
    r = simulate_pk(PKParameters(CL=1000, Vc=5), DosingRegimen(dose=10, route="iv_bolus"),
                    t_end=t_end)
    assert np.all(np.isfinite(r.conc))
    assert rel_err(r.conc, 2.0 * np.exp(-200 * r.t)) < 1e-9


def test_fast_absorption_is_stable_and_exact():
    ka, k = 300.0, 0.1
    r = simulate_pk(PKParameters(CL=3, Vc=30, ka=ka), DosingRegimen(dose=10), t_end=24)
    exact = 10 * ka / (30 * (ka - k)) * (np.exp(-k * r.t) - np.exp(-ka * r.t))
    assert rel_err(r.conc, exact) < 1e-9


# --- Bug: 2-compartment half-life and accumulation used k10, not beta -------
def test_two_compartment_matches_biexponential():
    pk = PKParameters(CL=108, Vc=20, Q=40, Vp=120)
    rates = pk.disposition_rates()
    a, b, k21, dose = rates["distribution"], rates["terminal"], 40 / 120, 100.0
    r = simulate_pk(pk, DosingRegimen(dose=dose, route="iv_bolus"), t_end=24)
    exact = dose / 20 * ((a - k21) / (a - b) * np.exp(-a * r.t)
                         + (k21 - b) / (a - b) * np.exp(-b * r.t))
    assert rel_err(r.conc, exact) < 1e-9


def test_two_compartment_half_life_and_accumulation_use_terminal_rate():
    pk = PKParameters(CL=108, Vc=20, Q=40, Vp=120)  # propofol-like
    beta = pk.disposition_rates()["terminal"]
    r = simulate_pk(pk, DosingRegimen(dose=100, route="iv_bolus", interval=6, n_doses=4),
                    t_end=48)
    m = compute_nca(r)
    assert m["model_half_life"] == pytest.approx(math.log(2) / beta, rel=1e-5)   # ~2.885 h
    assert m["model_half_life"] == pytest.approx(m["terminal_half_life"], rel=0.01)
    assert m["accumulation_ratio"] == pytest.approx(1 / (1 - math.exp(-beta * 6)), rel=1e-5)
    assert "distribution_half_life" in m


def test_flip_flop_accumulation_uses_absorption_rate():
    """When ka < k, absorption is the slow process and governs accumulation."""
    pk = PKParameters(CL=10, Vc=10, ka=0.1)  # k = 1/h, ka = 0.1/h
    r = simulate_pk(pk, DosingRegimen(dose=10, interval=12, n_doses=3), t_end=48)
    assert compute_nca(r)["accumulation_ratio"] == pytest.approx(
        1 / (1 - math.exp(-0.1 * 12)), rel=1e-5)


# --- Bug: repeat IV bolus peaks were dropped when segments were stitched -----
def test_repeat_iv_bolus_keeps_trough_and_peak():
    D, V, k, tau, n = 100.0, 20.0, 0.5, 6.0, 4
    r = simulate_pk(PKParameters(CL=k * V, Vc=V),
                    DosingRegimen(dose=D, route="iv_bolus", interval=tau, n_doses=n), t_end=30)
    q = math.exp(-k * tau)
    assert r.conc.max() == pytest.approx(D / V * (1 - q ** n) / (1 - q), rel=1e-9)
    at_18 = r.conc[np.isclose(r.t, 18.0)]                      # 4th dose time
    trough = D / V * (q + q ** 2 + q ** 3)
    assert sorted(at_18) == pytest.approx([trough, trough + D / V], rel=1e-9)


# --- Bug: AUC was a trapezoid estimate on the output grid --------------------
def test_auc_is_exact():
    D, V, k = 100.0, 20.0, 0.5
    r = simulate_pk(PKParameters(CL=k * V, Vc=V),
                    DosingRegimen(dose=D, route="iv_bolus", interval=6, n_doses=4), t_end=30)
    exact = sum(D / (V * k) * (1 - math.exp(-k * (30 - td))) for td in (0, 6, 12, 18))
    m = compute_nca(r)
    assert m["AUC_last"] == pytest.approx(exact, rel=1e-6)
    assert m["AUC_inf"] == pytest.approx(4 * D / (k * V), rel=1e-6)  # mass balance


def test_infusion_matches_analytical_during_and_after():
    R, T, CL, Vc = 50.0, 4.0, 5.0, 25.0
    k = CL / Vc
    r = simulate_pk(PKParameters(CL=CL, Vc=Vc),
                    DosingRegimen(dose=R * T, route="iv_infusion", infusion_duration=T), t_end=24)
    c_T = R / CL * (1 - math.exp(-k * T))
    exact = np.where(r.t <= T, R / CL * (1 - np.exp(-k * r.t)), c_T * np.exp(-k * (r.t - T)))
    assert rel_err(r.conc, exact) < 1e-9


def test_lag_time_shifts_curve():
    ka, k, lag = 1.2, 5 / 30, 1.5
    r = simulate_pk(PKParameters(CL=5, Vc=30, ka=ka, F=1, tlag=lag), DosingRegimen(dose=100),
                    t_end=48)
    s = np.clip(r.t - lag, 0, None)
    exact = 100 * ka / (30 * (ka - k)) * (np.exp(-k * s) - np.exp(-ka * s))
    assert rel_err(r.conc, exact) < 1e-9


# --- Bug: time of peak effect ignored inhibitory models ----------------------
def test_time_of_peak_effect_for_inhibitory_model():
    drug = Drug(name="I", pk=PKParameters(CL=3, Vc=30, ka=1),
                pd=PDParameters(model="imax", E0=100, Imax=80, IC50=0.1))
    res = simulate_drug(drug, DosingRegimen(dose=10), t_end=48)
    assert res.metrics["time_of_peak_effect"] == pytest.approx(
        res.t[int(np.argmin(res.effect))], rel=1e-5)
    assert res.metrics["time_of_peak_effect"] > 0
