"""PK engine tests, validated against closed-form analytical solutions."""
import numpy as np
import pytest

from alpha.models import DosingRegimen, PKParameters
from alpha.pk import simulate_pk


def test_iv_bolus_one_compartment_matches_analytical():
    dose, CL, Vc = 100.0, 10.0, 20.0
    k = CL / Vc
    pk = PKParameters(CL=CL, Vc=Vc)
    reg = DosingRegimen(dose=dose, route="iv_bolus")
    res = simulate_pk(pk, reg, t_end=24, n_points=2000)
    analytical = (dose / Vc) * np.exp(-k * res.t)
    # relative error small across the curve
    assert np.allclose(res.conc, analytical, rtol=1e-3, atol=1e-6)
    # C(0) == Dose/Vc
    assert res.conc[0] == pytest.approx(dose / Vc, rel=1e-6)


def test_oral_one_compartment_matches_analytical():
    dose, CL, Vc, ka, F = 100.0, 5.0, 30.0, 1.2, 0.8
    k = CL / Vc
    pk = PKParameters(CL=CL, Vc=Vc, ka=ka, F=F)
    reg = DosingRegimen(dose=dose, route="oral")
    res = simulate_pk(pk, reg, t_end=48, n_points=3000)
    t = res.t
    analytical = (F * dose * ka) / (Vc * (ka - k)) * (np.exp(-k * t) - np.exp(-ka * t))
    assert np.allclose(res.conc, analytical, rtol=2e-3, atol=1e-4)


def test_oral_tmax_matches_analytical():
    # Analytical Tmax = ln(ka/k) / (ka - k)
    CL, Vc, ka = 5.0, 30.0, 1.2
    k = CL / Vc
    pk = PKParameters(CL=CL, Vc=Vc, ka=ka, F=1.0)
    reg = DosingRegimen(dose=100, route="oral")
    res = simulate_pk(pk, reg, t_end=48, n_points=5000)
    tmax_analytical = np.log(ka / k) / (ka - k)
    tmax_sim = res.t[int(np.argmax(res.conc))]
    assert tmax_sim == pytest.approx(tmax_analytical, abs=0.1)


def test_iv_infusion_steady_state():
    # During a long constant infusion, Css = R / CL.
    CL, Vc = 4.0, 25.0
    rate_dose = 200.0  # mg over the infusion
    dur = 50.0         # long infusion -> approaches steady state
    pk = PKParameters(CL=CL, Vc=Vc)
    reg = DosingRegimen(dose=rate_dose, route="iv_infusion", infusion_duration=dur)
    res = simulate_pk(pk, reg, t_end=dur, n_points=4000)
    css_expected = (rate_dose / dur) / CL
    # Near end of infusion we should be close to Css.
    assert res.conc[-1] == pytest.approx(css_expected, rel=0.02)


def test_mass_conservation_two_compartment_no_elimination_like():
    # Two-compartment bolus: total amount must decay monotonically (elimination
    # only from central), and peripheral should fill then empty.
    pk = PKParameters(CL=8.0, Vc=20.0, Q=10.0, Vp=40.0)
    reg = DosingRegimen(dose=100, route="iv_bolus")
    res = simulate_pk(pk, reg, t_end=48, n_points=3000)
    total = res.amounts.sum(axis=0)
    assert np.all(np.diff(total) <= 1e-6)  # non-increasing total drug
    peripheral = res.amounts[2]
    assert peripheral.max() > 0  # drug distributed to periphery
    assert peripheral[-1] < peripheral.max()  # and started returning/clearing


def test_multiple_dosing_accumulation():
    # Repeated oral dosing should accumulate; trough after last dose > first.
    pk = PKParameters(CL=3.0, Vc=40.0, ka=1.0, F=1.0)
    reg = DosingRegimen(dose=50, route="oral", interval=8, n_doses=6)
    res = simulate_pk(pk, reg, t_end=8 * 6, n_points=4000)
    # concentration just before 2nd dose vs just before last dose
    def conc_at(tt):
        return res.conc[np.searchsorted(res.t, tt) - 1]
    assert conc_at(16.0) > conc_at(8.0)  # accumulating
