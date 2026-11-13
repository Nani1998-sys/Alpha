"""PD model tests."""
import numpy as np
import pytest

from alpha.models import PDParameters
from alpha.pd import effect_from_concentration, effect_site_concentration, simulate_pd


def test_emax_half_maximal_at_ec50():
    pd = PDParameters(model="emax", E0=0, Emax=100, EC50=5.0)
    e = effect_from_concentration(np.array([0.0, 5.0, 1e6]), pd)
    assert e[0] == pytest.approx(0.0)
    assert e[1] == pytest.approx(50.0)           # half-maximal at EC50
    assert e[2] == pytest.approx(100.0, rel=1e-3)  # approaches Emax


def test_sigmoid_emax_hill_steepness():
    shallow = PDParameters(model="sigmoid_emax", E0=0, Emax=100, EC50=5, hill=1)
    steep = PDParameters(model="sigmoid_emax", E0=0, Emax=100, EC50=5, hill=4)
    c = np.array([2.5, 5.0, 10.0])
    es = effect_from_concentration(c, steep)
    el = effect_from_concentration(c, shallow)
    assert es[1] == pytest.approx(50.0)  # still half-maximal at EC50
    # steeper curve is lower below EC50 and higher above EC50
    assert es[0] < el[0]
    assert es[2] > el[2]


def test_imax_inhibition():
    pd = PDParameters(model="imax", E0=100, Imax=80, IC50=2.0)
    e = effect_from_concentration(np.array([0.0, 2.0]), pd)
    assert e[0] == pytest.approx(100.0)
    assert e[1] == pytest.approx(60.0)  # 100 - 80*2/(2+2) = 60


def test_linear_and_loglinear():
    lin = PDParameters(model="linear", E0=1.0, slope=2.0)
    assert effect_from_concentration(np.array([3.0]), lin)[0] == pytest.approx(7.0)
    logl = PDParameters(model="loglinear", E0=0.0, slope=1.0)
    assert effect_from_concentration(np.array([np.e]), logl)[0] == pytest.approx(1.0)


def test_effect_compartment_tracks_input_at_steady_state():
    # With constant concentration, Ce converges to C.
    t = np.linspace(0, 50, 5000)
    c = np.full_like(t, 10.0)
    ce = effect_site_concentration(t, c, ke0=0.5)
    assert ce[-1] == pytest.approx(10.0, rel=1e-3)
    assert ce[0] == pytest.approx(0.0)


def test_effect_compartment_introduces_hysteresis():
    # Rising-then-falling concentration: peak of Ce lags peak of C.
    t = np.linspace(0, 24, 4000)
    c = 10 * (np.exp(-0.2 * t) - np.exp(-1.0 * t))  # oral-like shape
    ce = effect_site_concentration(t, c, ke0=0.3)
    assert t[int(np.argmax(ce))] > t[int(np.argmax(c))]
