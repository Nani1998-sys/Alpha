"""Pharmacodynamics (PD): what the drug does to the body.

Maps a concentration-time course to an effect-time course using a chosen
exposure-response model. Supports direct-response models and an optional
*effect compartment* (link model) that introduces hysteresis, so the effect
lags the plasma concentration.
"""
from __future__ import annotations

import numpy as np

from .models import PDParameters

_EPS = 1e-12


def effect_from_concentration(conc: np.ndarray, pd: PDParameters) -> np.ndarray:
    """Instantaneous (direct) effect for each concentration value."""
    pd.validate()
    c = np.asarray(conc, dtype=float)
    c = np.clip(c, 0.0, None)

    model = pd.model
    if model == "none":
        return np.zeros_like(c)
    if model == "linear":
        return pd.E0 + pd.slope * c
    if model == "loglinear":
        return pd.E0 + pd.slope * np.log(np.clip(c, _EPS, None))
    if model == "emax":
        return pd.E0 + pd.Emax * c / (pd.EC50 + c)
    if model == "sigmoid_emax":
        ch = np.power(c, pd.hill)
        return pd.E0 + pd.Emax * ch / (pd.EC50 ** pd.hill + ch)
    if model == "imax":
        return pd.E0 - pd.Imax * c / (pd.IC50 + c)
    if model == "sigmoid_imax":
        ch = np.power(c, pd.hill)
        return pd.E0 - pd.Imax * ch / (pd.IC50 ** pd.hill + ch)
    raise ValueError(f"Unhandled PD model {model!r}")


def effect_site_concentration(
    t: np.ndarray, conc: np.ndarray, ke0: float
) -> np.ndarray:
    """Integrate the effect-compartment ODE  dCe/dt = ke0 * (C - Ce).

    Uses an exact piecewise-linear-input solution between samples, which is
    accurate and unconditionally stable for any step size.
    """
    t = np.asarray(t, dtype=float)
    c = np.asarray(conc, dtype=float)
    ce = np.zeros_like(c)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            ce[i] = ce[i - 1]
            continue
        # Linear input C(tau) = c0 + m*tau over the interval; exact solution of
        # Ce' = ke0*(C - Ce).
        c0 = c[i - 1]
        m = (c[i] - c[i - 1]) / dt
        e = np.exp(-ke0 * dt)
        ce[i] = (
            ce[i - 1] * e
            + c0 * (1 - e)
            + m * (dt - (1 - e) / ke0)
        )
    return ce


def simulate_pd(t: np.ndarray, conc: np.ndarray, pd: PDParameters):
    """Return (driving_concentration, effect).

    If ``pd.ke0`` is set, the effect is driven by the effect-site concentration
    Ce (hysteresis); otherwise it is driven directly by plasma concentration.
    """
    pd.validate()
    if pd.ke0 is not None:
        driver = effect_site_concentration(t, conc, pd.ke0)
    else:
        driver = np.asarray(conc, dtype=float)
    return driver, effect_from_concentration(driver, pd)
