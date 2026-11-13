"""Reporting: text summaries and matplotlib plots of simulation results."""
from __future__ import annotations

from typing import Optional

from .models import Drug
from .simulate import SimulationResult


def text_report(drug: Drug, result: SimulationResult) -> str:
    """A human-readable summary of a simulation."""
    lines = []
    lines.append("=" * 64)
    lines.append(f"  ALPHA simulation report: {drug.name}")
    lines.append("=" * 64)
    if drug.smiles:
        lines.append(f"Structure (SMILES): {drug.smiles}")
    d = drug.descriptors or {}
    if d:
        lines.append(f"Formula: {d.get('formula', '?')}   "
                     f"MW: {d.get('mol_weight', '?')}   "
                     f"logP: {d.get('logp', '?')}   "
                     f"TPSA: {d.get('tpsa', '?')}")
        dl = d.get("drug_likeness", {})
        if dl:
            lip = dl.get("lipinski", {})
            lines.append(f"Lipinski Ro5: {'PASS' if lip.get('pass') else 'FAIL'} "
                         f"({lip.get('violations', '?')} violation(s))   "
                         f"QED: {d.get('qed', '?')}")

    lines.append("-" * 64)
    lines.append(f"Doses given at (h): {', '.join(f'{x:g}' for x in result.dose_times)}")
    lines.append(f"Concentration units: {result.conc_units}   "
                 f"Time units: {result.time_units}")
    lines.append("-" * 64)
    lines.append("Exposure metrics (NCA + model):")
    for k, v in result.metrics.items():
        lines.append(f"  {k:<22}: {v}")
    if result.effect is not None:
        lines.append("-" * 64)
        lines.append(f"PD model: {drug.pd.model}"
                     + (f" (effect units: {result.effect_units})"
                        if result.effect_units else ""))
    lines.append("=" * 64)
    return "\n".join(lines)


def plot_simulation(
    result: SimulationResult,
    path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
):
    """Plot concentration-time (and effect-time, if present). Returns the
    matplotlib Figure. Requires matplotlib."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_pd = result.effect is not None
    n = 2 if has_pd else 1
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(result.t, result.conc, color="#2563eb", lw=2, label="Plasma conc.")
    ax.set_ylabel(f"Concentration ({result.conc_units})")
    ax.set_title(title or f"{result.drug_name}: pharmacokinetics")
    for dt in result.dose_times:
        ax.axvline(dt, color="#94a3b8", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if has_pd:
        ax2 = axes[1]
        ax2.plot(result.t, result.effect, color="#dc2626", lw=2, label="Effect")
        lbl = f"Effect ({result.effect_units})" if result.effect_units else "Effect"
        ax2.set_ylabel(lbl)
        ax2.set_title(f"{result.drug_name}: pharmacodynamics")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel(f"Time ({result.time_units})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig
