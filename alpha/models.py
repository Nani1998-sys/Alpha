"""Core data models for the Alpha drug-testing platform.

These dataclasses describe a drug (its chemical structure + computed
descriptors), its pharmacokinetic (PK) and pharmacodynamic (PD) parameters,
and a dosing regimen. They are plain, JSON-serialisable containers; the
numerical work lives in :mod:`alpha.pk`, :mod:`alpha.pd` and
:mod:`alpha.simulate`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Supported administration routes.
ROUTES = ("iv_bolus", "iv_infusion", "oral")

# Supported PD model names.
PD_MODELS = (
    "none",
    "linear",
    "loglinear",
    "emax",
    "sigmoid_emax",
    "imax",
    "sigmoid_imax",
)


@dataclass
class PKParameters:
    """Pharmacokinetic parameters (clearance/volume parameterisation).

    A one-compartment model needs only ``CL`` and ``Vc`` (plus ``ka``/``F``
    for oral dosing). Supplying ``Q`` and ``Vp`` turns it into a
    two-compartment model.

    Units are up to the user but must be internally consistent. The defaults
    and examples in this project use: clearance in L/h, volumes in L, rate
    constants in 1/h, time in h, dose in mg, giving concentrations in mg/L.
    """

    CL: float                       # systemic clearance (e.g. L/h)
    Vc: float                       # central volume of distribution (e.g. L)
    ka: Optional[float] = None      # first-order absorption rate constant (1/h), oral only
    F: float = 1.0                  # bioavailability fraction (0-1), oral only
    tlag: float = 0.0               # absorption lag time (h), oral only
    Q: Optional[float] = None       # inter-compartmental clearance (L/h), 2-comp only
    Vp: Optional[float] = None      # peripheral volume of distribution (L), 2-comp only

    @property
    def n_compartments(self) -> int:
        return 2 if (self.Q is not None and self.Vp is not None) else 1

    def micro_constants(self) -> Dict[str, float]:
        """Return elimination / distribution micro rate constants (1/time)."""
        k10 = self.CL / self.Vc
        if self.n_compartments == 2:
            k12 = self.Q / self.Vc
            k21 = self.Q / self.Vp
        else:
            k12 = k21 = 0.0
        return {"k10": k10, "k12": k12, "k21": k21}

    def disposition_rates(self) -> Dict[str, float]:
        """Hybrid disposition rate constants.

        For a one-compartment model the only disposition rate is k10. For a
        two-compartment model the plasma curve after a bolus is
        bi-exponential, ``C = A*exp(-alpha*t) + B*exp(-beta*t)``, where
        alpha (fast, distribution) and beta (slow, terminal) are the roots of
        ``x^2 - (k10+k12+k21)*x + k10*k21 = 0``. The *terminal* rate beta,
        not k10, governs the drug's half-life and its accumulation.
        """
        mc = self.micro_constants()
        k10, k12, k21 = mc["k10"], mc["k12"], mc["k21"]
        if self.n_compartments == 1:
            return {"terminal": k10, "distribution": None}
        s = k10 + k12 + k21
        disc = math.sqrt(max(s * s - 4.0 * k10 * k21, 0.0))
        alpha = (s + disc) / 2.0
        # Product form avoids cancellation when beta << alpha.
        beta = (k10 * k21) / alpha
        return {"terminal": beta, "distribution": alpha}

    def validate(self) -> None:
        if self.CL <= 0:
            raise ValueError("CL (clearance) must be > 0")
        if self.Vc <= 0:
            raise ValueError("Vc (central volume) must be > 0")
        if not (0.0 < self.F <= 1.0):
            raise ValueError("F (bioavailability) must be in (0, 1]")
        if self.ka is not None and self.ka <= 0:
            raise ValueError("ka (absorption rate) must be > 0 when provided")
        if (self.Q is None) != (self.Vp is None):
            raise ValueError("Two-compartment model requires BOTH Q and Vp")
        if self.Q is not None and (self.Q <= 0 or self.Vp <= 0):
            raise ValueError("Q and Vp must be > 0 when provided")


@dataclass
class PDParameters:
    """Pharmacodynamic parameters linking concentration to effect.

    ``model`` selects the exposure-response relationship:

    * ``none``         - no PD (PK only)
    * ``linear``       - E = E0 + slope * C
    * ``loglinear``    - E = E0 + slope * ln(C)
    * ``emax``         - E = E0 + Emax * C / (EC50 + C)
    * ``sigmoid_emax`` - E = E0 + Emax * C^h / (EC50^h + C^h)   (Hill)
    * ``imax``         - E = E0 - Imax * C / (IC50 + C)          (inhibitory)
    * ``sigmoid_imax`` - E = E0 - Imax * C^h / (IC50^h + C^h)

    ``ke0`` (optional) enables an *effect compartment* so effect lags plasma
    concentration (hysteresis); the effect model is then driven by the
    effect-site concentration Ce instead of plasma C.
    """

    model: str = "none"
    E0: float = 0.0                 # baseline effect (no drug)
    Emax: Optional[float] = None    # maximal effect (stimulatory models)
    EC50: Optional[float] = None    # concentration for half-maximal effect
    Imax: Optional[float] = None    # maximal inhibition (fraction, 0-1, or absolute)
    IC50: Optional[float] = None    # concentration for half-maximal inhibition
    hill: float = 1.0               # Hill coefficient (sigmoidicity)
    slope: Optional[float] = None   # slope for linear / log-linear models
    ke0: Optional[float] = None     # effect-compartment equilibration rate (1/h)
    effect_units: str = ""          # free-text label for the effect axis

    def validate(self) -> None:
        if self.model not in PD_MODELS:
            raise ValueError(f"Unknown PD model {self.model!r}; choose from {PD_MODELS}")
        if self.model in ("emax", "sigmoid_emax"):
            if self.Emax is None or self.EC50 is None:
                raise ValueError(f"{self.model} requires Emax and EC50")
            if self.EC50 <= 0:
                raise ValueError("EC50 must be > 0")
        if self.model in ("imax", "sigmoid_imax"):
            if self.Imax is None or self.IC50 is None:
                raise ValueError(f"{self.model} requires Imax and IC50")
            if self.IC50 <= 0:
                raise ValueError("IC50 must be > 0")
        if self.model in ("linear", "loglinear") and self.slope is None:
            raise ValueError(f"{self.model} requires slope")
        if self.hill <= 0:
            raise ValueError("Hill coefficient must be > 0")
        if self.ke0 is not None and self.ke0 <= 0:
            raise ValueError("ke0 must be > 0 when provided")


@dataclass
class DosingRegimen:
    """A dosing schedule: amount, route, and repetition."""

    dose: float                          # amount per administration (e.g. mg)
    route: str = "oral"                  # one of ROUTES
    interval: Optional[float] = None     # dosing interval tau (h); None => single dose
    n_doses: int = 1                     # number of doses
    infusion_duration: Optional[float] = None  # h, required for iv_infusion
    start_time: float = 0.0              # time of first dose (h)

    def validate(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"Unknown route {self.route!r}; choose from {ROUTES}")
        if self.dose <= 0:
            raise ValueError("dose must be > 0")
        if self.n_doses < 1:
            raise ValueError("n_doses must be >= 1")
        if self.n_doses > 1 and (self.interval is None or self.interval <= 0):
            raise ValueError("interval must be > 0 for multiple dosing")
        if self.route == "iv_infusion" and (
            self.infusion_duration is None or self.infusion_duration <= 0
        ):
            raise ValueError("iv_infusion requires a positive infusion_duration")

    def dose_times(self) -> List[float]:
        tau = self.interval or 0.0
        return [self.start_time + i * tau for i in range(self.n_doses)]


@dataclass
class Drug:
    """A drug entry: chemical identity + computed descriptors + PK/PD."""

    name: str
    smiles: Optional[str] = None
    inchikey: Optional[str] = None
    descriptors: Dict[str, Any] = field(default_factory=dict)
    pk: Optional[PKParameters] = None
    pd: Optional[PDParameters] = None
    notes: str = ""
    target_name: str = ""        # the protein this drug acts on
    target_pdb: str = ""         # a PDB ID for that protein, for docking
    target_ligand: str = ""      # residue name of a co-crystallised ligand

    # ---- serialisation helpers -------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Drug":
        pk = PKParameters(**d["pk"]) if d.get("pk") else None
        pd = PDParameters(**d["pd"]) if d.get("pd") else None
        return cls(
            name=d["name"],
            smiles=d.get("smiles"),
            inchikey=d.get("inchikey"),
            descriptors=d.get("descriptors", {}) or {},
            pk=pk,
            pd=pd,
            notes=d.get("notes", ""),
            target_name=d.get("target_name", ""),
            target_pdb=d.get("target_pdb", ""),
            target_ligand=d.get("target_ligand", ""),
        )
