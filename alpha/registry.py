"""Drug registry: a simple JSON-backed database of drugs.

The headline feature: register a brand-new drug *just by giving its chemical
structure*. ``add_drug`` parses the SMILES, computes and stores descriptors
automatically, and attaches any provided PK/PD parameters.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from . import chem
from .models import Drug, PDParameters, PKParameters

DEFAULT_DB = os.environ.get(
    "ALPHA_DB", os.path.join(os.path.dirname(__file__), "data", "drugs.json")
)


class DrugNotFoundError(KeyError):
    """Raised when a drug name is not in the registry."""

    def __str__(self) -> str:  # KeyError would wrap the message in quotes
        return str(self.args[0])


class DrugRegistry:
    """A collection of drugs persisted to a JSON file."""

    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        self._drugs: Dict[str, Drug] = {}
        self.load()

    # ---- persistence -----------------------------------------------------
    def load(self) -> None:
        self._drugs = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("drugs", []):
                drug = Drug.from_dict(entry)
                self._drugs[drug.name.lower()] = drug

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        data = {"drugs": [d.to_dict() for d in self._drugs.values()]}
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    # ---- mutation --------------------------------------------------------
    def add_drug(
        self,
        name: str,
        smiles: Optional[str] = None,
        pk: Optional[PKParameters] = None,
        pd: Optional[PDParameters] = None,
        notes: str = "",
        target_name: str = "",
        target_pdb: str = "",
        target_ligand: str = "",
        overwrite: bool = False,
    ) -> Drug:
        """Add (or replace) a drug.

        If a chemical ``smiles`` is given, molecular descriptors are computed
        automatically and stored on the drug.
        """
        key = name.lower()
        if key in self._drugs and not overwrite:
            raise ValueError(
                f"Drug {name!r} already exists. To replace it, pass overwrite=True "
                "(or --overwrite on the command line)."
            )
        descriptors: Dict = {}
        inchikey = None
        if smiles:
            descriptors = chem.analyze_structure(smiles)
            inchikey = descriptors.get("inchikey")
            smiles = descriptors.get("canonical_smiles", smiles)
        if pk is not None:
            pk.validate()
        if pd is not None:
            pd.validate()
        drug = Drug(
            name=name,
            smiles=smiles,
            inchikey=inchikey,
            descriptors=descriptors,
            pk=pk,
            pd=pd,
            notes=notes,
            target_name=target_name,
            target_pdb=target_pdb,
            target_ligand=target_ligand,
        )
        self._drugs[key] = drug
        return drug

    def set_pk(self, name: str, pk: PKParameters) -> Drug:
        drug = self.get(name)
        pk.validate()
        drug.pk = pk
        return drug

    def set_pd(self, name: str, pd: PDParameters) -> Drug:
        drug = self.get(name)
        pd.validate()
        drug.pd = pd
        return drug

    def remove(self, name: str) -> None:
        self._drugs.pop(name.lower(), None)

    # ---- access ----------------------------------------------------------
    def get(self, name: str) -> Drug:
        key = name.lower()
        if key not in self._drugs:
            known = ", ".join(self.list_names()) or "none"
            raise DrugNotFoundError(
                f"No drug named {name!r} in {self.path}. Available: {known}"
            )
        return self._drugs[key]

    def has(self, name: str) -> bool:
        return name.lower() in self._drugs

    def list_names(self) -> List[str]:
        return sorted(d.name for d in self._drugs.values())

    def all(self) -> List[Drug]:
        return list(self._drugs.values())

    def find_by_inchikey(self, inchikey: str) -> Optional[Drug]:
        for d in self._drugs.values():
            if d.inchikey and d.inchikey == inchikey:
                return d
        return None

    def __len__(self) -> int:
        return len(self._drugs)
