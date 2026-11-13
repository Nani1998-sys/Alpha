"""Protein structures: fetch, parse and inspect 3D macromolecular structures.

Two sources, both free and public:

* **RCSB Protein Data Bank** (https://www.rcsb.org) - experimentally determined
  structures (X-ray, cryo-EM, NMR). Fetched by 4-character PDB ID.
* **AlphaFold DB** (https://alphafold.ebi.ac.uk) - computationally *predicted*
  structures covering most known proteins, keyed by UniProt accession. These
  are predictions, not measurements; every residue carries a pLDDT confidence
  score (stored in the B-factor column) and low-pLDDT regions are unreliable.

Downloads are cached on disk, so a structure is fetched once and then works
offline.

Only the standard library is used here, so importing this module costs
nothing. Docking (:mod:`alpha.docking`) and 3D viewing (:mod:`alpha.viewer3d`)
build on it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

RCSB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_ENTRY_API = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".alpha", "structures")

# Residues that are part of the macromolecule rather than a bound ligand.
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "ASX", "GLX", "UNK",
}
NUCLEOTIDES = {"A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "DI"}
SOLVENT = {"HOH", "DOD", "WAT", "SOL"}
# Common crystallisation additives: present because of how the crystal was
# grown, not because they are biologically meaningful binders.
CRYSTALLISATION_ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "MPD", "DMS", "TRS", "MES",
    "EPE", "ACT", "ACY", "FMT", "CIT", "TLA", "IMD", "BME", "NH4", "IOD",
    "CL", "BR", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CD",
    "CO", "HG", "PT", "AU", "CS", "RB", "SR", "BA", "LI", "F",
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


class StructureError(RuntimeError):
    """Raised when a structure cannot be fetched or parsed."""


# ---------------------------------------------------------------------------
# Atom / residue / ligand records
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occupancy: float
    bfactor: float           # for AlphaFold models this is the pLDDT score
    element: str
    hetatm: bool
    line: str                # the original PDB line

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def residue_key(self) -> Tuple[str, int, str, str]:
        return (self.chain, self.resseq, self.icode, self.resname)


@dataclass
class Ligand:
    """A non-polymer group bound to the structure (a HETATM residue)."""

    resname: str
    chain: str
    resseq: int
    atoms: List[Atom] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_heavy_atoms(self) -> int:
        return sum(1 for a in self.atoms if a.element not in ("H", "D"))

    @property
    def is_solvent(self) -> bool:
        return self.resname in SOLVENT

    @property
    def is_additive(self) -> bool:
        return self.resname in CRYSTALLISATION_ADDITIVES

    @property
    def label(self) -> str:
        return f"{self.resname}:{self.chain}{self.resseq}"

    def centroid(self) -> Tuple[float, float, float]:
        n = len(self.atoms)
        if not n:
            raise StructureError(f"Ligand {self.label} has no atoms")
        return (sum(a.x for a in self.atoms) / n,
                sum(a.y for a in self.atoms) / n,
                sum(a.z for a in self.atoms) / n)

    def extent(self) -> Tuple[float, float, float]:
        xs = [a.x for a in self.atoms]
        ys = [a.y for a in self.atoms]
        zs = [a.z for a in self.atoms]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    def to_pdb(self) -> str:
        return "".join(a.line for a in self.atoms) + "END\n"


@dataclass
class ProteinStructure:
    """A parsed 3D structure."""

    identifier: str
    pdb_text: str
    atoms: List[Atom]
    source: str = "local"            # "RCSB", "AlphaFold" or "local"
    title: str = ""
    method: str = ""
    resolution: Optional[float] = None
    uniprot: Optional[str] = None

    # ---- composition ---------------------------------------------------
    @property
    def is_predicted(self) -> bool:
        return self.source == "AlphaFold"

    @property
    def polymer_atoms(self) -> List[Atom]:
        return [a for a in self.atoms if not a.hetatm and a.element != "H"]

    @property
    def chains(self) -> List[str]:
        seen = []
        for a in self.atoms:
            if not a.hetatm and a.chain not in seen:
                seen.append(a.chain)
        return seen

    def residue_count(self, chain: Optional[str] = None) -> int:
        keys = {(a.chain, a.resseq, a.icode) for a in self.atoms
                if not a.hetatm and (chain is None or a.chain == chain)}
        return len(keys)

    def sequence(self, chain: Optional[str] = None) -> str:
        """One-letter amino-acid sequence, in file order."""
        out, seen = [], set()
        for a in self.atoms:
            if a.hetatm or a.resname not in AMINO_ACIDS:
                continue
            if chain is not None and a.chain != chain:
                continue
            key = (a.chain, a.resseq, a.icode)
            if key in seen:
                continue
            seen.add(key)
            out.append(THREE_TO_ONE.get(a.resname, "X"))
        return "".join(out)

    def ligands(self, include_solvent: bool = False,
                include_additives: bool = False,
                min_heavy_atoms: int = 1) -> List[Ligand]:
        """Bound non-polymer groups, largest first.

        By default waters and common crystallisation additives are left out,
        because they are usually artefacts of how the crystal was grown
        rather than meaningful binders.
        """
        groups: Dict[Tuple[str, int, str], Ligand] = {}
        for a in self.atoms:
            if not a.hetatm or a.resname in AMINO_ACIDS or a.resname in NUCLEOTIDES:
                continue
            key = (a.chain, a.resseq, a.resname)
            groups.setdefault(key, Ligand(a.resname, a.chain, a.resseq)).atoms.append(a)
        out = []
        for lig in groups.values():
            if lig.is_solvent and not include_solvent:
                continue
            if lig.is_additive and not include_additives:
                continue
            if lig.n_heavy_atoms < min_heavy_atoms:
                continue
            out.append(lig)
        return sorted(out, key=lambda g: -g.n_heavy_atoms)

    def waters(self) -> List[Ligand]:
        return [l for l in self.ligands(include_solvent=True, include_additives=True)
                if l.is_solvent]

    def get_ligand(self, resname: str, chain: Optional[str] = None) -> Ligand:
        for lig in self.ligands(include_solvent=True, include_additives=True):
            if lig.resname.upper() == resname.upper() and (chain is None or lig.chain == chain):
                return lig
        raise StructureError(
            f"No ligand {resname!r} in {self.identifier}. "
            f"Present: {', '.join(l.label for l in self.ligands()) or 'none'}")

    # ---- confidence (AlphaFold) ---------------------------------------
    def plddt_summary(self) -> Optional[Dict[str, float]]:
        """Mean pLDDT and the fraction of residues above the usual
        confidence thresholds. Only meaningful for AlphaFold models."""
        if not self.is_predicted:
            return None
        per_res: Dict[Tuple[str, int, str], float] = {}
        for a in self.polymer_atoms:
            per_res.setdefault((a.chain, a.resseq, a.icode), a.bfactor)
        vals = list(per_res.values())
        if not vals:
            return None
        n = len(vals)
        return {
            "mean_plddt": round(sum(vals) / n, 1),
            "very_high_>90": round(sum(v > 90 for v in vals) / n, 3),
            "confident_70_90": round(sum(70 <= v <= 90 for v in vals) / n, 3),
            "low_50_70": round(sum(50 <= v < 70 for v in vals) / n, 3),
            "very_low_<50": round(sum(v < 50 for v in vals) / n, 3),
        }

    # ---- geometry ------------------------------------------------------
    def residues_near(self, center: Sequence[float], radius: float = 5.0,
                      exclude_solvent: bool = True) -> List[Dict[str, object]]:
        """Polymer residues with any atom within ``radius`` of a point."""
        cx, cy, cz = center
        r2 = radius * radius
        best: Dict[Tuple[str, int, str, str], float] = {}
        for a in self.atoms:
            if a.hetatm or a.element == "H":
                continue
            if exclude_solvent and a.resname in SOLVENT:
                continue
            d2 = (a.x - cx) ** 2 + (a.y - cy) ** 2 + (a.z - cz) ** 2
            if d2 <= r2:
                k = a.residue_key
                if k not in best or d2 < best[k]:
                    best[k] = d2
        out = [{"chain": c, "resseq": s, "icode": i, "resname": n,
                "min_distance": round(d ** 0.5, 2)}
               for (c, s, i, n), d in best.items()]
        return sorted(out, key=lambda r: r["min_distance"])

    def pocket_residues(self, ligand: Ligand, radius: float = 4.5) -> List[Dict[str, object]]:
        """Residues lining the pocket occupied by a bound ligand."""
        r2 = radius * radius
        best: Dict[Tuple[str, int, str, str], float] = {}
        lig_xyz = [(a.x, a.y, a.z) for a in ligand.atoms if a.element not in ("H", "D")]
        for a in self.atoms:
            if a.hetatm or a.element == "H" or a.resname in SOLVENT:
                continue
            d2 = min((a.x - x) ** 2 + (a.y - y) ** 2 + (a.z - z) ** 2
                     for x, y, z in lig_xyz)
            if d2 <= r2:
                k = a.residue_key
                if k not in best or d2 < best[k]:
                    best[k] = d2
        out = [{"chain": c, "resseq": s, "icode": i, "resname": n,
                "min_distance": round(d ** 0.5, 2)}
               for (c, s, i, n), d in best.items()]
        return sorted(out, key=lambda r: (r["chain"], r["resseq"]))

    # ---- export --------------------------------------------------------
    def to_pdb(self, polymer_only: bool = False,
               keep_resnames: Iterable[str] = (),
               keep_residues: Iterable[str] = (),
               keep_hydrogens: bool = True) -> str:
        """PDB text for this structure, optionally stripped down.

        ``polymer_only`` drops every HETATM group except those matched by
        ``keep_resnames`` (every group of that name, e.g. ``["HEM"]``) or by
        ``keep_residues`` (individual groups as ``"CHAIN:RESSEQ"``, e.g.
        ``["B:301"]``). Use ``keep_residues`` for a single structural water:
        keeping every water would fill the binding site.
        """
        keep_names = {r.upper() for r in keep_resnames}
        keep_ids = {str(r).upper().replace(" ", "") for r in keep_residues}
        lines = []
        for a in self.atoms:
            if not keep_hydrogens and a.element in ("H", "D"):
                continue
            if polymer_only and a.hetatm:
                rid = f"{a.chain}:{a.resseq}".upper()
                if a.resname.upper() not in keep_names and rid not in keep_ids:
                    continue
            lines.append(a.line)
        lines.append("END\n")
        return "".join(lines)

    def bridging_waters(self, ligand: "Ligand", max_distance: float = 3.5,
                        min_protein_contacts: int = 1) -> List[Dict[str, object]]:
        """Waters that sit between a ligand and the protein.

        A water hydrogen-bonded to both the ligand and the protein is part of
        the binding site rather than bulk solvent - HIV-1 protease's "flap
        water" is the classic example - and keeping it usually improves
        docking. Returns the closest first, as ``"CHAIN:RESSEQ"`` ids ready
        to pass to ``to_pdb(keep_residues=...)`` or ``dock(keep_residues=...)``.
        """
        lig_xyz = [(a.x, a.y, a.z) for a in ligand.atoms if a.element not in ("H", "D")]
        if not lig_xyz:
            return []
        prot = [a for a in self.atoms
                if not a.hetatm and a.element in ("N", "O")]
        out = []
        for wat in self.waters():
            for a in wat.atoms:
                if a.element not in ("O", "N"):
                    continue
                d_lig = min(((a.x - x) ** 2 + (a.y - y) ** 2 + (a.z - z) ** 2) ** 0.5
                            for x, y, z in lig_xyz)
                if d_lig > max_distance:
                    continue
                n_prot = sum(1 for p in prot
                             if ((a.x - p.x) ** 2 + (a.y - p.y) ** 2
                                 + (a.z - p.z) ** 2) ** 0.5 <= max_distance)
                if n_prot >= min_protein_contacts:
                    out.append({"residue": f"{wat.chain}:{wat.resseq}",
                                "distance_to_ligand": round(d_lig, 2),
                                "protein_contacts": n_prot})
                    break
        return sorted(out, key=lambda w: w["distance_to_ligand"])


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------
def _f(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def parse_pdb(text: str, identifier: str = "structure",
              source: str = "local") -> ProteinStructure:
    """Parse PDB-format text into a :class:`ProteinStructure`.

    Only the first model of a multi-model file (NMR ensemble) is kept, and
    alternate locations other than the first are discarded, which is what
    docking and visualisation expect.
    """
    atoms: List[Atom] = []
    title_parts: List[str] = []
    method = ""
    resolution = None
    in_later_model = False
    seen_altloc: Dict[Tuple[str, int, str, str], str] = {}

    for line in text.splitlines(keepends=True):
        rec = line[:6]
        if rec == "MODEL ":
            in_later_model = line.split()[1].strip() != "1" if len(line.split()) > 1 else False
            continue
        if rec == "ENDMDL":
            in_later_model = True
            continue
        if rec == "TITLE ":
            title_parts.append(line[10:].strip())
            continue
        if rec == "EXPDTA":
            method = line[10:].strip()
            continue
        if line.startswith("REMARK   2 RESOLUTION."):
            for tok in line.split():
                v = _f(tok, -1.0)
                if v > 0:
                    resolution = v
                    break
            continue
        if rec not in ("ATOM  ", "HETATM") or in_later_model:
            continue
        try:
            altloc = line[16]
            key = (line[21], int(line[22:26]), line[26], line[12:16].strip())
            if altloc.strip():
                if key in seen_altloc and seen_altloc[key] != altloc:
                    continue
                seen_altloc[key] = altloc
            element = line[76:78].strip() or line[12:16].strip()[:1]
            atoms.append(Atom(
                serial=int(line[6:11]), name=line[12:16].strip(), altloc=altloc,
                resname=line[17:20].strip(), chain=line[21],
                resseq=int(line[22:26]), icode=line[26],
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                occupancy=_f(line[54:60], 1.0), bfactor=_f(line[60:66]),
                element=element.capitalize() if len(element) == 2 else element.upper(),
                hetatm=(rec == "HETATM"), line=line if line.endswith("\n") else line + "\n",
            ))
        except (ValueError, IndexError):
            continue

    if not atoms:
        raise StructureError(f"No atoms found in {identifier}")
    return ProteinStructure(
        identifier=identifier, pdb_text=text, atoms=atoms, source=source,
        title=" ".join(title_parts).strip(), method=method, resolution=resolution,
    )


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 30.0, as_json: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": "alpha-pkpd/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise StructureError(f"{url} returned HTTP {exc.code}") from exc
    except Exception as exc:
        raise StructureError(
            f"Could not reach {url}: {exc}. Check your internet connection; "
            "cached structures still work offline.") from exc
    return json.loads(raw) if as_json else raw.decode("utf-8", "replace")


def _cache_path(cache_dir: Optional[str], name: str) -> Optional[str]:
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def fetch_pdb(pdb_id: str, cache_dir: Optional[str] = DEFAULT_CACHE,
              with_metadata: bool = True) -> ProteinStructure:
    """Download an experimental structure from the RCSB PDB by its 4-character ID."""
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id[0].isdigit():
        raise StructureError(
            f"{pdb_id!r} is not a PDB ID. A PDB ID is 4 characters starting "
            "with a digit, for example 1HSG. To use a UniProt accession such "
            "as P00533, call fetch_alphafold() instead.")
    path = _cache_path(cache_dir, f"{pdb_id}.pdb")
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read()
    else:
        text = _http_get(RCSB_FILE_URL.format(pdb_id=pdb_id))
        if path:
            open(path, "w", encoding="utf-8").write(text)
    struct = parse_pdb(text, identifier=pdb_id, source="RCSB")
    if with_metadata:
        try:
            meta = _http_get(RCSB_ENTRY_API.format(pdb_id=pdb_id), as_json=True)
            struct.title = meta.get("struct", {}).get("title", struct.title)
            exptl = meta.get("exptl") or []
            if exptl:
                struct.method = exptl[0].get("method", struct.method)
            res = (meta.get("rcsb_entry_info") or {}).get("resolution_combined")
            if res:
                struct.resolution = res[0]
        except StructureError:
            pass  # metadata is a nicety; the coordinates are what matter
    return struct


def fetch_alphafold(accession: str, cache_dir: Optional[str] = DEFAULT_CACHE
                    ) -> ProteinStructure:
    """Download an AlphaFold *predicted* structure by UniProt accession.

    The B-factor column holds the per-residue pLDDT confidence (0-100). Use
    :meth:`ProteinStructure.plddt_summary` to check it: regions below 70 are
    unreliable and below 50 are often disordered.
    """
    accession = accession.strip().upper()
    path = _cache_path(cache_dir, f"AF-{accession}.pdb")
    if path and os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        struct = parse_pdb(text, identifier=f"AF-{accession}", source="AlphaFold")
        struct.uniprot = accession
        return struct
    entries = _http_get(ALPHAFOLD_API.format(accession=accession), as_json=True)
    if not entries:
        raise StructureError(f"AlphaFold has no model for UniProt {accession!r}")
    entry = entries[0]
    url = entry.get("pdbUrl")
    if not url:
        raise StructureError(f"AlphaFold entry for {accession} has no PDB file")
    text = _http_get(url)
    if path:
        open(path, "w", encoding="utf-8").write(text)
    struct = parse_pdb(text, identifier=entry.get("modelEntityId", f"AF-{accession}"),
                       source="AlphaFold")
    struct.uniprot = accession
    struct.title = entry.get("uniprotDescription", "") or struct.title
    struct.method = f"AlphaFold prediction ({entry.get('latestVersion', '')})".strip()
    return struct


def search_pdb(query: str, limit: int = 10) -> List[Dict[str, object]]:
    """Full-text search of the RCSB PDB. Returns entry IDs with match scores."""
    payload = {
        "query": {"type": "terminal", "service": "full_text",
                  "parameters": {"value": query}},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": int(limit)}},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        RCSB_SEARCH_API, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "alpha-pkpd/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return []
        raise StructureError(f"PDB search failed: HTTP {exc.code}") from exc
    except Exception as exc:
        raise StructureError(f"PDB search failed: {exc}") from exc
    return [{"pdb_id": r["identifier"], "score": round(r.get("score", 0.0), 3)}
            for r in body.get("result_set", [])]
