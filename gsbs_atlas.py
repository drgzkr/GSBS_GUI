"""
Schaefer 2018 parcellation: names, networks, and the official fsaverage
surface annotations.

Parcel-level values are painted on the surface through the annotation that
Schaefer et al. released for fsaverage (CBIG repository), not by projecting the
volumetric atlas with vol_to_surf: that projection assigns a different parcel
than the annotation to about a quarter of fsaverage vertices. The annotation's
parcel order is the same as the volumetric atlas's label order, so column i of
a Schaefer timeseries is annotation parcel i + 1.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

N_ROIS_OPTIONS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
NETWORK_OPTIONS = (7, 17)

_CBIG_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/"
    "{mesh}/label/{hemi}.Schaefer2018_{n}Parcels_{k}Networks_order.annot"
)


@dataclass(frozen=True)
class Schaefer:
    n_rois: int = 400
    yeo_networks: int = 17

    def __post_init__(self):
        if self.n_rois not in N_ROIS_OPTIONS or self.yeo_networks not in NETWORK_OPTIONS:
            raise ValueError(f"No Schaefer atlas with {self.n_rois} parcels and "
                             f"{self.yeo_networks} networks.")

    @property
    def label(self) -> str:
        return f"Schaefer {self.n_rois} / {self.yeo_networks} networks"

    def to_meta(self) -> dict:
        return {"atlas": "schaefer", "n_rois": self.n_rois, "yeo_networks": self.yeo_networks}

    @classmethod
    def from_meta(cls, meta: dict | None) -> "Schaefer | None":
        if not meta or meta.get("atlas") != "schaefer":
            return None
        return cls(int(meta["n_rois"]), int(meta["yeo_networks"]))


def cache_dir() -> Path:
    """Shared with nilearn's data directory (and neuro_tools)."""
    from nilearn.datasets import get_data_dirs
    return Path(get_data_dirs()[0]) / "schaefer_2018_fsaverage"


def annot_path(spec: Schaefer, hemi: str, mesh: str = "fsaverage") -> Path:
    """Local path of an annotation file, downloading it on first use."""
    h = {"left": "lh", "right": "rh"}.get(hemi, hemi)
    path = cache_dir() / mesh / f"{h}.Schaefer2018_{spec.n_rois}Parcels_{spec.yeo_networks}Networks_order.annot"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = _CBIG_URL.format(mesh=mesh, hemi=h, n=spec.n_rois, k=spec.yeo_networks)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(path)
    return path


@lru_cache(maxsize=8)
def surface_labels(spec: Schaefer, mesh: str = "fsaverage") -> dict:
    """Per-vertex parcel ids for each hemisphere: 1..n_rois, 0 = medial wall."""
    import nibabel as nib
    out = {}
    offset = 0
    for hemi in ("left", "right"):
        labels, _, names = nib.freesurfer.read_annot(str(annot_path(spec, hemi, mesh)))
        out[hemi] = np.where(labels > 0, labels + offset, 0).astype(np.int32)
        offset += len(names) - 1
    if offset != spec.n_rois:
        raise ValueError(f"Annotation has {offset} parcels, expected {spec.n_rois}.")
    return out


@dataclass(frozen=True)
class ParcelInfo:
    names: tuple[str, ...]          # full Schaefer names, e.g. 17Networks_LH_DefaultA_PFCd_1
    hemis: np.ndarray               # "L" / "R" per parcel
    networks: np.ndarray            # network name per parcel, e.g. "DefaultA"
    network_order: tuple[str, ...]  # networks in atlas order
    network_colours: dict           # network -> (r, g, b) in 0..1


@lru_cache(maxsize=8)
def parcel_info(spec: Schaefer) -> ParcelInfo:
    import nibabel as nib
    names, colours = [], []
    for hemi in ("left", "right"):
        _, ctab, hnames = nib.freesurfer.read_annot(str(annot_path(spec, hemi)))
        names += [n.decode() if isinstance(n, bytes) else str(n) for n in hnames[1:]]
        colours += [tuple(c[:3] / 255) for c in ctab[1:]]
    parts = [n.split("_") for n in names]
    hemis = np.array([p[1][0] for p in parts])
    networks = np.array([p[2] for p in parts])
    order = tuple(str(n) for n in dict.fromkeys(networks))
    net_colours = {}
    for net, col in zip(networks, colours):
        net_colours.setdefault(str(net), col)
    return ParcelInfo(tuple(names), hemis, networks, order, net_colours)


def short_name(full: str) -> str:
    """'17Networks_LH_DefaultA_PFCd_1' -> 'LH DefaultA PFCd 1'."""
    return " ".join(full.split("_")[1:])


def fetch_volume(spec: Schaefer, resolution_mm: int = 2):
    """The volumetric atlas image (MNI152) from nilearn."""
    from nilearn import datasets
    return datasets.fetch_atlas_schaefer_2018(
        n_rois=spec.n_rois, yeo_networks=spec.yeo_networks,
        resolution_mm=resolution_mm, verbose=0)["maps"]
