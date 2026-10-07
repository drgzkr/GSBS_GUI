"""
Fast parcel maps on the inflated fsaverage surface.

Drawing a surface in 3D with matplotlib takes 10 s (fsaverage5) to over two
minutes (fsaverage) for four views, far too slow to follow a slider. Instead,
each view is rasterised once into an image that records, per pixel, which
vertex is visible there and how it is lit. Drawing a parcel map is then a
per-pixel colour lookup: tens of milliseconds, with no 3D work at all.

The rasterised views are cached on disk (~/.cache/gsbs_gui), so the one-off
cost (a few seconds) is paid once per machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import gsbs_atlas as atlas

# Strip order: left lateral, left medial, right medial, right lateral.
VIEWS = (("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral"))
_CAMERA = {("left", "lateral"): (-1, 0, 0), ("left", "medial"): (1, 0, 0),
           ("right", "lateral"): (1, 0, 0), ("right", "medial"): (-1, 0, 0)}
_GAP = 6                  # pixels between views
_BAKE_VERSION = 1


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "gsbs_gui"


# ---------------------------------------------------------------------------
# Rasterising a view
# ---------------------------------------------------------------------------

def _rasterise(coords: np.ndarray, faces: np.ndarray, camera, height: int):
    """Orthographic z-buffered rasterisation of a mesh.

    Returns (vertex image, shading image): the nearest visible vertex at each
    pixel (-1 for background) and Lambert shading of its face (0..1).
    """
    cam = np.asarray(camera, float)
    cam /= np.linalg.norm(cam)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, cam)
    right /= np.linalg.norm(right)
    u, v, depth = coords @ right, coords @ up, coords @ cam

    scale = (height - 4) / (v.max() - v.min())
    px = (u - u.min()) * scale + 2
    py = (v.max() - v) * scale + 2
    width = int(np.ceil(px.max())) + 3

    a, b, c = (coords[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    lambert = np.abs(normal @ cam)

    fx, fy = px[faces], py[faces]
    x0 = np.floor(fx.min(1)).astype(int)
    y0 = np.floor(fy.min(1)).astype(int)
    span = int(np.ceil(max((fx.max(1) - x0).max(), (fy.max(1) - y0).max()))) + 1
    oy, ox = (g.ravel() for g in np.mgrid[0:span, 0:span])

    pix_l, dep_l, vert_l, shade_l = [], [], [], []
    for chunk in np.array_split(np.arange(len(faces)), max(1, len(faces) * span * span // 2_000_000)):
        X = x0[chunk, None] + ox + 0.5
        Y = y0[chunk, None] + oy + 0.5
        xa, xb, xc = (fx[chunk, i, None] for i in range(3))
        ya, yb, yc = (fy[chunk, i, None] for i in range(3))
        den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        den = np.where(den == 0, 1e-12, den)
        w1 = ((yb - yc) * (X - xc) + (xc - xb) * (Y - yc)) / den
        w2 = ((yc - ya) * (X - xc) + (xa - xc) * (Y - yc)) / den
        w3 = 1 - w1 - w2
        fi, ki = np.nonzero((w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6))
        f = chunk[fi]
        w = np.stack([w1[fi, ki], w2[fi, ki], w3[fi, ki]], 1)
        dep_l.append((w * depth[faces[f]]).sum(1))
        vert_l.append(faces[f, w.argmax(1)])
        pix_l.append((y0[f] + oy[ki]) * width + (x0[f] + ox[ki]))
        shade_l.append(lambert[f])

    pix, dep = np.concatenate(pix_l), np.concatenate(dep_l)
    vert, shade = np.concatenate(vert_l), np.concatenate(shade_l)
    keep = (pix >= 0) & (pix < height * width)
    pix, dep, vert, shade = pix[keep], dep[keep], vert[keep], shade[keep]
    order = np.lexsort((-dep, pix))                    # nearest to the camera first
    pix, vert, shade = pix[order], vert[order], shade[order]
    first = np.r_[True, pix[1:] != pix[:-1]]

    vimg = np.full(height * width, -1, np.int32)
    simg = np.zeros(height * width, np.float32)
    vimg[pix[first]] = vert[first]
    simg[pix[first]] = shade[first]
    return vimg.reshape(height, width), simg.reshape(height, width)


# ---------------------------------------------------------------------------
# The four-view strip
# ---------------------------------------------------------------------------

class SurfaceStrip:
    """Four rasterised views of the inflated surface side by side."""

    def __init__(self, vertex: np.ndarray, hemi: np.ndarray, shade: np.ndarray,
                 sulc: np.ndarray, mesh: str):
        self.vertex = vertex      # (H, W) vertex index within its hemisphere, -1 = background
        self.hemi = hemi          # (H, W) 0 = left, 1 = right
        self.shade = shade        # (H, W) Lambert shading
        self.sulc = sulc          # (H, W) sulcal depth (for the grey background)
        self.mesh = mesh
        self._layers: dict = {}
        self._cells = self._grid_cells()

    def _view_columns(self) -> list[tuple[int, int]]:
        """Column range of each view: the runs of columns that contain surface."""
        filled = (self.vertex >= 0).any(axis=0).astype(int)
        change = np.flatnonzero(np.diff(np.r_[0, filled, 0]))
        runs = list(zip(change[::2], change[1::2]))
        if len(runs) != len(VIEWS):
            raise ValueError(f"Expected {len(VIEWS)} views in the strip, found {len(runs)}.")
        return runs

    def _grid_cells(self, gap: int = _GAP) -> list[tuple[int, int, int, int]]:
        """2x2 layout: lateral views on top, medial below, left hemisphere on the left.

        Returns (grid x, grid y, strip x, width) per cell.
        """
        cols = self._view_columns()
        h = self.vertex.shape[0]
        order = ((0, 0, 0), (3, 1, 0), (1, 0, 1), (2, 1, 1))   # (view, column, row)
        left_w = max(cols[0][1] - cols[0][0], cols[1][1] - cols[1][0])
        cells = []
        for view, c, r in order:
            a, b = cols[view]
            gx = 0 if c == 0 else left_w + gap
            cells.append((gx, r * (h + gap), a, b - a))
        return cells

    @property
    def grid_shape(self) -> tuple[int, int]:
        h = self.vertex.shape[0]
        return 2 * h + _GAP, max(gx + w for gx, _, _, w in self._cells)

    def to_grid(self, img: np.ndarray) -> np.ndarray:
        """Rearrange a rendered strip into the 2x2 layout."""
        h = self.vertex.shape[0]
        out = np.zeros(self.grid_shape + img.shape[2:], img.dtype)
        for gx, gy, sx, w in self._cells:
            out[gy:gy + h, gx:gx + w] = img[:, sx:sx + w]
        return out

    def grid_to_strip(self, x: int, y: int) -> tuple[int, int] | None:
        h = self.vertex.shape[0]
        for gx, gy, sx, w in self._cells:
            if gx <= x < gx + w and gy <= y < gy + h:
                return sx + x - gx, y - gy
        return None

    @property
    def shape(self) -> tuple[int, int]:
        return self.vertex.shape

    @classmethod
    def load(cls, height: int = 200, mesh: str = "fsaverage") -> "SurfaceStrip":
        """Load from the disk cache, rasterising first if needed."""
        path = _cache_root() / f"strip_{mesh}_h{height}_v{_BAKE_VERSION}.npz"
        if path.exists():
            d = np.load(path)
            return cls(d["vertex"], d["hemi"], d["shade"], d["sulc"], mesh)

        from nilearn import datasets
        surf = datasets.load_fsaverage(mesh)["inflated"]
        sulc_data = datasets.load_fsaverage_data(mesh=mesh, data_type="sulcal")
        parts = []
        for hemi, view in VIEWS:
            part = surf.parts[hemi]
            vimg, simg = _rasterise(np.asarray(part.coordinates), np.asarray(part.faces),
                                    _CAMERA[(hemi, view)], height)
            sulc = np.asarray(sulc_data.data.parts[hemi])[np.maximum(vimg, 0)]
            parts.append((vimg, np.full(vimg.shape, hemi == "right", np.int8), simg,
                          np.where(vimg >= 0, sulc, 0).astype(np.float32)))

        def join(i, fill):
            pad = np.full((height, _GAP), fill, parts[0][i].dtype)
            out = []
            for j, p in enumerate(parts):
                out.append(p[i])
                if j < len(parts) - 1:
                    out.append(pad)
            return np.concatenate(out, axis=1)

        strip = cls(join(0, -1), join(1, 0), join(2, 0), join(3, 0), mesh)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, vertex=strip.vertex, hemi=strip.hemi,
                            shade=strip.shade, sulc=strip.sulc)
        tmp.rename(path)
        return strip

    def parcels(self, spec: atlas.Schaefer) -> "ParcelLayer":
        if spec not in self._layers:
            self._layers[spec] = ParcelLayer(self, spec)
        return self._layers[spec]


class ParcelLayer:
    """A SurfaceStrip combined with one parcellation: draws and picks parcels."""

    def __init__(self, strip: SurfaceStrip, spec: atlas.Schaefer):
        labels = atlas.surface_labels(spec, strip.mesh)
        self.spec = spec
        self.background = strip.vertex < 0
        v = np.maximum(strip.vertex, 0)
        parcel = np.where(strip.hemi == 1, labels["right"][v], labels["left"][v])
        self.parcel = np.where(self.background, -1, parcel).astype(np.int32)

        edge = np.zeros(self.parcel.shape, bool)
        edge[:, 1:] |= self.parcel[:, 1:] != self.parcel[:, :-1]
        edge[1:, :] |= self.parcel[1:, :] != self.parcel[:-1, :]
        self._edge = edge & ~self.background

        self._light = (0.5 + 0.5 * strip.shade)[..., None]
        grey = np.where(strip.sulc < 0, 0.80, 0.64)   # gyri light, sulci darker
        self._grey = grey[..., None]

    def render(self, values: np.ndarray, cmap, vmin: float, vmax: float,
               highlight=None, highlight_rgb=(1.0, 0.85, 0.2)) -> np.ndarray:
        """RGBA uint8 image of one value per parcel (values[i] is parcel i + 1).

        NaN values and the medial wall are drawn as grey anatomy.
        """
        values = np.asarray(values, float)
        idx = np.clip(self.parcel - 1, 0, len(values) - 1)
        val = np.where(self.parcel > 0, values[idx], np.nan)
        span = (vmax - vmin) or 1.0
        rgb = cmap(np.clip((val - vmin) / span, 0, 1))[..., :3]
        missing = np.isnan(val)
        rgb = np.where(missing[..., None], self._grey, rgb)
        rgb = np.where(self._edge[..., None], rgb * 0.8, rgb)
        rgb = rgb * self._light

        if highlight is not None:
            sel = np.isin(self.parcel, np.atleast_1d(highlight))
            grown = sel.copy()
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                grown |= np.roll(np.roll(sel, dy, 0), dx, 1)
            rgb = np.where((grown & ~sel & ~self.background)[..., None],
                           np.asarray(highlight_rgb), rgb)

        out = np.zeros(self.parcel.shape + (4,), np.uint8)
        out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(self.background, 0, 255)
        return out

    def pick(self, x: int, y: int) -> int | None:
        """Parcel id (1-based) at pixel (x, y), or None."""
        h, w = self.parcel.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        p = int(self.parcel[y, x])
        return p if p > 0 else None
