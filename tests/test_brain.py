"""Schaefer annotations and the rasterised surface renderer.

Needs the Schaefer fsaverage annotation and the fsaverage mesh; both are
downloaded once and cached. Skipped when they are unavailable (offline).
"""

import matplotlib
import numpy as np
import pytest

import gsbs_atlas as atlas

SPEC = atlas.Schaefer(400, 17)


@pytest.fixture(scope="module")
def strip():
    try:
        import gsbs_surface as surface
        s = surface.SurfaceStrip.load(height=90)
        s.parcels(SPEC)
        return s
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"surface data unavailable: {exc}")


class TestAtlas:

    def test_spec_validation(self):
        with pytest.raises(ValueError):
            atlas.Schaefer(450, 17)

    def test_meta_round_trip(self):
        assert atlas.Schaefer.from_meta(SPEC.to_meta() | {"kind": "parcels"}) == SPEC
        assert atlas.Schaefer.from_meta({"atlas": "other"}) is None

    def test_annotation_order_matches_volume_labels(self, strip):
        info = atlas.parcel_info(SPEC)
        assert len(info.names) == 400
        assert info.names[0].startswith("17Networks_LH_")
        assert info.names[200].startswith("17Networks_RH_")
        labels = atlas.surface_labels(SPEC)
        assert labels["left"].max() == 200 and labels["right"].min() == 0
        assert set(np.unique(labels["right"][labels["right"] > 0])) == set(range(201, 401))
        assert len(info.network_order) == 17

    def test_short_name(self):
        assert atlas.short_name("17Networks_LH_DefaultA_PFCd_1") == "LH DefaultA PFCd 1"


class TestSurface:

    def test_four_views(self, strip):
        assert len(strip._view_columns()) == 4

    def test_grid_round_trip(self, strip):
        layer = strip.parcels(SPEC)
        g = strip.to_grid(layer.parcel[..., None])[..., 0]
        ys, xs = np.nonzero(g > 0)
        for y, x in list(zip(ys, xs))[::997]:
            sx, sy = strip.grid_to_strip(x, y)
            assert layer.parcel[sy, sx] == g[y, x]

    def test_render_paints_each_parcel_its_value(self, strip):
        layer = strip.parcels(SPEC)
        values = np.full(400, np.nan)
        values[149] = 1.0
        img = layer.render(values, matplotlib.colormaps["RdBu_r"], -1, 1)
        inside = (layer.parcel == 150) & ~layer._edge
        assert inside.any()
        r, g, b = (img[inside][:, i].astype(int) for i in range(3))
        assert np.all(r > b)                                   # a high value renders red
        grey = img[(layer.parcel > 0) & (layer.parcel != 150) & ~layer._edge]
        assert np.all(np.abs(grey[:, 0].astype(int) - grey[:, 2].astype(int)) <= 1)
        assert np.all(img[layer.background][:, 3] == 0)

    def test_pick(self, strip):
        layer = strip.parcels(SPEC)
        ys, xs = np.nonzero(layer.parcel == 77)
        assert layer.pick(xs[0], ys[0]) == 77
        assert layer.pick(-1, -1) is None

    def test_highlight_draws_outline(self, strip):
        layer = strip.parcels(SPEC)
        values = np.zeros(400)
        cmap = matplotlib.colormaps["RdBu_r"]
        plain = layer.render(values, cmap, -1, 1)
        hl = layer.render(values, cmap, -1, 1, highlight=77)
        changed = np.any(plain != hl, axis=2)
        assert changed.any()
        assert not np.any(changed & (layer.parcel == 77))      # outline is outside the parcel
