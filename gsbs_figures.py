"""
Publication figures, independent of the GUI.

Each function takes plain arrays (as saved by the exported analysis scripts)
and returns a matplotlib Figure. Exported packages call these functions from
one small script per figure, so every figure file comes with the exact code
that drew it.
"""

from __future__ import annotations

import sys

import matplotlib

if "matplotlib.pyplot" not in sys.modules:      # standalone scripts draw off-screen
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

FULL_WIDTH = 7.2          # inches, two-column journal width
STYLE = {
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "pdf.fonttype": 42, "svg.fonttype": "none",
}
BOUNDARY = "#202020"
STIMULUS = "#2a9d4b"


def style():
    return plt.rc_context(STYLE)


def label_panel(ax, letter: str) -> None:
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segmentation(x, ks, tvals, best_k: int, k: int, edges, tr: float | None = None,
                 stim_trs=None, columns: str = "parcels"):
    """A: t-distance per number of states. B: time-by-time correlation with the states of
    solution k. C: the timeseries with the boundaries (and stimulus boundaries)."""
    x = np.asarray(x)
    edges = np.asarray(edges)
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 4.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1.05])
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(ks, tvals, color="#3a6ea5", lw=1)
        ax.plot([best_k], [tvals[list(ks).index(best_k)]], "*", color="#d1495b", ms=8,
                label=f"optimum: {best_k} states")
        if k != best_k:
            ax.axvline(k, color="#555555", ls="--", lw=0.8, label=f"shown: {k} states")
        ax.set_xlabel("Number of states")
        ax.set_ylabel("t-distance")
        ax.legend(frameon=False)
        label_panel(ax, "A")

        axc = fig.add_subplot(gs[0, 1])
        corr = np.corrcoef(x)
        im = axc.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="none")
        for a, b in zip(edges[:-1] - 0.5, edges[1:] - 0.5):
            axc.add_patch(Rectangle((a, a), b - a, b - a, fill=False, ec=BOUNDARY, lw=0.6))
        axc.set_xlabel("TR")
        axc.set_ylabel("TR")
        fig.colorbar(im, ax=axc, shrink=0.75, label="r")
        label_panel(axc, "B")

        axt = fig.add_subplot(gs[1, :])
        z = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
        lim = float(np.percentile(np.abs(z), 99)) or 1.0
        axt.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="none",
                   origin="lower")
        axt.vlines(edges[1:-1] - 0.5, -0.5, x.shape[1] - 0.5, color=BOUNDARY, lw=0.6)
        if stim_trs is not None and len(stim_trs):
            axt.vlines(np.asarray(stim_trs) - 0.5, x.shape[1] - 0.5, x.shape[1] * 1.06,
                       color=STIMULUS, lw=0.6, clip_on=False)
        axt.set_ylabel(columns.capitalize())
        axt.set_xlabel("TR" + (f" ({tr:g} s)" if tr else ""))
        label_panel(axt, "C")
    return fig


# ---------------------------------------------------------------------------
# Surface maps
# ---------------------------------------------------------------------------

def _surface(strip, spec, values, cmap, lo, hi, grid=False):
    img = strip.parcels(spec).render(np.asarray(values, float), matplotlib.colormaps[cmap], lo, hi)
    return strip.to_grid(img) if grid else img


def _network_axis(ax, info, net_values: dict, cmap: str, lo: float, hi: float,
                  title: str, counts: dict | None = None, sequential: bool = False):
    nets = [n for n in info.network_order]
    vals = [net_values.get(n, np.nan) for n in nets]
    norm = Normalize(lo, hi)
    cm = matplotlib.colormaps[cmap]
    y = np.arange(len(nets))
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        left = lo if sequential else 0
        ax.barh(i, v - left, left=left, color=cm(norm(v)), height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(nets, fontsize=5.5)
    ax.tick_params(axis="y", length=0, pad=9)
    for i, n in enumerate(nets):                     # network colour as a swatch, text stays dark
        ax.add_patch(Rectangle((-0.045, i - 0.32), 0.03, 0.64, transform=ax.get_yaxis_transform(),
                               color=info.network_colours[n], clip_on=False))
    ax.invert_yaxis()
    if not sequential:
        ax.axvline(0, color="#888888", lw=0.5)
    ax.set_title(title, fontsize=6.5)
    ax.tick_params(axis="x", labelsize=5.5)
    if counts:
        for i, n in enumerate(nets):
            if n in counts:
                ax.text(1.02, i, counts[n], transform=ax.get_yaxis_transform(), fontsize=5,
                        va="center", color="#555555")


def roi_map(strip, spec, info, values, cmap: str, lo: float, hi: float, label: str,
            net_values: dict | None = None, counts: dict | None = None, sequential=False):
    """Values per ROI on the inflated surface, with network means."""
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 2.3), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 0.08])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(_surface(strip, spec, values, cmap, lo, hi))
        ax.axis("off")
        cax = fig.add_subplot(gs[1, 0])
        fig.colorbar(plt.cm.ScalarMappable(Normalize(lo, hi), cmap), cax=cax,
                     orientation="horizontal", label=label)
        if net_values is not None:
            axn = fig.add_subplot(gs[:, 1])
            _network_axis(axn, info, net_values, cmap, lo, hi, "Network means", counts, sequential)
    return fig


def boundary(strip, spec, info, before, after, change, lim: float, clim: float,
             labels=("Before", "After", "Change")):
    """The states either side of one boundary and the change between them."""
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 2.6), constrained_layout=True)
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.55], height_ratios=[1, 0.07])
        for i, (vals, cmap, l) in enumerate(((before, "RdBu_r", lim), (after, "RdBu_r", lim),
                                             (change, "PuOr_r", clim))):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(_surface(strip, spec, vals, cmap, -l, l, grid=True))
            ax.set_title(labels[i])
            ax.axis("off")
            label_panel(ax, "ABC"[i])
        cax = fig.add_subplot(gs[1, :2])
        fig.colorbar(plt.cm.ScalarMappable(Normalize(-lim, lim), "RdBu_r"), cax=cax,
                     orientation="horizontal", label="mean pattern (z)")
        cax2 = fig.add_subplot(gs[1, 2])
        fig.colorbar(plt.cm.ScalarMappable(Normalize(-clim, clim), "PuOr_r"), cax=cax2,
                     orientation="horizontal", label="after − before")
        axn = fig.add_subplot(gs[:, 3])
        nets = {n: float(np.nanmean(np.asarray(change)[info.networks == n])) for n in info.network_order}
        _network_axis(axn, info, nets, "PuOr_r", -clim, clim, "Network change")
        label_panel(axn, "D")
    return fig


# ---------------------------------------------------------------------------
# Boundary overlap
# ---------------------------------------------------------------------------

def overlap_pair(ref_trs, test_trs, n_time: int, null, observed: int, expected: float,
                 ref_label: str, test_label: str, tr: float | None = None, tolerance: int = 0):
    """A: both boundary sets, matches highlighted. B: overlap after shuffling state durations."""
    ref = np.asarray(ref_trs)
    test = np.asarray(test_trs)
    matched = np.array([np.any(np.abs(ref - b) <= tolerance) for b in test]) if len(ref) else \
        np.zeros(len(test), bool)
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 2.4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1])
        ax = fig.add_subplot(gs[0])
        ax.vlines(ref, 1.1, 1.9, color=STIMULUS, lw=0.6)
        ax.vlines(test[~matched], 0.1, 0.9, color="#3a6ea5", lw=0.6)
        ax.vlines(test[matched], 0.1, 0.9, color="#d1495b", lw=1.0)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels([test_label, ref_label])
        ax.set_xlim(0, n_time)
        ax.set_xlabel("TR" + (f" ({tr:g} s)" if tr else ""))
        ax.spines["left"].set_visible(False)
        label_panel(ax, "A")
        axh = fig.add_subplot(gs[1])
        null = np.asarray(null)
        bins = np.arange(min(null.min(), observed) - 0.5, max(null.max(), observed) + 1.5)
        axh.hist(null, bins=bins, color="#bbbbbb")
        axh.axvline(observed, color="#d1495b", lw=1.2, label=f"observed ({observed})")
        axh.axvline(expected, color="#555555", ls="--", lw=0.8, label=f"chance ({expected:.1f})")
        axh.set_xlabel("Shared boundaries")
        axh.set_ylabel("Shuffles")
        axh.legend(frameon=False)
        label_panel(axh, "B")
    return fig


def overlap_map(strip, spec, info, or_values, lim: float, delay_values=None, delays=None,
                net_or=None, net_delay=None, counts=None):
    """A: OR per ROI (significant ROIs only). B: best HRF delay per ROI (if scanned)."""
    rows = 2 if delay_values is not None else 1
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 2.3 * rows), constrained_layout=True)
        gs = fig.add_gridspec(2 * rows, 2, width_ratios=[5, 1],
                              height_ratios=[1, 0.08] * rows)
        panels = [(or_values, "RdBu_r", -lim, lim, "OR", net_or, False)]
        if delay_values is not None:
            panels.append((delay_values, "viridis", float(min(delays)), float(max(delays)),
                           "best HRF delay (s)", net_delay, True))
        for i, (vals, cmap, lo, hi, label, nets, seq) in enumerate(panels):
            ax = fig.add_subplot(gs[2 * i, 0])
            ax.imshow(_surface(strip, spec, vals, cmap, lo, hi))
            ax.axis("off")
            label_panel(ax, "AB"[i])
            cax = fig.add_subplot(gs[2 * i + 1, 0])
            fig.colorbar(plt.cm.ScalarMappable(Normalize(lo, hi), cmap), cax=cax,
                         orientation="horizontal", label=label)
            if nets is not None:
                axn = fig.add_subplot(gs[2 * i:2 * i + 2, 1])
                _network_axis(axn, info, nets, cmap, lo, hi, "Network means", counts, seq)
    return fig


# ---------------------------------------------------------------------------
# State clusters
# ---------------------------------------------------------------------------

def cluster_colours(n: int):
    base = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [base(i) for i in range(n)]


def clusters(centroids, labels, edges, transitions, time_fraction, tr=None,
             strip=None, spec=None, info=None):
    """A: cluster mean patterns. B: cluster of every state over time, and time per cluster.
    C: transitions between clusters."""
    centroids = np.asarray(centroids)
    labels = np.asarray(labels)
    edges = np.asarray(edges)
    n = len(centroids)
    cols = cluster_colours(n)
    lim = float(np.percentile(np.abs(centroids), 98)) or 1.0
    ncol = min(n, 4)
    nrow = (n + ncol - 1) // ncol
    with style():
        fig = plt.figure(figsize=(FULL_WIDTH, 1.55 * nrow + 2.3), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[1.55 * nrow, 2.3])
        top = outer[0].subgridspec(nrow, ncol)
        for i in range(n):
            ax = fig.add_subplot(top[i // ncol, i % ncol])
            if strip is not None:
                ax.imshow(_surface(strip, spec, centroids[i], "RdBu_r", -lim, lim, grid=True))
                ax.axis("off")
            else:
                ax.imshow(centroids[i][None], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_yticks([])
                ax.set_xlabel("voxel")
            ax.set_title(f"C{i + 1}: {int(np.sum(labels == i))} states, "
                         f"{100 * time_fraction[i]:.0f}% of time", color=cols[i], fontsize=6.5)
            if i == 0:
                label_panel(ax, "A")
        bottom = outer[1].subgridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[0.35, 1])
        axt = fig.add_subplot(bottom[0, 0])
        line = np.repeat(labels, np.diff(edges))
        cmap = ListedColormap(["#999999"] + [matplotlib.colors.to_hex(c) for c in cols])
        axt.imshow((line + 1)[None], aspect="auto", cmap=cmap, vmin=-0.5, vmax=n + 0.5,
                   interpolation="none")
        axt.set_yticks([])
        axt.set_xlabel("TR" + (f" ({tr:g} s)" if tr else ""))
        label_panel(axt, "B")
        axf = fig.add_subplot(bottom[1, 0])
        axf.barh(range(n), 100 * np.asarray(time_fraction), color=cols, height=0.6)
        axf.set_yticks(range(n))
        axf.set_yticklabels([f"C{i + 1}" for i in range(n)])
        axf.invert_yaxis()
        axf.set_xlabel("% of the time course")
        axm = fig.add_subplot(bottom[:, 1])
        t = np.asarray(transitions, float)
        rows = t.sum(1, keepdims=True)
        P = np.divide(t, rows, out=np.full_like(t, np.nan), where=rows > 0)
        np.fill_diagonal(P, np.nan)
        im = axm.imshow(P, cmap="magma", vmin=0, vmax=1)
        for a in range(n):
            for b in range(n):
                if a != b and t[a, b] > 0 and n <= 12:
                    axm.text(b, a, int(t[a, b]), ha="center", va="center", fontsize=5.5,
                             color="black" if P[a, b] > 0.6 else "white")
        axm.set_xticks(range(n))
        axm.set_yticks(range(n))
        axm.set_xticklabels([f"C{i + 1}" for i in range(n)])
        axm.set_yticklabels([f"C{i + 1}" for i in range(n)])
        axm.set_xlabel("next cluster")
        axm.set_ylabel("current cluster")
        fig.colorbar(im, ax=axm, shrink=0.8, label="P(next | current)")
        label_panel(axm, "C")
    return fig
