"""
State clusters window: cluster the states of the segmentation shown in the main
window, and show the cluster patterns, the cluster timeline, and the transitions
between clusters. It follows the main window: changing the solution there
reclusters here with the same settings.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.patches
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import gsbs_atlas as atlas
import gsbs_clusters as C
from gsbs_brain_panel import BrainImage, ColourBar, NetworkBars

_BG = "#1e1e2e"
_DIM = "#a6adc8"
_NOISE = "#6c7086"


def cluster_colours(n: int) -> list:
    if n <= 20:
        base = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
        return [base(i)[:3] for i in range(n)]
    return [matplotlib.colormaps["hsv"](i / n)[:3] for i in range(n)]


class ClusterWindow(QMainWindow):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("State clusters")
        self.resize(1250, 820)
        self.obj = None
        self.k = None
        self.meta = None
        self.tr = None
        self.patterns = self.edges = None
        self.clusters: C.Clusters | None = None
        self.scan_result: C.Scan | None = None
        self._strip = None

        split = QSplitter()
        split.addWidget(self._build_controls())
        self.tabs = QTabWidget()
        self._build_clusters_tab()
        self.tl_fig, self.tl_canvas = self._canvas()
        self.tabs.addTab(self.tl_canvas, "Timeline")
        self.sim_fig, self.sim_canvas = self._canvas()
        self.tabs.addTab(self.sim_canvas, "Similarity")
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 930])
        self.setCentralWidget(split)

    # ---- controls -----------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        box = QGroupBox("Segmentation")
        bv = QVBoxLayout(box)
        self.seg_label = QLabel("")
        self.seg_label.setWordWrap(True)
        bv.addWidget(self.seg_label)
        hint = QLabel("Follows the solution shown in the main window.")
        hint.setStyleSheet(f"color: {_DIM};")
        hint.setWordWrap(True)
        bv.addWidget(hint)
        v.addWidget(box)

        meth = QGroupBox("Clustering")
        mv = QVBoxLayout(meth)
        self.method_combo = QComboBox()
        for m in C.METHODS:
            self.method_combo.addItem(C.METHOD_LABELS[m], m)
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        mv.addWidget(self.method_combo)
        self.param_stack = QStackedWidget()
        p1 = QWidget()
        f1 = QFormLayout(p1)
        self.ncl_spin = QSpinBox()
        self.ncl_spin.setRange(2, 500)
        self.ncl_spin.setValue(5)
        f1.addRow("Clusters:", self.ncl_spin)
        p2 = QWidget()
        f2 = QFormLayout(p2)
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.001, 2.0)
        self.eps_spin.setDecimals(3)
        self.eps_spin.setSingleStep(0.01)
        self.eps_spin.setValue(0.5)
        self.eps_spin.setToolTip("Cosine distance within which states are neighbours")
        self.ms_spin = QSpinBox()
        self.ms_spin.setRange(1, 1000)
        self.ms_spin.setValue(2)
        f2.addRow("eps:", self.eps_spin)
        f2.addRow("min samples:", self.ms_spin)
        self.param_stack.addWidget(p1)
        self.param_stack.addWidget(p2)
        mv.addWidget(self.param_stack)
        row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan (silhouette)")
        self.scan_btn.setToolTip("Try a range of settings and use the one with the best silhouette.")
        self.scan_btn.clicked.connect(self.run_scan)
        self.cluster_btn = QPushButton("Cluster")
        self.cluster_btn.setObjectName("run_btn")
        self.cluster_btn.clicked.connect(self.run_cluster)
        row.addWidget(self.scan_btn)
        row.addWidget(self.cluster_btn)
        mv.addLayout(row)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        mv.addWidget(self.result_label)
        v.addWidget(meth)

        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("CSV with one row per state (TRs and cluster), and an .npz "
                                   "with the cluster patterns and the transition counts.")
        self.export_btn.clicked.connect(self.export)
        self.export_btn.setEnabled(False)
        v.addWidget(self.export_btn)
        v.addStretch()
        w.setMinimumWidth(300)
        return w

    def _method_changed(self) -> None:
        self.param_stack.setCurrentIndex(1 if self.method_combo.currentData() == "dbscan" else 0)

    def _canvas(self):
        fig = Figure(facecolor=_BG, constrained_layout=True)
        return fig, FigureCanvasQTAgg(fig)

    def _build_clusters_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        v = QVBoxLayout(host)
        self.maps_cbar = ColourBar()
        v.addWidget(self.maps_cbar)
        self.maps_grid = QGridLayout()
        v.addLayout(self.maps_grid)
        v.addStretch()
        scroll.setWidget(host)
        self.tabs.addTab(scroll, "Clusters")

    # ---- input --------------------------------------------------------------------

    def set_segmentation(self, obj, k: int, name: str, meta: dict | None) -> None:
        """Called by the main window whenever the shown solution changes."""
        if obj is self.obj and k == self.k:
            return
        self.obj, self.k, self.meta = obj, k, meta
        self.tr = meta.get("tr") if meta else None
        self.patterns, self.edges = C.state_patterns(obj, k)
        self.scan_result = None
        self.seg_label.setText(f"<b>{name}</b><br>{k} states · {int(self.edges[-1])} TRs")
        self.ncl_spin.setMaximum(max(2, k - 1))
        self.run_cluster()

    # ---- analysis -----------------------------------------------------------------

    def _params(self) -> dict:
        if self.method_combo.currentData() == "dbscan":
            return {"eps": self.eps_spin.value(), "min_samples": self.ms_spin.value()}
        return {"n_clusters": min(self.ncl_spin.value(), len(self.patterns) - 1)}

    def run_scan(self) -> None:
        if self.patterns is None or len(self.patterns) < 4:
            return
        method = self.method_combo.currentData()
        self.scan_result = C.scan(self.patterns, method)
        best = self.scan_result.best
        if best is None:
            self.result_label.setText("No setting gave two or more clusters.")
            return
        if method == "dbscan":
            self.eps_spin.setValue(best["eps"])
            self.ms_spin.setValue(best["min_samples"])
        else:
            self.ncl_spin.setValue(best["n_clusters"])
        self.run_cluster()

    def run_cluster(self) -> None:
        if self.patterns is None or len(self.patterns) < 3:
            self.clusters = None
            self.result_label.setText("Need at least 3 states to cluster.")
            self.export_btn.setEnabled(False)
            return
        method = self.method_combo.currentData()
        labels = C.cluster(self.patterns, method, **self._params())
        self.clusters = C.summarise(self.patterns, self.edges, labels, method)
        cl = self.clusters
        text = f"{cl.n} clusters of {len(self.patterns)} states · silhouette {cl.silhouette:.2f}"
        noise = int(np.sum(cl.labels < 0))
        if noise:
            text += f" · {noise} noise states"
        self.result_label.setText(text)
        self.export_btn.setEnabled(cl.n > 0)
        self._draw_maps()
        self._draw_timeline()
        self._draw_similarity()

    # ---- drawing ---------------------------------------------------------------------

    def _spec(self):
        spec = atlas.Schaefer.from_meta(self.meta)
        if (spec is None or self.meta.get("kind") != "parcels"
                or self.patterns.shape[1] != spec.n_rois):
            return None
        return spec

    def _surface(self):
        brain = getattr(self.main, "brain", None)
        if brain is not None and brain._big is not None:
            return brain._big
        if self._strip is None:
            import gsbs_surface as surface
            self._strip = surface.SurfaceStrip.load(height=200)
        return self._strip

    def _draw_maps(self) -> None:
        while self.maps_grid.count():
            w = self.maps_grid.takeAt(0).widget()
            if w is not None:
                w.hide()                           # deleteLater alone leaves it drawn until deleted
                w.setParent(None)
                w.deleteLater()
        cl = self.clusters
        if cl is None or cl.n == 0:
            self.maps_cbar.set_scale(None, 0, 1)
            return
        spec = self._spec()
        lim = float(np.percentile(np.abs(cl.centroids), 98)) or 1.0
        cmap = matplotlib.colormaps["RdBu_r"]
        cols = cluster_colours(cl.n)
        strip = layer = info = None
        if spec is not None:
            try:
                strip = self._surface()
                layer = strip.parcels(spec)
                info = atlas.parcel_info(spec)
            except Exception:                                      # noqa: BLE001
                strip = None
        per_row = 2 if strip is not None else 1
        for i in range(cl.n):
            box = QFrame()
            box.setStyleSheet(f"QFrame {{ border-left: 4px solid "
                              f"{matplotlib.colors.to_hex(cols[i])}; }}")
            bv = QVBoxLayout(box)
            head = QLabel(f"<b>Cluster {i + 1}</b> · {cl.n_states[i]} state"
                          f"{'s' if cl.n_states[i] != 1 else ''} · "
                          f"{100 * cl.time_fraction[i]:.0f}% of the time")
            head.setStyleSheet("border: none;")
            bv.addWidget(head)
            row = QHBoxLayout()
            img = BrainImage(170 if strip is not None else 40)
            img.setStyleSheet("border: none;")
            if strip is not None:
                img.grid = True
                img.set_rgba(strip.to_grid(layer.render(cl.centroids[i], cmap, -lim, lim)))
                row.addWidget(img, stretch=1)
                bars = NetworkBars()
                bars.setStyleSheet("border: none;")
                bars.set_values(info, cl.centroids[i], lim, "RdBu_r")
                row.addWidget(bars)
            else:
                rgba = (cmap(np.clip((cl.centroids[i] + lim) / (2 * lim), 0, 1)) * 255).astype(np.uint8)
                img.set_rgba(np.repeat(rgba[None], 12, axis=0))
                row.addWidget(img, stretch=1)
            bv.addLayout(row)
            self.maps_grid.addWidget(box, i // per_row, i % per_row)
        self.maps_cbar.set_scale("RdBu_r", -lim, lim, "cluster mean of z-scored state patterns"
                                 + ("" if strip is not None else " (one column per voxel)"))

    def _draw_timeline(self) -> None:
        fig = self.tl_fig
        fig.clear()
        cl = self.clusters
        n = cl.n
        cols = cluster_colours(n)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.2], width_ratios=[1.4, 1])

        # one row: the cluster of every TR, state boundaries as thin gaps
        ax = fig.add_subplot(gs[0, :])
        line = np.repeat(cl.labels, np.diff(self.edges))
        cmap = ListedColormap([_NOISE] + [matplotlib.colors.to_hex(c) for c in cols])
        ax.imshow((line + 1)[None], aspect="auto", interpolation="none", cmap=cmap,
                  vmin=-0.5, vmax=n + 0.5, extent=(-0.5, len(line) - 0.5, 0, 1))
        for b in self.edges[1:-1]:
            ax.axvline(b - 0.5, color=_BG, lw=0.6)
        ax.set_yticks([])
        ax.set_xlabel("TR" + (f" ({self.tr:g} s each)" if self.tr else ""))
        ax.set_title(f"Cluster of each state ({self.k} states"
                     + ("; grey: noise" if np.any(cl.labels < 0) else "") + ")", fontsize=9)

        # transitions between different clusters
        counts = C.transitions(cl.labels, n)
        rows = counts.sum(1, keepdims=True)
        P = np.divide(counts, rows, out=np.full_like(counts, np.nan), where=rows > 0)
        axm = fig.add_subplot(gs[1, 0])
        im = axm.imshow(P, cmap="magma", vmin=0, vmax=1, interpolation="none")
        for i in range(n):
            axm.add_patch(matplotlib.patches.Rectangle((i - 0.5, i - 0.5), 1, 1, color=_BG))
            for j in range(n):
                if i != j and counts[i, j] > 0 and n <= 15:
                    axm.text(j, i, f"{int(counts[i, j])}", ha="center", va="center", fontsize=7,
                             color="black" if P[i, j] > 0.6 else "white")
        ticks = range(n)
        axm.set_xticks(ticks)
        axm.set_yticks(ticks)
        axm.set_xticklabels([f"C{i + 1}" for i in ticks], fontsize=7)
        axm.set_yticklabels([f"C{i + 1}" for i in ticks], fontsize=7)
        for t, c in zip(axm.get_xticklabels() + axm.get_yticklabels(), cols + cols):
            t.set_color(c)
        axm.set_xlabel("next cluster")
        axm.set_ylabel("current cluster")
        axm.set_title(f"Transitions: P(next | current), counts written in "
                      f"({int(counts.sum())} in total)", fontsize=9)
        fig.colorbar(im, ax=axm, shrink=0.8)

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.barh(range(n), 100 * cl.time_fraction, color=cols)
        ax2.set_yticks(range(n))
        ax2.set_yticklabels([f"C{i + 1}" for i in range(n)], fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlabel("% of the time")
        for a in fig.axes:
            a.set_facecolor(_BG)
        self.tl_canvas.draw_idle()

    def _draw_similarity(self) -> None:
        fig = self.sim_fig
        fig.clear()
        cl = self.clusters
        method = self.method_combo.currentData()
        has_scan = self.scan_result is not None and self.scan_result.method == method
        gs = fig.add_gridspec(1, 2 if has_scan else 1)
        z = C.zscore_rows(self.patterns)
        r = z @ z.T / z.shape[1]
        order = np.lexsort((np.arange(len(cl.labels)), np.where(cl.labels < 0, 10**6, cl.labels)))
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(r[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="none")
        lab = cl.labels[order]
        cols = cluster_colours(cl.n)
        bounds = np.r_[0, np.flatnonzero(np.diff(lab)) + 1, len(lab)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if lab[a] >= 0:
                ax.add_patch(matplotlib.patches.Rectangle((a - 0.5, a - 0.5), b - a, b - a,
                                                          fill=False, ec=cols[lab[a]], lw=2))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("State × state correlation, sorted by cluster", fontsize=9)
        if has_scan:
            sc = self.scan_result
            ax3 = fig.add_subplot(gs[0, 1])
            if sc.method == "dbscan":
                eps = np.array([g["eps"] for g in sc.grid])
                ms = np.array([g["min_samples"] for g in sc.grid])
                for m in np.unique(ms):
                    ax3.plot(eps[ms == m], sc.scores[ms == m], marker=".", label=f"min samples {m}")
                ax3.set_xlabel("eps")
                ax3.legend(fontsize=7)
            else:
                ax3.plot([g["n_clusters"] for g in sc.grid], sc.scores, marker=".", color="#89b4fa")
                ax3.set_xlabel("number of clusters")
            ax3.set_ylabel("silhouette")
            ax3.set_title("Scan", fontsize=9)
        for a in fig.axes:
            a.set_facecolor(_BG)
        self.sim_canvas.draw_idle()

    # ---- export -------------------------------------------------------------------------

    def export(self) -> None:
        if self.clusters is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export clusters", "state_clusters.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        cl = self.clusters
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["state", "start_tr", "end_tr", "length_trs", "cluster"])
            for i, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
                w.writerow([i + 1, int(a), int(b) - 1, int(b - a),
                            int(cl.labels[i]) + 1 if cl.labels[i] >= 0 else 0])
        np.savez(Path(path).with_suffix(".npz"), centroids=cl.centroids, labels=cl.labels,
                 edges=self.edges, time_fraction=cl.time_fraction,
                 transition_counts=C.transitions(cl.labels, cl.n), k=self.k,
                 method=self.method_combo.currentData(), silhouette=cl.silhouette)
        self.statusBar().showMessage(f"Exported {path} and {Path(path).with_suffix('.npz').name}")
