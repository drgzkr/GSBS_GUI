"""
Boundary overlap window: compare any two boundary sets (the shown segmentation,
the whole brain, an ROI, the stimulus), or every fitted ROI against one
reference, with the result mapped on the ROI map.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import gsbs_atlas as atlas
import gsbs_core as core
import gsbs_overlap as ov
from gsbs_nifti import WHOLE_BRAIN

_BG = "#1e1e2e"
_DIM = "#a6adc8"


def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


class _AllRois(QObject):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)

    def __init__(self, ref, rois, n_time, tolerance, n_perm):
        super().__init__()
        self.ref, self.rois, self.n_time = ref, rois, n_time
        self.tolerance, self.n_perm = tolerance, n_perm

    def run(self):
        out = {}
        for i, (roi, bounds) in enumerate(self.rois):
            out[roi] = ov.overlap(self.ref, bounds, self.n_time, self.tolerance, self.n_perm, seed=roi)
            self.progress.emit(i + 1, len(self.rois))
        self.done.emit(out)


class OverlapWindow(QMainWindow):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Boundary overlap")
        self.resize(1150, 760)
        self._thread = None
        split = QSplitter()
        split.addWidget(self._build_controls())
        self.fig = Figure(facecolor=_BG, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        split.addWidget(self.canvas)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 730])
        self.setCentralWidget(split)

    # ---- UI ---------------------------------------------------------------------------

    def _build_controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        pair = QGroupBox("Compare two boundary sets")
        f = QFormLayout(pair)
        self.ref_combo = QComboBox()
        self.test_combo = QComboBox()
        self.test_combo.setToolTip("The test segmentation's state durations are shuffled for "
                                   "the null, and OA is scaled by its number of boundaries.")
        f.addRow("Reference:", self.ref_combo)
        f.addRow("Test (shuffled):", self.test_combo)
        self.tol_spin = QSpinBox()
        self.tol_spin.setRange(0, 5)
        self.tol_spin.setPrefix("±")
        self.tol_spin.setSuffix(" TR")
        f.addRow("Tolerance:", self.tol_spin)
        self.perm_spin = QSpinBox()
        self.perm_spin.setRange(100, 100000)
        self.perm_spin.setSingleStep(500)
        self.perm_spin.setValue(1000)
        f.addRow("Shuffles:", self.perm_spin)
        b = QPushButton("Compare")
        b.setObjectName("run_btn")
        b.clicked.connect(self.compare)
        f.addRow(b)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        f.addRow(self.result_label)
        v.addWidget(pair)

        allbox = QGroupBox("Every fitted ROI against the reference")
        av = QVBoxLayout(allbox)
        self.all_btn = QPushButton("Compute and map on the ROI map")
        self.all_btn.clicked.connect(self.compare_all)
        av.addWidget(self.all_btn)
        self.all_progress = QProgressBar()
        self.all_progress.setFixedHeight(12)
        self.all_progress.hide()
        av.addWidget(self.all_progress)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("ROI", "O/n", "OA", "OR", "p", "q (FDR)"))
        self.table.verticalHeader().hide()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._open_roi_row)
        av.addWidget(self.table, stretch=1)
        self.all_note = QLabel("Double-click a row to open that ROI.")
        self.all_note.setStyleSheet(f"color: {_DIM};")
        self.all_note.setWordWrap(True)
        av.addWidget(self.all_note)
        v.addWidget(allbox, stretch=1)
        w.setMinimumWidth(380)
        return w

    # ---- sources --------------------------------------------------------------------------

    def refresh_sources(self) -> None:
        items = []
        m = self.main
        if m.gsbs_object is not None:
            items.append((f"Shown segmentation ({m._k} states)", ("shown", None)))
        s = m.session
        if s is not None:
            summ = s.summary()
            if WHOLE_BRAIN in summ:
                items.append((f"Whole brain (best, {summ[WHOLE_BRAIN].nstates} states)",
                              ("target", WHOLE_BRAIN)))
        sb = m.stimulus.stimulus_bounds() if m.stimulus.has_content else None
        if sb is not None:
            st = m.stimulus.settings()
            items.append((f"Stimulus: {st['label'].lower()} (HRF delay {st['delay_s']:g} s)",
                          ("stimulus", None)))
        if s is not None:
            info = atlas.parcel_info(s.spec)
            for t in sorted(k for k in s.summary() if k != WHOLE_BRAIN):
                items.append((f"ROI {t}: {atlas.short_name(info.names[t - 1])} "
                              f"(best, {s.summary()[t].nstates} states)", ("target", t)))
        for combo, default in ((self.ref_combo, ("stimulus", None)), (self.test_combo, ("shown", None))):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for label, key in items:
                combo.addItem(label, key)
            i = combo.findData(current if current is not None else default)
            if i < 0:
                i = combo.findData(default)
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)
        if self.ref_combo.currentData() == self.test_combo.currentData() and self.ref_combo.count() > 1:
            self.ref_combo.setCurrentIndex(1 if self.test_combo.currentIndex() == 0 else 0)

    def _bounds(self, key) -> tuple[np.ndarray, int]:
        kind, target = key
        m = self.main
        if kind == "shown":
            return m._edges[1:-1], int(m.x.shape[0])
        if kind == "stimulus":
            return m.stimulus.stimulus_bounds(), int(m.stimulus.n_time)
        s = m.session
        return s.best_bounds(target), int(s.n_time)

    # ---- comparisons -------------------------------------------------------------------------

    def compare(self) -> None:
        if self.ref_combo.currentData() is None or self.test_combo.currentData() is None:
            return
        ref, t1 = self._bounds(self.ref_combo.currentData())
        test, t2 = self._bounds(self.test_combo.currentData())
        if t1 != t2:
            QMessageBox.warning(self, "Different lengths",
                                f"The two sets cover {t1} and {t2} TRs; they must come from the same scan.")
            return
        r = ov.overlap(ref, test, t1, self.tol_spin.value(), self.perm_spin.value())
        self.result_label.setText(f"{ov.describe(r)}<br><span style='color:{_DIM}'>"
                                  f"{r.n_ref} reference vs {r.n_test} test boundaries over {t1} TRs"
                                  f"</span>")
        self._plot(ref, test, t1, r, self.ref_combo.currentText(), self.test_combo.currentText())

    def _plot(self, ref, test, n_time, r, ref_name: str, test_name: str) -> None:
        fig = self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4])
        ax = fig.add_subplot(gs[0])
        ref = np.asarray(ref)
        test = np.asarray(test)
        tol = self.tol_spin.value()
        matched = np.array([np.any(np.abs(ref - b) <= tol) for b in test]) if len(ref) else np.zeros(len(test), bool)
        ax.vlines(ref, 1.1, 1.9, color="#a6e3a1", lw=1)
        ax.vlines(test[~matched], 0.1, 0.9, color="#89b4fa", lw=1)
        ax.vlines(test[matched], 0.1, 0.9, color="#f9e2af", lw=2)
        ax.set_yticks([0.5, 1.5])
        short = lambda t: t if len(t) <= 34 else t[:33] + "\u2026"
        ax.set_yticklabels([f"test: {short(test_name)}", f"reference: {short(ref_name)}"], fontsize=7)
        ax.set_xlim(0, n_time)
        ax.set_xlabel("TR")
        ax.set_title("Boundaries (yellow: test boundaries that overlap the reference)", fontsize=9)
        ax2 = fig.add_subplot(gs[1])
        rng = np.random.default_rng(0)
        null = [ov.count_matches(ref, ov.shuffle_states(test, n_time, rng), tol)
                for _ in range(self.perm_spin.value())]
        ax2.hist(null, bins=np.arange(min(null + [r.observed]) - 0.5, max(null + [r.observed]) + 1.5),
                 color="#585b70")
        ax2.axvline(r.observed, color="#f38ba8", lw=2, label=f"observed O = {r.observed}")
        ax2.axvline(r.expected, color=_DIM, ls="--", lw=1, label=f"chance E = {r.expected:.1f}")
        ax2.legend(fontsize=8)
        ax2.set_xlabel("overlapping boundaries after shuffling the test state durations")
        ax2.set_ylabel("shuffles")
        for a in fig.axes:
            a.set_facecolor(_BG)
        self.canvas.draw_idle()

    def compare_all(self) -> None:
        s = self.main.session
        key = self.ref_combo.currentData()
        if s is None or key is None or self._thread is not None:
            if s is None:
                QMessageBox.information(self, "No session", "Open or import a session first.")
            return
        ref, n_time = self._bounds(key)
        if n_time != s.n_time:
            QMessageBox.warning(self, "Different lengths", "The reference does not cover this session's scan.")
            return
        fitted = sorted(k for k in s.summary() if k != WHOLE_BRAIN)
        self._excluded = [t for t in fitted if s.summary()[t].at_limit]
        rois = [(t, s.best_bounds(t)) for t in fitted if t not in self._excluded]
        if not rois:
            QMessageBox.information(self, "No ROI fits", "Fit some ROIs first (batch in the session panel).")
            return
        self._ref_label = self.ref_combo.currentText().split(" (")[0]
        self.all_progress.setRange(0, len(rois))
        self.all_progress.setValue(0)
        self.all_progress.show()
        self.all_btn.setEnabled(False)
        self._worker = _AllRois(ref, rois, n_time, self.tol_spin.value(), self.perm_spin.value())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda i, n: self.all_progress.setValue(i))
        self._worker.done.connect(self._all_done)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._all_finished)
        self._thread.start()

    def _all_finished(self) -> None:
        self._thread = None
        self.all_progress.hide()
        self.all_btn.setEnabled(True)

    def _all_done(self, results: dict) -> None:
        s = self.main.session
        info = atlas.parcel_info(s.spec)
        rois = sorted(results)
        q = fdr_bh([results[r].p for r in rois])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rois))
        from gsbs_session_panel import _NumItem
        for row, (roi, qv) in enumerate(zip(rois, q)):
            r = results[roi]
            name = _NumItem(roi, f"{roi:>3} {atlas.short_name(info.names[roi - 1])}")
            name.setData(Qt.ItemDataRole.UserRole, roi)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(f"{r.observed}/{r.n_test}"))
            for col, val, fmt in ((2, r.oa, "{:.2f}"), (3, r.or_, "{:.2f}"), (4, r.p, "{:.3g}"), (5, qv, "{:.3g}")):
                self.table.setItem(row, col, _NumItem(val, fmt.format(val)))
        self.table.setSortingEnabled(True)
        self.table.sortItems(3, Qt.SortOrder.DescendingOrder)
        values = np.full(s.spec.n_rois, np.nan)
        for roi in rois:
            values[roi - 1] = results[roi].or_
        label = f"Overlap with {self._ref_label} (OR)"
        self.main.brain.set_extra_metric("overlap", label, values, "OR")
        self.main.brain_dock.show()
        self.main.brain.setCurrentIndex(3)
        n_sig = int(np.sum(q < 0.05))
        excl = (f" {len(self._excluded)} ROI{'s' if len(self._excluded) != 1 else ''} left out: best "
                f"solution at kmax (no optimum)." if self._excluded else "")
        self.all_note.setText(f"{len(rois)} ROIs · {n_sig} with q < .05 (Benjamini–Hochberg "
                              f"over the ROIs) · mapped on the ROI map as “{label}”.{excl} "
                              f"Double-click a row to open that ROI.")

    def _open_roi_row(self, row: int, _col: int) -> None:
        roi = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.main.activate_target(int(roi))
