"""
Session panel (ROI browser and batch runs) and the NIfTI import dialog.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import gsbs_atlas as atlas
import gsbs_batch as batch
import gsbs_core as core
import gsbs_nifti as nifti
from gsbs_nifti import WHOLE_BRAIN

_DIM = "#a6adc8"
_POLL_MS = 200


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


class _NumItem(QTableWidgetItem):
    """Table item that sorts by a number (blank cells last)."""

    def __init__(self, value, text=None):
        super().__init__("" if value is None else (text if text is not None else str(value)))
        self.value = np.inf if value is None else value

    def __lt__(self, other):
        return self.value < getattr(other, "value", np.inf)


# ---------------------------------------------------------------------------
# Session panel
# ---------------------------------------------------------------------------

class SessionPanel(QWidget):
    targetActivated = pyqtSignal(object)      # WHOLE_BRAIN or ROI id
    resultSaved = pyqtSignal(object)          # a batch fit finished for this target

    COLS = ("ROI", "Vox", "States", "Dur", "")
    _STATUS = {"fitted": "\u2713", "running": "\u27f3", "failed": "\u2717", "too small": "\u2013",
               "at kmax": "\u26a0"}

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self._get_params = get_params         # () -> (engine, params dict)
        self.session: nifti.Session | None = None
        self.runner: batch.BatchRunner | None = None
        self._rows: dict = {}                 # target -> row index
        self._failed: dict = {}
        self._running_targets: set = set()
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._build()

    # ---- UI ---------------------------------------------------------------------

    def _build(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("No session")
        self.title.setStyleSheet("font-weight: bold;")
        self.title.setWordWrap(True)
        v.addWidget(self.title)
        self.info = QLabel("")
        self.info.setStyleSheet(f"color: {_DIM};")
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        f = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter ROIs…")
        self.search.textChanged.connect(self._apply_filter)
        f.addWidget(self.search, stretch=1)
        self.net_combo = QComboBox()
        self.net_combo.currentIndexChanged.connect(self._apply_filter)
        f.addWidget(self.net_combo)
        v.addLayout(f)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        tips = ("", "Voxels in the ROI", "Number of states of the best solution",
                "Median state duration of the best solution",
                "\u2713 fitted  \u26a0 best at kmax (no optimum)  \u27f3 running  "
                "\u2717 failed  \u2013 too few voxels")
        for i, tip in enumerate(tips):
            self.table.horizontalHeaderItem(i).setToolTip(tip)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        v.addWidget(self.table, stretch=1)

        box = QGroupBox("Batch GSBS")
        b = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("ROIs without a result", "missing")
        self.scope_combo.addItem("Selected ROIs", "selected")
        self.scope_combo.addItem("All ROIs (refit)", "all")
        self.scope_combo.currentIndexChanged.connect(self.update_estimate)
        r1.addWidget(self.scope_combo, stretch=1)
        r1.addWidget(QLabel("workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 256)
        self.workers_spin.valueChanged.connect(self.update_estimate)
        r1.addWidget(self.workers_spin)
        b.addLayout(r1)
        self.estimate = QLabel("")
        self.estimate.setWordWrap(True)
        self.estimate.setStyleSheet(f"color: {_DIM};")
        b.addWidget(self.estimate)
        r2 = QHBoxLayout()
        self.start_btn = QPushButton("Run batch")
        self.start_btn.setObjectName("run_btn")
        self.start_btn.clicked.connect(self.start_batch)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_batch)
        for w in (self.start_btn, self.pause_btn, self.cancel_btn):
            r2.addWidget(w)
        b.addLayout(r2)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(14)
        self.progress.setFormat("%v / %m")
        b.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        b.addWidget(self.status)
        self.workers_label = QLabel("")
        self.workers_label.setStyleSheet(f"color: {_DIM}; font-family: monospace; font-size: 8pt;")
        self.workers_label.setWordWrap(True)
        b.addWidget(self.workers_label)
        v.addWidget(box)
        self._set_batch_running(False)

    # ---- session ------------------------------------------------------------------

    def set_session(self, session: nifti.Session) -> None:
        self.cancel_batch(ask=False)
        self.session = session
        meta = session.meta
        self.title.setText(session.name)
        tr = f"TR {session.tr:g} s" if session.tr else "TR unknown"
        missing = meta.get("missing_parcels", [])
        self.info.setText(
            f"{session.n_time} TRs · {tr} · {session.spec.label}"
            + (f" · {len(missing)} parcels without voxels" if missing else "")
            + f"\nSource: {Path(meta['source']).name}")
        self.info.setToolTip(meta["source"])
        info = atlas.parcel_info(session.spec)
        self.net_combo.blockSignals(True)
        self.net_combo.clear()
        self.net_combo.addItem("All networks", None)
        for net in info.network_order:
            self.net_combo.addItem(net, net)
        self.net_combo.blockSignals(False)
        self._fill_table()
        self._failed.clear()
        self._suggest_workers()
        self.update_estimate()

    def _fill_table(self) -> None:
        s = self.session
        info = atlas.parcel_info(s.spec)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(s.spec.n_rois + 1)
        targets = [WHOLE_BRAIN] + list(range(1, s.spec.n_rois + 1))
        for row, target in enumerate(targets):
            if target == WHOLE_BRAIN:
                name = _NumItem(-1, f"Whole brain ({s.spec.n_rois} parcels)")
                nvox = None
            else:
                name = _NumItem(target, f"{target:>3}  {atlas.short_name(info.names[target - 1])}")
                nvox = int(s.voxel_counts[target - 1])
            name.setData(Qt.ItemDataRole.UserRole, target)
            name.setToolTip(name.text().strip())
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, _NumItem(nvox))
        self.table.setSortingEnabled(True)
        self.table.sortItems(0)
        self._reindex()
        for target in targets:
            self.update_row(target)

    def _reindex(self) -> None:
        self._rows = {self.table.item(r, 0).data(Qt.ItemDataRole.UserRole): r
                      for r in range(self.table.rowCount())}

    def update_row(self, target) -> None:
        if self.session is None or target not in self._rows:
            return
        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self._reindex()
        row = self._rows[target]
        s = self.session.summary().get(target)
        if s is not None:
            dur = s.median_duration * self.session.tr if self.session.tr else s.median_duration
            unit = "s" if self.session.tr else "TR"
            self.table.setItem(row, 2, _NumItem(s.nstates))
            self.table.setItem(row, 3, _NumItem(dur, f"{dur:.3g} {unit}"))
        else:
            self.table.setItem(row, 2, _NumItem(None))
            self.table.setItem(row, 3, _NumItem(None))
        if target in self._running_targets:
            status = "running"
        elif s is not None and s.at_limit:
            status = "at kmax"
        elif target in self._failed:
            status = "failed"
        elif s is not None:
            status = "fitted"
        elif target != WHOLE_BRAIN and self.session.voxel_counts[target - 1] < 2:
            status = "too small"
        else:
            status = ""
        item = QTableWidgetItem(self._STATUS.get(status, ""))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        tips = {"at kmax": "The t-distance was still rising at kmax: no optimum was found, so "
                           "the number of states is set by the search limit. Refit with a "
                           "larger kmax, or treat this ROI as having no clear state structure."}
        item.setToolTip(self._failed[target].strip().splitlines()[-1]
                        if target in self._failed else tips.get(status, status))
        self.table.setItem(row, 4, item)
        self.table.setSortingEnabled(sorting)
        self._reindex()

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        net = self.net_combo.currentData()
        networks = atlas.parcel_info(self.session.spec).networks if self.session else None
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            target = item.data(Qt.ItemDataRole.UserRole)
            rnet = None if target == WHOLE_BRAIN else networks[target - 1]
            hide = ((text and text not in item.text().lower())
                    or (net is not None and rnet is not None and rnet != net))
            self.table.setRowHidden(r, bool(hide))

    def _selection_changed(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) == 1:
            target = self.table.item(rows.pop(), 0).data(Qt.ItemDataRole.UserRole)
            self.targetActivated.emit(target)
        self.update_estimate()

    def select_target(self, target) -> None:
        """Select a row without emitting targetActivated."""
        self._reindex()
        if target not in self._rows:
            return
        self.table.blockSignals(True)
        self.table.selectRow(self._rows[target])
        self.table.blockSignals(False)
        self.table.scrollToItem(self.table.item(self._rows[target], 0))

    def selected_targets(self) -> list:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]

    # ---- batch --------------------------------------------------------------------

    def _suggest_workers(self) -> None:
        if self.session is None:
            return
        _, params = self._get_params()
        n, why = batch.suggest_workers(self.session.n_time, int(self.session.voxel_counts.max()),
                                       params["kmax"], params["statewise_detection"])
        self.workers_spin.blockSignals(True)
        self.workers_spin.setValue(n)
        self.workers_spin.blockSignals(False)
        self.workers_spin.setToolTip(f"Suggested: {why}")

    def _batch_targets(self) -> list:
        s = self.session
        rois = [r for r in range(1, s.spec.n_rois + 1) if s.voxel_counts[r - 1] >= 2]
        scope = self.scope_combo.currentData()
        if scope == "missing":
            done = s.summary()
            return [r for r in rois if r not in done]
        if scope == "selected":
            return [t for t in self.selected_targets() if t in rois or t == WHOLE_BRAIN]
        return rois

    def _prior_seconds(self) -> float | None:
        summary = self.session.summary()
        roi_times = [v.fit_seconds for k, v in summary.items() if k != WHOLE_BRAIN]
        if roi_times:
            return float(np.median(roi_times))
        if WHOLE_BRAIN in summary:
            return summary[WHOLE_BRAIN].fit_seconds
        return None

    def update_estimate(self) -> None:
        if self.session is None or (self.runner is not None and self.runner.active):
            return
        targets = self._batch_targets()
        n = len(targets)
        workers = min(self.workers_spin.value(), max(1, n))
        per = self._prior_seconds()
        text = f"{n} fit{'s' if n != 1 else ''}"
        if n and per is not None:
            text += (f" · ≈ {fmt_duration(per)} each · {workers} workers "
                     f"→ ≈ {fmt_duration(np.ceil(n / workers) * per)}")
            text += ("\n(estimate from the ROI fits so far)" if any(
                k != WHOLE_BRAIN for k in self.session.summary())
                     else "\n(estimate from the whole-brain fit; voxel count barely matters)")
        elif n:
            text += " · time per fit unknown until the whole-brain fit or the first ROIs finish"
        self.estimate.setText(text)
        self.start_btn.setEnabled(n > 0)

    def start_batch(self) -> None:
        if self.session is None:
            return
        targets = self._batch_targets()
        if not targets:
            return
        engine, params = self._get_params()
        workers = self.workers_spin.value()
        n_time = self.session.n_time
        need = core.estimate_fit_bytes(n_time, int(self.session.voxel_counts.max()),
                                       params["kmax"], params["statewise_detection"]) * workers
        avail = core.available_bytes()
        if avail and need > 0.85 * avail:
            reply = QMessageBox.warning(
                self, "Memory",
                f"{workers} workers need an estimated {core.format_bytes(need)}; "
                f"{core.format_bytes(avail)} is free.\n\nStart anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._failed = {t: m for t, m in self._failed.items() if t not in targets}
        self.runner = batch.BatchRunner(self.session.path, targets, engine, params,
                                        workers, prior_seconds=self._prior_seconds())
        self.runner.start()
        self.progress.setRange(0, len(targets))
        self.progress.setValue(0)
        self.status.setText(f"Starting {self.runner.n_workers} workers…")
        self._set_batch_running(True)
        self._timer.start()

    def _toggle_pause(self) -> None:
        if self.runner is None:
            return
        if self.runner.paused:
            self.runner.resume()
        else:
            self.runner.pause()
        self.pause_btn.setText("Resume" if self.runner.paused else "Pause")

    def cancel_batch(self, ask: bool = True) -> bool:
        if self.runner is None or not self.runner.active:
            return True
        if ask:
            reply = QMessageBox.question(
                self, "Cancel batch",
                "Stop the batch? Fits in progress are lost; finished results are kept, "
                "and running the batch again continues where it stopped.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return False
        self.runner.cancel()
        self._finish("Batch cancelled.")
        return True

    def _set_batch_running(self, running: bool) -> None:
        self.start_btn.setVisible(not running)
        self.pause_btn.setVisible(running)
        self.cancel_btn.setVisible(running)
        self.progress.setVisible(running)
        self.scope_combo.setEnabled(not running)
        self.workers_spin.setEnabled(not running)
        self.pause_btn.setText("Pause")
        if not running:
            self.workers_label.setText("")

    def _poll(self) -> None:
        r = self.runner
        if r is None:
            return
        st = r.poll()
        for kind, target, payload in st.events:
            self._running_targets.discard(target)
            if kind == "finished":
                self.session.add_summary(target, nifti.TargetSummary(**payload))
                self._failed.pop(target, None)
                self.resultSaved.emit(target)
            else:
                self._failed[target] = payload
            self.update_row(target)
        previous = self._running_targets
        self._running_targets = {t for t, _, _ in st.running.values()}
        for t in previous ^ self._running_targets:
            self.update_row(t)

        self.progress.setValue(st.finished + st.failed)
        text = f"{st.finished} done"
        if st.failed:
            text += f", {st.failed} failed"
        text += f", {len(st.running)} running, {st.pending} waiting"
        if st.paused:
            text += " · paused (running fits will finish)"
        elif st.eta_seconds is not None:
            text += f" · ≈ {fmt_duration(st.eta_seconds)} left"
        self.status.setText(text)
        lines = []
        for wid, (target, frac, elapsed) in sorted(st.running.items()):
            label = "whole brain" if target == WHOLE_BRAIN else f"ROI {target}"
            lines.append(f"w{wid + 1}: {label:<12} {frac * 100:3.0f}%  {fmt_duration(elapsed)}")
        self.workers_label.setText("\n".join(lines))
        if not r.active:
            self._finish(f"Batch finished: {st.finished} fitted"
                         + (f", {st.failed} failed (hover the status for the error)" if st.failed else "")
                         + ".")

    def _finish(self, message: str) -> None:
        self._timer.stop()
        for t in list(self._running_targets):
            self._running_targets.discard(t)
            self.update_row(t)
        self._set_batch_running(False)
        self.status.setText(message)
        self.update_estimate()


# ---------------------------------------------------------------------------
# NIfTI import dialog
# ---------------------------------------------------------------------------

class ImportDialog(QDialog):
    """Collects import settings, runs the import in a child process, returns the session folder."""

    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import NIfTI")
        self.setMinimumWidth(560)
        self.session_dir: str | None = None
        self._proc = self._queue = None
        self._created_dir = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

        v = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.nifti_edit = QLineEdit()
        self.nifti_edit.setPlaceholderText("4D NIfTI in MNI space (.nii / .nii.gz)")
        self.nifti_edit.editingFinished.connect(self._nifti_changed)
        row.addWidget(self.nifti_edit, stretch=1)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse_nifti(start_dir))
        row.addWidget(b)
        form.addRow("NIfTI:", row)
        self.nifti_info = QLabel("")
        self.nifti_info.setStyleSheet(f"color: {_DIM};")
        form.addRow("", self.nifti_info)

        arow = QHBoxLayout()
        self.n_rois = QComboBox()
        for n in atlas.N_ROIS_OPTIONS:
            self.n_rois.addItem(f"{n} parcels", n)
        self.n_rois.setCurrentIndex(atlas.N_ROIS_OPTIONS.index(400))
        self.networks = QComboBox()
        for k in atlas.NETWORK_OPTIONS:
            self.networks.addItem(f"{k} networks", k)
        self.networks.setCurrentIndex(1)
        arow.addWidget(self.n_rois)
        arow.addWidget(self.networks)
        arow.addStretch()
        form.addRow("Schaefer atlas:", arow)

        self.tr_spin = QDoubleSpinBox()
        self.tr_spin.setRange(0, 60)
        self.tr_spin.setDecimals(3)
        self.tr_spin.setSuffix(" s")
        self.tr_spin.setSpecialValueText("unknown")
        trow = QHBoxLayout()
        trow.addWidget(self.tr_spin)
        self.tr_note = QLabel("")
        self.tr_note.setStyleSheet("color: #fab387;")
        self.tr_note.setWordWrap(True)
        trow.addWidget(self.tr_note, stretch=1)
        form.addRow("TR:", trow)

        crow = QHBoxLayout()
        self.conf_edit = QLineEdit()
        self.conf_edit.setPlaceholderText("optional fMRIPrep confounds .tsv (motion, WM, CSF are used)")
        crow.addWidget(self.conf_edit, stretch=1)
        cb = QPushButton("Browse…")
        cb.clicked.connect(self._browse_confounds)
        crow.addWidget(cb)
        form.addRow("Confounds:", crow)

        prow = QHBoxLayout()
        self.detrend = QCheckBox("Detrend")
        self.detrend.setChecked(True)
        self.detrend.setToolTip(
            "Remove linear drift from every voxel. fMRIPrep outputs are not detrended, and "
            "drift makes GSBS split the run into a few long states (on nilearn's sample scan: "
            "2 states without detrending, 30 with it). Detrending data that were already "
            "detrended changes nothing.")
        self.hp_spin = QDoubleSpinBox()
        self.hp_spin.setRange(0, 1)
        self.hp_spin.setDecimals(4)
        self.hp_spin.setSingleStep(0.001)
        self.hp_spin.setSuffix(" Hz")
        self.hp_spin.setSpecialValueText("no high-pass")
        self.zparcels = QCheckBox("Z-score parcel timeseries")
        self.zparcels.setToolTip("Voxels are always z-scored before averaging, as in the GSBS "
                                 "scripts. This also z-scores the parcel averages, as "
                                 "neuro_tools.compute_roi_averaged_matrix does.")
        for w in (self.detrend, self.hp_spin, self.zparcels):
            prow.addWidget(w)
        prow.addStretch()
        form.addRow("Preprocessing:", prow)

        drow = QHBoxLayout()
        self.dir_edit = QLineEdit()
        drow.addWidget(self.dir_edit, stretch=1)
        db = QPushButton("Browse…")
        db.clicked.connect(self._browse_dir)
        drow.addWidget(db)
        form.addRow("Session folder:", drow)

        self.fit_after = QCheckBox("Fit whole-brain GSBS when the import finishes")
        self.fit_after.setChecked(True)
        form.addRow("", self.fit_after)
        v.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        v.addWidget(self.progress)
        self.stage = QLabel("")
        v.addWidget(self.stage)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.import_btn = self.buttons.addButton("Import", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self._cancel)
        v.addWidget(self.buttons)

    # ---- inputs -----------------------------------------------------------------------

    def _browse_nifti(self, start_dir: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "4D NIfTI", start_dir,
                                              "NIfTI (*.nii *.nii.gz);;All files (*.*)")
        if path:
            self.nifti_edit.setText(path)
            self._nifti_changed()

    def _browse_confounds(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Confounds", str(Path(self.nifti_edit.text()).parent),
                                              "TSV (*.tsv);;All files (*.*)")
        if path:
            self.conf_edit.setText(path)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Session folder", self.dir_edit.text())
        if path:
            self.dir_edit.setText(path)

    def _nifti_changed(self) -> None:
        path = self.nifti_edit.text().strip()
        if not path:
            return
        try:
            info = nifti.peek_nifti(path)
        except Exception as exc:                             # noqa: BLE001
            self.nifti_info.setText(f"Cannot read: {exc}")
            return
        x, y, z, t = info["shape"]
        mm = "×".join(f"{v:g}" for v in info["voxel_mm"])
        self.nifti_info.setText(
            f"{x}×{y}×{z} voxels ({mm} mm), {t} volumes · loading needs "
            f"≈ {core.format_bytes(info['bytes'] * 1.5)} of memory")
        if info["tr"]:
            self.tr_spin.setValue(info["tr"])
        self.tr_note.setText(info["tr_note"])
        if not self.dir_edit.text():
            self.dir_edit.setText(str(nifti.default_session_dir(path)))
        stem = nifti.nifti_stem(path)
        conf = Path(path).parent / (stem.split("_space-")[0].replace("_bold", "") + "_desc-confounds_timeseries.tsv")
        if not self.conf_edit.text() and conf.exists():
            self.conf_edit.setText(str(conf))

    def prefill(self, nifti_path: str, tr: float | None = None,
                confounds: str | None = None, note: str = "") -> None:
        """Fill the form for a known dataset (e.g. the online example)."""
        self.nifti_edit.setText(nifti_path)
        self._nifti_changed()
        if tr:
            self.tr_spin.setValue(tr)
            self.tr_note.setText(note)
            self.tr_note.setStyleSheet(f"color: {_DIM};")
        self.conf_edit.setText(confounds or "")

    # ---- run --------------------------------------------------------------------------

    def _start(self) -> None:
        path = self.nifti_edit.text().strip()
        out = self.dir_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Import", "Choose a NIfTI file.")
            return
        if not out:
            QMessageBox.warning(self, "Import", "Choose a session folder.")
            return
        if (Path(out) / "session.json").exists():
            reply = QMessageBox.question(
                self, "Session exists",
                f"{out} already holds an imported session.\n\nOpen it instead? "
                "(Choose No to pick a different folder.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.session_dir = out
                self.accept()
            return
        self._created_dir = not Path(out).exists()
        params = nifti.ImportParams(
            nifti=path, session_dir=out,
            n_rois=self.n_rois.currentData(), yeo_networks=self.networks.currentData(),
            tr=self.tr_spin.value() or None, confounds=self.conf_edit.text().strip() or None,
            detrend=self.detrend.isChecked(), high_pass=self.hp_spin.value() or None,
            zscore_parcels=self.zparcels.isChecked())
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._proc = ctx.Process(target=nifti.run_import, args=(self._queue, params), daemon=True)
        self._proc.start()
        self.progress.show()
        self.progress.setValue(0)
        self.import_btn.setEnabled(False)
        for w in self.findChildren(QLineEdit) + self.findChildren(QComboBox):
            w.setEnabled(False)
        self._timer.start()

    def _poll(self) -> None:
        final = None
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue_mod.Empty:
                break
            if msg[0] == "stage":
                self.stage.setText(msg[1] + "…")
                self.progress.setValue(int(msg[2] * 100))
            else:
                final = msg
        if final is None and not self._proc.is_alive():
            final = ("error", f"The import process ended unexpectedly (exit code "
                              f"{self._proc.exitcode}; possibly out of memory).")
        if final is None:
            return
        self._timer.stop()
        self._proc.join(2)
        self._proc = None
        if final[0] == "done":
            self.session_dir = final[1]
            self.accept()
        else:
            self._reset_inputs()
            box = QMessageBox(QMessageBox.Icon.Critical, "Import failed",
                              final[1].strip().splitlines()[-1], parent=self)
            box.setDetailedText(final[1])
            box.exec()

    def _reset_inputs(self) -> None:
        self.progress.hide()
        self.stage.setText("")
        self.import_btn.setEnabled(True)
        for w in self.findChildren(QLineEdit) + self.findChildren(QComboBox):
            w.setEnabled(True)

    def _cancel(self) -> None:
        if self._proc is not None:
            self._timer.stop()
            self._proc.terminate()
            self._proc.join(2)
            self._proc = None
            self._remove_partial()
            self._reset_inputs()
            self.stage.setText("Import cancelled.")
            return
        self.reject()

    def _remove_partial(self) -> None:
        """Delete only the files this import writes, and the folder if it created it."""
        out = Path(self.dir_edit.text().strip())
        if (out / "session.json").exists():
            return
        for name in ("voxels.npy", "voxel_labels.npy", "voxel_ijk.npy", "parcels.npy"):
            (out / name).unlink(missing_ok=True)
        if self._created_dir:
            for d in (out / "results", out):
                try:
                    d.rmdir()                                # only if empty
                except OSError:
                    pass

    def closeEvent(self, e) -> None:
        if self._proc is not None:
            self._cancel()
        super().closeEvent(e)
