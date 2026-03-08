#!/usr/bin/env python3
"""
GSBS GUI — GridSpec layout variant (PyQt6)

Uses a single matplotlib figure with GridSpec so all subplot proportions
are handled internally by matplotlib — no Qt layout fighting.
"""

import sys

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from statesegmentation import gsbs

plt.style.use("dark_background")

_FIG_BG = "#1e1e2e"
_ACCENT = "#89b4fa"
_RED    = "#f38ba8"
_GREEN  = "#a6e3a1"
_ORANGE = "#fab387"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _GsbsWorker(QObject):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, kmax: int, data: np.ndarray,
                 statewise: bool, finetune: int) -> None:
        super().__init__()
        self._kmax      = kmax
        self._data      = data
        self._statewise = statewise
        self._finetune  = finetune

    def run(self) -> None:
        try:
            obj = gsbs.GSBS(
                kmax=self._kmax,
                x=self._data,
                statewise_detection=self._statewise,
                finetune=self._finetune,
            )
            obj.fit(showProgressBar=False)
            self.finished.emit(obj)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class GSBSApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GSBS GUI")
        self.setMinimumSize(800, 600)

        self.roi_data:    np.ndarray | None = None
        self.gsbs_object                    = None
        self._corr_cache: np.ndarray | None = None
        self._gsbs_running: bool            = False
        self._thread: QThread | None        = None
        self._worker: _GsbsWorker | None    = None
        self._pending_k: int                = 0

        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(150)
        self._plot_timer.timeout.connect(self._deferred_refresh)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(8, 8, 8, 4)
        vbox.setSpacing(4)

        self._build_controls(vbox)
        self._build_run_status_row(vbox)
        self._build_solution_explorer(vbox)
        self._build_canvas(vbox)
        self.statusBar().showMessage("Ready.")

    def _build_controls(self, parent: QVBoxLayout) -> None:
        group  = QGroupBox("Controls")
        vstack = QVBoxLayout(group)
        vstack.setSpacing(6)

        # Row 0: file path + load
        row0 = QHBoxLayout()
        row0.setSpacing(6)
        row0.addWidget(QLabel("File (.npy):"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Browse or type path…")
        row0.addWidget(self.path_edit, stretch=1)
        self._btn(row0, "Browse",    self._browse_data)
        self._btn(row0, "Load File", self.load_file)
        vstack.addLayout(row0)

        # Row 1: parameters + run
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("kmax:"))
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(1, 9999)
        self.kmax_spin.setValue(10)
        row1.addWidget(self.kmax_spin)
        row1.addWidget(QLabel("finetune:"))
        self.finetune_spin = QSpinBox()
        self.finetune_spin.setRange(0, 100)
        self.finetune_spin.setValue(1)
        row1.addWidget(self.finetune_spin)
        self.statewise_cb = QCheckBox("Statewise detection")
        row1.addWidget(self.statewise_cb)
        self.run_btn = QPushButton("Run GSBS")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self.run_gsbs)
        row1.addWidget(self.run_btn)
        row1.addStretch()
        vstack.addLayout(row1)

        # Row 2: save
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Save as:"))
        self.save_edit = QLineEdit("gsbs_result.npy")
        row2.addWidget(self.save_edit, stretch=1)
        self._btn(row2, "Browse",           self._browse_save)
        self._btn(row2, "Save GSBS Object", self.save_gsbs_object)
        vstack.addLayout(row2)

        parent.addWidget(group)

    def _build_run_status_row(self, parent: QVBoxLayout) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(16)
        self.progress.hide()
        parent.addWidget(self.progress)

        self.patience_label = QLabel(
            "\u23f3  This can take tens of minutes with large kmax or "
            "statewise detection \u2014 please be patient.")
        self.patience_label.setWordWrap(True)
        self.patience_label.setObjectName("patience_label")
        self.patience_label.hide()
        parent.addWidget(self.patience_label)

    def _build_solution_explorer(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Solution Explorer")
        row   = QHBoxLayout(group)

        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setRange(0, 1)
        self.k_slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.k_slider, stretch=1)

        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(0, 1)
        self.k_spinbox.setFixedWidth(70)
        self.k_spinbox.valueChanged.connect(self._on_spinbox)
        row.addWidget(self.k_spinbox)

        self.best_k_label = QLabel("Best k: \u2014")
        row.addWidget(self.best_k_label)

        parent.addWidget(group)

    def _build_canvas(self, parent: QVBoxLayout) -> None:
        """Single figure, GridSpec — matplotlib owns all layout proportions.

          ┌──────────────┬──────────────────────────┐
          │              │  T-dist Curve             │
          │  Correlation ├──────────────────────────┤
          │  Matrix      │  Voxel Timeseries         │
          │  (square)    ├──────────────────────────┤
          │              │  State Activity Timeseries│
          └──────────────┴──────────────────────────┘
        """
        self.fig = plt.figure(constrained_layout=True)
        self.fig.patch.set_facecolor(_FIG_BG)

        gs = self.fig.add_gridspec(
            3, 2,
            width_ratios=[1, 1.8],
            height_ratios=[2, 1.5, 1.5],
        )

        self.ax_corrmat = self.fig.add_subplot(gs[:, 0])
        self.ax_tdist   = self.fig.add_subplot(gs[0, 1])
        self.ax_raw     = self.fig.add_subplot(gs[1, 1])
        self.ax_state   = self.fig.add_subplot(gs[2, 1], sharex=self.ax_raw)

        for ax in (self.ax_corrmat, self.ax_tdist, self.ax_raw, self.ax_state):
            ax.set_facecolor(_FIG_BG)

        self.ax_corrmat.set_title("Correlation Matrix", fontsize=9)
        self.ax_tdist.set_title("T-dist Curve", fontsize=9)
        self.ax_raw.set_title("Voxel Timeseries", fontsize=9)
        self.ax_raw.set_ylabel("Channels")
        self.ax_state.set_title("State Activity Timeseries", fontsize=9)
        self.ax_state.set_ylabel("Channels")
        self.ax_state.set_xlabel("Timepoints")

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        parent.addWidget(self.canvas, stretch=1)
        parent.addWidget(NavigationToolbar2QT(self.canvas, self))

    # ── widget helpers ──────────────────────────────────────────────────

    @staticmethod
    def _btn(layout: QHBoxLayout, text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        layout.addWidget(btn)
        return btn

    # ------------------------------------------------------------------
    # File dialogs
    # ------------------------------------------------------------------

    def _browse_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select data or GSBS object", "",
            "NumPy files (*.npy);;All files (*.*)")
        if path:
            self.path_edit.setText(path)

    def _browse_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GSBS object", "gsbs_result.npy",
            "NumPy files (*.npy);;All files (*.*)")
        if path:
            self.save_edit.setText(path)

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_file(self) -> None:
        """Load a .npy file — auto-detects GSBS object vs raw ROI data."""
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", "Enter or browse to a .npy file first.")
            return
        try:
            raw = np.load(path, allow_pickle=True)

            try:
                obj = raw.item()
                if hasattr(obj, "tdists") and hasattr(obj, "all_bounds"):
                    self.gsbs_object = obj
                    self.roi_data    = obj.x
                    self._corr_cache = np.corrcoef(obj.x)
                    kmax   = len(obj.tdists) - 1
                    best_k = int(np.argmax(obj.tdists))
                    self.kmax_spin.setValue(kmax)
                    self._configure_slider(kmax, best_k)
                    self._status(f"Loaded GSBS object  kmax={kmax}  best k={best_k}")
                    self._refresh_all_plots(best_k)
                    return
            except (ValueError, AttributeError):
                pass

            if raw.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {raw.shape}")
            self.roi_data    = raw
            self.gsbs_object = None
            self._corr_cache = None
            self._clear_all_plots()
            self._status(f"Loaded ROI data  shape={raw.shape}")
            self._plot_roi_data()

        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def save_gsbs_object(self) -> None:
        if self.gsbs_object is None:
            QMessageBox.warning(self, "Nothing to save", "No GSBS object in memory.")
            return
        path = self.save_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", "Enter a save path first.")
            return
        try:
            np.save(path, self.gsbs_object, allow_pickle=True)
            self._status(f"Saved GSBS object \u2192 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    # ------------------------------------------------------------------
    # GSBS computation (QThread + signals)
    # ------------------------------------------------------------------

    def run_gsbs(self) -> None:
        if self.roi_data is None:
            QMessageBox.warning(self, "No data", "Load ROI data first.")
            return
        self._gsbs_running = True
        self.run_btn.setEnabled(False)
        self.progress.show()
        self.patience_label.show()
        self._status("Running GSBS\u2026")

        self._worker = _GsbsWorker(
            kmax      = self.kmax_spin.value(),
            data      = self.roi_data,
            statewise = self.statewise_cb.isChecked(),
            finetune  = self.finetune_spin.value(),
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._gsbs_done)
        self._worker.error.connect(self._gsbs_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _gsbs_done(self, obj) -> None:
        self.gsbs_object   = obj
        self._gsbs_running = False
        self._corr_cache   = np.corrcoef(obj.x)
        self.progress.hide()
        self.patience_label.hide()
        self.run_btn.setEnabled(True)
        kmax   = len(obj.tdists) - 1
        best_k = int(np.argmax(obj.tdists))
        self._configure_slider(kmax, best_k)
        self._status(f"GSBS done  kmax={kmax}  best k={best_k}")
        self._refresh_all_plots(best_k)

    def _gsbs_error(self, msg: str) -> None:
        self._gsbs_running = False
        self.progress.hide()
        self.patience_label.hide()
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "GSBS error", msg)
        self._status("GSBS failed.")

    # ------------------------------------------------------------------
    # Slider / k control
    # ------------------------------------------------------------------

    def _configure_slider(self, kmax: int, value: int) -> None:
        for w in (self.k_slider, self.k_spinbox):
            w.blockSignals(True)
            w.setRange(1, kmax)
            w.setValue(value)
            w.blockSignals(False)
        self.best_k_label.setText(f"Best k: {value}")

    def _on_slider(self, value: int) -> None:
        if self.gsbs_object is None:
            return
        self.k_spinbox.blockSignals(True)
        self.k_spinbox.setValue(value)
        self.k_spinbox.blockSignals(False)
        self._pending_k = value
        self._plot_timer.start()

    def _on_spinbox(self, value: int) -> None:
        if self.gsbs_object is None:
            return
        self.k_slider.blockSignals(True)
        self.k_slider.setValue(value)
        self.k_slider.blockSignals(False)
        self._pending_k = value
        self._plot_timer.start()

    def _deferred_refresh(self) -> None:
        if self.gsbs_object is None:
            return
        kmax = len(self.gsbs_object.tdists) - 1
        k    = max(1, min(self._pending_k, kmax))
        self._refresh_all_plots(k)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _clear_all_plots(self) -> None:
        for ax in (self.ax_corrmat, self.ax_tdist, self.ax_raw, self.ax_state):
            ax.clear()
        self.canvas.draw()

    def _plot_roi_data(self) -> None:
        data = self.roi_data
        self.ax_corrmat.clear()
        self.ax_corrmat.imshow(np.corrcoef(data), cmap="viridis",
                               vmin=-1, vmax=1, aspect="equal")
        self.ax_corrmat.set_title("Correlation Matrix", fontsize=9)
        self.ax_raw.clear()
        self.ax_raw.imshow(data.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        self.ax_raw.set_title("Voxel Timeseries", fontsize=9)
        self.canvas.draw()

    def _refresh_all_plots(self, k: int) -> None:
        obj    = self.gsbs_object
        bounds = np.where(obj.all_bounds[k] > 0)[0]
        n_time = obj.x.shape[0]

        self.ax_tdist.clear()
        self.ax_tdist.plot(obj.tdists, color=_ACCENT)
        self.ax_tdist.axvline(x=k, color=_RED, linestyle="--",
                              linewidth=1.2, label=f"k = {k}")
        self.ax_tdist.set_xlabel("k (boundaries)")
        self.ax_tdist.set_ylabel("T-dist")
        self.ax_tdist.set_title("T-dist Curve", fontsize=9)
        self.ax_tdist.legend(fontsize=8)

        self.ax_corrmat.clear()
        self.ax_corrmat.imshow(self._corr_cache, cmap="viridis",
                               vmin=-1, vmax=1, aspect="equal")
        self.ax_corrmat.set_title("Correlation Matrix", fontsize=9)
        edges = np.concatenate(([0], bounds, [n_time]))
        for i in range(len(edges) - 1):
            x0, x1 = edges[i], edges[i + 1]
            self.ax_corrmat.add_patch(patches.Rectangle(
                (x0, x0), x1 - x0, x1 - x0,
                linewidth=1.5, edgecolor=_RED, facecolor="none"))

        self.ax_raw.clear()
        self.ax_raw.imshow(obj.x.T, aspect="auto", origin="lower")
        self.ax_raw.set_ylabel("Channels")
        self.ax_raw.set_title("Voxel Timeseries", fontsize=9)
        for b in bounds:
            self.ax_raw.axvline(x=b, color=_RED, linewidth=0.8)

        self.ax_state.clear()
        self.ax_state.imshow(self._state_timeseries(k).T,
                             aspect="auto", origin="lower")
        self.ax_state.set_ylabel("Channels")
        self.ax_state.set_xlabel("Timepoints")
        self.ax_state.set_title("State Activity Timeseries", fontsize=9)
        for b in bounds:
            self.ax_state.axvline(x=b, color=_RED, linewidth=0.8)

        self.canvas.draw()

    def _state_timeseries(self, k: int) -> np.ndarray:
        obj      = self.gsbs_object
        patterns = obj.get_state_patterns(k=k)
        bounds   = np.where(obj.all_bounds[k] > 0)[0]
        n_time   = obj.x.shape[0]
        state_ts = np.empty_like(obj.x)
        edges    = np.concatenate(([0], bounds, [n_time]))
        for i, pattern in enumerate(patterns):
            state_ts[edges[i]:edges[i + 1]] = pattern
        return state_ts

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._gsbs_running:
            reply = QMessageBox.question(
                self, "GSBS running",
                "GSBS is still running. Closing will cancel the computation.\n\nClose anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self._thread is not None and self._thread.isRunning():
                self._thread.terminate()
                self._thread.wait(2000)
        plt.close("all")
        event.accept()

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)


# ---------------------------------------------------------------------------
# Dark palette + stylesheet
# ---------------------------------------------------------------------------

def _setup_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    p = QPalette()
    dark     = QColor("#1e1e2e")
    mid      = QColor("#313244")
    text     = QColor("#cdd6f4")
    accent   = QColor("#89b4fa")
    disabled = QColor("#585b70")

    p.setColor(QPalette.ColorRole.Window,          dark)
    p.setColor(QPalette.ColorRole.WindowText,      text)
    p.setColor(QPalette.ColorRole.Base,            mid)
    p.setColor(QPalette.ColorRole.AlternateBase,   dark)
    p.setColor(QPalette.ColorRole.Text,            text)
    p.setColor(QPalette.ColorRole.Button,          mid)
    p.setColor(QPalette.ColorRole.ButtonText,      text)
    p.setColor(QPalette.ColorRole.Highlight,       accent)
    p.setColor(QPalette.ColorRole.HighlightedText, dark)
    p.setColor(QPalette.ColorRole.ToolTipBase,     dark)
    p.setColor(QPalette.ColorRole.ToolTipText,     text)
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.WindowText, disabled)
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.Text, disabled)
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.ButtonText, disabled)
    app.setPalette(p)

    app.setStyleSheet(f"""
        QPushButton#run_btn {{
            background-color: {_GREEN};
            color: #1e1e2e;
            font-weight: bold;
            padding: 4px 10px;
        }}
        QPushButton#run_btn:hover     {{ background-color: #94e2d5; }}
        QPushButton#run_btn:disabled  {{ background-color: {disabled.name()}; color: #1e1e2e; }}

        QLabel#patience_label {{ color: {_ORANGE}; }}

        QProgressBar {{ text-align: center; border-radius: 3px; }}
        QProgressBar::chunk {{ background-color: {_GREEN}; border-radius: 3px; }}
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    _setup_dark_theme(app)
    window = GSBSApp()
    window.show()
    if len(sys.argv) > 1:
        window.path_edit.setText(sys.argv[1])
        QTimer.singleShot(200, window.load_file)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
