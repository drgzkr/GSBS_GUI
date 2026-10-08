#!/usr/bin/env python3
"""
GSBS GUI (PyQt6)

Fit GSBS to an ROI timeseries and explore every solution from the t-distance
curve. The fit runs in a child process (see gsbs_core.run_fit): it can be
cancelled at any time, and the curve and boundaries are drawn live while it runs.

Whole-brain work: File > Import NIfTI extracts Schaefer parcel timeseries into a
session folder (gsbs_nifti). The session panel lists every ROI for voxel-level
fits and runs batches (gsbs_batch); the brain panel maps parcel-level state
patterns on the fsaverage surface (gsbs_brain_panel).
"""

import multiprocessing as mp
import queue as queue_mod
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import gsbs_atlas as atlas
import gsbs_core as core
import gsbs_nifti as nifti
from gsbs_brain_panel import BrainPanel
from gsbs_nifti import WHOLE_BRAIN
from gsbs_session_panel import ImportDialog, SessionPanel
from gsbs_stimulus_panel import StimulusPanel

plt.style.use("dark_background")

_FIG_BG = "#1e1e2e"
_TEXT   = "#cdd6f4"
_ACCENT = "#89b4fa"
_RED    = "#f38ba8"
_GREEN  = "#a6e3a1"
_ORANGE = "#fab387"
_YELLOW = "#f9e2af"
_BOUND  = "#000000"     # boundaries drawn over RdBu images: black reads on red, blue and white
_CMAP   = "RdBu_r"

_POLL_MS = 50


class _ExampleFetcher(QObject):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.stop = False

    def _progress(self, done: int, total: int) -> None:
        if self.stop:
            raise nifti.DownloadCancelled()
        self.progress.emit(done, total)

    def run(self) -> None:
        try:
            if self.key == nifti.READY_DEMO.key:
                path = nifti.install_demo_session(progress=self._progress)
                self.done.emit({"session": str(path)})
                return
            self.done.emit(nifti.fetch_example(self.key, progress=self._progress))
        except nifti.DownloadCancelled:
            self.cancelled.emit()
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class GSBSApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GSBS GUI")
        self.setMinimumSize(900, 650)

        self.x: np.ndarray | None = None          # data being viewed
        self.gsbs_object = None                   # fitted result, if any
        self.truth: np.ndarray | None = None      # known boundaries (demo only)
        self._ks = np.array([], int)              # valid solutions of gsbs_object
        self._k = 0                               # k on display
        self._edges = None                        # state edges of the solution on display
        self._sel = 0                             # selected state (index into the solution)
        self.meta: dict | None = None             # what the data columns are (gsbs_gui_meta)
        self.session: nifti.Session | None = None
        self.target = None                        # session target on display, if any

        self._proc = None
        self._queue = None
        self._run_started = 0.0
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_MS)
        self._poll_timer.timeout.connect(self._poll_fit)

        self._build_ui()
        self._build_docks()
        self._build_menu()
        self._set_running(False)
        self._update_memory_label()

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
        self.statusBar().showMessage("Open a .npy file, or try the demo.")

    def _build_controls(self, parent: QVBoxLayout) -> None:
        group  = QGroupBox("Data and parameters")
        vstack = QVBoxLayout(group)
        vstack.setSpacing(6)

        # Row 0: file
        row0 = QHBoxLayout()
        self.open_btn = self._btn(row0, "Open…", self._browse_data)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "ROI data (timepoints × voxels) or a saved GSBS object, as .npy")
        self.path_edit.returnPressed.connect(self.load_file)
        row0.addWidget(self.path_edit, stretch=1)

        self.demo_btn = QToolButton()
        self.demo_btn.setText("Demo")
        self.demo_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.demo_btn)
        act = QAction("Fitted demo result (instant)", self)
        act.triggered.connect(lambda: self.load_demo(fitted=True))
        menu.addAction(act)
        act = QAction("Demo data only (run the fit yourself)", self)
        act.triggered.connect(lambda: self.load_demo(fitted=False))
        menu.addAction(act)
        menu.addSeparator()
        act = QAction(f"{nifti.READY_DEMO.title} \u00b7 \u2248 {nifti.READY_DEMO.size_mb:g} MB "
                      f"download once\u2026", self)
        act.setToolTip(nifti.READY_DEMO.description)
        act.triggered.connect(lambda: self.load_online_example(nifti.READY_DEMO.key))
        menu.addAction(act)
        online = menu.addMenu("Online example NIfTI (import a whole-brain scan)")
        online.setToolTipsVisible(True)
        for ex in nifti.EXAMPLES:
            act = QAction(f"{ex.title} \u00b7 {ex.volumes} TRs \u00b7 \u2248 {ex.size_mb:g} MB\u2026", self)
            act.setToolTip(ex.description)
            act.triggered.connect(lambda _=False, key=ex.key: self.load_online_example(key))
            online.addAction(act)
        menu.setToolTipsVisible(True)
        self.demo_btn.setMenu(menu)
        row0.addWidget(self.demo_btn)
        self.import_btn = self._btn(row0, "Import NIfTI\u2026", self.import_nifti)
        self.import_btn.setToolTip("Extract Schaefer parcel and ROI voxel timeseries from a 4D NIfTI")

        self.columns_combo = QComboBox()
        self.columns_combo.setToolTip(
            "What the data columns are. Choosing a Schaefer atlas maps state patterns on the "
            "brain surface. The 7- and 17-network versions order parcels differently, so pick "
            "the one the data was extracted with.")
        self.columns_combo.currentIndexChanged.connect(self._on_columns_changed)
        self.columns_combo.hide()
        row0.addWidget(self.columns_combo)

        self.data_label = QLabel("No data")
        self.data_label.setObjectName("dim_label")
        row0.addWidget(self.data_label)
        vstack.addLayout(row0)

        # Row 1: main parameters + run
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Engine:"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("fast_GSBS", "fast_GSBS")
        self.engine_combo.addItem("GSBS (reference)", "GSBS")
        self.engine_combo.setToolTip(
            "fast_GSBS gives bit-identical results to GSBS, typically 5–10× faster.")
        row1.addWidget(self.engine_combo)

        row1.addWidget(QLabel("kmax:"))
        self.kmax_spin = QSpinBox()
        self.kmax_spin.setRange(2, 100000)
        self.kmax_spin.setValue(10)
        self.kmax_spin.setToolTip(
            "Largest number of states to search. Half the number of timepoints is a "
            "reasonable choice and is set when data is loaded.")
        row1.addWidget(self.kmax_spin)

        row1.addWidget(QLabel("finetune:"))
        self.finetune_spin = QSpinBox()
        self.finetune_spin.setRange(-1, 100000)
        self.finetune_spin.setValue(1)
        self.finetune_spin.setSpecialValueText("all TRs")
        self.finetune_spin.setToolTip(
            "TRs around each boundary searched when finetuning. 0 = no finetuning; "
            "'all TRs' (−1) searches everywhere and is slow.")
        row1.addWidget(self.finetune_spin)

        self.statewise_cb = QCheckBox("Statewise detection")
        self.statewise_cb.setChecked(True)
        self.statewise_cb.setToolTip(
            "2-D search that can place a whole new state (two boundaries) per step. "
            "On by default in statesegmentation.")
        row1.addWidget(self.statewise_cb)

        self.adv_btn = QToolButton()
        self.adv_btn.setText("Advanced ▸")
        self.adv_btn.setCheckable(True)
        self.adv_btn.toggled.connect(self._toggle_advanced)
        row1.addWidget(self.adv_btn)

        row1.addStretch()
        self.mem_label = QLabel("")
        self.mem_label.setObjectName("dim_label")
        row1.addWidget(self.mem_label)
        self.run_btn = QPushButton("Run GSBS")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._run_or_cancel)
        row1.addWidget(self.run_btn)
        vstack.addLayout(row1)

        # Row 2: advanced parameters (hidden by default)
        self.adv_row = QWidget()
        row2 = QHBoxLayout(self.adv_row)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("finetune order:"))
        self.ft_order_combo = QComboBox()
        self.ft_order_combo.addItem("weakest → strongest", True)
        self.ft_order_combo.addItem("strongest → weakest", False)
        row2.addWidget(self.ft_order_combo)
        row2.addWidget(QLabel("dmin:"))
        self.dmin_spin = QSpinBox()
        self.dmin_spin.setRange(1, 1000)
        self.dmin_spin.setValue(1)
        self.dmin_spin.setToolTip(
            "Minimum distance between timepoints whose correlations enter the t-distance.")
        row2.addWidget(self.dmin_spin)
        row2.addWidget(QLabel("blocksize:"))
        self.blocksize_spin = QSpinBox()
        self.blocksize_spin.setRange(1, 100000)
        self.blocksize_spin.setValue(50)
        self.blocksize_spin.setToolTip("Block size for the candidate-boundary pre-search.")
        row2.addWidget(self.blocksize_spin)
        row2.addStretch()
        self.adv_row.hide()
        vstack.addWidget(self.adv_row)

        for w in (self.kmax_spin,):
            w.valueChanged.connect(self._update_memory_label)
        self.statewise_cb.toggled.connect(self._update_memory_label)

        parent.addWidget(group)

    def _build_run_status_row(self, parent: QVBoxLayout) -> None:
        self.run_row = QWidget()
        row = QHBoxLayout(self.run_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(16)
        self.progress.setFormat("%v / %m iterations")
        row.addWidget(self.progress, stretch=1)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("dim_label")
        row.addWidget(self.eta_label)
        self.run_row.hide()
        parent.addWidget(self.run_row)

    def _build_solution_explorer(self, parent: QVBoxLayout) -> None:
        group = QGroupBox("Solutions")
        row   = QHBoxLayout(group)

        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setRange(0, 0)
        self.k_slider.setToolTip(
            "Steps through every solution GSBS produced. With statewise detection some "
            "numbers of states are skipped, because two boundaries were added at once.")
        self.k_slider.valueChanged.connect(self._on_slider)
        row.addWidget(self.k_slider, stretch=1)

        row.addWidget(QLabel("states:"))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(0, 0)
        self.k_spinbox.setFixedWidth(80)
        self.k_spinbox.valueChanged.connect(self._on_spinbox)
        row.addWidget(self.k_spinbox)

        self.best_btn = QPushButton("Best")
        self.best_btn.setToolTip("Jump to the solution with the highest t-distance (B)")
        self.best_btn.clicked.connect(self.go_to_best)
        row.addWidget(self.best_btn)
        QShortcut(QKeySequence("B"), self, activated=self.go_to_best)

        self.best_k_label = QLabel("best: —")
        row.addWidget(self.best_k_label)

        self.duration_label = QLabel("")
        self.duration_label.setObjectName("dim_label")
        self.duration_label.setToolTip("Median duration of the states in the solution shown, "
                                       "with the shortest and longest state")
        row.addWidget(self.duration_label)

        self.save_btn = QPushButton("Save result…")
        self.save_btn.clicked.connect(self.save_gsbs_object)
        row.addWidget(self.save_btn)

        self.cluster_btn = QPushButton("Cluster states…")
        self.cluster_btn.setToolTip("Cluster state patterns from chosen solutions into recurring states")
        self.cluster_btn.clicked.connect(self.open_clusters)
        row.addWidget(self.cluster_btn)

        parent.addWidget(group)

    def _build_canvas(self, parent: QVBoxLayout) -> None:
        """Single figure, GridSpec — matplotlib owns all layout proportions.

          ┌──────────────┬──────────────────────────┐
          │              │  T-distance curve         │
          │  Correlation ├──────────────────────────┤
          │  matrix      │  Voxel timeseries         │
          │  (square)    ├──────────────────────────┤
          │              │  State mean patterns      │
          └──────────────┴──────────────────────────┘
        """
        self.fig = plt.figure(constrained_layout=True)
        self.fig.patch.set_facecolor(_FIG_BG)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1, 1.8], height_ratios=[2, 1.5, 1.5])
        self.ax_corrmat = self.fig.add_subplot(gs[:, 0])
        self.ax_tdist   = self.fig.add_subplot(gs[0, 1])
        self.ax_raw     = self.fig.add_subplot(gs[1, 1])
        self.ax_state   = self.fig.add_subplot(gs[2, 1], sharex=self.ax_raw)
        self._cbar = None
        self._artists: dict = {}
        self._reset_axes()

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        parent.addWidget(self.canvas, stretch=1)
        parent.addWidget(self.toolbar)

    def _build_docks(self) -> None:
        self.session_panel = SessionPanel(lambda: (self.engine_combo.currentData(), self._params()))
        self.session_panel.targetActivated.connect(self.activate_target)
        self.session_panel.resultSaved.connect(self._on_batch_result)
        self.session_dock = QDockWidget("Session", self)
        self.session_dock.setObjectName("session_dock")
        self.session_dock.setWidget(self.session_panel)
        self.session_dock.setMinimumWidth(360)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.session_dock)
        self.session_dock.hide()

        self.brain = BrainPanel()
        self.brain.overlap_sources = self._overlap_references
        self.brain.stateSelected.connect(self._select_state)
        self.brain.roiSelected.connect(self._open_roi)
        self.brain_dock = QDockWidget("Brain", self)
        self.brain_dock.setObjectName("brain_dock")
        self.brain_dock.setWidget(self.brain)
        self.brain_dock.setMinimumHeight(250)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.brain_dock)
        self.brain_dock.hide()

        self.stimulus = StimulusPanel()
        self.stimulus.stateSelected.connect(self._select_state)
        self.stimulus.boundariesChanged.connect(self._draw_stimulus_ticks)
        self.stimulus.boundariesChanged.connect(lambda: self.brain.refresh_overlap_sources())
        self.stimulus.transcriptLoaded.connect(self._attach_transcript)
        self.stimulus.eventsLoaded.connect(self._attach_events)
        self.stimulus_dock = QDockWidget("Stimulus", self)
        self.stimulus_dock.setObjectName("stimulus_dock")
        self.stimulus_dock.setWidget(self.stimulus)
        self.stimulus_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.stimulus_dock)
        self.stimulus_dock.hide()
        self._manual_transcript = None
        self._manual_events = None

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        for text, slot, keys in (
                ("&Open data or result\u2026", self._browse_data, "Ctrl+O"),
                ("&Import NIfTI\u2026", self.import_nifti, "Ctrl+I"),
                ("Open &session folder\u2026", self._browse_session, "Ctrl+Shift+O"),
                ("&Save result\u2026", self.save_gsbs_object, "Ctrl+S"),
                ("&Export analysis package (scripts, figures, Methods/Results)\u2026", self.export_package, "Ctrl+E"),
                (None, None, None),
                ("&Quit", self.close, "Ctrl+Q")):
            if text is None:
                m.addSeparator()
                continue
            act = QAction(text, self)
            act.setShortcut(QKeySequence(keys))
            act.triggered.connect(slot)
            m.addAction(act)
        a = self.menuBar().addMenu("&Analysis")
        act = QAction("State &clusters\u2026", self)
        act.setShortcut(QKeySequence("Ctrl+K"))
        act.triggered.connect(self.open_clusters)
        a.addAction(act)
        act = QAction("Boundary &overlap\u2026", self)
        act.setShortcut(QKeySequence("Ctrl+B"))
        act.triggered.connect(self.open_overlap)
        a.addAction(act)
        act = QAction("Stimulus panel (load transcript or events)", self)
        act.triggered.connect(lambda: (self.stimulus_dock.show(), self.stimulus_dock.raise_()))
        a.addAction(act)
        v = self.menuBar().addMenu("&View")
        v.addAction(self.session_dock.toggleViewAction())
        v.addAction(self.brain_dock.toggleViewAction())
        v.addAction(self.stimulus_dock.toggleViewAction())

    @staticmethod
    def _btn(layout: QHBoxLayout, text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        layout.addWidget(btn)
        return btn

    def _toggle_advanced(self, on: bool) -> None:
        self.adv_row.setVisible(on)
        self.adv_btn.setText("Advanced ▾" if on else "Advanced ▸")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _params(self) -> dict:
        return {
            "kmax": self.kmax_spin.value(),
            "statewise_detection": self.statewise_cb.isChecked(),
            "finetune": self.finetune_spin.value(),
            "finetune_order": self.ft_order_combo.currentData(),
            "dmin": self.dmin_spin.value(),
            "blocksize": self.blocksize_spin.value(),
        }

    def _show_params(self, p: dict) -> None:
        self.kmax_spin.setValue(p["kmax"])
        self.statewise_cb.setChecked(p["statewise_detection"])
        self.finetune_spin.setValue(p["finetune"])
        self.ft_order_combo.setCurrentIndex(0 if p["finetune_order"] else 1)
        self.dmin_spin.setValue(p["dmin"])
        self.blocksize_spin.setValue(p["blocksize"])

    def _memory_estimate(self) -> tuple[int, int | None]:
        n_time, n_vox = self.x.shape
        need = core.estimate_fit_bytes(n_time, n_vox, self.kmax_spin.value(),
                                       self.statewise_cb.isChecked())
        return need, core.available_bytes()

    def _update_memory_label(self) -> None:
        if self.x is None:
            self.mem_label.setText("")
            return
        need, avail = self._memory_estimate()
        text = f"≈ {core.format_bytes(need)} peak"
        colour = ""
        if avail:
            text += f" of {core.format_bytes(avail)} free"
            frac = need / avail
            colour = _RED if frac > 0.85 else _ORANGE if frac > 0.5 else ""
        self.mem_label.setText(text)
        self.mem_label.setStyleSheet(f"color: {colour};" if colour else "")
        self.mem_label.setToolTip(
            "Estimated peak memory for the fit plus the result. It grows with the square "
            "of the number of timepoints (the time-by-time covariance).")

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def _browse_data(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ROI data or GSBS object", self.path_edit.text(),
            "NumPy files (*.npy);;All files (*.*)")
        if path:
            self.path_edit.setText(path)
            self.load_file()

    def load_demo(self, fitted: bool) -> None:
        self.path_edit.setText(str(core.DEMO_OBJECT if fitted else core.DEMO_DATA))
        self.load_file()

    def load_file(self) -> None:
        """Load a .npy file — auto-detects GSBS object vs raw ROI data."""
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", "Enter or browse to a .npy file first.")
            return
        try:
            loaded = core.load_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        is_demo = Path(path).resolve() in (core.DEMO_DATA, core.DEMO_OBJECT)
        self.truth = np.load(core.DEMO_TRUTH) if is_demo and core.DEMO_TRUTH.exists() else None
        self.target = None
        if self.session is not None:
            self.session_panel.table.clearSelection()

        meta = getattr(loaded.obj, "gsbs_gui_meta", None) if loaded.obj is not None else None
        self._set_dataset(loaded.x, loaded.obj, meta)
        n_time, n_cols = loaded.x.shape
        if loaded.obj is not None:
            self._status(f"Loaded GSBS result ({type(loaded.obj).__name__})  "
                         f"kmax={loaded.obj.kmax}  best={core.best_k(loaded.obj)} states")
        else:
            self._status(f"Loaded data  {n_time} timepoints × {n_cols} columns  "
                         "— set parameters and run GSBS")

    def _set_dataset(self, x: np.ndarray, obj, meta: dict | None, label: str | None = None) -> None:
        """Show new data (and its fitted result, if any)."""
        self.x = x
        self.gsbs_object = None
        self._ks = np.array([], int)
        self._edges = None
        self._sel = 0
        self.meta = dict(meta) if meta else None
        n_time, n_cols = x.shape
        self._update_columns_combo()
        self.data_label.setText(label or f"{n_time} TRs × {n_cols} {self._col_noun()}")
        self._init_plots()
        self._update_brain_data()
        if obj is not None:
            self._show_params(core.fit_params(obj))
            self._set_result(obj)
        else:
            self.kmax_spin.setValue(max(2, n_time // 2))
            self._configure_explorer()
        self._update_memory_label()

    # ------------------------------------------------------------------
    # Column meaning (voxels or Schaefer parcels)
    # ------------------------------------------------------------------

    def _spec(self) -> atlas.Schaefer | None:
        return atlas.Schaefer.from_meta(self.meta)

    def _is_parcels(self) -> bool:
        spec = self._spec()
        return (spec is not None and self.meta.get("kind") == "parcels"
                and self.x is not None and self.x.shape[1] == spec.n_rois)

    def _col_noun(self) -> str:
        return "parcels" if self._is_parcels() else "voxels"

    def _update_columns_combo(self) -> None:
        c = self.columns_combo
        c.blockSignals(True)
        c.clear()
        c.addItem("Columns: voxels", None)
        n_cols = self.x.shape[1]
        for k in atlas.NETWORK_OPTIONS:
            if n_cols in atlas.N_ROIS_OPTIONS:
                c.addItem(f"Columns: Schaefer {n_cols} / {k} networks", (n_cols, k))
        spec = self._spec()
        if self._is_parcels():
            c.setCurrentIndex(max(0, c.findData((spec.n_rois, spec.yeo_networks))))
        c.blockSignals(False)
        c.setVisible(c.count() > 1 and self.target is None)

    def _on_columns_changed(self) -> None:
        choice = self.columns_combo.currentData()
        if choice is None:
            self.meta = None
        else:
            tr = self.meta.get("tr") if self.meta else None
            self.meta = {"kind": "parcels", **atlas.Schaefer(*choice).to_meta(), "tr": tr}
        noun = self._col_noun()
        self.data_label.setText(f"{self.x.shape[0]} TRs × {self.x.shape[1]} {noun}")
        for ax in (self.ax_raw, self.ax_state):
            ax.set_ylabel(noun.capitalize())
        self._update_brain_data()
        if self.gsbs_object is not None:
            self.show_k(self._k)
        else:
            self.canvas.draw_idle()

    def _update_brain_data(self) -> None:
        spec = self._spec() or (self.session.spec if self.session else None)
        self.brain.set_atlas(spec)
        missing = (self.session.meta.get("missing_parcels", [])
                   if self.session is not None and self.target == WHOLE_BRAIN else [])
        tr = self.meta.get("tr") if self.meta else None
        self.brain.set_parcel_data(self.x if self._is_parcels() else None, tr, missing)
        self.brain.set_current_roi(self.target if isinstance(self.target, int) else None)
        self.brain_dock.setVisible(self.brain.has_content)
        self._update_stimulus_data()

    # ------------------------------------------------------------------
    # Sessions (imported NIfTIs)
    # ------------------------------------------------------------------

    def import_nifti(self) -> None:
        if self._proc is not None:
            return
        start = str(Path(self.path_edit.text().strip() or ".").parent)
        dlg = ImportDialog(self, start_dir=start)
        if dlg.exec() == ImportDialog.DialogCode.Accepted and dlg.session_dir:
            self.open_session(dlg.session_dir, fit_whole_brain=dlg.fit_after.isChecked())

    def load_online_example(self, key: str = "pixar") -> None:
        """Download an example scan (once) and open the import dialog for it."""
        if self._proc is not None or getattr(self, "_fetch_thread", None) is not None:
            return
        ex = nifti.READY_DEMO if key == nifti.READY_DEMO.key else nifti.EXAMPLE_BY_KEY[key]
        self._example = ex
        self._status(f"Getting the example NIfTI (\u2248 {ex.size_mb:g} MB, downloaded once)\u2026")
        self.demo_btn.setEnabled(False)
        self._fetch_dialog = QProgressDialog(f"Downloading {ex.title}\u2026", "Cancel", 0, 100, self)
        self._fetch_dialog.setWindowTitle("Example data")
        self._fetch_dialog.setMinimumDuration(400)
        self._fetch_dialog.setAutoClose(False)
        self._fetch_dialog.setAutoReset(False)
        self._fetcher = _ExampleFetcher(key)
        self._fetch_dialog.canceled.connect(lambda: setattr(self._fetcher, "stop", True))
        self._fetch_thread = QThread(self)
        self._fetcher.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetcher.run)
        self._fetcher.progress.connect(self._example_progress)
        self._fetcher.done.connect(self._example_ready)
        self._fetcher.failed.connect(self._example_failed)
        self._fetcher.cancelled.connect(lambda: self._status("Example download cancelled."))
        for sig in (self._fetcher.done, self._fetcher.failed, self._fetcher.cancelled):
            sig.connect(self._fetch_thread.quit)
        self._fetch_thread.finished.connect(self._fetch_finished)
        self._fetch_thread.start()

    def _example_progress(self, done: int, total: int) -> None:
        if total:
            self._fetch_dialog.setValue(int(100 * done / total))
            self._fetch_dialog.setLabelText(
                f"Downloading {self._example.title}\u2026\n"
                f"{done / 1e6:.0f} of {total / 1e6:.0f} MB")

    def _fetch_finished(self) -> None:
        self._fetch_thread = None
        self._fetch_dialog.close()
        self.demo_btn.setEnabled(self._proc is None)

    def _example_failed(self, msg: str) -> None:
        self._status("Example download failed.")
        QMessageBox.critical(self, "Example data", f"Could not get the example:\n{msg}")

    def _example_ready(self, files: dict) -> None:
        ex = self._example
        self._fetch_dialog.close()
        if "session" in files:                       # ready-made session: open it directly
            self.open_session(files["session"])
            self._status("Opened the ready-made demo session: " + ex.description)
            return
        self._status("Example ready: " + ex.description)
        dlg = ImportDialog(self)
        dlg.setWindowTitle("Import NIfTI \u2014 online example")
        dlg.prefill(files["nifti"], tr=files["tr"], confounds=files["confounds"],
                    note=ex.tr_note or "From the dataset description.")
        if ex.url is not None:
            dlg.stage.setText("These data are already confound-cleaned (motion, CSF/WM "
                              "components, high-pass, polynomial trends), so no confounds "
                              "file is applied.")
        if dlg.exec() == ImportDialog.DialogCode.Accepted and dlg.session_dir:
            transcript = files.get("transcript")
            if transcript is not None:
                import gsbs_nifti as _n
                sess = _n.Session(dlg.session_dir)
                if sess.transcript() is None:
                    sess.attach_transcript(transcript)
            self.open_session(dlg.session_dir, fit_whole_brain=dlg.fit_after.isChecked())

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open session folder",
                                                str(Path(self.path_edit.text().strip() or ".").parent))
        if path:
            self.open_session(path)

    def open_session(self, path: str, fit_whole_brain: bool = False) -> None:
        if self._proc is not None:
            QMessageBox.information(self, "Busy", "Wait for or cancel the running fit first.")
            return
        if not self.session_panel.cancel_batch(ask=True):
            return
        try:
            session = nifti.Session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Session", str(exc))
            return
        self.session = session
        self.session_panel.set_session(session)
        self.session_dock.show()
        self.brain.set_session(session)
        self.activate_target(WHOLE_BRAIN)
        if fit_whole_brain and self.gsbs_object is None:
            self.run_gsbs()

    def activate_target(self, target) -> None:
        """Show the whole-brain parcels or one ROI's voxels from the session."""
        if self.session is None:
            return
        if self._proc is not None:
            self._status("A fit is running — wait for it or cancel it before switching.")
            self.session_panel.select_target(self.target)
            return
        s = self.session
        try:
            x = s.data(target)
            obj = s.load_result(target)
        except Exception as exc:
            QMessageBox.critical(self, "Session", str(exc))
            return
        self.truth = None
        self.target = target
        self.path_edit.setText(str(s.path))
        noun = "parcels" if target == WHOLE_BRAIN else "voxels"
        label = f"{s.target_label(target)} · {x.shape[0]} TRs × {x.shape[1]} {noun}"
        self._set_dataset(x, obj, s.target_meta(target), label=label)
        self.session_panel.select_target(target)
        self._status(f"{s.target_label(target)}: "
                     + ("fitted, best = %d states" % core.best_k(obj) if obj is not None
                        else "not fitted yet — Run GSBS, or run a batch from the session panel"))

    def open_clusters(self) -> None:
        from gsbs_cluster_window import ClusterWindow
        if getattr(self, "cluster_window", None) is None:
            self.cluster_window = ClusterWindow(self)
        win = self.cluster_window
        self._update_cluster_window(force=True)
        win.show()
        win.raise_()
        win.activateWindow()

    def _update_cluster_window(self, force: bool = False) -> None:
        """Hand the shown segmentation to the cluster window, if it is open."""
        win = getattr(self, "cluster_window", None)
        if win is None or self.gsbs_object is None or not (force or win.isVisible()):
            return
        name = self.data_label.text().split(" \u00b7 ")[0]
        win.set_segmentation(self.gsbs_object, self._k, name, self.meta)

    # ------------------------------------------------------------------
    # Stimulus and boundary overlap
    # ------------------------------------------------------------------

    def _update_stimulus_data(self) -> None:
        if self.session is not None:
            transcript, events = self.session.transcript(), self.session.event_times()
        else:
            transcript, events = self._manual_transcript, self._manual_events
        tr = self.meta.get("tr") if self.meta else None
        n_time = self.x.shape[0] if self.x is not None else 0
        self.stimulus.set_data(transcript, events, tr, n_time)
        if self.stimulus.has_content:
            self.stimulus_dock.show()
        self._draw_stimulus_ticks()

    def _attach_transcript(self, transcript) -> None:
        if self.session is not None:
            self.session.attach_transcript(transcript)
        else:
            self._manual_transcript = transcript
        self._update_stimulus_data()
        if self._edges is not None:
            self.stimulus.set_solution(self._edges, self._k)

    def _attach_events(self, times) -> None:
        if self.session is not None:
            self.session.attach_events(times)
        else:
            self._manual_events = times
        self._update_stimulus_data()
        if self._edges is not None:
            self.stimulus.set_solution(self._edges, self._k)

    def _draw_stimulus_ticks(self) -> None:
        """Stimulus boundaries as short ticks under the top of the timeseries."""
        a = self._artists
        if not a or self.x is None:
            return
        if "stim_ticks" in a:
            a["stim_ticks"].remove()
            del a["stim_ticks"]
        sb = self.stimulus.stimulus_bounds() if self.stimulus.has_content else None
        if sb is not None and len(sb):
            n_vox = self.x.shape[1]
            top = n_vox - 0.5
            segs = [[(b - 0.5, top), (b - 0.5, top - 0.12 * n_vox)] for b in sb]
            a["stim_ticks"] = LineCollection(segs, colors="#a6e3a1", linewidths=1.0, zorder=6)
            self.ax_raw.add_collection(a["stim_ticks"])
        self.canvas.draw_idle()

    def _overlap_references(self) -> list[dict]:
        """References for the brain panel's per-ROI overlap map."""
        refs = []
        tr = self.session.tr if self.session is not None else None
        times = self.stimulus.stimulus_times() if self.stimulus.has_content else None
        if times is not None and len(times) and tr:
            st = self.stimulus.settings()
            refs.append({"label": f"Stimulus: {st['label'].lower()}", "kind": "stimulus",
                         "times": np.asarray(times, float), "tr": tr, "delay": st["delay_s"],
                         "config": {"kind": "stimulus", "boundaries": st["kind"],
                                    "label": st["label"], "pause_s": st["pause_s"]}})
        if self.session is not None and WHOLE_BRAIN in self.session.summary():
            s = self.session.summary()[WHOLE_BRAIN]
            refs.append({"label": f"Whole brain (best, {s.nstates} states)", "kind": "neural",
                         "bounds": self.session.best_bounds(WHOLE_BRAIN),
                         "config": {"kind": "neural", "target": WHOLE_BRAIN}})
        return refs

    def export_package(self) -> None:
        from gsbs_export_dialog import ExportDialog
        ExportDialog(self).exec()

    def open_overlap(self) -> None:
        from gsbs_overlap_window import OverlapWindow
        if getattr(self, "overlap_window", None) is None:
            self.overlap_window = OverlapWindow(self)
        self.overlap_window.refresh_sources()
        self.overlap_window.show()
        self.overlap_window.raise_()

    def _open_roi(self, roi: int) -> None:
        if self.session is not None:
            self.activate_target(int(roi))

    def _on_batch_result(self, target) -> None:
        self.brain.refresh_roi_map()
        if target == self.target and self._proc is None:
            k = self._k
            self.activate_target(target)
            if k in self._ks:
                self.show_k(k)

    def save_gsbs_object(self) -> None:
        if self.gsbs_object is None:
            QMessageBox.warning(self, "Nothing to save", "No GSBS result in memory.")
            return
        stem = Path(self.path_edit.text().strip() or "gsbs").stem
        default = str(Path(self.path_edit.text().strip() or ".").parent / f"{stem}_gsbs.npy")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GSBS result", default, "NumPy files (*.npy);;All files (*.*)")
        if not path:
            return
        try:
            if self.meta and not hasattr(self.gsbs_object, "gsbs_gui_meta"):
                self.gsbs_object.gsbs_gui_meta = dict(self.meta)
            core.save_object(self.gsbs_object, path)
            self._status(f"Saved GSBS result → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    # ------------------------------------------------------------------
    # Fitting (child process)
    # ------------------------------------------------------------------

    def _run_or_cancel(self) -> None:
        if self._proc is not None:
            self.cancel_gsbs()
        else:
            self.run_gsbs()

    def run_gsbs(self) -> None:
        if self.x is None:
            QMessageBox.warning(self, "No data", "Load ROI data first.")
            return
        params = self._params()
        if params["kmax"] >= self.x.shape[0]:
            QMessageBox.warning(self, "kmax too large",
                                f"kmax must be smaller than the number of timepoints "
                                f"({self.x.shape[0]}).")
            return
        need, avail = self._memory_estimate()
        if avail and need > 0.85 * avail:
            reply = QMessageBox.warning(
                self, "Memory",
                f"This fit needs an estimated {core.format_bytes(need)}, and only "
                f"{core.format_bytes(avail)} is free. It may be very slow or be killed "
                f"by the system.\n\nRun anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.gsbs_object = None
        self._ks = np.array([], int)
        self._edges = None
        self._configure_explorer()
        self._init_plots()
        self.brain.set_solution(None, 0)

        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        self._proc = ctx.Process(
            target=core.run_fit, daemon=True,
            args=(self._queue, self.engine_combo.currentData(), self.x, params))
        self._proc.start()
        self._run_started = time.monotonic()
        self.progress.setRange(0, max(1, params["kmax"] - 1))
        self.progress.setValue(0)
        self.eta_label.setText("starting…")
        self._set_running(True)
        self._poll_timer.start()
        self._status(f"Running {self.engine_combo.currentText()}…")

    def cancel_gsbs(self) -> None:
        self._stop_child()
        self._set_running(False)
        self._init_plots()
        self._status("Fit cancelled.")

    def _stop_child(self) -> None:
        self._poll_timer.stop()
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(2)
        if self._queue is not None:
            self._queue.cancel_join_thread()
            self._queue.close()
        self._proc = self._queue = None

    def _drain(self) -> tuple[tuple | None, tuple | None]:
        """Read everything queued: (latest progress message, final message)."""
        progress = final = None
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue_mod.Empty:
                return progress, final
            if msg[0] == "progress":
                progress = msg
            else:
                return progress, msg

    def _poll_fit(self) -> None:
        if self._proc is None:
            return
        progress, final = self._drain()
        if final is None and not self._proc.is_alive():
            # The child flushes its queue before exiting, so read once more.
            progress2, final = self._drain()
            progress = progress2 or progress
            if final is None:
                code = self._proc.exitcode
                hint = " (possibly out of memory)" if code in (-9, 137) else ""
                final = ("error", f"The fit process ended unexpectedly, "
                                  f"exit code {code}{hint}.")
        if progress is not None:
            self._show_progress(*progress[1:])
        if final is not None:
            kind, payload = final
            self._stop_child()
            self._set_running(False)
            if kind == "done":
                self._fit_done(payload)
            else:
                self._fit_error(payload)

    def _show_progress(self, done, total, ks, tvals, bounds_row) -> None:
        self.progress.setValue(min(done, self.progress.maximum()))
        elapsed = time.monotonic() - self._run_started
        text = f"{_fmt_time(elapsed)} elapsed"
        if done >= 3 and elapsed > 1:
            remaining = elapsed / done * max(0, total - done)
            text += f" · ≈ {_fmt_time(remaining)} left"
        self.eta_label.setText(text)
        n_states = len(core.boundary_indices(bounds_row)) + 1
        self._draw_solution(bounds_row, ks, tvals, k=n_states, best=None, live=True)

    def _fit_done(self, obj) -> None:
        elapsed = time.monotonic() - self._run_started
        if self.meta:
            obj.gsbs_gui_meta = dict(self.meta) | {"fit_seconds": elapsed}
        if self.session is not None and self.target is not None:
            try:
                self.session.record(self.target, obj, elapsed, type(obj).__name__)
                self.session_panel.update_row(self.target)
                self.session_panel.update_estimate()
                self.brain.refresh_roi_map()
            except Exception as exc:
                QMessageBox.warning(self, "Session", f"Could not save the result: {exc}")
        self._set_result(obj)
        self._status(f"{type(obj).__name__} done in {_fmt_time(elapsed)}  "
                     f"kmax={obj.kmax}  best={core.best_k(obj)} states")

    def _fit_error(self, msg: str) -> None:
        self._init_plots()
        box = QMessageBox(QMessageBox.Icon.Critical, "GSBS error",
                          msg.strip().splitlines()[-1], parent=self)
        box.setDetailedText(msg)
        box.exec()
        self._status("GSBS failed.")

    def _set_running(self, running: bool) -> None:
        self.run_btn.setText("Cancel" if running else "Run GSBS")
        self.run_btn.setObjectName("cancel_btn" if running else "run_btn")
        self.run_btn.style().unpolish(self.run_btn)
        self.run_btn.style().polish(self.run_btn)
        self.run_row.setVisible(running)
        for w in (self.open_btn, self.path_edit, self.demo_btn, self.engine_combo,
                  self.kmax_spin, self.finetune_spin, self.statewise_cb, self.adv_row,
                  self.import_btn, self.columns_combo, self.session_panel.table):
            w.setEnabled(not running)
        self._configure_explorer()

    # ------------------------------------------------------------------
    # Solution explorer
    # ------------------------------------------------------------------

    def _set_result(self, obj) -> None:
        self.gsbs_object = obj
        self._ks = core.solution_ks(obj)
        self._configure_explorer()
        self.show_k(core.best_k(obj))

    def _configure_explorer(self) -> None:
        has = self.gsbs_object is not None and len(self._ks) > 0 and self._proc is None
        for w in (self.k_slider, self.k_spinbox, self.best_btn, self.save_btn, self.cluster_btn):
            w.setEnabled(has)
        for w in (self.k_slider, self.k_spinbox):
            w.blockSignals(True)
        if has:
            self.k_slider.setRange(0, len(self._ks) - 1)
            self.k_spinbox.setRange(int(self._ks[0]), int(self._ks[-1]))
            limit = core.best_at_limit(self.gsbs_object)
            self.best_k_label.setText(f"best: {core.best_k(self.gsbs_object)}"
                                      + (" \u26a0" if limit else ""))
            self.best_k_label.setStyleSheet(f"color: {_ORANGE};" if limit else "")
            self.best_k_label.setToolTip(
                "The t-distance was still rising at the largest k searched: no optimum was "
                "found, so this 'best' is set by kmax. Refit with a larger kmax, or treat the "
                "data as having no clear state structure." if limit else "")
        else:
            self.k_slider.setRange(0, 0)
            self.k_spinbox.setRange(0, 0)
            self.best_k_label.setText("best: —")
            self.duration_label.setText("")
        for w in (self.k_slider, self.k_spinbox):
            w.blockSignals(False)

    def show_k(self, k: int) -> None:
        """Display the solution with k states (k must be in self._ks)."""
        idx = int(np.searchsorted(self._ks, k))
        self._k = int(self._ks[idx])
        for w, v in ((self.k_slider, idx), (self.k_spinbox, self._k)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        obj = self.gsbs_object
        row = obj.all_bounds[self._k]
        edges = core.state_edges(row, self.x.shape[0])
        if self._edges is not None and self._sel < len(self._edges) - 1:
            mid = (self._edges[self._sel] + self._edges[self._sel + 1]) // 2
            self._sel = int(np.searchsorted(edges, mid, side="right") - 1)
        else:
            self._sel = 0
        self._sel = int(np.clip(self._sel, 0, len(edges) - 2))
        self._edges = edges
        self.duration_label.setText(self._duration_text(edges))
        self.stimulus.set_solution(edges, self._k)
        self._draw_solution(row, self._ks, obj.tdists[self._ks],
                            k=self._k, best=core.best_k(obj), live=False)
        self.brain.set_solution(row, self._k, obj)
        self.brain.set_selected_state(self._sel)
        self.stimulus.set_selected(self._sel)
        self._update_cluster_window()

    def _duration_text(self, edges: np.ndarray) -> str:
        """'median state 6 s (4 TRs), 2–40 s' for the solution with these state edges."""
        lengths = np.diff(edges)
        tr = self.meta.get("tr") if self.meta else None
        med = float(np.median(lengths))
        if tr:
            return (f"median state {med * tr:.3g} s ({med:g} TRs), "
                    f"{lengths.min() * tr:.3g}\u2013{lengths.max() * tr:.3g} s")
        return f"median state {med:g} TRs, {lengths.min()}\u2013{lengths.max()} TRs"

    def _select_state(self, i: int) -> None:
        if self._edges is None:
            return
        self._sel = int(np.clip(i, 0, len(self._edges) - 2))
        self._draw_selection()
        self.canvas.draw_idle()
        self.brain.set_selected_state(self._sel)
        self.stimulus.set_selected(self._sel)

    def go_to_best(self) -> None:
        if self.gsbs_object is not None and self._proc is None:
            self.show_k(core.best_k(self.gsbs_object))

    def _on_slider(self, idx: int) -> None:
        if self.gsbs_object is not None:
            self.show_k(int(self._ks[idx]))

    def _on_spinbox(self, k: int) -> None:
        """Snap to the nearest solution in the direction the user moved."""
        if self.gsbs_object is None:
            return
        ks = self._ks
        if k > self._k:
            candidates = ks[ks >= k]
            target = candidates[0] if len(candidates) else ks[-1]
        elif k < self._k:
            candidates = ks[ks <= k]
            target = candidates[-1] if len(candidates) else ks[0]
        else:
            target = k
        self.show_k(int(target))

    def _on_canvas_click(self, event) -> None:
        """Click the t-distance curve to pick a solution, or a timepoint to select its state."""
        if (event.button != 1 or event.xdata is None or self.gsbs_object is None
                or self._proc is not None or str(self.toolbar.mode)):
            return
        if event.inaxes is self.ax_tdist:
            self.show_k(int(self._ks[np.argmin(np.abs(self._ks - event.xdata))]))
        elif event.inaxes in (self.ax_raw, self.ax_state, self.ax_corrmat) and self._edges is not None:
            tr = int(round(event.xdata))
            self._select_state(int(np.searchsorted(self._edges, tr, side="right") - 1))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    #
    # Artists are created once per dataset (_init_plots) and only their data is
    # updated per solution (_draw_solution), so stepping through k is cheap.
    # Pixel i covers [i - 0.5, i + 0.5], so a state starting at TR b has its
    # edge drawn at b - 0.5.

    def _reset_axes(self) -> None:
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        for ax in (self.ax_corrmat, self.ax_tdist, self.ax_raw, self.ax_state):
            ax.clear()
            ax.set_facecolor(_FIG_BG)
        self.ax_corrmat.set_title("Time × time correlation", fontsize=9)
        self.ax_corrmat.set_xlabel("Timepoints")
        self.ax_corrmat.set_ylabel("Timepoints")
        self.ax_tdist.set_title("T-distance", fontsize=9)
        self.ax_tdist.set_xlabel("Number of states")
        self.ax_tdist.set_ylabel("t")
        self.ax_raw.set_title("Voxel timeseries", fontsize=9)
        self.ax_raw.set_ylabel(self._col_noun().capitalize())
        self.ax_state.set_title("State mean patterns", fontsize=9)
        self.ax_state.set_ylabel(self._col_noun().capitalize())
        self.ax_state.set_xlabel("Timepoints")
        self._artists = {}

    def _init_plots(self) -> None:
        self._reset_axes()
        if self.x is None:
            self.canvas.draw_idle()
            return
        x = self.x
        n_time, n_vox = x.shape
        a = self._artists

        corr = np.corrcoef(x)
        im = self.ax_corrmat.imshow(corr, cmap=_CMAP, vmin=-1, vmax=1,
                                    interpolation="none", aspect="equal")
        self._cbar = self.fig.colorbar(im, ax=self.ax_corrmat, location="bottom",
                                       shrink=0.6, pad=0.02, label="r")
        a["boxes"] = LineCollection([], colors=_BOUND, linewidths=1.0)
        self.ax_corrmat.add_collection(a["boxes"])

        lim = float(np.percentile(np.abs(x - x.mean(0)), 99)) or 1.0
        kw = dict(cmap=_CMAP, vmin=-lim, vmax=lim, interpolation="none",
                  aspect="auto", origin="lower")
        self.ax_raw.imshow((x - x.mean(0)).T, **kw)
        a["state_im"] = self.ax_state.imshow(np.zeros((n_vox, n_time)), **kw)
        a["state_im"].set_visible(False)
        for name, ax in (("raw_lines", self.ax_raw), ("state_lines", self.ax_state)):
            a[name] = LineCollection([], colors=_BOUND, linewidths=0.9)
            ax.add_collection(a[name])

        if self.truth is not None:
            self.ax_raw.plot(self.truth - 0.5, np.full(len(self.truth), n_vox - 0.5),
                             "v", ms=5, color=_YELLOW, mec="black", mew=0.5,
                             clip_on=False, zorder=5)
            self.ax_tdist.axvline(len(self.truth) + 1, color=_YELLOW, ls=":", lw=1.2,
                                  label=f"true: {len(self.truth) + 1} states (▼ demo)")

        a["sel_box"] = LineCollection([], colors=_YELLOW, linewidths=2.0, zorder=6)
        self.ax_corrmat.add_collection(a["sel_box"])
        a["sel_spans"] = []

        (a["curve"],) = self.ax_tdist.plot([], [], color=_ACCENT, lw=1.3, marker=".", ms=3)
        a["current"] = self.ax_tdist.axvline(0, color=_RED, ls="--", lw=1.2, visible=False)
        (a["best"],) = self.ax_tdist.plot([], [], "*", color=_GREEN, ms=11, mec="black",
                                          mew=0.5, zorder=4)
        self.ax_tdist.set_xlim(0, self.kmax_spin.value() + 2)
        self.canvas.draw_idle()

    def _draw_solution(self, bounds_row, ks, tvals, k: int, best: int | None,
                       live: bool) -> None:
        if not self._artists:
            return
        a = self._artists
        x = self.x
        n_time, n_vox = x.shape
        edges = core.state_edges(bounds_row, n_time) - 0.5

        a["boxes"].set_segments([
            [(e0, e0), (e1, e0), (e1, e1), (e0, e1), (e0, e0)]
            for e0, e1 in zip(edges[:-1], edges[1:])])
        inner = edges[1:-1]
        segs = [[(b, -0.5), (b, n_vox - 0.5)] for b in inner]
        a["raw_lines"].set_segments(segs)
        a["state_lines"].set_segments(segs)
        st = core.state_timeseries(x, bounds_row)
        a["state_im"].set_data((st - x.mean(0)).T)
        a["state_im"].set_visible(True)

        a["curve"].set_data(ks, tvals)
        a["current"].set_xdata([k, k])
        a["current"].set_visible(True)
        a["current"].set_label(f"current: {k} states" if live else f"shown: {k} states")
        if best is not None:
            a["best"].set_data([best], [self.gsbs_object.tdists[best]])
            a["best"].set_label(f"best: {best} states")
        else:
            a["best"].set_data([], [])
            a["best"].set_label("_")
        if len(ks):
            self.ax_tdist.set_xlim(0, max(self.kmax_spin.value(), int(ks[-1])) + 2)
            lo, hi = float(np.min(tvals)), float(np.max(tvals))
            pad = 0.08 * (hi - lo or 1.0)
            self.ax_tdist.set_ylim(lo - pad, hi + pad)
        self.ax_tdist.legend(fontsize=7, loc="lower right")

        prefix = "Fitting… " if live else ""
        noun = "Parcel" if self._is_parcels() else "Voxel"
        self.ax_raw.set_title(f"{noun} timeseries · {prefix}{k} states", fontsize=9)
        if live:
            self._clear_selection()
        else:
            self._draw_selection()
        self.canvas.draw_idle()

    def _clear_selection(self) -> None:
        a = self._artists
        for span in a.get("sel_spans", []):
            span.remove()
        a["sel_spans"] = []
        if "sel_box" in a:
            a["sel_box"].set_segments([])

    def _draw_selection(self) -> None:
        """Highlight the selected state on the timeseries and the correlation matrix."""
        self._clear_selection()
        a = self._artists
        if not a or self._edges is None:
            return
        e0, e1 = self._edges[self._sel] - 0.5, self._edges[self._sel + 1] - 0.5
        a["sel_box"].set_segments([[(e0, e0), (e1, e0), (e1, e1), (e0, e1), (e0, e0)]])
        for ax in (self.ax_raw, self.ax_state):
            a["sel_spans"].append(ax.axvspan(e0, e1, facecolor=_YELLOW, alpha=0.22,
                                             edgecolor=_YELLOW, lw=1.2, zorder=4))

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._proc is not None:
            reply = QMessageBox.question(
                self, "GSBS running",
                "GSBS is still running. Closing will cancel the fit.\n\nClose anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_child()
        if not self.session_panel.cancel_batch(ask=True):
            event.ignore()
            return
        for t in (getattr(self.brain, "_thread", None), getattr(self, "_fetch_thread", None)):
            if t is not None and t.isRunning():          # surface preparation / example download
                if t is getattr(self, "_fetch_thread", None):
                    self._fetcher.stop = True
                t.quit()
                t.wait(10000)
        plt.close("all")
        event.accept()

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)


def _fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Dark palette + stylesheet
# ---------------------------------------------------------------------------

def _setup_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")

    p = QPalette()
    dark     = QColor("#1e1e2e")
    mid      = QColor("#313244")
    text     = QColor(_TEXT)
    accent   = QColor(_ACCENT)
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
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(p)

    app.setStyleSheet(f"""
        QPushButton#run_btn {{
            background-color: {_GREEN}; color: #1e1e2e;
            font-weight: bold; padding: 4px 12px;
        }}
        QPushButton#run_btn:hover    {{ background-color: #94e2d5; }}
        QPushButton#run_btn:disabled {{ background-color: {disabled.name()}; }}
        QPushButton#cancel_btn {{
            background-color: {_RED}; color: #1e1e2e;
            font-weight: bold; padding: 4px 12px;
        }}
        QLabel#dim_label {{ color: #a6adc8; }}
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
    window.resize(1200, 820)
    window.show()
    if len(sys.argv) > 1:
        window.path_edit.setText(sys.argv[1])
        QTimer.singleShot(200, window.load_file)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
