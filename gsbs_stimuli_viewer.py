#!/usr/bin/env python3
"""
GSBS Stimuli Boundary Viewer

Standalone popup for inspecting stimulus frames around each state boundary.

Usage (from another script):
    from gsbs_stimuli_viewer import StimuliViewer
    dlg = StimuliViewer(boundaries=[47, 112, 203], stim_paths=[...], parent=self)
    dlg.exec()

Or run directly:
    python gsbs_stimuli_viewer.py /path/to/stimuli_folder /path/to/gsbs_object.npy [k]
"""

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_IMG_W   = 160
_IMG_H   = 120
_FIG_BG  = "#1e1e2e"
_RED     = "#f38ba8"
_TEXT    = "#cdd6f4"
_MID     = "#313244"
_ACCENT  = "#89b4fa"
_ORANGE  = "#fab387"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_stim_paths(folder: str) -> list[str]:
    """Return sorted list of image paths from *folder*."""
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )


# ---------------------------------------------------------------------------
# Thumbnail strip widget
# ---------------------------------------------------------------------------

class _ThumbnailStrip(QWidget):
    """Horizontal row of labelled image thumbnails inside a scroll area."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setSpacing(6)
        self._row.setContentsMargins(4, 4, 4, 4)

    def update_frames(self, boundary_tp: int, stim_paths: list[str],
                      n_context: int) -> None:
        # Clear existing widgets
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        n_total = len(stim_paths)
        for offset in range(-n_context, n_context + 1):
            tp = boundary_tp + offset
            is_boundary = offset == 0

            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            if is_boundary:
                frame.setStyleSheet(
                    f"QFrame {{ border: 2px solid {_RED}; border-radius: 4px; "
                    f"background: {_MID}; }}")
            else:
                frame.setStyleSheet(
                    f"QFrame {{ border: 1px solid #45475a; border-radius: 4px; "
                    f"background: {_MID}; }}")

            fvbox = QVBoxLayout(frame)
            fvbox.setSpacing(3)
            fvbox.setContentsMargins(4, 4, 4, 4)

            img_label = QLabel()
            img_label.setFixedSize(_IMG_W, _IMG_H)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("border: none;")

            if 0 <= tp < n_total:
                pix = QPixmap(stim_paths[tp])
                if not pix.isNull():
                    pix = pix.scaled(
                        _IMG_W, _IMG_H,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    img_label.setPixmap(pix)
                else:
                    img_label.setText("(load error)")
                    img_label.setStyleSheet("border: none; color: #f38ba8;")
            else:
                img_label.setText("—")
                img_label.setStyleSheet("border: none; color: #585b70;")

            tp_text = f"t = {tp}"
            if is_boundary:
                tp_text += "  ◄"
            tp_label = QLabel(tp_text)
            tp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tp_label.setStyleSheet(
                f"border: none; color: {_RED}; font-weight: bold;"
                if is_boundary else f"border: none; color: {_TEXT};"
            )

            fvbox.addWidget(img_label)
            fvbox.addWidget(tp_label)
            self._row.addWidget(frame)

        self._row.addStretch()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class StimuliViewer(QDialog):
    """
    Pop-up viewer showing stimulus frames around each state boundary.

    Parameters
    ----------
    boundaries : list[int]
        Timepoint indices of boundaries (as returned by ``np.where``).
    stim_paths : list[str]
        Sorted list of image file paths; one per timepoint.
    initial_idx : int
        Which boundary to show first (0-based index into *boundaries*).
    parent : QWidget, optional
    """

    def __init__(self, boundaries: list[int], stim_paths: list[str],
                 initial_idx: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stimulus Boundary Viewer")
        self.resize(900, 360)

        self.boundaries  = boundaries
        self.stim_paths  = stim_paths
        self._cur_idx    = -1          # forces first draw

        self._build_ui()
        self._goto(initial_idx)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setSpacing(6)
        vbox.setContentsMargins(8, 8, 8, 8)

        # ── top bar: header + context spinbox ───────────────────────────
        top = QHBoxLayout()
        self.header_label = QLabel()
        self.header_label.setStyleSheet(f"font-weight: bold; color: {_TEXT};")
        top.addWidget(self.header_label, stretch=1)

        top.addWidget(QLabel("Context frames:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(1, 10)
        self.ctx_spin.setValue(3)
        self.ctx_spin.setToolTip("Number of frames to show before and after the boundary")
        self.ctx_spin.valueChanged.connect(self._on_ctx_changed)
        top.addWidget(self.ctx_spin)
        vbox.addLayout(top)

        # ── scrollable thumbnail strip ───────────────────────────────────
        self._strip = _ThumbnailStrip()
        scroll = QScrollArea()
        scroll.setWidget(self._strip)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(_IMG_H + 60)   # image + label + padding
        vbox.addWidget(scroll)

        # ── navigation bar ───────────────────────────────────────────────
        nav = QHBoxLayout()

        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self._prev)
        nav.addWidget(self.prev_btn)

        self.b_slider = QSlider(Qt.Orientation.Horizontal)
        self.b_slider.setRange(0, max(0, len(self.boundaries) - 1))
        self.b_slider.valueChanged.connect(self._on_b_slider)
        nav.addWidget(self.b_slider, stretch=1)

        self.b_spinbox = QSpinBox()
        self.b_spinbox.setRange(1, len(self.boundaries))
        self.b_spinbox.setFixedWidth(65)
        self.b_spinbox.setToolTip("Boundary number")
        self.b_spinbox.valueChanged.connect(self._on_b_spinbox)
        nav.addWidget(self.b_spinbox)

        nav.addWidget(QLabel(f"/ {len(self.boundaries)}"))

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._next)
        nav.addWidget(self.next_btn)

        vbox.addLayout(nav)

        # ── close button ─────────────────────────────────────────────────
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setFixedWidth(100)
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(close_btn)
        vbox.addLayout(close_row)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _goto(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.boundaries) - 1))
        if idx == self._cur_idx:
            return
        self._cur_idx = idx
        b = self.boundaries[idx]

        self.header_label.setText(
            f"Boundary {idx + 1} of {len(self.boundaries)}  —  timepoint {b}")

        for w in (self.b_slider, self.b_spinbox):
            w.blockSignals(True)
        self.b_slider.setValue(idx)
        self.b_spinbox.setValue(idx + 1)
        for w in (self.b_slider, self.b_spinbox):
            w.blockSignals(False)

        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self.boundaries) - 1)

        self._strip.update_frames(b, self.stim_paths, self.ctx_spin.value())

    def _prev(self)  -> None: self._goto(self._cur_idx - 1)
    def _next(self)  -> None: self._goto(self._cur_idx + 1)

    def _on_b_slider(self, value: int) -> None:
        self.b_spinbox.blockSignals(True)
        self.b_spinbox.setValue(value + 1)
        self.b_spinbox.blockSignals(False)
        self._goto(value)

    def _on_b_spinbox(self, value: int) -> None:
        self.b_slider.blockSignals(True)
        self.b_slider.setValue(value - 1)
        self.b_slider.blockSignals(False)
        self._goto(value - 1)

    def _on_ctx_changed(self) -> None:
        if self._cur_idx >= 0:
            b = self.boundaries[self._cur_idx]
            self._strip.update_frames(b, self.stim_paths, self.ctx_spin.value())


# ---------------------------------------------------------------------------
# Dark theme (mirrors gsbs_gui.py)
# ---------------------------------------------------------------------------

def _setup_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    dark     = QColor(_FIG_BG)
    mid      = QColor(_MID)
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
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       disabled)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    app.setPalette(p)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Usage:
        python gsbs_stimuli_viewer.py <stimuli_folder> [gsbs_object.npy] [k]
    """
    app = QApplication(sys.argv)
    _setup_dark_theme(app)

    args = sys.argv[1:]

    # -- stimuli folder --
    if args:
        folder = args[0]
    else:
        folder = QFileDialog.getExistingDirectory(None, "Select stimuli folder")
        if not folder:
            sys.exit(0)

    stim_paths = load_stim_paths(folder)
    if not stim_paths:
        QMessageBox.critical(None, "No images",
                             f"No image files found in:\n{folder}")
        sys.exit(1)

    # -- GSBS object --
    boundaries = None
    if len(args) >= 2:
        gsbs_path = args[1]
        try:
            raw = np.load(gsbs_path, allow_pickle=True)
            obj = raw.item()
            k = int(args[2]) if len(args) >= 3 else int(np.argmax(obj.tdists))
            boundaries = list(np.where(obj.all_bounds[k] > 0)[0])
        except Exception as exc:
            QMessageBox.critical(None, "Load error", str(exc))
            sys.exit(1)

    if not boundaries:
        # Fallback: evenly spaced demo boundaries
        n = len(stim_paths)
        boundaries = list(range(n // 5, n, n // 5))

    dlg = StimuliViewer(boundaries, stim_paths)
    dlg.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
