"""
Brain panel: parcel-level state maps on the fsaverage surface.

Tabs
    State     the selected state's mean pattern, with network means
    Boundary  the states either side of a boundary and the change between them
    Gallery   every state of the current solution, rendered progressively
    ROI map   one statistic per ROI from the session's fits; click to open an ROI

Maps are drawn with gsbs_surface (pre-rasterised views, ~40 ms per map), only
for the visible tab, and cached. The surface views are prepared once in a
background thread, so the window never waits for them.
"""

from __future__ import annotations

from collections import OrderedDict

import matplotlib
import numpy as np
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

import gsbs_atlas as atlas
import gsbs_core as core
import gsbs_overlap as ov
import gsbs_surface as surface

_TEXT = "#cdd6f4"
_DIM = "#a6adc8"
_SEL = QColor("#f9e2af")
_STATE_CMAP = "RdBu_r"
_CHANGE_CMAP = "PuOr_r"
_METRIC_CMAP = "viridis"
_CACHE_SIZE = 64
_TILE_W, _TILE_H = 190, 135


def _cmap(name):
    return matplotlib.colormaps[name]


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------

class BrainImage(QWidget):
    """Shows one rendered strip, scaled to fit; reports clicks and hovers in image pixels."""

    clicked = pyqtSignal(int, int)
    double_clicked = pyqtSignal(int, int)
    hovered = pyqtSignal(int, int, QPointF)

    def __init__(self, min_height: int = 70, parent=None):
        super().__init__(parent)
        self._arr = None
        self._img = None
        self._placeholder = ""
        self.selected = False
        self.grid = False            # True when showing the 2x2 layout instead of the strip
        self.setMouseTracking(True)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_rgba(self, arr: np.ndarray | None) -> None:
        if arr is None:
            self._arr = self._img = None
        else:
            self._arr = np.ascontiguousarray(arr)
            h, w = self._arr.shape[:2]
            self._img = QImage(self._arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.set_rgba(None)

    def _rect(self) -> QRectF | None:
        if self._img is None:
            return None
        w, h = self._img.width(), self._img.height()
        s = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)

    def _to_image(self, pos) -> tuple[int, int] | None:
        r = self._rect()
        if r is None or not r.contains(pos):
            return None
        s = r.width() / self._img.width()
        return int((pos.x() - r.x()) / s), int((pos.y() - r.y()) / s)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        r = self._rect()
        if r is None:
            p.setPen(QColor(_DIM))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
        else:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(r, self._img)
        if self.selected:
            p.setPen(QPen(_SEL, 2))
            p.drawRect(self.rect().adjusted(1, 1, -1, -1))
        p.end()

    def mousePressEvent(self, e) -> None:
        xy = self._to_image(e.position())
        if xy and e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(*xy)

    def mouseDoubleClickEvent(self, e) -> None:
        xy = self._to_image(e.position())
        if xy:
            self.double_clicked.emit(*xy)

    def mouseMoveEvent(self, e) -> None:
        xy = self._to_image(e.position())
        if xy:
            self.hovered.emit(xy[0], xy[1], e.globalPosition())
        else:
            QToolTip.hideText()


class ColourBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._cmap, self._lo, self._hi, self._label = None, 0.0, 1.0, ""

    def set_scale(self, cmap: str, lo: float, hi: float, label: str = "") -> None:
        self._cmap, self._lo, self._hi, self._label = cmap, lo, hi, label
        self.update()

    def paintEvent(self, _e) -> None:
        if self._cmap is None:
            return
        p = QPainter(self)
        w = min(260, self.width() - 20)
        x0 = (self.width() - w) / 2
        grad = QLinearGradient(x0, 0, x0 + w, 0)
        cm = _cmap(self._cmap)
        for t in np.linspace(0, 1, 11):
            grad.setColorAt(t, QColor.fromRgbF(*cm(t)[:3]))
        p.fillRect(QRectF(x0, 2, w, 9), grad)
        p.setPen(QColor(_DIM))
        f = QFont(p.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        p.drawText(QRectF(x0 - 40, 13, 80, 14), Qt.AlignmentFlag.AlignCenter, f"{self._lo:.3g}")
        p.drawText(QRectF(x0 + w - 40, 13, 80, 14), Qt.AlignmentFlag.AlignCenter, f"{self._hi:.3g}")
        p.drawText(QRectF(x0, 13, w, 14), Qt.AlignmentFlag.AlignCenter, self._label)
        p.end()


class NetworkBars(QWidget):
    """Mean value per Yeo network, as horizontal bars around zero (or from a minimum).

    `counts` adds a short text per network (e.g. "2/9"); `sequential=(lo, hi)`
    draws bars from the left edge for quantities without a natural zero.
    """

    def __init__(self, parent=None, width: int = 150):
        super().__init__(parent)
        self.setFixedWidth(width)
        self.setMinimumHeight(18 + 17 * 9)          # 17 networks at the smallest row height
        self._rows: list = []
        self._lim, self._cmap, self._title = 1.0, _STATE_CMAP, ""
        self._counts: dict = {}
        self._seq = None

    def set_values(self, info: atlas.ParcelInfo | None, values: np.ndarray | None,
                   lim: float, cmap: str, title: str = "Network means",
                   counts: dict | None = None, sequential: tuple | None = None) -> None:
        self._rows = []
        if info is not None and values is not None:
            for net in info.network_order:
                v = values[info.networks == net]
                v = v[~np.isnan(v)]
                self._rows.append((net, float(v.mean()) if len(v) else np.nan,
                                   info.network_colours[net]))
        self._lim, self._cmap, self._title = lim or 1.0, cmap, title
        self._counts = counts or {}
        self._seq = sequential
        self.update()

    def paintEvent(self, _e) -> None:
        if not self._rows:
            return
        p = QPainter(self)
        f = QFont(p.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        p.setPen(QColor(_DIM))
        p.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self._title)
        top, n = 18, len(self._rows)
        row_h = max(9.0, min(14.0, (self.height() - top - 2) / n))
        f.setPixelSize(max(7, min(11, int(row_h - 1))))     # labels shrink with the rows
        p.setFont(f)
        name_w, sw = 66, 6
        count_w = 30 if self._counts else 0
        bar_x0 = name_w + sw + 4
        bar_w = self.width() - bar_x0 - 4 - count_w
        mid = bar_x0 + bar_w / 2 if self._seq is None else bar_x0
        cm = _cmap(self._cmap)
        for i, (net, val, col) in enumerate(self._rows):
            y = top + i * row_h
            p.fillRect(QRectF(0, y + 2, sw, row_h - 4), QColor.fromRgbF(*col))
            p.setPen(QColor(_TEXT))
            p.drawText(QRectF(sw + 3, y, name_w, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, net)
            if not np.isnan(val):
                if self._seq is None:
                    frac = float(np.clip(val / self._lim, -1, 1))
                    p.fillRect(QRectF(mid, y + 2, frac * bar_w / 2, row_h - 4),
                               QColor.fromRgbF(*cm(0.5 + 0.5 * frac)[:3]))
                else:
                    lo, hi = self._seq
                    frac = float(np.clip((val - lo) / ((hi - lo) or 1.0), 0, 1))
                    p.fillRect(QRectF(bar_x0, y + 2, max(1.0, frac * bar_w), row_h - 4),
                               QColor.fromRgbF(*cm(frac)[:3]))
            if net in self._counts:
                p.setPen(QColor(_DIM))
                p.drawText(QRectF(self.width() - count_w - 2, y, count_w, row_h),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           self._counts[net])
        p.setPen(QPen(QColor("#585b70"), 1))
        p.drawLine(QPointF(mid, top), QPointF(mid, top + n * row_h))
        p.end()


# ---------------------------------------------------------------------------
# Surface preparation (background thread)
# ---------------------------------------------------------------------------

class _Prepare(QObject):
    ready = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def run(self) -> None:
        try:
            big = surface.SurfaceStrip.load(height=200)
            small = surface.SurfaceStrip.load(height=90)
            big.parcels(self.spec)
            small.parcels(self.spec)
            atlas.parcel_info(self.spec)
            self.ready.emit(big, small)
        except Exception as exc:                             # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

class _OverlapWorker(QObject):
    """Overlap of every ROI's best boundaries with one reference at each delay
    (gsbs_analysis.roi_overlap_map, the same code the exported scripts run)."""
    progress = pyqtSignal(int, int)
    done = pyqtSignal(object)

    def __init__(self, session, refs, delays, n_perm):
        super().__init__()
        self.session, self.refs, self.delays, self.n_perm = session, refs, delays, n_perm

    def run(self):
        import gsbs_analysis as A
        self.done.emit(A.roi_overlap_map(self.session, self.refs, self.delays, self.n_perm,
                                         progress=self.progress.emit))


class BrainPanel(QTabWidget):
    stateSelected = pyqtSignal(int)      # 0-based state index in the current solution
    roiSelected = pyqtSignal(int)        # 1-based parcel id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spec: atlas.Schaefer | None = None
        self.info: atlas.ParcelInfo | None = None
        self._big = self._small = None
        self._preparing = None
        self._thread = None

        self.x = None                    # (T, P) parcel data of the current dataset, or None
        self.tr = None
        self.missing = np.array([], int)
        self.edges = None                # state edges of the current solution
        self.patterns = None             # (n_states, P)
        self.lim = 1.0
        self.strengths = None
        self.k = 0
        self.sel = 0

        self.session = None
        self.current_roi = None
        self._extra: dict = {}           # key -> (values per parcel, unit)
        self.overlap_sources = None      # callable -> list of reference dicts (set by the main window)
        self._ovl = None                 # last per-ROI overlap result
        self._ovl_thread = None

        self._cache: OrderedDict = OrderedDict()
        self._render_pending = False
        self._gallery_next = 0
        self._gallery_timer = QTimer(self)
        self._gallery_timer.setInterval(0)
        self._gallery_timer.timeout.connect(self._gallery_step)

        self._build_state_tab()
        self._build_boundary_tab()
        self._build_gallery_tab()
        self._build_roi_tab()
        self._build_overlap_tab()
        self.currentChanged.connect(lambda _: self._schedule())
        self._update_tabs()

    # ---- construction -------------------------------------------------------

    @staticmethod
    def _header() -> QLabel:
        lab = QLabel("")
        lab.setStyleSheet(f"color: {_TEXT};")
        return lab

    def _build_state_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 4, 6, 2)
        self.state_header = self._header()
        v.addWidget(self.state_header)
        row = QHBoxLayout()
        self.state_img = BrainImage(120)
        self._wire_image(self.state_img, lambda: self._state_values())
        row.addWidget(self.state_img, stretch=1)
        self.state_bars = NetworkBars()
        row.addWidget(self.state_bars)
        v.addLayout(row, stretch=1)
        self.state_cbar = ColourBar()
        v.addWidget(self.state_cbar)
        self.addTab(w, "State")

    def _build_boundary_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 4, 6, 2)
        head = QHBoxLayout()
        self.prev_btn = QToolButton(text="◀")
        self.next_btn = QToolButton(text="▶")
        self.prev_btn.clicked.connect(lambda: self._step_boundary(-1))
        self.next_btn.clicked.connect(lambda: self._step_boundary(+1))
        head.addWidget(self.prev_btn)
        head.addWidget(self.next_btn)
        self.boundary_header = self._header()
        head.addWidget(self.boundary_header, stretch=1)
        v.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(0)
        self.bnd_imgs, self.bnd_labels = [], []
        for i, text in enumerate(("Before", "After", "Change")):
            lab = QLabel(text)
            lab.setStyleSheet(f"color: {_DIM};")
            lab.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            img = BrainImage(120)
            img.grid = True
            self._wire_image(img, (lambda i=i: self._boundary_values()[i]))
            grid.addWidget(lab, 0, i)
            grid.addWidget(img, 1, i)
            self.bnd_imgs.append(img)
            self.bnd_labels.append(lab)
        self.bnd_bars = NetworkBars()
        grid.addWidget(self.bnd_bars, 0, 3, 3, 1)
        self.bnd_cbar = ColourBar()
        self.chg_cbar = ColourBar()
        grid.addWidget(self.bnd_cbar, 2, 0, 1, 2)
        grid.addWidget(self.chg_cbar, 2, 2)
        grid.setRowStretch(1, 1)
        v.addLayout(grid, stretch=1)
        self.addTab(w, "Boundary")

    def _build_gallery_tab(self) -> None:
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._gallery_host = QWidget()
        self.gallery_grid = QGridLayout(self._gallery_host)
        self.gallery_grid.setSpacing(6)
        self.gallery_scroll.setWidget(self._gallery_host)
        self.gallery_scroll.viewport().installEventFilter(self)
        self._tiles: list[BrainImage] = []
        self._tile_boxes: list[QWidget] = []
        self._gallery_cols = 0
        self.addTab(self.gallery_scroll, "Gallery")

    def _build_roi_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 4, 6, 2)
        head = QHBoxLayout()
        head.addWidget(QLabel("Show:"))
        self.metric_combo = QComboBox()
        for label, key in (("Number of states (best)", "nstates"),
                           ("Median state duration", "median_duration"),
                           ("Voxels per ROI", "voxels"),
                           ("Fit time", "fit_seconds")):
            self.metric_combo.addItem(label, key)
        self.metric_combo.currentIndexChanged.connect(lambda _: self._invalidate("roi"))
        head.addWidget(self.metric_combo)
        self.roi_header = self._header()
        head.addWidget(self.roi_header, stretch=1)
        v.addLayout(head)
        self.roi_img = BrainImage(120)
        self.roi_img.clicked.connect(self._roi_clicked)
        self.roi_img.hovered.connect(self._roi_hover)
        v.addWidget(self.roi_img, stretch=1)
        self.roi_cbar = ColourBar()
        v.addWidget(self.roi_cbar)
        self.addTab(w, "ROI map")

    def _wire_image(self, img: BrainImage, values_fn) -> None:
        img.hovered.connect(lambda x, y, g: self._hover(img, x, y, g, values_fn))
        img.double_clicked.connect(lambda x, y: self._open_parcel_at(img, x, y))

    # ---- public API -----------------------------------------------------------

    def set_atlas(self, spec: atlas.Schaefer | None) -> None:
        if spec == self.spec:
            return
        self.spec = spec
        self.info = None
        self._cache.clear()
        if spec is not None:
            self._prepare(spec)
        self._update_tabs()

    def set_parcel_data(self, x: np.ndarray | None, tr: float | None,
                        missing=()) -> None:
        """Parcel-level data of the shown dataset (None for voxel-level data)."""
        self.x = x
        self.tr = tr
        self.missing = np.asarray(missing, int)
        self.edges = self.patterns = None
        self._cache.clear()
        self._reset_gallery()
        self._update_tabs()

    def set_solution(self, bounds_row: np.ndarray | None, k: int, obj=None) -> None:
        if self.x is None or bounds_row is None:
            self.edges = self.patterns = None
            self._update_tabs()
            return
        n_time = self.x.shape[0]
        edges = core.state_edges(bounds_row, n_time)
        if self.edges is not None and np.array_equal(edges, self.edges):
            return
        old_mid = None
        if self.edges is not None and self.sel < len(self.edges) - 1:
            old_mid = (self.edges[self.sel] + self.edges[self.sel + 1]) // 2
        self.edges = edges
        self.k = k
        self.patterns = np.add.reduceat(self.x, edges[:-1], axis=0) / np.diff(edges)[:, None]
        if len(self.missing):
            self.patterns[:, self.missing - 1] = np.nan
        self.lim = float(np.nanpercentile(np.abs(self.patterns), 98)) or 1.0
        self.strengths = None
        if obj is not None:
            try:
                s = obj.get_strengths(k)
                self.strengths = s[s > 0]
            except Exception:                                # noqa: BLE001
                self.strengths = None
        self._cache = OrderedDict((key, v) for key, v in self._cache.items() if key[0] == "roi")
        self.sel = 0 if old_mid is None else self.state_at(old_mid)
        self._reset_gallery()
        self._update_tabs()

    def state_at(self, tr: int) -> int:
        return int(np.clip(np.searchsorted(self.edges, tr, side="right") - 1,
                           0, len(self.edges) - 2))

    def set_selected_state(self, i: int) -> None:
        if self.patterns is None:
            return
        i = int(np.clip(i, 0, len(self.patterns) - 1))
        if i != self.sel:
            if self._tiles:
                self._tiles[self.sel].selected = False
                self._tiles[self.sel].update()
            self.sel = i
            if self._tiles:
                self._tiles[i].selected = True
                self._tiles[i].update()
            self._schedule()

    def set_session(self, session) -> None:
        self.session = session
        self._ovl = None
        self.refresh_overlap_sources()
        self._invalidate("roi")
        self._update_tabs()

    def set_current_roi(self, roi: int | None) -> None:
        self.current_roi = roi
        self._invalidate("roi")

    def refresh_roi_map(self) -> None:
        self._invalidate("roi")

    @property
    def has_content(self) -> bool:
        return self.spec is not None and (self.x is not None or self.session is not None)

    # ---- preparation -------------------------------------------------------------

    def _prepare(self, spec) -> None:
        if self._big is not None:
            try:
                self._big.parcels(spec)
                self._small.parcels(spec)
                self.info = atlas.parcel_info(spec)
                self._schedule()
                return
            except Exception:                                # noqa: BLE001
                pass
        for img in self._all_images():
            img.set_placeholder("Preparing brain surface (first time only: downloads the "
                                "Schaefer annotation and draws the views)…")
        self._preparing = _Prepare(spec)
        self._thread = QThread(self)
        self._preparing.moveToThread(self._thread)
        self._thread.started.connect(self._preparing.run)
        self._preparing.ready.connect(self._prepared)
        self._preparing.failed.connect(self._prepare_failed)
        self._preparing.ready.connect(self._thread.quit)
        self._preparing.failed.connect(self._thread.quit)
        self._thread.start()

    def _prepared(self, big, small) -> None:
        self._big, self._small = big, small
        self.info = atlas.parcel_info(self.spec)
        self._preparing = None
        self._schedule()

    def _prepare_failed(self, msg: str) -> None:
        self._preparing = None
        for img in self._all_images():
            img.set_placeholder(f"Could not prepare the brain surface: {msg}")

    def _all_images(self):
        return [self.state_img, *self.bnd_imgs, self.roi_img, *self._tiles]

    @property
    def _ready(self) -> bool:
        return self._big is not None and self.info is not None and self.spec is not None

    # ---- rendering -------------------------------------------------------------

    def _invalidate(self, kind: str) -> None:
        for key in [k for k in self._cache if k[0] == kind]:
            del self._cache[key]
        self._schedule()

    def _schedule(self) -> None:
        if not self._render_pending:
            self._render_pending = True
            QTimer.singleShot(0, self._render_visible)

    def _render(self, key, strip, values, cmap, lim_lo, lim_hi, highlight=None):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = strip.parcels(self.spec).render(values, _cmap(cmap), lim_lo, lim_hi,
                                              highlight=highlight)
        self._cache[key] = img
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return img

    def _render_visible(self) -> None:
        self._render_pending = False
        if not self._ready:
            return
        tab = self.currentIndex()
        if tab == 0:
            self._render_state()
        elif tab == 1:
            self._render_boundary()
        elif tab == 2:
            if not self._gallery_timer.isActive() and self._gallery_next < len(self._tiles):
                self._gallery_timer.start()
        elif tab == 3:
            self._render_roi_map()
        elif tab == 4:
            self._render_overlap()

    def _state_values(self):
        return None if self.patterns is None else self.patterns[self.sel]

    def _span_text(self, i: int) -> str:
        a, b = int(self.edges[i]), int(self.edges[i + 1])
        text = f"TRs {a}–{b - 1} ({b - a} TRs"
        if self.tr:
            text += f", {(b - a) * self.tr:.0f} s"
        return text + ")"

    def _render_state(self) -> None:
        if self.patterns is None:
            return
        i = self.sel
        self.state_header.setText(f"State {i + 1} of {len(self.patterns)} · "
                                  f"{self._span_text(i)} · double-click a parcel to open its ROI")
        img = self._render(("state", self.k, i), self._big, self.patterns[i],
                           _STATE_CMAP, -self.lim, self.lim, highlight=self.current_roi)
        self.state_img.set_rgba(img)
        self.state_bars.set_values(self.info, self.patterns[i], self.lim, _STATE_CMAP)
        self.state_cbar.set_scale(_STATE_CMAP, -self.lim, self.lim, "mean pattern")

    def _boundary_index(self) -> int:
        """Boundary b sits between state b-1 and state b (b >= 1)."""
        return max(1, self.sel) if len(self.patterns) > 1 else 0

    def _boundary_values(self):
        if self.patterns is None or len(self.patterns) < 2:
            return (None, None, None)
        b = self._boundary_index()
        before, after = self.patterns[b - 1], self.patterns[b]
        return before, after, after - before

    def _render_boundary(self) -> None:
        if self.patterns is None:
            return
        if len(self.patterns) < 2:
            self.boundary_header.setText("This solution has a single state, so no boundaries.")
            for img in self.bnd_imgs:
                img.set_placeholder("")
            return
        b = self._boundary_index()
        tr = int(self.edges[b])
        text = f"Boundary {b} of {len(self.patterns) - 1} at TR {tr}"
        if self.tr:
            text += f" ({tr * self.tr:.0f} s)"
        if self.strengths is not None and b - 1 < len(self.strengths):
            text += f" · strength {self.strengths[b - 1]:.2f}"
        self.boundary_header.setText(text)
        self.bnd_labels[0].setText(f"State {b}")
        self.bnd_labels[1].setText(f"State {b + 1}")
        before, after, change = self._boundary_values()
        clim = float(np.nanpercentile(np.abs(self.patterns[1:] - self.patterns[:-1]), 98)) or 1.0
        hl = self.current_roi
        g = self._big.to_grid
        self.bnd_imgs[0].set_rgba(g(self._render(("state", self.k, b - 1), self._big, before,
                                                 _STATE_CMAP, -self.lim, self.lim, hl)))
        self.bnd_imgs[1].set_rgba(g(self._render(("state", self.k, b), self._big, after,
                                                 _STATE_CMAP, -self.lim, self.lim, hl)))
        self.bnd_imgs[2].set_rgba(g(self._render(("change", self.k, b), self._big, change,
                                                 _CHANGE_CMAP, -clim, clim, hl)))
        self.bnd_bars.set_values(self.info, change, clim, _CHANGE_CMAP, "Network change")
        self.bnd_cbar.set_scale(_STATE_CMAP, -self.lim, self.lim, "state")
        self.chg_cbar.set_scale(_CHANGE_CMAP, -clim, clim, "after − before")
        self.prev_btn.setEnabled(b > 1)
        self.next_btn.setEnabled(b < len(self.patterns) - 1)

    def _step_boundary(self, step: int) -> None:
        if self.patterns is None:
            return
        b = int(np.clip(self._boundary_index() + step, 1, len(self.patterns) - 1))
        self.set_selected_state(b)
        self.stateSelected.emit(b)

    # ---- gallery -----------------------------------------------------------------

    def _reset_gallery(self) -> None:
        self._gallery_timer.stop()
        while self.gallery_grid.count():
            item = self.gallery_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._tiles = []
        self._tile_boxes = []
        self._gallery_cols = 0
        self._gallery_next = 0
        if self.patterns is None:
            return
        for i in range(len(self.patterns)):
            box = QWidget()
            bv = QVBoxLayout(box)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.setSpacing(0)
            lab = QLabel(f"{i + 1} · {int(self.edges[i + 1] - self.edges[i])} TRs")
            lab.setStyleSheet(f"color: {_DIM}; font-size: 8pt;")
            tile = BrainImage(_TILE_H - 16)
            tile.grid = True
            box.setFixedSize(_TILE_W, _TILE_H)
            tile.selected = i == self.sel
            tile.set_placeholder("…")
            tile.clicked.connect(lambda _x, _y, i=i: self._tile_clicked(i))
            self._wire_image(tile, (lambda i=i: self.patterns[i]))
            bv.addWidget(lab)
            bv.addWidget(tile)
            self._tiles.append(tile)
            self._tile_boxes.append(box)
        self._layout_gallery()
        if self.currentIndex() == 2:
            self._schedule()

    def _layout_gallery(self) -> None:
        """Place tiles in as many columns as fit the visible width."""
        cols = max(1, (self.gallery_scroll.viewport().width() - 12) // (_TILE_W + 6))
        if cols == self._gallery_cols:
            return
        self._gallery_cols = cols
        for box in self._tile_boxes:
            self.gallery_grid.removeWidget(box)
        for i, box in enumerate(self._tile_boxes):
            self.gallery_grid.addWidget(box, i // cols, i % cols)
        self.gallery_grid.setColumnStretch(cols, 1)
        self.gallery_grid.setRowStretch(len(self._tile_boxes) // cols + 1, 1)

    def eventFilter(self, obj, event):
        if obj is self.gallery_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._layout_gallery()
        return super().eventFilter(obj, event)

    def _gallery_step(self) -> None:
        if not self._ready or self._gallery_next >= len(self._tiles) or self.patterns is None:
            self._gallery_timer.stop()
            return
        i = self._gallery_next
        self._gallery_next += 1
        img = self._small.parcels(self.spec).render(self.patterns[i], _cmap(_STATE_CMAP),
                                                    -self.lim, self.lim)
        self._tiles[i].set_rgba(self._small.to_grid(img))

    def _tile_clicked(self, i: int) -> None:
        self.set_selected_state(i)
        self.stateSelected.emit(i)

    # ---- ROI map -----------------------------------------------------------------

    def set_extra_metric(self, key: str, label: str, values: np.ndarray, unit: str) -> None:
        """Add (or update) a metric computed elsewhere, e.g. boundary overlap, and show it."""
        self._extra[key] = (np.asarray(values, float), unit)
        i = self.metric_combo.findData(key)
        if i < 0:
            self.metric_combo.addItem(label, key)
            i = self.metric_combo.count() - 1
        else:
            self.metric_combo.setItemText(i, label)
        self.metric_combo.setCurrentIndex(i)
        self._invalidate("roi")

    def _roi_metric(self) -> tuple[np.ndarray, str]:
        key = self.metric_combo.currentData()
        n = self.spec.n_rois
        if key in self._extra:
            return self._extra[key]
        vals = np.full(n, np.nan)
        if self.session is None:
            return vals, ""
        if key == "voxels":
            vals = self.session.voxel_counts.astype(float)
            vals[vals == 0] = np.nan
            return vals, "voxels"
        for target, s in self.session.summary().items():
            if isinstance(target, int):
                v = getattr(s, key)
                if key == "median_duration" and self.session.tr:
                    v = v * self.session.tr
                vals[target - 1] = v
        unit = {"nstates": "states", "fit_seconds": "s",
                "median_duration": "s" if self.session.tr else "TRs"}[key]
        return vals, unit

    def _render_roi_map(self) -> None:
        if self.session is None:
            self.roi_img.set_placeholder("Import a NIfTI or open a session to map ROI results.")
            return
        vals, unit = self._roi_metric()
        done = int(np.sum(~np.isnan(vals)))
        if done == 0:
            self.roi_header.setText("No ROI fits yet: run a batch from the session panel, or "
                                    "click an ROI to open and fit it")
            self.roi_cbar.set_scale(None, 0, 1)
            self.roi_img.set_rgba(self._render(("roi", "empty", self.current_roi), self._big,
                                               vals, _METRIC_CMAP, 0, 1, highlight=self.current_roi))
            return
        finite = vals[~np.isnan(vals)]
        lo, hi = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if len(finite) else (0, 1)
        if hi <= lo:
            hi = lo + 1
        key = ("roi", self.metric_combo.currentData(), self.current_roi, done,
               round(lo, 6), round(hi, 6))
        img = self._render(key, self._big, vals, _METRIC_CMAP, lo, hi, highlight=self.current_roi)
        self.roi_img.set_rgba(img)
        self.roi_cbar.set_scale(_METRIC_CMAP, lo, hi, unit)
        n_ok = int(np.sum(self.session.voxel_counts >= 2))
        self.roi_header.setText(f"{done} of {n_ok} ROIs · click an ROI to open it")

    def _pick(self, img: BrainImage, x: int, y: int) -> int | None:
        """Parcel under image pixel (x, y) of a BrainImage, whatever its layout."""
        if not self._ready:
            return None
        strip = self._small if img in self._tiles else self._big
        if img.grid:
            xy = strip.grid_to_strip(x, y)
            if xy is None:
                return None
            x, y = xy
        return strip.parcels(self.spec).pick(x, y)

    def _roi_clicked(self, x: int, y: int) -> None:
        if self.session is not None:
            parcel = self._pick(self.roi_img, x, y)
            if parcel is not None:
                self.roiSelected.emit(parcel)

    def _roi_hover(self, x: int, y: int, gpos) -> None:
        if not self._ready or self.session is None:
            return
        parcel = self._pick(self.roi_img, x, y)
        if parcel is None:
            QToolTip.hideText()
            return
        vals, unit = self._roi_metric()
        v = vals[parcel - 1]
        val = "not fitted" if np.isnan(v) else f"{v:.3g} {unit}"
        QToolTip.showText(gpos.toPoint(), f"ROI {parcel}: "
                          f"{atlas.short_name(self.info.names[parcel - 1])}\n{val}")

    # ---- shared hover / double-click ----------------------------------------------

    def _hover(self, img: BrainImage, x: int, y: int, gpos, values_fn) -> None:
        if not self._ready:
            return
        parcel = self._pick(img, x, y)
        values = values_fn()
        if parcel is None or values is None:
            QToolTip.hideText()
            return
        v = values[parcel - 1]
        QToolTip.showText(gpos.toPoint(), f"ROI {parcel}: "
                          f"{atlas.short_name(self.info.names[parcel - 1])}\n"
                          + ("no data" if np.isnan(v) else f"{v:+.3f}"))

    def _open_parcel_at(self, img: BrainImage, x: int, y: int) -> None:
        if self.session is not None:
            parcel = self._pick(img, x, y)
            if parcel is not None:
                self.roiSelected.emit(parcel)

    # ---- boundary overlap per ROI ------------------------------------------------

    _OVL_SHOW = (("OR", "or"), ("OR where p < .05, FDR across ROIs", "or_q"),
                 ("OR where p < .05, uncorrected", "or_p"),
                 ("Best HRF delay", "delay"), ("Best HRF delay where p < .05, FDR", "delay_q"))

    def _build_overlap_tab(self) -> None:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 4, 6, 2)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Reference:"))
        self.ovl_ref = QComboBox()
        self.ovl_ref.setMinimumContentsLength(24)
        self.ovl_ref.currentIndexChanged.connect(self._ovl_ref_changed)
        self.ovl_ref.activated.connect(lambda _: setattr(self, "_ovl_ref_chosen", True))
        self._ovl_ref_chosen = False     # keep the user's choice across refreshes, else the first
        r1.addWidget(self.ovl_ref)
        r1.addWidget(QLabel("HRF delay:"))
        self.ovl_delay = QComboBox()
        self.ovl_delay.setToolTip(
            "Stimulus boundaries are placed this much later than they occur. Scanning a grid "
            "picks each ROI's best delay; its p-value then comes from the best OR over all "
            "delays in every shuffle, which corrects for the search.")
        r1.addWidget(self.ovl_delay)
        self.ovl_btn = QPushButton("Compute")
        self.ovl_btn.setObjectName("run_btn")
        self.ovl_btn.clicked.connect(self.compute_overlap)
        r1.addWidget(self.ovl_btn)
        self.ovl_progress = QProgressBar()
        self.ovl_progress.setFixedHeight(12)
        self.ovl_progress.setFixedWidth(140)
        self.ovl_progress.setTextVisible(False)
        self.ovl_progress.hide()
        r1.addWidget(self.ovl_progress)
        r1.addWidget(QLabel("Show:"))
        self.ovl_show = QComboBox()
        for label, key in self._OVL_SHOW:
            self.ovl_show.addItem(label, key)
        self.ovl_show.currentIndexChanged.connect(lambda _: self._invalidate("ovl"))
        r1.addWidget(self.ovl_show)
        r1.addStretch()
        v.addLayout(r1)
        self.ovl_header = self._header()
        self.ovl_header.setStyleSheet(f"color: {_DIM};")
        v.addWidget(self.ovl_header)
        self.ovl_img = BrainImage(120)
        self.ovl_img.clicked.connect(lambda x, y: self._ovl_click(x, y))
        self.ovl_img.hovered.connect(self._ovl_hover)
        row = QHBoxLayout()
        row.addWidget(self.ovl_img, stretch=1)
        self.ovl_bars = NetworkBars(width=185)
        self.ovl_bars.setToolTip(
            "Mean over all fitted ROIs of each network (unthresholded: averaging only the "
            "significant ROIs would inflate every network). Right: ROIs with q < .05 / "
            "fitted ROIs in the network.")
        row.addWidget(self.ovl_bars)
        v.addLayout(row, stretch=1)
        self.ovl_cbar = ColourBar()
        v.addWidget(self.ovl_cbar)
        self.addTab(w, "Overlap")

    def refresh_overlap_sources(self) -> None:
        refs = self.overlap_sources() if self.overlap_sources else []
        current = self.ovl_ref.currentText()
        delay_idx = self.ovl_delay.currentIndex()
        self.ovl_ref.blockSignals(True)
        self.ovl_ref.clear()
        for r in refs:
            self.ovl_ref.addItem(r["label"], r)
        i = self.ovl_ref.findText(current) if self._ovl_ref_chosen else 0
        self.ovl_ref.setCurrentIndex(max(0, i))
        self.ovl_ref.blockSignals(False)
        self._ovl_ref_changed()
        if 0 <= delay_idx < self.ovl_delay.count():
            self.ovl_delay.setCurrentIndex(delay_idx)

    def _ovl_ref_changed(self) -> None:
        ref = self.ovl_ref.currentData()
        self.ovl_delay.clear()
        if ref is None or ref["kind"] != "stimulus":
            self.ovl_delay.addItem("\u2014 (neural reference)", None)
            self.ovl_delay.setEnabled(False)
        else:
            tr = ref["tr"]
            self.ovl_delay.setEnabled(True)
            self.ovl_delay.addItem(f"{ref['delay']:g} s (stimulus panel)", [ref["delay"]])
            grid = [round(i * tr, 3) for i in range(int(9 / tr) + 1)]
            self.ovl_delay.addItem(f"Scan {grid[0]:g}\u2013{grid[-1]:g} s in TR steps "
                                   f"({len(grid)} delays)", grid)
        self.ovl_btn.setEnabled(ref is not None and self.session is not None)

    def compute_overlap(self) -> None:
        ref = self.ovl_ref.currentData()
        s = self.session
        if ref is None or s is None or self._ovl_thread is not None:
            return
        import gsbs_stimulus as stim
        delays = self.ovl_delay.currentData()
        if ref["kind"] == "stimulus":
            refs = [stim.to_trs(ref["times"], ref["tr"], s.n_time, d) for d in delays]
        else:
            refs, delays = [np.asarray(ref["bounds"], int)], None
        fitted = sorted(k for k in s.summary() if isinstance(k, int))
        rois = [t for t in fitted if not s.summary()[t].at_limit]
        if not rois:
            self.ovl_header.setText("No fitted ROIs yet: run a batch from the session panel.")
            return
        self._ovl_pending = {"ref": ref["label"], "ref_config": ref.get("config", {})}
        self.ovl_progress.setRange(0, len(rois))
        self.ovl_progress.setValue(0)
        self.ovl_progress.show()
        self.ovl_btn.setEnabled(False)
        self.ovl_header.setText(f"Computing {len(rois)} ROIs \u00d7 {len(refs)} "
                                f"delay{'s' if len(refs) != 1 else ''}\u2026")
        self._ovl_worker = _OverlapWorker(s, refs, delays, 1000)
        self._ovl_thread = QThread(self)
        self._ovl_worker.moveToThread(self._ovl_thread)
        self._ovl_thread.started.connect(self._ovl_worker.run)
        self._ovl_worker.progress.connect(lambda i, n: self.ovl_progress.setValue(i))
        self._ovl_worker.done.connect(self._ovl_done)
        self._ovl_worker.done.connect(self._ovl_thread.quit)
        self._ovl_thread.finished.connect(self._ovl_finished)
        self._ovl_thread.start()

    def _ovl_finished(self) -> None:
        self._ovl_thread = None
        self.ovl_progress.hide()
        self.ovl_btn.setEnabled(True)

    def _ovl_done(self, res: dict) -> None:
        meta = self._ovl_pending
        grid = res["grid"]
        self._ovl = {
            **meta, "result": res, "rois": np.array(res["rois"], int), "excluded": res["excluded"],
            "delays": res["delays"], "or": np.array(res["or"]),
            "best_delay": np.array([np.nan if d is None else d for d in res["best_delay"]], float),
            "p": np.array(res["p"]), "q": np.array(res["q"]), "grid": grid,
            "or_grid": np.array(res["or_grid"]),
        }
        has_delay = grid
        model = self.ovl_show.model()
        for i, (_, key) in enumerate(self._OVL_SHOW):
            model.item(i).setEnabled(has_delay or not key.startswith("delay"))
        if not has_delay and self.ovl_show.currentData().startswith("delay"):
            self.ovl_show.setCurrentIndex(0)
        self._invalidate("ovl")

    def _ovl_values(self) -> tuple[np.ndarray, str, float, float, str]:
        """Per-parcel values, colour map, limits and unit for the current display."""
        o = self._ovl
        n = self.spec.n_rois
        vals = np.full(n, np.nan)
        key = self.ovl_show.currentData()
        keep = np.ones(len(o["rois"]), bool)
        if key in ("or_q", "delay_q"):
            keep = o["q"] < 0.05
        elif key == "or_p":
            keep = o["p"] < 0.05
        if key.startswith("delay"):
            vals[o["rois"][keep] - 1] = o["best_delay"][keep]
            d = np.asarray(o["delays"], float)
            return vals, "viridis", float(d.min()), float(d.max()), "best HRF delay (s)"
        vals[o["rois"][keep] - 1] = o["or"][keep]
        lim = float(np.nanpercentile(np.abs(o["or"]), 98)) if len(o["or"]) else 1.0
        return vals, "RdBu_r", -(lim or 1.0), lim or 1.0, "OR (0 = chance, 1 = all boundaries shared)"

    def _render_overlap(self) -> None:
        if self.session is None:
            self.ovl_img.set_placeholder("Open a session and fit ROIs to map boundary overlap.")
            return
        if self.ovl_ref.count() == 0:
            self.refresh_overlap_sources()
        if self._ovl is None:
            self.ovl_header.setText("Choose a reference and press Compute.")
            self.ovl_cbar.set_scale(None, 0, 1)
            self.ovl_bars.set_values(None, None, 1.0, "viridis")
            self.ovl_img.set_rgba(self._render(("ovl", "empty", self.current_roi), self._big,
                                               np.full(self.spec.n_rois, np.nan), "viridis", 0, 1,
                                               highlight=self.current_roi))
            return
        o = self._ovl
        vals, cmap, lo, hi, unit = self._ovl_values()
        key = ("ovl", id(o), self.ovl_show.currentData(), self.current_roi)
        self.ovl_img.set_rgba(self._render(key, self._big, vals, cmap, lo, hi,
                                           highlight=self.current_roi))
        self.ovl_cbar.set_scale(cmap, lo, hi, unit)
        self._ovl_network_bars(cmap, lo, hi)
        n_q = int(np.sum(o["q"] < 0.05))
        delays = o["delays"]
        dtext = ("" if delays is None else
                 f" \u00b7 delays {delays[0]:g}\u2013{delays[-1]:g} s, best per ROI (p corrected "
                 f"for the search)" if o["grid"] else f" \u00b7 delay {delays[0]:g} s")
        excl = (f" \u00b7 {len(o['excluded'])} left out (best at kmax)" if o["excluded"] else "")
        self.ovl_header.setText(f"{o['ref']}{dtext} \u00b7 {len(o['rois'])} ROIs, {n_q} with "
                                f"q < .05{excl} \u00b7 grey: not fitted or not significant")

    def _ovl_network_bars(self, cmap: str, lo: float, hi: float) -> None:
        """Network means of the shown quantity over all fitted ROIs, with significant/fitted counts."""
        o = self._ovl
        delay = self.ovl_show.currentData().startswith("delay")
        per_roi = o["best_delay"] if delay else o["or"]
        vals = np.full(self.spec.n_rois, np.nan)
        vals[o["rois"] - 1] = per_roi
        nets = self.info.networks[o["rois"] - 1]
        sig = o["q"] < 0.05
        counts = {net: f"{int(np.sum(sig[nets == net]))}/{int(np.sum(nets == net))}"
                  for net in self.info.network_order if np.any(nets == net)}
        title = "Mean best delay (s)" if delay else "Mean OR"
        self.ovl_bars.set_values(self.info, vals, max(abs(lo), abs(hi)), cmap,
                                 f"{title} \u00b7 q<.05/fitted", counts,
                                 sequential=(lo, hi) if delay else None)

    def _ovl_lookup(self, parcel: int):
        o = self._ovl
        if o is None:
            return None
        i = np.flatnonzero(o["rois"] == parcel)
        return None if len(i) == 0 else int(i[0])

    def _ovl_hover(self, x: int, y: int, gpos) -> None:
        if not self._ready or self.session is None:
            return
        parcel = self._pick(self.ovl_img, x, y)
        if parcel is None:
            QToolTip.hideText()
            return
        name = f"ROI {parcel}: {atlas.short_name(self.info.names[parcel - 1])}"
        i = self._ovl_lookup(parcel)
        if i is None:
            QToolTip.showText(gpos.toPoint(), name + "\nnot fitted / left out")
            return
        o = self._ovl
        text = f"{name}\nOR {o['or'][i]:.2f} \u00b7 p {o['p'][i]:.3g} \u00b7 q {o['q'][i]:.3g}"
        if o["grid"]:
            text += f"\nbest delay {o['best_delay'][i]:g} s"
        QToolTip.showText(gpos.toPoint(), text)

    def _ovl_click(self, x: int, y: int) -> None:
        parcel = self._pick(self.ovl_img, x, y)
        if parcel is not None and self.session is not None:
            self.roiSelected.emit(parcel)

    # ---- tab availability ----------------------------------------------------------

    def _update_tabs(self) -> None:
        has_states = self.spec is not None and self.patterns is not None
        for i in (0, 1, 2):
            self.setTabEnabled(i, has_states)
        self.setTabEnabled(3, self.spec is not None and self.session is not None)
        self.setTabEnabled(4, self.spec is not None and self.session is not None)
        if not has_states and self.isTabEnabled(3) and self.currentIndex() not in (3, 4):
            self.setCurrentIndex(3)
        elif has_states and not getattr(self, "_had_states", False) and self.currentIndex() in (3, 4):
            self.setCurrentIndex(0)          # state maps just became available
        self._had_states = has_states
        if has_states:
            for t in self._tiles:
                t.selected = False
            if self._tiles and self.sel < len(self._tiles):
                self._tiles[self.sel].selected = True
        self._schedule()
