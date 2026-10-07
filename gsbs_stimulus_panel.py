"""
Stimulus panel: the transcript laid out as the neural states of the solution
shown, with stimulus boundaries marked, and the overlap between those
boundaries and the neural ones.
"""

from __future__ import annotations

import html

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import gsbs_overlap as ov
import gsbs_stimulus as stim

_DIM = "#a6adc8"
_MARK = "#a6e3a1"
_SEL_BG = "#45475a"


class StimulusPanel(QWidget):
    stateSelected = pyqtSignal(int)
    boundariesChanged = pyqtSignal()
    transcriptLoaded = pyqtSignal(object)      # a Transcript the user loaded from a file
    eventsLoaded = pyqtSignal(object)          # event onsets (s) the user loaded

    def __init__(self, parent=None):
        super().__init__(parent)
        self.transcript: stim.Transcript | None = None
        self.events: np.ndarray | None = None
        self.tr: float | None = None
        self.n_time = 0
        self.edges = None
        self.k = 0
        self.sel = 0
        self._build()

    def _build(self) -> None:
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Boundaries:"))
        self.kind_combo = QComboBox()
        self.kind_combo.currentIndexChanged.connect(self._settings_changed)
        r1.addWidget(self.kind_combo)
        self.pause_spin = QDoubleSpinBox()
        self.pause_spin.setRange(0.1, 30)
        self.pause_spin.setValue(1.0)
        self.pause_spin.setSingleStep(0.25)
        self.pause_spin.setPrefix("≥ ")
        self.pause_spin.setSuffix(" s")
        self.pause_spin.setToolTip("Silence between two words that counts as a pause")
        self.pause_spin.valueChanged.connect(self._settings_changed)
        r1.addWidget(self.pause_spin)
        r1.addStretch()
        v.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("HRF delay:"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0, 15)
        self.delay_spin.setValue(4.5)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setSuffix(" s")
        self.delay_spin.setToolTip("Words and stimulus boundaries are placed this much later, "
                                   "where the BOLD response to them is expected.")
        self.delay_spin.valueChanged.connect(self._settings_changed)
        r2.addWidget(self.delay_spin)
        r2.addWidget(QLabel("tolerance:"))
        self.tol_spin = QSpinBox()
        self.tol_spin.setRange(0, 5)
        self.tol_spin.setPrefix("±")
        self.tol_spin.setSuffix(" TR")
        self.tol_spin.setToolTip("Boundaries this many TRs apart still count as overlapping")
        self.tol_spin.valueChanged.connect(self._settings_changed)
        r2.addWidget(self.tol_spin)
        r2.addStretch()
        v.addLayout(r2)

        r3 = QHBoxLayout()
        b1 = QPushButton("Load transcript…")
        b1.setToolTip("TSV/CSV with columns word, onset[, offset] in seconds from the first volume")
        b1.clicked.connect(self._load_transcript)
        b2 = QPushButton("Load events…")
        b2.setToolTip("Event onsets in seconds from the first volume: a BIDS events.tsv "
                      "(onset column) or one number per line")
        b2.clicked.connect(self._load_events)
        r3.addWidget(b1)
        r3.addWidget(b2)
        r3.addStretch()
        v.addLayout(r3)

        self.overlap_label = QLabel("")
        self.overlap_label.setWordWrap(True)
        v.addWidget(self.overlap_label)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(self._anchor)
        v.addWidget(self.text, stretch=1)
        self.source_label = QLabel("")
        self.source_label.setStyleSheet(f"color: {_DIM}; font-size: 8pt;")
        self.source_label.setWordWrap(True)
        v.addWidget(self.source_label)

    # ---- data ------------------------------------------------------------------

    def set_data(self, transcript, events, tr: float | None, n_time: int) -> None:
        self.transcript, self.events, self.tr, self.n_time = transcript, events, tr, n_time
        self.source_label.setText(transcript.source if transcript is not None else "")
        current = self.kind_combo.currentData()
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        if transcript is not None:
            self.kind_combo.addItem(stim.KINDS["sentences"], "sentences")
            self.kind_combo.addItem(stim.KINDS["pauses"], "pauses")
        if events is not None and len(events):
            self.kind_combo.addItem(f"{stim.KINDS['events']} ({len(events)})", "events")
        i = self.kind_combo.findData(current)
        self.kind_combo.setCurrentIndex(i if i >= 0 else 0)
        self.kind_combo.blockSignals(False)
        self._settings_changed()

    @property
    def has_content(self) -> bool:
        return self.transcript is not None or (self.events is not None and len(self.events) > 0)

    def set_solution(self, edges, k: int) -> None:
        self.edges, self.k = edges, k
        self.sel = min(self.sel, len(edges) - 2) if edges is not None else 0
        self._refresh()

    def set_selected(self, i: int) -> None:
        if self.edges is None or i == self.sel:
            return
        self.sel = int(i)
        self._render_text()

    def stimulus_bounds(self) -> np.ndarray | None:
        """Stimulus boundary TRs (after the HRF delay), or None."""
        kind = self.kind_combo.currentData()
        if kind is None or not self.tr:
            return None
        delay = self.delay_spin.value()
        if kind == "events":
            times = self.events
        else:
            times = self.transcript.boundary_times(kind, self.pause_spin.value())
        return stim.to_trs(times, self.tr, self.n_time, delay)

    def stimulus_times(self) -> np.ndarray | None:
        """Stimulus boundary times in seconds, before the HRF delay."""
        kind = self.kind_combo.currentData()
        if kind is None:
            return None
        if kind == "events":
            return np.asarray(self.events, float)
        return self.transcript.boundary_times(kind, self.pause_spin.value())

    def settings(self) -> dict:
        return {"kind": self.kind_combo.currentData(), "label": self.kind_combo.currentText(),
                "pause_s": self.pause_spin.value(), "delay_s": self.delay_spin.value(),
                "tolerance": self.tol_spin.value()}

    # ---- updates -----------------------------------------------------------------

    def _settings_changed(self) -> None:
        self.pause_spin.setVisible(self.kind_combo.currentData() == "pauses")
        self._refresh()
        self.boundariesChanged.emit()

    def _refresh(self) -> None:
        self._update_overlap()
        self._render_text()

    def _update_overlap(self) -> None:
        if not self.tr:
            self.overlap_label.setText("The stimulus can only be placed on the scan when the TR "
                                       "is known.")
            return
        sb = self.stimulus_bounds()
        if sb is None or self.edges is None:
            self.overlap_label.setText("")
            return
        nb = self.edges[1:-1]
        r = ov.overlap(sb, nb, self.n_time, tolerance=self.tol_spin.value())
        self.overlap_label.setText(
            f"<b>{len(nb)} neural vs {len(sb)} stimulus boundaries</b> ({self.kind_combo.currentText().lower()})"
            f"<br>{ov.describe(r)}<br><span style='color:{_DIM}'>OA, OR: 0 = chance, 1 = every "
            f"boundary matched. p from {r.n_perm} shuffles of the state durations.</span>")

    def _render_text(self) -> None:
        if self.transcript is None or self.edges is None or not self.tr:
            msg = ("No transcript. Online Narratives examples bring one; otherwise load a "
                   "transcript (word, onset in seconds).") if self.transcript is None else ""
            self.text.setHtml(f"<p style='color:{_DIM}'>{html.escape(msg)}</p>")
            return
        t = self.transcript
        delay = self.delay_spin.value()
        trs = t.word_trs(self.tr, delay)
        state = np.clip(np.searchsorted(self.edges, trs, side="right") - 1, 0, len(self.edges) - 2)
        starts = set()
        kind = self.kind_combo.currentData()
        if kind in ("sentences", "pauses"):
            times = set(np.round(t.boundary_times(kind, self.pause_spin.value()), 3))
            starts = {i for i in range(len(t)) if round(float(t.onset[i]), 3) in times and t.timed[i]}
        parts = []
        for s in range(len(self.edges) - 1):
            idx = np.flatnonzero(state == s)
            a, b = int(self.edges[s]), int(self.edges[s + 1])
            head = (f"State {s + 1} · TRs {a}–{b - 1} · "
                    f"{a * self.tr:.0f}–{b * self.tr:.0f} s (words shifted by {delay:g} s)")
            words = []
            for i in idx:
                w = html.escape(t.words[i])
                if not t.timed[i]:
                    w = f"<span style='color:{_DIM}'>{w}</span>"
                if i in starts:
                    w = f"<span style='color:{_MARK}'>│</span>{w}"
                words.append(w)
            body = " ".join(words) if words else f"<span style='color:{_DIM}'>(no speech)</span>"
            bg = f" style='background:{_SEL_BG}'" if s == self.sel else ""
            parts.append(f"<a name='s{s}'></a><div{bg}><p><a href='state:{s}' "
                         f"style='color:{_DIM}; text-decoration:none'>{head}</a><br>{body}</p></div>")
        self.text.setHtml("".join(parts))
        self.text.scrollToAnchor(f"s{self.sel}")

    def _anchor(self, url) -> None:
        s = url.toString()
        if s.startswith("state:"):
            i = int(s.split(":")[1])
            self.set_selected(i)
            self.stateSelected.emit(i)

    # ---- loading -------------------------------------------------------------------

    def _load_transcript(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load transcript", "",
                                              "Transcript (*.tsv *.csv *.txt);;All files (*.*)")
        if not path:
            return
        try:
            t = stim.Transcript.load(path)
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Transcript", str(exc))
            return
        self.transcriptLoaded.emit(t)

    def _load_events(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load event onsets", "",
                                              "Events (*.tsv *.csv *.txt);;All files (*.*)")
        if not path:
            return
        try:
            times = stim.load_event_times(path)
        except Exception as exc:                                   # noqa: BLE001
            QMessageBox.critical(self, "Events", f"Could not read onsets: {exc}")
            return
        self.eventsLoaded.emit(times)
