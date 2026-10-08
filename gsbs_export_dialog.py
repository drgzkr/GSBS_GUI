"""
Export dialog: collects the analysis the GUI holds into a configuration, writes
an analysis package (gsbs_export), runs it, checks that it reproduces the GUI's
numbers, and previews the generated text and figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import gsbs_core as core
import gsbs_export as ex
from gsbs_nifti import WHOLE_BRAIN

_DIM = "#a6adc8"


# ---------------------------------------------------------------------------
# Configuration from the GUI state
# ---------------------------------------------------------------------------

def collect_config(main) -> tuple[dict, dict]:
    """(analysis configuration, extra inputs for the package) for the current GUI state."""
    if main.gsbs_object is None:
        raise ValueError("Fit or open a result first: the export starts from the segmentation shown.")
    cfg = {"package": {"created": ex.now(), "gsbs_gui_commit": ex.gui_commit(),
                       "software": ex.software_versions()}}
    extras = {"session": None, "array": None, "gui_result": None, "transcript": None, "events": None}
    s = main.session
    in_session = s is not None and main.target is not None
    if in_session:
        meta = s.meta
        key = ex.example_key(meta["source"])
        pre = meta["preprocessing"]
        cfg["data"] = {
            "kind": "nifti", "example": key, "path": None if key else meta["source"],
            "tr": s.tr, "n_time": s.n_time, "atlas": s.spec.to_meta(), "columns": "parcels",
            "import": {"n_rois": s.spec.n_rois, "yeo_networks": s.spec.yeo_networks, "tr": s.tr,
                       "confounds": (bool(pre["confounds"]) if key else pre["confounds"]),
                       "detrend": pre["detrend"], "high_pass": pre["high_pass"],
                       "zscore_parcels": pre["zscore_parcels"]},
            "confound_columns": pre.get("confound_columns", []),
            "atlas_resolution_mm": meta["atlas"].get("resolution_mm"),
            "missing_parcels": len(meta.get("missing_parcels", [])),
        }
        summ = s.summary()
        targets = sorted(summ, key=lambda t: (t != WHOLE_BRAIN, t if isinstance(t, int) else 0))
        cfg["fits"] = {"targets": [t if t == WHOLE_BRAIN else int(t) for t in targets],
                       "params": {str(t): summ[t].params for t in targets},
                       "engine": summ[main.target].engine if main.target in summ else "fast_GSBS"}
        cfg["segmentation"] = {"target": main.target, "k": int(main._k)}
        extras["session"] = str(s.path)
        transcript = s.transcript()
        if transcript is not None:
            cfg["transcript"] = {"source": transcript.source}
            if not (key or "").startswith("narratives"):
                extras["transcript"] = transcript
        events = s.event_times()
        if events is not None and len(events):
            extras["events"] = events
    else:
        m = main.meta or {}
        cfg["data"] = {"kind": "array", "example": None, "path": main.path_edit.text().strip(),
                       "tr": m.get("tr"), "n_time": int(main.x.shape[0]),
                       "atlas": ({k: m[k] for k in ("atlas", "n_rois", "yeo_networks")}
                                 if main._is_parcels() else None),
                       "columns": main._col_noun()}
        cfg["fits"] = {"targets": ["data"], "params": {"data": core.fit_params(main.gsbs_object)},
                       "engine": type(main.gsbs_object).__name__}
        cfg["segmentation"] = {"target": "data", "k": int(main._k)}
        extras["array"] = np.asarray(main.x)
        extras["gui_result"] = main.gsbs_object
        extras["transcript"] = main._manual_transcript
        extras["events"] = main._manual_events
        if main._manual_transcript is not None:
            cfg["transcript"] = {"source": main._manual_transcript.source}
    if cfg["fits"]["engine"] not in ("fast_GSBS", "GSBS"):
        cfg["fits"]["engine"] = "fast_GSBS"

    st = main.stimulus
    if st.has_content and cfg["data"].get("tr") and st.stimulus_bounds() is not None:
        se = st.settings()
        cfg["stimulus"] = {"boundaries": se["kind"], "label": se["label"], "pause_s": se["pause_s"],
                           "delay_s": se["delay_s"], "tolerance": se["tolerance"]}

    o = getattr(main.brain, "_ovl", None)
    if in_session and o is not None and o.get("ref_config"):
        rc = dict(o["ref_config"])
        ref = ({"kind": "stimulus", "boundaries": rc["boundaries"], "label": rc["label"],
                "pause_s": rc["pause_s"]} if rc["kind"] == "stimulus"
               else {"kind": "neural", "target": rc["target"]})
        cfg["overlap_map"] = {"reference": ref, "delays": o["delays"],
                              "n_perm": o["result"]["n_perm"], "alpha": 0.05}
        if ref["kind"] == "stimulus" and "stimulus" not in cfg:
            cfg["stimulus"] = {"boundaries": rc["boundaries"], "label": rc["label"],
                               "pause_s": rc["pause_s"], "delay_s": o["delays"][0], "tolerance": 0}

    cw = getattr(main, "cluster_window", None)
    if cw is not None and cw.clusters is not None and cw.obj is main.gsbs_object:
        params = cw._params()
        sc = cw.scan_result
        method = cw.method_combo.currentData()
        chosen = bool(sc is not None and sc.method == method and sc.best == params)
        scan = None
        if chosen:
            if method == "dbscan":
                eps = [g["eps"] for g in sc.grid]
                scan = {"eps": [min(eps), max(eps)], "n_eps": len(set(eps)),
                        "min_samples": sorted({g["min_samples"] for g in sc.grid})}
            else:
                n = [g["n_clusters"] for g in sc.grid]
                scan = {"n_clusters": [min(n), max(n)]}
        cfg["clusters"] = {"target": cfg["segmentation"]["target"], "k": int(cw.k),
                           "method": method, "params": params,
                           "selected_by_silhouette": chosen, "scan": scan}
    return cfg, extras


def gui_results(main, cfg: dict) -> dict:
    """What the GUI currently shows, for comparison with the package's results.json."""
    out = {"best_k": core.best_k(main.gsbs_object)}
    o = getattr(main.brain, "_ovl", None)
    if cfg.get("overlap_map") and o is not None:
        out["overlap_map"] = {"rois": o["result"]["rois"], "or": o["result"]["or"], "p": o["result"]["p"]}
    cw = getattr(main, "cluster_window", None)
    if cfg.get("clusters") and cw is not None and cw.clusters is not None:
        out["clusters"] = {"labels": cw.clusters.labels.tolist()}
    return out


def compare(gui: dict, res: dict) -> list[tuple[str, bool]]:
    checks = [("Optimal number of states", gui["best_k"] == res["segmentation"]["best_k"])]
    if "overlap_map" in gui:
        a, b = gui["overlap_map"], res.get("overlap_map", {})
        checks.append(("Per-ROI overlap (OR and p)", a["rois"] == b.get("rois")
                       and np.allclose(a["or"], b.get("or", []), equal_nan=True)
                       and np.allclose(a["p"], b.get("p", []))))
    if "clusters" in gui:
        checks.append(("Cluster labels", gui["clusters"]["labels"] == res.get("clusters", {}).get("labels")))
    return checks


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class _Runner(QObject):
    line = pyqtSignal(str)
    done = pyqtSignal(int)

    def __init__(self, out, args):
        super().__init__()
        self.out, self.args = out, args

    def run(self):
        self.done.emit(ex.run_package(self.out, self.args, log=self.line.emit))


class ExportDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Export analysis package")
        self.resize(980, 760)
        self._thread = None
        try:
            self.cfg, self.extras = collect_config(main)
            self.gui = gui_results(main, self.cfg)
            error = None
        except Exception as exc:                                    # noqa: BLE001
            self.cfg = None
            error = str(exc)

        v = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        row = QHBoxLayout()
        row.addWidget(QLabel("Folder:"))
        self.dir_edit = QLineEdit(self._default_dir())
        row.addWidget(self.dir_edit, stretch=1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row.addWidget(b)
        self.export_btn = QPushButton("Export and run")
        self.export_btn.setObjectName("run_btn")
        self.export_btn.clicked.connect(self.export)
        row.addWidget(self.export_btn)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.dir_edit.text())))
        self.open_btn.setEnabled(False)
        row.addWidget(self.open_btn)
        v.addLayout(row)
        self.tabs = QTabWidget()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")
        self.methods_view = QTextBrowser()
        self.results_view = QTextBrowser()
        self.tabs.addTab(self.methods_view, "Methods")
        self.tabs.addTab(self.results_view, "Results")
        self.fig_scroll = QScrollArea()
        self.fig_scroll.setWidgetResizable(True)
        self.tabs.addTab(self.fig_scroll, "Figures")
        v.addWidget(self.tabs, stretch=1)

        if error:
            self.summary.setText(f"⚠ {error}")
            self.export_btn.setEnabled(False)
        else:
            self.summary.setText(self._describe())

    def _default_dir(self) -> str:
        m = self.main
        if m.session is not None:
            return str(m.session.path.parent / f"{m.session.name}_analysis")
        base = Path(m.path_edit.text().strip() or Path.home() / "gsbs").with_suffix("")
        return str(Path(str(base) + "_analysis"))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Package folder", self.dir_edit.text())
        if path:
            self.dir_edit.setText(path)

    def _describe(self) -> str:
        c = self.cfg
        parts = [f"<b>Data:</b> {c['data'].get('example') or Path(str(c['data'].get('path'))).name} "
                 f"({c['data']['n_time']} TRs)",
                 f"<b>Fits:</b> {len(c['fits']['targets'])} ({c['fits']['engine']})",
                 f"<b>Segmentation shown:</b> {c['segmentation']['target']}, {c['segmentation']['k']} states"]
        if c.get("stimulus"):
            parts.append(f"<b>Stimulus:</b> {c['stimulus']['label'].lower()}, delay {c['stimulus']['delay_s']:g} s")
        if c.get("overlap_map"):
            d = c["overlap_map"]["delays"]
            parts.append("<b>Overlap map:</b> " + ("stimulus" if c["overlap_map"]["reference"]["kind"] == "stimulus"
                                                     else "whole brain") + (f", {len(d)} delays" if d and len(d) > 1 else ""))
        if c.get("clusters"):
            parts.append(f"<b>Clusters:</b> {c['clusters']['method']} on {c['clusters']['k']} states")
        figs = ex.figure_ids(c)
        return (" · ".join(parts) + f"<br><span style='color:{_DIM}'>Figures: "
                + ", ".join(figs) + ". The package re-runs these analyses from its own scripts "
                "(reusing the GUI's fits) and checks that it reproduces the GUI's numbers.</span>")

    def export(self) -> None:
        out = Path(self.dir_edit.text().strip())
        if not out.name:
            return
        self.log.clear()
        self.tabs.setCurrentIndex(0)
        e = self.extras
        ex.write_package(self.cfg, out, gui_result=e["gui_result"], transcript=e["transcript"],
                         events=e["events"], array=e["array"])
        self.log.appendPlainText(f"Wrote package to {out}")
        args = ["--skip-prepare", "--skip-fit"]
        if e["session"]:
            args += ["--session", e["session"]]
        self.export_btn.setEnabled(False)
        self._runner = _Runner(out, args)
        self._thread = QThread(self)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.line.connect(self.log.appendPlainText)
        self._runner.done.connect(lambda code: self._finished(out, code))
        self._runner.done.connect(self._thread.quit)
        self._thread.start()

    def _finished(self, out: Path, code: int) -> None:
        self.export_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        if code:
            self.log.appendPlainText(f"✗ run_all.py failed (exit code {code}); see the log above.")
            return
        res = json.loads((out / "results" / "results.json").read_text())
        checks = compare(self.gui, res)
        for name, ok in checks:
            self.log.appendPlainText(f"{'✓' if ok else '✗'} {name} "
                                     f"{'reproduced' if ok else 'DIFFERS from the GUI'}")
        self.methods_view.setMarkdown((out / "manuscript" / "methods.md").read_text())
        self.results_view.setMarkdown((out / "manuscript" / "results.md").read_text())
        self._show_figures(out)
        self.tabs.setCurrentIndex(1)

    def _show_figures(self, out: Path) -> None:
        host = QWidget()
        v = QVBoxLayout(host)
        for png in sorted((out / "figures").glob("*.png")):
            cap = out / "figures" / f"fig_{png.stem}_caption.md"
            lab = QLabel()
            pix = QPixmap(str(png))
            lab.setPixmap(pix.scaledToWidth(min(880, pix.width()), Qt.TransformationMode.SmoothTransformation))
            v.addWidget(lab)
            text = QLabel(cap.read_text().replace("**", "") if cap.exists() else png.name)
            text.setWordWrap(True)
            text.setStyleSheet(f"color: {_DIM};")
            v.addWidget(text)
            v.addWidget(QLabel(f"<span style='color:{_DIM}'>code: figures/fig_{png.stem}.py</span>"))
        v.addStretch()
        self.fig_scroll.setWidget(host)
