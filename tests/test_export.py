"""Analysis packages: generated text, and replication of GUI results without the GUI."""

import json
import re
import shutil
import time

import numpy as np
import pytest

import gsbs_export as ex
import gsbs_report as report

AMERICAN = r"\b(analyz|normaliz|organiz|recogniz|behavior|color|neighbor|modeling|labeled)\w*"


def _check_text(text: str) -> None:
    assert "\u2014" not in text, "em dash in generated text"
    assert not re.search(AMERICAN, text), re.search(AMERICAN, text)
    assert not re.search(r"\bvery\b|\bclearly\b|\bobviously\b", text)


class TestReport:

    @pytest.fixture
    def cfg_res(self):
        cfg = {"package": {"software": {"python": "3.12", "statesegmentation": "0.1.0"}, "gsbs_gui_commit": None},
               "data": {"kind": "nifti", "example": None, "tr": 2.0, "atlas": {"atlas": "schaefer", "n_rois": 100, "yeo_networks": 7},
                        "import": {"detrend": True, "confounds": None, "high_pass": None, "zscore_parcels": False}},
               "fits": {"targets": ["whole_brain"], "engine": "fast_GSBS"},
               "segmentation": {"target": "whole_brain", "k": 12},
               "stimulus": {"boundaries": "events", "label": "Loaded events", "pause_s": 1, "delay_s": 4.5, "tolerance": 0}}
        res = {"segmentation": {"n_time": 300, "n_columns": 100, "k": 12, "best_k": 12, "best_at_limit": False,
                                "kmax": 150, "params": {"kmax": 150, "statewise_detection": True, "finetune": 1, "dmin": 1},
                                "engine": "fast_GSBS", "t_best": 12.3, "median_length_s": 40.0, "min_length_s": 6.0,
                                "max_length_s": 90.0, "median_length_trs": 20, "min_length_trs": 3, "max_length_trs": 45,
                                "boundaries": list(range(20, 300, 25)), "strengths": [0.5] * 11},
               "stimulus_overlap": {"n_ref": 30, "n_test": 11, "n_time": 300, "tolerance": 0, "observed": 6,
                                    "expected": 1.1, "oa": 0.5, "or": 0.5, "p": 0.000999, "n_perm": 1000, "null": [1] * 10}}
        return cfg, res

    def test_text_conventions(self, cfg_res, tmp_path):
        cfg, res = cfg_res
        out = report.write_all(cfg, res, tmp_path)
        md = (tmp_path / "methods.md").read_text() + (tmp_path / "results.md").read_text()
        _check_text(md)
        assert "[TODO:" in md                                   # unknown data described as a TODO
        assert "p < 0.001" in md
        # figures cited in the order they are numbered
        cited = [int(n) for n in re.findall(r"Fig\. (\d)", (tmp_path / "results.md").read_text().split("## Figure captions")[0])]
        assert cited == sorted(cited) and set(cited) == set(range(1, len(out["figures"]) + 1))
        tex = (tmp_path / "methods.tex").read_text()
        assert "\\todo{" in tex and "\\citep{Geerligs2022}" in tex and "\\citet{Geerligs2021}" in tex
        bib = (tmp_path / "references.bib").read_text()
        for key in set(re.findall(r"\\cite[pt]\{([^}]+)\}", tex + (tmp_path / "results.tex").read_text())):
            for k in key.split(","):
                assert f"{{{k}," in bib

    def test_plural_and_numbers(self):
        assert report.plural(2, "boundary") == "2 boundaries"
        assert report.plural(1, "state") == "1 state"
        assert report.plural(0, "parcel", words=True) == "no parcels"
        assert report.fmt_p(0.000999, 1000) == "p < 0.001"


class TestReplication:

    def test_array_package_reproduces_gui(self, qapp, tmp_path):
        """Export from the GUI (demo data with events and clusters), then run the package
        from scratch, refitting, and compare with the GUI."""
        import gsbs_gui
        from gsbs_export_dialog import collect_config, compare, gui_results
        w = gsbs_gui.GSBSApp()
        try:
            w.load_demo(fitted=True)
            w.meta = {"tr": 1.5}
            rng = np.random.default_rng(0)
            w._attach_events(np.sort(rng.uniform(5, 440, 30)))
            w.show_k(int(w._ks[w._ks >= 20][0]))
            w.open_clusters()
            w.cluster_window.run_scan()
            cfg, extras = collect_config(w)
            gui = gui_results(w, cfg)
            gui_seg = list(w._edges[1:-1])
            assert cfg["data"]["kind"] == "array" and cfg.get("stimulus") and cfg.get("clusters")
            out = ex.write_package(cfg, tmp_path / "pkg", gui_result=extras["gui_result"],
                                   transcript=extras["transcript"], events=extras["events"],
                                   array=extras["array"])
            (out / "work" / "result.npy").unlink()              # force a refit from the data
            log = []
            assert ex.run_package(out, [], log=log.append) == 0, "\n".join(log[-20:])
            res = json.loads((out / "results" / "results.json").read_text())
            assert all(ok for _, ok in compare(gui, res))
            assert res["segmentation"]["boundaries"] == [int(b) for b in gui_seg]
            for f in ex.figure_ids(cfg):
                for ext in ("png", "pdf", "svg"):
                    assert (out / "figures" / f"{f}.{ext}").exists()
                assert (out / "figures" / f"fig_{f}.py").exists()
                assert (out / "figures" / f"fig_{f}_caption.md").exists()
            for name in ("methods.md", "results.md", "methods.tex", "results.tex", "figures.tex", "references.bib"):
                assert (out / "manuscript" / name).exists()
            _check_text((out / "manuscript" / "methods.md").read_text())
        finally:
            w.close()


class TestReportAudit:
    """Each choice the analysis makes is stated, and each dropped item is counted."""

    def _render(self, cfg, res):
        m, r, c = report.methods(cfg, res), report.results(cfg, res), report.captions(cfg, res)
        return (" ".join(m.render(x, False) for _, p in m.paragraphs for x in p),
                " ".join(r.render(x, False) for _, p in r.paragraphs for x in p), c)

    def test_engine_order_blocksize_confounds_parcels(self):
        cfg, res = _base()
        cfg["fits"]["engine"] = "fast_GSBS"
        res["segmentation"]["engine"] = "GSBS"                  # saved objects are plain GSBS
        res["segmentation"]["params"].update(finetune_order=False, blocksize=30)
        cfg["data"].update(confound_columns=["trans_x", "csf"], atlas_resolution_mm=2, missing_parcels=3)
        cfg["data"]["import"]["confounds"] = "conf.tsv"
        methods, _, _ = self._render(cfg, res)
        assert "We used fast_GSBS" in methods
        assert "from the strongest to the weakest" in methods
        assert "blocks of 30 timepoints" in methods
        assert "2 confound time series from the preprocessing output (trans_x, csf)" in methods
        assert "2 mm, 7-network version" in methods
        assert "3 parcels that contained no voxels" in methods

    def test_stimulus_bookkeeping_and_tolerance(self):
        cfg, res = _base()
        res["stimulus_info"] = {"kind": "sentences", "n_times": 40, "n_outside": 2, "n_trs": 35,
                                "n_merged": 3, "n_words": 500, "n_untimed": 20}
        cfg["stimulus"].update(boundaries="sentences", tolerance=1)
        res["stimulus_overlap"]["tolerance"] = 1
        cfg["transcript"] = {"source": "x"}
        methods, results, caps = self._render(cfg, res)
        assert "(20 of 500, 4.0%) could not start a boundary" in methods
        assert "Three boundaries that fell in the same TR" in methods and "2 boundaries that fell outside" in methods
        assert "leaving 35 stimulus boundaries" in methods
        assert "fell within 1 TR of" in results and "within 1 TR of a stimulus boundary" in caps["stimulus_overlap"]
        assert "Neural state boundaries were always the test segmentation" in methods

    def test_cluster_choice_is_stated(self):
        cfg, res = _base()
        res["clusters"] = {"k": 12, "labels": [0, 0, 1, 1, -1, 2, 2, 0, 1, 2, 2, 0], "centroids": [[0]] * 3,
                           "time_fraction": [0.4, 0.3, 0.3], "silhouette": 0.3,
                           "transitions": [[0, 1, 0], [0, 0, 1], [2, 0, 0]], "edges": list(range(13))}
        cfg["clusters"] = {"target": "whole_brain", "k": 12, "method": "hierarchical",
                           "params": {"n_clusters": 3}, "selected_by_silhouette": False, "scan": None}
        assert "[TODO: justify the number of clusters.]" in self._render(cfg, res)[0]
        cfg["clusters"].update(selected_by_silhouette=True, scan={"n_clusters": [2, 11]})
        assert "across solutions with 2 to 11 clusters" in self._render(cfg, res)[0]
        cfg["clusters"].update(method="dbscan", params={"eps": 0.4, "min_samples": 3},
                               scan={"eps": [0.1, 0.9], "n_eps": 12, "min_samples": [2, 3, 5]})
        methods, results, caps = self._render(cfg, res)
        assert "over 12 radii from 0.1 to 0.9 and minimum numbers of 2, 3 and 5 states" in methods
        assert "not assigned to any cluster were not counted" in methods and "scikit-learn" in methods
        assert "with 1 state not assigned to any cluster" in results
        assert "not assigned to any cluster are grey" in caps["clusters"]


def _base():
    cfg = {"package": {"software": {"python": "3.12", "statesegmentation": "0.1.0"}, "gsbs_gui_commit": None},
           "data": {"kind": "nifti", "example": None, "tr": 2.0, "atlas": {"atlas": "schaefer", "n_rois": 100, "yeo_networks": 7},
                    "import": {"detrend": True, "confounds": None, "high_pass": None, "zscore_parcels": False}},
           "fits": {"targets": ["whole_brain"], "engine": "fast_GSBS"},
           "segmentation": {"target": "whole_brain", "k": 12},
           "stimulus": {"boundaries": "events", "label": "Loaded events", "pause_s": 1, "delay_s": 4.5, "tolerance": 0}}
    res = {"segmentation": {"n_time": 300, "n_columns": 100, "k": 12, "best_k": 12, "best_at_limit": False,
                            "kmax": 150, "params": {"kmax": 150, "statewise_detection": True, "finetune": 1, "dmin": 1},
                            "engine": "fast_GSBS", "t_best": 12.3, "median_length_s": 40.0, "min_length_s": 6.0,
                            "max_length_s": 90.0, "median_length_trs": 20, "min_length_trs": 3, "max_length_trs": 45,
                            "boundaries": list(range(20, 300, 25)), "strengths": [0.5] * 11},
           "stimulus_overlap": {"n_ref": 30, "n_test": 11, "n_time": 300, "tolerance": 0, "observed": 6,
                                "expected": 1.1, "oa": 0.5, "or": 0.5, "p": 0.000999, "n_perm": 1000, "null": [1] * 10}}
    return cfg, res
