"""GUI tests (offscreen Qt)."""

import time

import numpy as np
import pytest

import gsbs_core as core


@pytest.fixture
def win(qapp):
    import gsbs_gui
    w = gsbs_gui.GSBSApp()
    yield w
    w._stop_child()
    w.close()


def _pump(qapp, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)


class TestFittedDemo:

    def test_opens_on_best_solution(self, win):
        win.load_demo(fitted=True)
        obj = win.gsbs_object
        assert win._k == core.best_k(obj)
        assert win.k_spinbox.value() == win._k
        assert win.k_slider.maximum() == len(core.solution_ks(obj)) - 1
        assert win.truth is not None
        assert win.kmax_spin.value() == obj.kmax          # not len(tdists) - 1

    def test_slider_only_reaches_real_solutions(self, win):
        win.load_demo(fitted=True)
        obj = win.gsbs_object
        for idx in range(0, win.k_slider.maximum() + 1, 7):
            win.k_slider.setValue(idx)
            assert len(core.boundary_indices(obj.all_bounds[win._k])) + 1 == win._k

    def test_spinbox_snaps_in_direction_of_travel(self, win):
        win.load_demo(fitted=True)
        ks = list(win._ks)
        gap = next(k for k in range(ks[0], ks[-1]) if k not in ks)
        prev = max(k for k in ks if k < gap)
        nxt = min(k for k in ks if k > gap)
        win.show_k(prev)
        win.k_spinbox.setValue(gap)
        assert win._k == nxt
        win.k_spinbox.setValue(gap)
        assert win._k == prev

    def test_box_edges_sit_between_pixels(self, win):
        win.load_demo(fitted=True)
        segs = win._artists["boxes"].get_segments()
        assert segs[0][0][0] == -0.5
        first_bound = core.boundary_indices(win.gsbs_object.all_bounds[win._k])[0]
        assert segs[1][0][0] == first_bound - 0.5


class TestRun:

    def test_live_fit_then_explore(self, win, qapp):
        win.load_demo(fitted=False)
        assert win.gsbs_object is None and win.truth is not None
        win.kmax_spin.setValue(30)
        win.run_gsbs()
        assert win.run_btn.text() == "Cancel"
        deadline = time.monotonic() + 60
        while win._proc is not None and time.monotonic() < deadline:
            _pump(qapp, 0.1)
        assert win.gsbs_object is not None, win.statusBar().currentMessage()
        assert type(win.gsbs_object).__name__ == "fast_GSBS"
        assert win.run_btn.text() == "Run GSBS"
        assert win.progress.value() == win.progress.maximum()
        assert win._k == core.best_k(win.gsbs_object)

    def test_cancel(self, win, qapp):
        win.load_demo(fitted=False)
        win.run_gsbs()
        _pump(qapp, 0.5)
        proc = win._proc
        win.cancel_gsbs()
        assert win._proc is None and not proc.is_alive()
        assert win.gsbs_object is None
        assert win.open_btn.isEnabled()


class TestSession:

    @pytest.fixture
    def session_dir(self, tmp_path):
        """A tiny session built from the demo data's shape conventions (100 parcels)."""
        import json
        rng = np.random.default_rng(1)
        n_time, n_rois = 60, 100
        counts = np.full(n_rois, 4)
        voxels = rng.standard_normal((counts.sum(), n_time)).astype(np.float32)
        labels = np.repeat(np.arange(1, n_rois + 1), counts)
        np.save(tmp_path / "voxels.npy", voxels)
        np.save(tmp_path / "voxel_labels.npy", labels)
        parcels = np.add.reduceat(voxels, np.r_[0, np.cumsum(counts)[:-1]], axis=0) / 4
        np.save(tmp_path / "parcels.npy", parcels.T.astype(float))
        (tmp_path / "results").mkdir()
        (tmp_path / "session.json").write_text(json.dumps({
            "source": "synthetic.nii.gz", "shape": [1, 1, 1, n_time], "tr": 2.0,
            "atlas": {"atlas": "schaefer", "n_rois": n_rois, "yeo_networks": 7},
            "voxels_per_parcel": counts.tolist(), "missing_parcels": []}))
        return tmp_path

    def test_open_fit_and_switch_targets(self, win, qapp, session_dir):
        win.open_session(str(session_dir), fit_whole_brain=True)
        assert win.target == core_whole_brain() and win._is_parcels()
        deadline = time.monotonic() + 60
        while win._proc is not None and time.monotonic() < deadline:
            _pump(qapp, 0.1)
        assert win.gsbs_object is not None
        assert win.session.result_path(core_whole_brain()).exists()
        assert win.brain.patterns is not None
        assert len(win.brain.patterns) == core.best_k(win.gsbs_object)

        win._select_state(2)
        assert win.brain.sel == 2

        win.brain.roiSelected.emit(7)                     # e.g. a click on the ROI map
        assert win.target == 7
        assert win.x.shape == (60, 4)
        assert not win._is_parcels() and win.brain.patterns is None
        assert win.gsbs_object is None

        win.session_panel.select_target(core_whole_brain())
        win.activate_target(core_whole_brain())
        assert win.gsbs_object is not None                 # saved result reloads


def core_whole_brain():
    from gsbs_nifti import WHOLE_BRAIN
    return WHOLE_BRAIN


class TestOnlineExample:

    def test_opens_prefilled_import_dialog(self, win, qapp, monkeypatch):
        import gsbs_gui
        import gsbs_nifti
        try:
            gsbs_nifti.fetch_example()                          # cache it, or skip offline
        except Exception as exc:                               # noqa: BLE001
            pytest.skip(f"example unavailable offline: {exc}")
        seen = {}

        def fake_exec(dlg):
            seen.update(nifti=dlg.nifti_edit.text(), tr=dlg.tr_spin.value(),
                        confounds=dlg.conf_edit.text(), detrend=dlg.detrend.isChecked(),
                        session_dir=dlg.dir_edit.text())
            return 0                                            # rejected
        monkeypatch.setattr(gsbs_gui.ImportDialog, "exec", fake_exec)
        win.load_online_example()
        deadline = time.monotonic() + 60
        while not seen and time.monotonic() < deadline:
            _pump(qapp, 0.1)
        assert seen["nifti"].endswith("desc-preproc_bold.nii.gz")
        assert seen["tr"] == 2.0
        assert seen["confounds"].endswith("Confounds_regressors.tsv")
        assert seen["detrend"] and seen["session_dir"].endswith("_gsbs")
        _pump(qapp, 0.3)
        assert win.demo_btn.isEnabled()


class TestClusterWindow:

    def test_clusters_follow_the_shown_solution(self, win, qapp):
        win.load_demo(fitted=True)
        win.open_clusters()
        cw = win.cluster_window
        assert cw.k == win._k and len(cw.patterns) == win._k
        cw.run_scan()
        assert cw.clusters is not None and cw.clusters.n >= 2
        assert cw.maps_grid.count() == cw.clusters.n
        from PyQt6.QtWidgets import QFrame
        tiles = [w for w in cw.findChildren(QFrame) if w.styleSheet().startswith("QFrame { border-left")]
        assert len(tiles) == cw.clusters.n                      # no tiles left from earlier runs
        other = int(win._ks[win._ks > win._k][0])
        win.show_k(other)                                        # main slider moves
        assert cw.k == other and len(cw.clusters.labels) == other
        cw.close()


class TestDurations:

    def test_median_duration_follows_solution(self, win):
        win.load_demo(fitted=True)                                # no TR: reported in TRs
        lengths = np.diff(core.state_edges(win.gsbs_object.all_bounds[win._k], win.x.shape[0]))
        assert win.duration_label.text().startswith(f"median state {np.median(lengths):g} TRs")
        win.meta = {"tr": 2.0}
        win.show_k(int(win._ks[0]))
        lengths = np.diff(core.state_edges(win.gsbs_object.all_bounds[win._k], win.x.shape[0]))
        assert f"{np.median(lengths) * 2:.3g} s" in win.duration_label.text()


class TestStimulus:

    def test_transcript_drives_text_ticks_and_overlap(self, win, qapp, tmp_path):
        import gsbs_stimulus as stim
        win.load_demo(fitted=True)
        win.meta = {"tr": 1.5}
        n = win.x.shape[0]
        onsets = np.arange(0, n * 1.5 - 10, 2.0)
        words = [f"w{i}." if i % 5 == 4 else f"w{i}" for i in range(len(onsets))]
        t = stim.Transcript(words, onsets, onsets + 0.5, np.ones(len(words), bool),
                            np.array([w.endswith(".") for w in words]), "test")
        win._attach_transcript(t)
        assert win.stimulus_dock.isVisible() or win.stimulus.has_content
        sb = win.stimulus.stimulus_bounds()
        assert len(sb) > 0 and "O = " in win.stimulus.overlap_label.text()
        assert "stim_ticks" in win._artists
        win._select_state(3)
        assert "background" in win.stimulus.text.toHtml() or win.stimulus.sel == 3

    def test_overlap_window_pairs(self, win, qapp):
        win.load_demo(fitted=True)
        win.meta = {"tr": 1.5}
        win._attach_events(np.array([30.0, 90.0, 150.0, 210.0]))
        win.open_overlap()
        ow = win.overlap_window
        keys = [ow.ref_combo.itemData(i) for i in range(ow.ref_combo.count())]
        assert ("shown", None) in keys and ("stimulus", None) in keys
        ow.ref_combo.setCurrentIndex(keys.index(("stimulus", None)))
        ow.test_combo.setCurrentIndex(keys.index(("shown", None)))
        ow.perm_spin.setValue(200)
        ow.compare()
        assert "OA =" in ow.result_label.text()
        ow.close()


class TestOverlapMap:

    def test_grid_finds_planted_delay(self, win, qapp, tmp_path):
        import json

        import gsbs_nifti as nifti
        import gsbs_stimulus as stim
        rng = np.random.default_rng(2)
        n_time, n_rois, tr = 300, 100, 2.0
        counts = np.full(n_rois, 3)
        np.save(tmp_path / "voxels.npy", rng.standard_normal((counts.sum(), n_time)).astype(np.float32))
        np.save(tmp_path / "voxel_labels.npy", np.repeat(np.arange(1, n_rois + 1), counts))
        np.save(tmp_path / "parcels.npy", rng.standard_normal((n_time, n_rois)))
        (tmp_path / "results").mkdir()
        (tmp_path / "session.json").write_text(json.dumps({
            "source": "synthetic.nii.gz", "shape": [1, 1, 1, n_time], "tr": tr,
            "atlas": {"atlas": "schaefer", "n_rois": n_rois, "yeo_networks": 7},
            "voxels_per_parcel": counts.tolist(), "missing_parcels": []}))
        s = nifti.Session(tmp_path)
        events = np.sort(rng.choice(np.arange(10, 560), 40, replace=False)).astype(float)
        s.attach_events(events)
        planted = stim.to_trs(events, tr, n_time, 6.0)                 # follows the events at 6 s
        for roi, bounds in ((1, planted), (2, np.sort(rng.choice(np.arange(1, n_time), 40, replace=False)))):
            s.add_summary(roi, nifti.TargetSummary(nstates=len(bounds) + 1, n_time=n_time, fit_seconds=1,
                                                   engine="fast_GSBS", bounds=[int(b) for b in bounds]))
        win.open_session(str(tmp_path))
        b = win.brain
        b.refresh_overlap_sources()
        assert b.ovl_ref.currentData()["kind"] == "stimulus"
        b.ovl_delay.setCurrentIndex(1)                                   # the delay grid
        b.compute_overlap()
        deadline = time.monotonic() + 120
        while b._ovl_thread is not None and time.monotonic() < deadline:
            _pump(qapp, 0.05)
        o = b._ovl
        assert o["grid"] and o["rois"].tolist() == [1, 2]
        assert o["best_delay"][0] == 6.0 and o["or"][0] > 0.9 and o["p"][0] < 0.01
        assert o["p"][1] > 0.05
        b.ovl_show.setCurrentIndex(1)                                    # thresholded OR
        vals, *_ = b._ovl_values()
        assert not np.isnan(vals[0]) and np.isnan(vals[1])
        b._render_overlap()
        net = b.info.networks[0]                                         # ROI 1's network
        same = [r for r in (1, 2) if b.info.networks[r - 1] == net]
        assert b.ovl_bars._counts[net] == f"1/{len(same)}"               # significant / fitted
        row = dict((n, v) for n, v, _ in b.ovl_bars._rows)
        assert np.isclose(row[net], np.mean(o["or"][[i for i, r in enumerate(o["rois"]) if r in same]]))


class TestReadyDemoMenu:

    def test_opens_session_directly(self, win, qapp, monkeypatch):
        import gsbs_gui
        import gsbs_nifti
        ex = gsbs_nifti.EXAMPLE_BY_KEY["narratives-sherlock"]
        if not (gsbs_nifti.example_cache_dir() / ex.url.rsplit("/", 1)[1]).exists():
            pytest.skip("example NIfTI not downloaded on this machine")
        monkeypatch.setattr(gsbs_gui.ImportDialog, "exec",
                            lambda self: pytest.fail("the ready-made demo needs no import dialog"))
        win.load_online_example(gsbs_nifti.READY_DEMO.key)
        deadline = time.monotonic() + 120
        while win.session is None and time.monotonic() < deadline:
            _pump(qapp, 0.1)
        assert win.session is not None and win.gsbs_object is not None
        assert win.stimulus.has_content and len(win.session.summary()) == 13
