"""NIfTI import, sessions, and batch runs on a small synthetic 4D image."""

import time

import numpy as np
import pytest

import gsbs_nifti as nifti

N_TIME = 40


class _ListQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


@pytest.fixture(scope="module")
def bold(tmp_path_factory):
    """A 4 mm MNI-space 4D image: random voxels plus a strong linear drift and an empty voxel."""
    import nibabel as nib
    from nilearn.datasets import load_mni152_template
    ref = load_mni152_template(resolution=4)
    rng = np.random.default_rng(0)
    shape = ref.shape[:3] + (N_TIME,)
    data = rng.standard_normal(shape).astype(np.float32)
    data += 5 * np.linspace(-1, 1, N_TIME, dtype=np.float32)
    data[ref.shape[0] // 2, ref.shape[1] // 2, ref.shape[2] // 2] = 0     # constant voxel
    img = nib.Nifti1Image(data, ref.affine)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 2.0
    path = tmp_path_factory.mktemp("nifti") / "sub-01_bold.nii.gz"
    nib.save(img, path)
    return path


@pytest.fixture(scope="module")
def session(bold, tmp_path_factory):
    out = tmp_path_factory.mktemp("sess") / "sub-01_gsbs"
    q = _ListQueue()
    nifti.run_import(q, nifti.ImportParams(nifti=str(bold), session_dir=str(out),
                                           n_rois=100, yeo_networks=7))
    assert q.items[-1][0] == "done", q.items[-1]
    return nifti.Session(out)


class TestImport:

    def test_peek_reads_header(self, bold):
        info = nifti.peek_nifti(bold)
        assert info["shape"][3] == N_TIME and info["tr"] == 2.0

    def test_default_session_dir(self, bold):
        assert nifti.default_session_dir(bold).name == "sub-01_bold_gsbs"

    def test_shapes_and_metadata(self, session):
        assert session.parcels().shape == (N_TIME, 100)
        assert session.tr == 2.0
        assert session.spec.n_rois == 100 and session.spec.yeo_networks == 7
        assert session.voxel_counts.sum() == np.load(session.path / "voxels.npy").shape[0]

    def test_roi_voxels_are_zscored_and_contiguous(self, session):
        x = session.roi_voxels(10)
        assert x.shape == (N_TIME, session.voxel_counts[9])
        np.testing.assert_allclose(x.mean(0), 0, atol=1e-5)
        np.testing.assert_allclose(x.std(0), 1, atol=1e-4)
        labels = np.load(session.path / "voxel_labels.npy")
        assert np.all(np.diff(labels) >= 0)

    def test_parcels_are_mean_of_zscored_voxels(self, session):
        for roi in (1, 50, 100):
            np.testing.assert_allclose(session.parcels()[:, roi - 1],
                                       session.roi_voxels(roi).mean(1), atol=1e-5)

    def test_detrend_removes_drift(self, bold, tmp_path):
        q = _ListQueue()
        nifti.run_import(q, nifti.ImportParams(nifti=str(bold), session_dir=str(tmp_path / "d"),
                                               n_rois=100, yeo_networks=7, detrend=True))
        drifted = abs(np.corrcoef(np.arange(N_TIME), nifti.Session(tmp_path / "d").parcels()[:, 0])[0, 1])
        assert drifted < 0.2

    def test_bad_input_reports_error(self, tmp_path):
        q = _ListQueue()
        nifti.run_import(q, nifti.ImportParams(nifti=str(tmp_path / "missing.nii.gz"),
                                               session_dir=str(tmp_path / "x")))
        assert q.items[-1][0] == "error"


class TestResults:

    def test_record_and_reload(self, session):
        from statesegmentation import fast_GSBS
        x = session.roi_voxels(3)
        obj = fast_GSBS(kmax=10, x=x)
        obj.fit(showProgressBar=False)
        s = session.record(3, obj, 1.5, "fast_GSBS")
        assert s.nstates == obj.nstates
        back = session.load_result(3)
        assert back.gsbs_gui_meta["roi"] == 3 and back.gsbs_gui_meta["kind"] == "roi_voxels"
        fresh = nifti.Session(session.path)              # summary persisted on disk
        assert fresh.summary()[3].nstates == obj.nstates


class TestBatch:

    def test_runs_pauses_and_reports(self, session):
        import gsbs_batch as batch
        targets = [5, 6, 7, 8]
        r = batch.BatchRunner(session.path, targets, "fast_GSBS",
                              dict(kmax=10, statewise_detection=True, finetune=1), n_workers=2)
        r.start()
        finished = {}
        deadline = time.monotonic() + 120
        while r.active and time.monotonic() < deadline:
            for kind, target, payload in r.poll().events:
                assert kind == "finished", payload
                finished[target] = payload
            time.sleep(0.05)
        assert sorted(finished) == targets
        for t in targets:
            assert session.result_path(t).exists()
            assert finished[t]["nstates"] == session.load_result(t).nstates

    def test_cancel_stops_workers(self, session):
        import gsbs_batch as batch
        r = batch.BatchRunner(session.path, list(range(20, 40)), "GSBS", dict(kmax=15), n_workers=2)
        r.start()
        time.sleep(1.5)
        r.poll()
        r.cancel()
        assert not r.active

    def test_suggest_workers_respects_cpus(self):
        import os

        import gsbs_batch as batch
        n, why = batch.suggest_workers(200, 100, 100, True)
        assert 1 <= n <= max(1, (os.cpu_count() or 2) - 1) and why


class TestOnlineExample:

    def test_fetch_example(self):
        try:
            files = nifti.fetch_example()
        except Exception as exc:                               # noqa: BLE001
            pytest.skip(f"example unavailable offline: {exc}")
        info = nifti.peek_nifti(files["nifti"])
        assert info["shape"][3] == 168
        assert not info["tr_units_known"]                     # why the TR is set explicitly
        assert files["tr"] == 2.0
        assert files["confounds"].endswith(".tsv")


class TestExamples:

    def test_tr_written_in_ms_but_labelled_seconds(self, tmp_path):
        import nibabel as nib
        img = nib.Nifti1Image(np.zeros((4, 4, 4, 10), np.float32), np.eye(4))
        img.header.set_xyzt_units("mm", "sec")
        img.header["pixdim"][4] = 1500.0
        nib.save(img, tmp_path / "a.nii.gz")
        info = nifti.peek_nifti(tmp_path / "a.nii.gz")
        assert info["tr"] == 1.5 and "milliseconds" in info["tr_note"]

    def test_catalogue(self):
        keys = [e.key for e in nifti.EXAMPLES]
        assert keys[0] == "pixar" and len(set(keys)) == len(keys)
        for e in nifti.EXAMPLES:
            assert e.tr > 0 and e.volumes > 100
            if e.url:
                assert e.url.startswith("https://") and e.url.endswith("desc-clean_bold.nii.gz")

    @pytest.fixture
    def local_example(self, tmp_path, monkeypatch, bold):
        import dataclasses
        ex = dataclasses.replace(nifti.EXAMPLE_BY_KEY["narratives-sherlock"],
                                 url=bold.as_uri())
        monkeypatch.setitem(nifti.EXAMPLE_BY_KEY, ex.key, ex)
        monkeypatch.setattr(nifti, "example_cache_dir", lambda: tmp_path / "cache")
        return ex

    def test_download_and_cache(self, local_example, tmp_path):
        seen = []
        files = nifti.fetch_example(local_example.key, progress=lambda d, t: seen.append((d, t)))
        assert files["confounds"] is None and files["tr"] == 1.5
        assert nifti.peek_nifti(files["nifti"])["shape"][3] == N_TIME
        assert seen and seen[-1][0] == seen[-1][1]
        seen.clear()
        nifti.fetch_example(local_example.key, progress=lambda d, t: seen.append(d))
        assert not seen                                          # cached: no second download

    def test_cancel_leaves_nothing(self, local_example, tmp_path):
        def stop(done, total):
            raise nifti.DownloadCancelled()
        with pytest.raises(nifti.DownloadCancelled):
            nifti.fetch_example(local_example.key, progress=stop)
        assert not list((tmp_path / "cache").glob("*"))


class TestReadyDemo:

    def test_package_is_complete(self):
        import gzip
        import io
        import json

        import gsbs_core as core
        import gsbs_stimulus as stim
        d = nifti.READY_DEMO_DIR
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "results" / "summary.json").read_text())
        assert set(manifest["fitted"]) == set(summary)
        for key in manifest["fitted"]:
            name = "whole_brain" if key == "whole_brain" else f"roi_{int(key):04d}"
            with gzip.open(d / "results" / f"{name}.npy.gz", "rb") as f:
                obj = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
            assert obj.x.shape[0] == manifest["n_time"]
            assert core.best_k(obj) == summary[key]["nstates"]
        t = stim.Transcript.load(d / "transcript.tsv")
        assert len(t) > 2000 and t.sentence_end.sum() > 100

    def test_install_when_nifti_cached(self):
        ex = nifti.EXAMPLE_BY_KEY["narratives-sherlock"]
        if not (nifti.example_cache_dir() / ex.url.rsplit("/", 1)[1]).exists():
            pytest.skip("example NIfTI not downloaded on this machine")
        s = nifti.Session(nifti.install_demo_session())
        assert len(s.summary()) == 13 and s.transcript() is not None
        wb = s.load_result(nifti.WHOLE_BRAIN)
        assert np.allclose(wb.x, s.parcels(), atol=1e-5)
