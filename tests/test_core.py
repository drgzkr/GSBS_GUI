"""Tests for gsbs_core: reading results, I/O, memory, and the child fit runner."""

import numpy as np
import pytest

import gsbs_core as core


def _demo():
    return core.load_file(core.DEMO_OBJECT).obj


class TestSolutionKs:

    def test_skips_statewise_duplicate_indices(self):
        # 6 timepoints; index 2 duplicates the 3-state solution at index 3
        all_bounds = np.zeros((6, 6), int)
        all_bounds[2, [2, 4]] = 3
        all_bounds[3, [2, 4]] = 3
        all_bounds[4, [1, 2, 4]] = [4, 3, 3]
        obj = type("O", (), {"all_bounds": all_bounds})()
        assert core.solution_ks(obj).tolist() == [3, 4]

    def test_every_solution_has_k_states(self):
        obj = _demo()
        for k in core.solution_ks(obj):
            assert len(core.boundary_indices(obj.all_bounds[k])) + 1 == k

    def test_excludes_unfilled_and_placeholder_indices(self):
        obj = _demo()
        ks = core.solution_ks(obj)
        assert ks.min() >= 2
        assert ks.max() <= obj.kmax + 1
        assert len(obj.tdists) == obj.kmax + 2

    def test_best_k_is_a_solution(self):
        obj = _demo()
        assert core.best_k(obj) in core.solution_ks(obj)


class TestStateTimeseries:

    def test_each_state_is_its_mean(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        bounds = np.array([0, 0, 1, 0, 0, 2])
        st = core.state_timeseries(x, bounds)
        np.testing.assert_allclose(st[:2], x[:2].mean(0, keepdims=True).repeat(2, 0))
        np.testing.assert_allclose(st[2:5], x[2:5].mean(0, keepdims=True).repeat(3, 0))
        np.testing.assert_allclose(st[5], x[5])

    def test_edges(self):
        assert core.state_edges(np.array([0, 0, 1, 0, 2]), 5).tolist() == [0, 2, 4, 5]


class TestLoadSave:

    def test_loads_raw_array(self, tmp_path):
        p = tmp_path / "d.npy"
        np.save(p, np.random.default_rng(0).random((20, 5)))
        loaded = core.load_file(p)
        assert loaded.obj is None and loaded.x.shape == (20, 5)

    @pytest.mark.parametrize("arr, msg", [
        (np.zeros(10), "2D"),
        (np.zeros((2, 5)), "timepoints"),
        (np.array([[1.0, np.nan]] * 5), "NaN"),
    ])
    def test_rejects_bad_arrays(self, tmp_path, arr, msg):
        p = tmp_path / "bad.npy"
        np.save(p, arr)
        with pytest.raises(ValueError, match=msg):
            core.load_file(p)

    def test_rejects_unfitted_object(self, tmp_path):
        from statesegmentation import GSBS
        p = tmp_path / "u.npy"
        np.save(p, GSBS(kmax=3, x=np.random.default_rng(0).random((10, 4))), allow_pickle=True)
        with pytest.raises(ValueError, match="not been fitted"):
            core.load_file(p)

    def test_save_is_plain_gsbs_and_identical(self, tmp_path):
        from statesegmentation import GSBS, fast_GSBS
        obj = _demo()
        obj.__class__ = fast_GSBS
        p = tmp_path / "r.npy"
        core.save_object(obj, p)
        back = core.load_file(p).obj
        assert type(back) is GSBS
        assert type(obj) is fast_GSBS                      # original untouched
        np.testing.assert_array_equal(back.all_bounds, obj.all_bounds)
        np.testing.assert_array_equal(back.tdists, obj.tdists)

    def test_fit_params_read_from_object(self):
        p = core.fit_params(_demo())
        assert p["kmax"] == 150 and p["statewise_detection"] is True


class TestMemory:

    def test_grows_quadratically_with_time(self):
        small = core.estimate_fit_bytes(1000, 100, 10, True)
        big = core.estimate_fit_bytes(2000, 100, 10, True)
        assert 3.5 < big / small < 4.1

    def test_statewise_costs_more(self):
        assert (core.estimate_fit_bytes(1000, 100, 10, True)
                > core.estimate_fit_bytes(1000, 100, 10, False))

    def test_format(self):
        assert core.format_bytes(1536 * 1024**2) == "1.5 GB"


class _ListQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class TestRunFit:

    @pytest.mark.parametrize("engine", core.ENGINES)
    def test_matches_direct_fit_and_reports_progress(self, engine, monkeypatch):
        from statesegmentation import GSBS
        monkeypatch.setattr(core, "PROGRESS_INTERVAL_S", 0.0)
        x = np.load(core.DEMO_DATA)[:80]
        params = dict(kmax=20, statewise_detection=True, finetune=1)
        q = _ListQueue()
        core.run_fit(q, engine, x, params)

        kind, obj = q.items[-1]
        assert kind == "done"
        ref = GSBS(x=x, **params)
        ref.fit(showProgressBar=False)
        np.testing.assert_array_equal(obj.all_bounds, ref.all_bounds)
        np.testing.assert_array_equal(obj.tdists, ref.tdists)

        progress = [m for m in q.items if m[0] == "progress"]
        assert progress, "no progress messages"
        _, done, total, ks, tvals, bounds = progress[-1]
        assert done >= total == 19
        np.testing.assert_array_equal(ks, core.solution_ks(obj))
        np.testing.assert_array_equal(tvals, obj.tdists[ks])

    def test_reports_errors(self):
        q = _ListQueue()
        core.run_fit(q, "fast_GSBS", np.zeros((10, 3)), dict(kmax=3, nonsense=1))
        assert q.items[-1][0] == "error"
        assert "nonsense" in q.items[-1][1]
