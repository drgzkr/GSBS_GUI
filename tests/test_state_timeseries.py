"""Tests for GSBSApp._state_timeseries — pure numpy logic."""

import numpy as np
import pytest

from tests.fixtures import app, make_fake_gsbs  # noqa: F401 — fixtures used by pytest


class TestStateTimeseries:

    def test_output_shape(self, app):
        obj = make_fake_gsbs(n_time=20, n_channels=5, k=2, boundary_at=10)
        app.gsbs_object = obj
        result = app._state_timeseries(k=2)
        assert result.shape == (20, 5)

    def test_each_segment_is_constant(self, app):
        """Every row within a state window must equal that state's pattern."""
        n_time, n_channels, k = 20, 5, 2
        obj = make_fake_gsbs(n_time=n_time, n_channels=n_channels,
                             k=k, boundary_at=10)
        app.gsbs_object = obj
        patterns = obj.get_state_patterns(k=k)
        result = app._state_timeseries(k=k)

        # First state: rows 0..9 should all equal patterns[0]
        for t in range(10):
            np.testing.assert_array_equal(result[t], patterns[0])

        # Second state: rows 10..19 should all equal patterns[1]
        for t in range(10, n_time):
            np.testing.assert_array_equal(result[t], patterns[1])

    def test_single_state_no_boundaries(self, app):
        """k=1 means no boundaries; entire timeseries gets patterns[0]."""
        n_time, n_channels = 15, 4
        rng = np.random.default_rng(0)
        import types
        obj = types.SimpleNamespace()
        obj.x = rng.random((n_time, n_channels))
        obj.all_bounds = {1: np.zeros(n_time)}
        pattern = rng.random((1, n_channels))
        obj.get_state_patterns = lambda k: pattern

        app.gsbs_object = obj
        result = app._state_timeseries(k=1)

        assert result.shape == (n_time, n_channels)
        for t in range(n_time):
            np.testing.assert_array_equal(result[t], pattern[0])

    def test_multiple_boundaries(self, app):
        """k=4, three boundaries — verify each of the four segments."""
        import types
        n_time, n_channels, k = 40, 3, 4
        rng = np.random.default_rng(7)
        obj = types.SimpleNamespace()
        obj.x = rng.random((n_time, n_channels))

        bounds_arr = np.zeros(n_time)
        for b in [10, 20, 30]:
            bounds_arr[b] = 1.0
        obj.all_bounds = {k: bounds_arr}

        patterns = rng.random((k, n_channels))
        obj.get_state_patterns = lambda k: patterns

        app.gsbs_object = obj
        result = app._state_timeseries(k=k)

        segments = [(0, 10), (10, 20), (20, 30), (30, 40)]
        for idx, (start, end) in enumerate(segments):
            for t in range(start, end):
                np.testing.assert_array_equal(result[t], patterns[idx],
                    err_msg=f"Mismatch at t={t} (segment {idx})")

    def test_output_dtype_matches_input(self, app):
        obj = make_fake_gsbs(n_time=20, n_channels=5, k=2)
        app.gsbs_object = obj
        result = app._state_timeseries(k=2)
        assert result.dtype == obj.x.dtype

    def test_full_coverage_no_unset_rows(self, app):
        """No row in the output should be left at zeros from np.empty."""
        obj = make_fake_gsbs(n_time=20, n_channels=5, k=2, boundary_at=10)
        app.gsbs_object = obj
        result = app._state_timeseries(k=2)
        # Every row was written — none can be all-zero (patterns are random > 0)
        for t in range(20):
            assert not np.all(result[t] == 0)
