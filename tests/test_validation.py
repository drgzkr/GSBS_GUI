"""Tests for input validation in load_roi_data and load_gsbs_object."""

import types
import unittest.mock as mock

import numpy as np
import pytest

from tests.fixtures import app  # noqa: F401


def _set_path(app, path: str):
    """Configure the mocked data_path_var to return a specific path."""
    app.data_path_var = mock.MagicMock()
    app.data_path_var.get.return_value = path


class TestLoadROIData:

    def test_accepts_valid_2d_array(self, app):
        data = np.random.rand(30, 10)
        _set_path(app, "/fake/data.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=data), \
             mock.patch("run_gsbs_gui.messagebox"):
            app.load_roi_data()
        assert app.roi_data is not None
        np.testing.assert_array_equal(app.roi_data, data)

    def test_rejects_1d_array(self, app):
        data = np.random.rand(30)
        _set_path(app, "/fake/data.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=data), \
             mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_roi_data()
        assert app.roi_data is None
        mb.showerror.assert_called_once()
        assert "Expected 2D" in mb.showerror.call_args[0][1]

    def test_rejects_3d_array(self, app):
        data = np.random.rand(10, 5, 3)
        _set_path(app, "/fake/data.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=data), \
             mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_roi_data()
        assert app.roi_data is None
        mb.showerror.assert_called_once()

    def test_empty_path_shows_warning(self, app):
        _set_path(app, "   ")
        with mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_roi_data()
        mb.showwarning.assert_called_once()
        assert app.roi_data is None

    def test_bad_file_shows_error(self, app):
        _set_path(app, "/fake/missing.npy")
        with mock.patch("run_gsbs_gui.np.load", side_effect=FileNotFoundError("not found")), \
             mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_roi_data()
        mb.showerror.assert_called_once()
        assert app.roi_data is None


class TestLoadGSBSObject:

    def _make_load_result(self, obj):
        """np.load(...).item() chain."""
        wrapped = mock.MagicMock()
        wrapped.item.return_value = obj
        return wrapped

    def test_accepts_valid_gsbs_object(self, app):
        obj = types.SimpleNamespace()
        obj.tdists = np.array([0.0, 0.5, 0.9, 0.7])
        obj.all_bounds = {3: np.zeros(20)}
        obj.x = np.random.rand(20, 5)
        obj.get_state_patterns = lambda k: np.random.rand(k, 5)

        _set_path(app, "/fake/gsbs.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=self._make_load_result(obj)), \
             mock.patch("run_gsbs_gui.messagebox"), \
             mock.patch.object(app, "_refresh_all_plots"):
            app.load_gsbs_object()

        assert app.gsbs_object is obj

    def test_rejects_object_missing_tdists(self, app):
        obj = types.SimpleNamespace()
        obj.all_bounds = {}      # no tdists

        _set_path(app, "/fake/gsbs.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=self._make_load_result(obj)), \
             mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_gsbs_object()

        assert app.gsbs_object is None
        mb.showerror.assert_called_once()

    def test_rejects_object_missing_all_bounds(self, app):
        obj = types.SimpleNamespace()
        obj.tdists = np.array([0.0, 1.0])  # no all_bounds

        _set_path(app, "/fake/gsbs.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=self._make_load_result(obj)), \
             mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_gsbs_object()

        assert app.gsbs_object is None
        mb.showerror.assert_called_once()

    def test_empty_path_shows_warning(self, app):
        _set_path(app, "")
        with mock.patch("run_gsbs_gui.messagebox") as mb:
            app.load_gsbs_object()
        mb.showwarning.assert_called_once()

    def test_sets_best_k_from_tdists(self, app):
        obj = types.SimpleNamespace()
        obj.tdists = np.array([0.0, 0.3, 0.9, 0.5])   # argmax = 2
        obj.all_bounds = {2: np.zeros(20)}
        obj.x = np.random.rand(20, 5)
        obj.get_state_patterns = lambda k: np.random.rand(k, 5)

        _set_path(app, "/fake/gsbs.npy")
        with mock.patch("run_gsbs_gui.np.load", return_value=self._make_load_result(obj)), \
             mock.patch("run_gsbs_gui.messagebox"), \
             mock.patch.object(app, "_refresh_all_plots") as mock_plot:
            app.load_gsbs_object()

        mock_plot.assert_called_once_with(2)
