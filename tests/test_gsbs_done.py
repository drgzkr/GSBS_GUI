"""Tests for the _gsbs_done and _gsbs_error callbacks (called from thread)."""

import unittest.mock as mock

import numpy as np
import pytest

from tests.fixtures import app, make_fake_gsbs  # noqa: F401


class TestGSBSDone:

    def test_sets_gsbs_object(self, app):
        obj = make_fake_gsbs()
        with mock.patch.object(app, "_refresh_all_plots"):
            app._gsbs_done(obj)
        assert app.gsbs_object is obj

    def test_re_enables_run_button(self, app):
        obj = make_fake_gsbs()
        with mock.patch.object(app, "_refresh_all_plots"):
            app._gsbs_done(obj)
        app.run_btn.config.assert_called_with(state="normal")

    def test_stops_progress_bar(self, app):
        obj = make_fake_gsbs()
        with mock.patch.object(app, "_refresh_all_plots"):
            app._gsbs_done(obj)
        app.progress.stop.assert_called()
        app.progress.grid_remove.assert_called()

    def test_calls_refresh_with_best_k(self, app):
        obj = make_fake_gsbs(k=3)
        # tdists: best_k = 3 (argmax)
        assert int(np.argmax(obj.tdists)) == 3

        with mock.patch.object(app, "_refresh_all_plots") as mock_plot:
            app._gsbs_done(obj)

        mock_plot.assert_called_once_with(3)

    def test_configures_slider_range(self, app):
        obj = make_fake_gsbs(k=4)   # tdists has 5 entries → kmax=4
        with mock.patch.object(app, "_refresh_all_plots"):
            app._gsbs_done(obj)
        app.k_slider.config.assert_called_with(from_=1, to=4)


class TestGSBSError:

    def test_re_enables_run_button(self, app):
        with mock.patch("run_gsbs_gui.messagebox"):
            app._gsbs_error("something went wrong")
        app.run_btn.config.assert_called_with(state="normal")

    def test_stops_progress_bar(self, app):
        with mock.patch("run_gsbs_gui.messagebox"):
            app._gsbs_error("oops")
        app.progress.stop.assert_called()
        app.progress.grid_remove.assert_called()

    def test_shows_error_dialog(self, app):
        with mock.patch("run_gsbs_gui.messagebox") as mb:
            app._gsbs_error("GSBS exploded")
        mb.showerror.assert_called_once()
        assert "GSBS exploded" in mb.showerror.call_args[0][1]

    def test_does_not_set_gsbs_object(self, app):
        app.gsbs_object = None
        with mock.patch("run_gsbs_gui.messagebox"):
            app._gsbs_error("fail")
        assert app.gsbs_object is None
