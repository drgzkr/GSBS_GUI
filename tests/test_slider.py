"""Tests for slider state and _on_slider callback."""

import unittest.mock as mock

import numpy as np
import pytest

from tests.fixtures import app, make_fake_gsbs  # noqa: F401


class TestOnSlider:

    def test_noop_when_no_gsbs_object(self, app):
        app.gsbs_object = None
        with mock.patch.object(app, "_refresh_all_plots") as mock_plot:
            app._on_slider()
        mock_plot.assert_not_called()

    def test_calls_refresh_with_current_k(self, app):
        app.gsbs_object = make_fake_gsbs(k=3)
        app.k_var = mock.MagicMock()
        app.k_var.get.return_value = 3.0   # Scale returns float

        with mock.patch.object(app, "_refresh_all_plots") as mock_plot:
            app._on_slider()

        mock_plot.assert_called_once_with(3)

    def test_rounds_float_k(self, app):
        """Scale can produce floats like 2.6 — must round to nearest int."""
        app.gsbs_object = make_fake_gsbs(k=3)
        app.k_var = mock.MagicMock()
        app.k_var.get.return_value = 2.6

        with mock.patch.object(app, "_refresh_all_plots") as mock_plot:
            app._on_slider()

        mock_plot.assert_called_once_with(3)

    def test_updates_k_label(self, app):
        app.gsbs_object = make_fake_gsbs(k=2)
        app.k_var = mock.MagicMock()
        app.k_var.get.return_value = 2.0

        with mock.patch.object(app, "_refresh_all_plots"):
            app._on_slider()

        app.k_label.config.assert_called_with(text="k = 2")


class TestConfigureSlider:

    def test_sets_slider_range(self, app):
        app._configure_slider(kmax=10, value=5)
        app.k_slider.config.assert_called_with(from_=1, to=10)

    def test_sets_slider_value(self, app):
        app._configure_slider(kmax=10, value=5)
        app.k_var.set.assert_called_with(5)

    def test_updates_label(self, app):
        app._configure_slider(kmax=10, value=5)
        app.k_label.config.assert_called_with(text="k = 5")
