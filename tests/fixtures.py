"""Shared test helpers and fixtures."""

import types
import unittest.mock as mock

import numpy as np
import pytest


def make_fake_gsbs(n_time: int = 20, n_channels: int = 5,
                   k: int = 2, boundary_at: int = 10):
    """Minimal GSBS-like object for testing."""
    rng = np.random.default_rng(42)
    obj = types.SimpleNamespace()
    obj.x = rng.random((n_time, n_channels))

    bounds_arr = np.zeros(n_time)
    if k > 1:
        bounds_arr[boundary_at] = 1.0
    obj.all_bounds = {k: bounds_arr}

    patterns = rng.random((k, n_channels))
    obj.get_state_patterns = lambda k: patterns

    # tdists: best_k = k (argmax of the array)
    obj.tdists = np.zeros(k + 1)
    obj.tdists[k] = 1.0

    return obj


@pytest.fixture
def app():
    """GSBSApp instantiated against a fully mocked Tk root."""
    import matplotlib.pyplot as plt
    import run_gsbs_gui  # noqa: PLC0415 — imported here so conftest patches land first

    root = mock.MagicMock()
    instance = run_gsbs_gui.GSBSApp(root)
    yield instance
    plt.close("all")
