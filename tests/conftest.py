"""
Headless test setup.

Patches tkinter, the TkAgg backend, and statesegmentation at the
sys.modules level *before* run_gsbs_gui is imported anywhere, so tests
run without a display server or the actual statesegmentation package.

Must live at module scope (not inside a fixture) to fire during
pytest's collection phase.
"""

import sys
import unittest.mock as mock

import matplotlib
matplotlib.use("Agg")

# Prevent the app's `matplotlib.use("TkAgg")` from switching away from Agg.
mock.patch("matplotlib.use", return_value=None).start()

_tk_mock = mock.MagicMock()
for _mod in (
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "matplotlib.backends.backend_tkagg",
    "statesegmentation",
    "statesegmentation.gsbs",
):
    sys.modules.setdefault(_mod, _tk_mock)
