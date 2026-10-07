"""
GSBS GUI — Qt-free core.

Everything the GUI needs that does not touch widgets: reading GSBS results
correctly, loading and saving, memory estimates, and the child-process fit
runner that streams live progress back to the GUI.
"""

from __future__ import annotations

import copy
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_DATA = DATA_DIR / "demo_roi_data.npy"
DEMO_OBJECT = DATA_DIR / "demo_GSBS_object.npy"
DEMO_TRUTH = DATA_DIR / "demo_true_bounds.npy"

ENGINES = ("fast_GSBS", "GSBS")


# ---------------------------------------------------------------------------
# Reading a GSBS result
# ---------------------------------------------------------------------------
#
# GSBS stores one solution per index k of `all_bounds` and `tdists`, where k
# is the number of states. Indices 0 and 1 are never filled. With statewise
# detection, an iteration can add two boundaries at once; the index it skips
# over (k) then holds the (k+1)-state solution and a t-distance copied from
# k-1. Only indices whose boundary row really has k states are solutions.

def solution_ks(obj) -> np.ndarray:
    """Every k for which the fit produced a genuine k-state solution."""
    return _solution_ks(obj.all_bounds)


def _solution_ks(all_bounds: np.ndarray) -> np.ndarray:
    n_states = (np.asarray(all_bounds) > 0).sum(axis=1) + 1
    ks = np.arange(len(n_states))
    return ks[(ks >= 2) & (n_states == ks)]


def best_k(obj) -> int:
    return int(np.argmax(obj.tdists))


def best_at_limit(obj) -> bool:
    """The best solution is at the largest k searched: the t-distance was still rising,
    so the fit found no optimum and its 'best' is set by kmax."""
    return int(np.max(solution_ks(obj))) <= best_k(obj)


def boundary_indices(bounds_row: np.ndarray) -> np.ndarray:
    """TR indices at which a new state starts."""
    return np.flatnonzero(np.asarray(bounds_row) > 0)


def state_edges(bounds_row: np.ndarray, n_time: int) -> np.ndarray:
    """[0, b1, b2, ..., n_time]: state i spans edges[i]:edges[i+1]."""
    return np.concatenate(([0], boundary_indices(bounds_row), [n_time]))


def state_timeseries(x: np.ndarray, bounds_row: np.ndarray) -> np.ndarray:
    """Each timepoint replaced by the mean pattern of the state it is in."""
    edges = state_edges(bounds_row, x.shape[0])
    out = np.empty_like(x, dtype=float)
    for a, b in zip(edges[:-1], edges[1:]):
        out[a:b] = x[a:b].mean(axis=0)
    return out


def fit_params(obj) -> dict:
    """The parameters an object was fitted with (older versions lack some)."""
    return {
        "kmax": int(obj.kmax),
        "statewise_detection": bool(getattr(obj, "statewise_detection", True)),
        "finetune": int(getattr(obj, "finetune", 1)),
        "finetune_order": bool(getattr(obj, "finetune_order", True)),
        "dmin": int(getattr(obj, "dmin", 1)),
        "blocksize": int(getattr(obj, "blocksize", 50)),
    }


# ---------------------------------------------------------------------------
# Loading and saving
# ---------------------------------------------------------------------------

@dataclass
class Loaded:
    x: np.ndarray
    obj: object | None = None          # a fitted GSBS object, if the file held one


def is_fitted_gsbs(obj) -> bool:
    return (hasattr(obj, "all_bounds") and hasattr(obj, "x")
            and getattr(obj, "_argmax", None) is not None)


def load_file(path: str | os.PathLike) -> Loaded:
    """Load raw ROI data (timepoints x voxels) or a fitted GSBS object."""
    raw = np.load(path, allow_pickle=True)
    if raw.dtype == object and raw.ndim == 0:
        obj = raw.item()
        if is_fitted_gsbs(obj):
            return Loaded(x=np.asarray(obj.x), obj=obj)
        if hasattr(obj, "all_bounds"):
            raise ValueError("This GSBS object has not been fitted yet.")
        raise ValueError(f"File holds a {type(obj).__name__}, not a GSBS object or array.")
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D (timepoints x voxels) array, got shape {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"Expected numeric data, got dtype {raw.dtype}.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Data contains NaN or infinite values.")
    if raw.shape[0] < 4:
        raise ValueError(f"Only {raw.shape[0]} timepoints; GSBS needs more.")
    return Loaded(x=raw.astype(float, copy=False))


def save_object(obj, path: str | os.PathLike) -> None:
    """Save a fitted object as a plain `GSBS`, so it loads without this fork.

    fast_GSBS adds no state of its own, so the pickled data is identical.
    """
    from statesegmentation import GSBS
    portable = copy.copy(obj)
    portable.__class__ = GSBS
    np.save(path, portable, allow_pickle=True)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

# Peak memory of fast_GSBS.fit() grows with T^2 (time-by-time covariance and
# its upper-triangle indices). Measured on fast_GSBS 0.1.0 at T = 1000-3000.
_BYTES_PER_T2 = {False: 30.0, True: 46.0}


def estimate_fit_bytes(n_time: int, n_vox: int, kmax: int, statewise: bool) -> int:
    fit = _BYTES_PER_T2[bool(statewise)] * n_time**2
    result = 8 * (kmax + 2) * n_time + 3 * 8 * n_time * n_vox
    gui = 8 * n_time**2                  # the correlation matrix the GUI draws
    return int(fit + result + gui)


def available_bytes() -> int | None:
    """Memory available to new processes, or None if it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return ""


# ---------------------------------------------------------------------------
# Child-process fit
# ---------------------------------------------------------------------------
#
# The fit runs in a separate process so it can be cancelled by killing it
# (numpy cannot be interrupted safely inside a thread) and so a crash or
# out-of-memory kill cannot take the GUI down. Progress comes from replacing
# the `tqdm` that statesegmentation's fit() loop calls `update()` on: after
# each update, all_bounds and tdists for every finished k are filled in, so
# the parent can draw the curve and boundaries while the fit runs.
#
# Messages on the queue:
#   ("progress", done, total, ks, tdists[ks], bounds_row)   ks = solutions so far
#   ("done", fitted_object)
#   ("error", traceback_text)

PROGRESS_INTERVAL_S = 0.2


class _QueueProgress:
    """Stands in for tqdm inside GSBS.fit()."""

    def __init__(self, obj, queue, total=None, **_):
        self._obj, self._queue = obj, queue
        self.total, self.n = int(total or 0), 0
        self._last_sent = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        self.n += n
        now = time.monotonic()
        if now - self._last_sent >= PROGRESS_INTERVAL_S or self.n >= self.total:
            self._last_sent = now
            last = 1 + self.n                       # highest k filled so far
            ks = _solution_ks(self._obj.all_bounds[: last + 1])
            self._queue.put(("progress", self.n, self.total, ks,
                             np.array(self._obj._tdists[ks]),
                             np.array(self._obj._bounds)))


def run_fit(queue, engine: str, x: np.ndarray, params: dict) -> None:
    """Child-process entry point."""
    try:
        import statesegmentation.fast_gsbs as fast_mod
        import statesegmentation.gsbs as base_mod

        cls = fast_mod.fast_GSBS if engine == "fast_GSBS" else base_mod.GSBS
        obj = cls(x=x, **params)

        def factory(*args, **kwargs):
            return _QueueProgress(obj, queue, **kwargs)

        fast_mod.tqdm = base_mod.tqdm = factory
        obj.fit(showProgressBar=True)
        queue.put(("done", obj))
    except BaseException:                           # noqa: BLE001 — report everything
        queue.put(("error", traceback.format_exc()))
