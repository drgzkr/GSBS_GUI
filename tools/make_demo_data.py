#!/usr/bin/env python3
"""
Generate the bundled demo dataset: a synthetic ROI timeseries with known
neural-state boundaries, plus a pre-fitted GSBS object for it.

Writes to data/:
    demo_roi_data.npy      (timepoints x voxels) float64
    demo_true_bounds.npy   where each state change should appear in the BOLD signal:
                           the neural onset TR of every state after the first,
                           shifted by the HRF peak lag (the HRF smears and delays
                           every change by a few TRs, so GSBS cannot find the
                           neural onsets themselves)
    demo_GSBS_object.npy   fitted with fast_GSBS, saved as a plain GSBS object

Deterministic (fixed seed); re-running reproduces the same files.
"""

from pathlib import Path

import numpy as np
from scipy.stats import gamma

from statesegmentation import fast_GSBS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_TIME   = 300     # TRs
N_VOX    = 120
TR       = 1.5     # seconds, only used to sample the HRF
MEAN_DUR = 11      # mean state duration in TRs
MIN_DUR  = 4
RHO      = 0.35    # correlation between consecutive state patterns
NOISE_SD = 1.6
SEED     = 2026


def _hrf(tr: float, length_s: float = 24.0) -> np.ndarray:
    """SPM-style double-gamma HRF sampled at the TR, unit sum."""
    t = np.arange(0, length_s, tr)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return h / h.sum()


def make_demo(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    durations = []
    while sum(durations) < N_TIME:
        durations.append(MIN_DUR + rng.poisson(MEAN_DUR - MIN_DUR))
    durations[-1] -= sum(durations) - N_TIME
    if durations[-1] < MIN_DUR:          # fold a too-short tail into the previous state
        durations[-2] += durations.pop()
    onsets = np.cumsum([0] + durations[:-1])

    patterns = np.empty((len(durations), N_VOX))
    patterns[0] = rng.standard_normal(N_VOX)
    for i in range(1, len(durations)):
        patterns[i] = RHO * patterns[i - 1] + np.sqrt(1 - RHO**2) * rng.standard_normal(N_VOX)

    signal = np.repeat(patterns, durations, axis=0)
    hrf = _hrf(TR)
    padded = np.vstack([np.repeat(signal[:1], len(hrf), axis=0), signal])
    bold = np.apply_along_axis(lambda c: np.convolve(c, hrf)[: len(c)], 0, padded)[len(hrf):]
    bold /= bold.std(0, keepdims=True)

    x = bold + NOISE_SD * rng.standard_normal(bold.shape)
    lag = int(np.argmax(hrf))
    return x, onsets[1:] + lag


def main() -> None:
    x, true_bounds = make_demo()
    obj = fast_GSBS(kmax=N_TIME // 2, x=x, statewise_detection=True, finetune=1)
    obj.fit(showProgressBar=True)

    from statesegmentation import GSBS
    obj.__class__ = GSBS                 # loads with upstream statesegmentation too

    DATA_DIR.mkdir(exist_ok=True)
    np.save(DATA_DIR / "demo_roi_data.npy", x)
    np.save(DATA_DIR / "demo_true_bounds.npy", true_bounds)
    np.save(DATA_DIR / "demo_GSBS_object.npy", obj, allow_pickle=True)

    found = np.flatnonzero(obj.bounds)
    hits = sum(np.min(np.abs(found - b)) <= 1 for b in true_bounds)
    print(f"{N_TIME} TRs x {N_VOX} voxels, {len(true_bounds) + 1} true states")
    print(f"GSBS best: {obj.nstates} states; {hits}/{len(true_bounds)} true boundaries (BOLD time) found within 1 TR")


if __name__ == "__main__":
    main()
