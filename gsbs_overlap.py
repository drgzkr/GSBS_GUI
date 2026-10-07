"""
Boundary overlap between two segmentations (neural or stimulus).

Conventions follow Geerligs et al. (2022) and neuro_tools.compute_boundary_overlap:
a boundary is the TR at which a new state or event starts; O is the number of
boundaries the two sets share; E = n_ref * n_test / T is the overlap expected by
chance;

    OA = (O - E) / (n_test - E)               absolute overlap, 0 = chance, 1 = every test boundary matched
    OR = (O - E) / (min(n_ref, n_test) - E)   relative overlap, corrected for unequal numbers

Significance comes from shuffling the durations of the test segmentation's
states (as overlap_permutation_test_optimized in the GSBS notebooks), keeping
its number and lengths of states. Here every state is shuffled, including the
last, so the total length is preserved, and p = (shuffles with overlap >= O + 1)
/ (n + 1).

With a tolerance of t TRs, a boundary matches one within +-t TRs (each boundary
matched at most once), and E is taken from the shuffles instead of the formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def count_matches(ref: np.ndarray, test: np.ndarray, tolerance: int = 0) -> int:
    """Boundaries of `test` with a distinct boundary of `ref` within `tolerance` TRs."""
    ref = np.sort(np.asarray(ref))
    test = np.sort(np.asarray(test))
    if tolerance == 0:
        return int(len(np.intersect1d(ref, test)))
    used = np.zeros(len(ref), bool)
    n = 0
    for b in test:
        lo, hi = np.searchsorted(ref, b - tolerance), np.searchsorted(ref, b + tolerance, "right")
        cand = [j for j in range(lo, hi) if not used[j]]
        if cand:
            j = min(cand, key=lambda j: abs(ref[j] - b))
            used[j] = True
            n += 1
    return n


def shuffle_states(bounds: np.ndarray, n_time: int, rng) -> np.ndarray:
    """Boundaries after randomly reordering the state durations."""
    edges = np.r_[0, np.sort(bounds), n_time]
    durations = rng.permutation(np.diff(edges))
    return np.cumsum(durations)[:-1]


@dataclass
class Overlap:
    n_ref: int
    n_test: int
    n_time: int
    tolerance: int
    observed: int
    expected: float
    oa: float
    or_: float
    null_mean: float
    p: float
    n_perm: int


def overlap(ref, test, n_time: int, tolerance: int = 0, n_perm: int = 1000,
            seed: int = 0) -> Overlap:
    """Overlap of `test` boundaries (shuffled for the null) with `ref` boundaries."""
    ref = np.unique(np.asarray(ref, int))
    test = np.unique(np.asarray(test, int))
    ref = ref[(ref > 0) & (ref < n_time)]
    test = test[(test > 0) & (test < n_time)]
    o = count_matches(ref, test, tolerance)
    rng = np.random.default_rng(seed)
    null = np.array([count_matches(ref, shuffle_states(test, n_time, rng), tolerance)
                     for _ in range(n_perm)]) if n_perm and len(test) else np.zeros(0)
    expected = len(ref) * len(test) / n_time if tolerance == 0 else (
        float(null.mean()) if len(null) else np.nan)

    def scaled(denom):
        return float((o - expected) / denom) if denom > 0 else np.nan

    return Overlap(
        n_ref=len(ref), n_test=len(test), n_time=n_time, tolerance=tolerance, observed=o,
        expected=float(expected), oa=scaled(len(test) - expected),
        or_=scaled(min(len(ref), len(test)) - expected),
        null_mean=float(null.mean()) if len(null) else np.nan,
        p=float((np.sum(null >= o) + 1) / (len(null) + 1)) if len(null) else np.nan,
        n_perm=int(len(null)))


def describe(r: Overlap) -> str:
    tol = f" (±{r.tolerance} TR)" if r.tolerance else ""
    return (f"O = {r.observed}{tol} of {r.n_test} boundaries · E = {r.expected:.1f} · "
            f"OA = {r.oa:.2f} · OR = {r.or_:.2f} · p = {r.p:.3g}")


# ---------------------------------------------------------------------------
# Several references at once (an HRF delay grid) for many test segmentations
# ---------------------------------------------------------------------------

@dataclass
class GridOverlap:
    """Overlap of one test segmentation with each of several references (e.g. delays)."""
    observed: np.ndarray            # (R,) overlapping boundaries per reference
    expected: np.ndarray            # (R,)
    oa: np.ndarray                  # (R,)
    or_: np.ndarray                 # (R,)
    p: np.ndarray                   # (R,) per reference
    best: int                       # reference with the highest OR
    p_best: float                   # p of the best OR against the max-over-references null
    n_test: int
    n_perm: int


def grid_overlap(refs: list, test, n_time: int, tolerance: int = 0, n_perm: int = 1000,
                 seed: int = 0) -> GridOverlap:
    """Overlap of `test` with every reference in `refs`, sharing one set of shuffles.

    Choosing the best of several references (HRF delays) inflates its OR, so
    `p_best` compares it with the best OR over all references in each shuffle
    (a max-statistic null), which corrects for having searched the grid.
    """
    refs = [np.unique(np.asarray(r, int)) for r in refs]
    refs = [r[(r > 0) & (r < n_time)] for r in refs]
    test = np.unique(np.asarray(test, int))
    test = test[(test > 0) & (test < n_time)]
    R = len(refs)
    n_ref = np.array([len(r) for r in refs], float)
    nt = len(test)

    if tolerance == 0:
        masks = np.zeros((R, n_time), bool)
        for i, r in enumerate(refs):
            masks[i, r] = True

        def counts(b):
            return masks[:, b].sum(1)
    else:
        def counts(b):
            return np.array([count_matches(r, b, tolerance) for r in refs])

    obs = counts(test).astype(float)
    rng = np.random.default_rng(seed)
    null = np.array([counts(shuffle_states(test, n_time, rng)) for _ in range(n_perm)],
                    float).reshape(n_perm, R) if (n_perm and nt) else np.zeros((0, R))
    expected = n_ref * nt / n_time if tolerance == 0 else (
        null.mean(0) if len(null) else np.full(R, np.nan))

    def scale(o, denom):
        return np.where(denom > 0, (o - expected) / np.where(denom > 0, denom, 1), np.nan)

    or_den = np.minimum(n_ref, nt) - expected
    oa = scale(obs, nt - expected)
    or_ = scale(obs, or_den)
    if len(null):
        p = (np.sum(null >= obs, 0) + 1) / (len(null) + 1)
        null_or = scale(null, or_den)
        best = int(np.nanargmax(or_)) if np.any(~np.isnan(or_)) else 0
        null_max = np.nanmax(null_or, 1)
        p_best = float((np.sum(null_max >= or_[best]) + 1) / (len(null) + 1))
    else:
        p = np.full(R, np.nan)
        best, p_best = 0, np.nan
    return GridOverlap(obs, expected, oa, or_, p, best, p_best, nt, int(len(null)))


def fdr_bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaNs stay NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    adj = q[order] * len(q) / np.arange(1, len(q) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(len(q))
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out
