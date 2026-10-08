"""
Analysis pipeline shared by the GUI and the exported replication scripts.

Every number the GUI reports for an exported analysis comes from these
functions, called with the same arguments and random seeds as the scripts in
an exported package, so a package reproduces the GUI's results exactly.
"""

from __future__ import annotations

import numpy as np

import gsbs_clusters as C
import gsbs_core as core
import gsbs_overlap as ov
import gsbs_stimulus as stim

N_PERM = 1000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Segmentation summaries
# ---------------------------------------------------------------------------

def segmentation_summary(obj, k: int | None = None, tr: float | None = None) -> dict:
    """Numbers describing one solution of a fitted result (default: the best)."""
    best = core.best_k(obj)
    k = best if k is None else int(k)
    x = np.asarray(obj.x)
    edges = core.state_edges(obj.all_bounds[k], x.shape[0])
    lengths = np.diff(edges)
    ks = core.solution_ks(obj)
    out = {
        "n_time": int(x.shape[0]), "n_columns": int(x.shape[1]), "tr": tr,
        "k": k, "best_k": best, "best_at_limit": bool(core.best_at_limit(obj)),
        "kmax": int(obj.kmax), "params": core.fit_params(obj), "engine": type(obj).__name__,
        "t_best": float(obj.tdists[best]), "t_shown": float(obj.tdists[k]),
        "n_solutions": int(len(ks)),
        "boundaries": [int(b) for b in edges[1:-1]],
        "median_length_trs": float(np.median(lengths)),
        "min_length_trs": int(lengths.min()), "max_length_trs": int(lengths.max()),
    }
    if tr:
        for key in ("median", "min", "max"):
            out[f"{key}_length_s"] = float(out[f"{key}_length_trs"] * tr)
    try:
        s = obj.get_strengths(k)
        out["strengths"] = [float(v) for v in s[s > 0]]
    except Exception:                                              # noqa: BLE001
        out["strengths"] = []
    return out


def state_patterns(obj, k: int) -> tuple[np.ndarray, np.ndarray]:
    return C.state_patterns(obj, k)


def roi_timescales(session) -> dict:
    """Best number of states and median state duration of every fitted ROI."""
    summ = session.summary()
    rois = sorted(t for t in summ if isinstance(t, int))
    usable = [t for t in rois if not summ[t].at_limit]
    out = {"rois": usable, "excluded": [t for t in rois if summ[t].at_limit],
           "nstates": [int(summ[t].nstates) for t in usable],
           "median_length_trs": [float(summ[t].median_duration) for t in usable]}
    if session.tr:
        out["median_length_s"] = [v * session.tr for v in out["median_length_trs"]]
    return out


# ---------------------------------------------------------------------------
# Stimulus boundaries
# ---------------------------------------------------------------------------

def stimulus_times(transcript, events, kind: str, pause_s: float) -> np.ndarray:
    """Stimulus boundary times (s from the first volume, before the HRF delay)."""
    if kind == "events":
        return np.asarray(events, float)
    return transcript.boundary_times(kind, pause_s)


def stimulus_trs(times_s, tr: float, n_time: int, delay_s: float) -> np.ndarray:
    return stim.to_trs(times_s, tr, n_time, delay_s)


def stimulus_info(transcript, events, kind: str, pause_s: float, tr: float, n_time: int,
                  delay_s: float) -> dict:
    """Bookkeeping for the Methods: what was dropped or merged on the way to boundary TRs."""
    times = stimulus_times(transcript, events, kind, pause_s)
    shifted = np.floor((np.asarray(times, float) + delay_s) / tr).astype(int)
    inside = shifted[(shifted > 0) & (shifted < n_time)]
    out = {"kind": kind, "n_times": int(len(times)), "n_outside": int(len(shifted) - len(inside)),
           "n_trs": int(len(np.unique(inside))), "n_merged": int(len(inside) - len(np.unique(inside)))}
    if transcript is not None and kind != "events":
        out |= {"n_words": int(len(transcript)), "n_untimed": int((~transcript.timed).sum())}
    return out


# ---------------------------------------------------------------------------
# Boundary overlap
# ---------------------------------------------------------------------------

def pair_overlap(ref_trs, test_trs, n_time: int, tolerance: int = 0, n_perm: int = N_PERM,
                 seed: int = 0) -> dict:
    """Overlap of a test segmentation with a reference, with the null counts for figures."""
    r = ov.overlap(ref_trs, test_trs, n_time, tolerance, n_perm, seed)
    rng = np.random.default_rng(seed)
    null = [ov.count_matches(ref_trs, ov.shuffle_states(np.asarray(test_trs), n_time, rng), tolerance)
            for _ in range(n_perm)]
    return {"n_ref": r.n_ref, "n_test": r.n_test, "n_time": n_time, "tolerance": tolerance,
            "observed": r.observed, "expected": r.expected, "oa": r.oa, "or": r.or_,
            "p": r.p, "n_perm": r.n_perm, "null": [int(v) for v in null]}


def roi_overlap_map(session, refs: list, delays: list | None, n_perm: int = N_PERM,
                    progress=None) -> dict:
    """Overlap of every usable fitted ROI's best boundaries with one reference per delay.

    `refs` holds the reference boundary TRs, one array per delay (one array for
    a neural reference). With several delays each ROI's best delay is chosen and
    its p comes from the max-over-delays null. Seeds are the ROI numbers.
    """
    summ = session.summary()
    fitted = sorted(t for t in summ if isinstance(t, int))
    excluded = [t for t in fitted if summ[t].at_limit]
    rois = [t for t in fitted if t not in excluded]
    grid = delays is not None and len(delays) > 1
    rows = []
    for i, roi in enumerate(rois):
        g = ov.grid_overlap(refs, session.best_bounds(roi), session.n_time, n_perm=n_perm, seed=roi)
        rows.append(g)
        if progress:
            progress(i + 1, len(rois))
    p = np.array([g.p_best if grid else g.p[0] for g in rows])
    best = np.array([g.best for g in rows], int)
    return {
        "rois": rois, "excluded": excluded, "delays": delays, "grid": grid, "n_perm": n_perm,
        "or": [float(g.or_[g.best] if grid else g.or_[0]) for g in rows],
        "oa": [float(g.oa[g.best] if grid else g.oa[0]) for g in rows],
        "observed": [float(g.observed[g.best] if grid else g.observed[0]) for g in rows],
        "n_test": [int(g.n_test) for g in rows],
        "p": p.tolist(), "q": ov.fdr_bh(p).tolist(),
        "best_delay": ([float(np.asarray(delays)[b]) for b in best] if delays is not None
                       else [None] * len(rois)),
        "or_grid": [g.or_.tolist() for g in rows],
    }


# ---------------------------------------------------------------------------
# State clusters
# ---------------------------------------------------------------------------

def state_clusters(obj, k: int, method: str, params: dict) -> dict:
    patterns, edges = C.state_patterns(obj, k)
    labels = C.cluster(patterns, method, **params)
    cl = C.summarise(patterns, edges, labels, method)
    return {"k": k, "method": method, "params": params, "labels": cl.labels.tolist(),
            "edges": edges.tolist(), "centroids": cl.centroids.tolist(),
            "n_states": cl.n_states.tolist(), "time_fraction": cl.time_fraction.tolist(),
            "silhouette": cl.silhouette, "transitions": C.transitions(cl.labels, cl.n).tolist()}


# ---------------------------------------------------------------------------
# Network summaries
# ---------------------------------------------------------------------------

def network_means(values_by_roi: dict, info, significant: dict | None = None) -> dict:
    """Mean of a per-ROI value within each Yeo network, with significant/fitted counts."""
    out = {}
    for net in info.network_order:
        rois = [r for r in values_by_roi if info.networks[r - 1] == net]
        if not rois:
            continue
        vals = np.array([values_by_roi[r] for r in rois], float)
        out[net] = {"mean": float(np.nanmean(vals)), "n": len(rois),
                    "n_significant": int(sum(bool(significant.get(r)) for r in rois))
                    if significant is not None else None}
    return out
