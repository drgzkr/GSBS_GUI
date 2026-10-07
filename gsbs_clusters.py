"""
Clustering the states of one GSBS segmentation into groups of similar patterns.

The input is a single solution (k states) of a fitted result: each state's mean
pattern is one item. Patterns are z-scored across columns, so the clustering
groups states by pattern shape, not amplitude.

Model selection uses the silhouette computed with the clustering's own
distance; DBSCAN noise states count as 0 (neither a cluster, nor left out, which
would reward declaring states noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gsbs_core as core

METHODS = ("hierarchical", "dbscan", "kmeans")
METHOD_LABELS = {"hierarchical": "Hierarchical (average linkage, correlation distance)",
                 "dbscan": "DBSCAN (cosine distance)",
                 "kmeans": "k-means (z-scored patterns)"}


def state_patterns(obj, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean pattern per state, state edges) of solution k."""
    x = np.asarray(obj.x)
    edges = core.state_edges(obj.all_bounds[k], x.shape[0])
    return np.add.reduceat(x, edges[:-1], axis=0) / np.diff(edges)[:, None], edges


def zscore_rows(p: np.ndarray) -> np.ndarray:
    p = p - p.mean(1, keepdims=True)
    sd = p.std(1, keepdims=True)
    return np.divide(p, sd, out=np.zeros_like(p), where=sd > 0)


def _metric(method: str) -> str:
    return "cosine" if method == "dbscan" else "correlation"


def distances(patterns: np.ndarray, method: str) -> np.ndarray:
    from scipy.spatial.distance import cdist
    z = zscore_rows(patterns)
    d = np.clip(cdist(z, z, metric=_metric(method)), 0, None)
    np.fill_diagonal(d, 0.0)
    return d


def cluster(patterns: np.ndarray, method: str, D: np.ndarray | None = None, **params) -> np.ndarray:
    """Labels 0..C-1 per state, or -1 for DBSCAN noise."""
    n = len(patterns)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=int(params["n_clusters"]), n_init=10,
                      random_state=0).fit(zscore_rows(patterns)).labels_
    D = distances(patterns, method) if D is None else D
    if method == "hierarchical":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        if n < 2:
            return np.zeros(n, int)
        tree = linkage(squareform(D, checks=False), method="average")
        return fcluster(tree, t=int(params["n_clusters"]), criterion="maxclust") - 1
    if method == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=float(params["eps"]), min_samples=int(params["min_samples"]),
                      metric="precomputed").fit(D).labels_
    raise ValueError(f"Unknown method {method}")


def silhouette(patterns: np.ndarray, labels: np.ndarray, method: str,
               D: np.ndarray | None = None) -> float:
    from sklearn.metrics import silhouette_score
    keep = labels >= 0
    n_cl = len(set(labels[keep]))
    if n_cl < 2 or n_cl >= keep.sum():
        return np.nan
    D = distances(patterns, method) if D is None else D
    return float(silhouette_score(D[np.ix_(keep, keep)], labels[keep], metric="precomputed")
                 * keep.mean())


@dataclass
class Scan:
    method: str
    grid: list
    scores: np.ndarray

    @property
    def best(self) -> dict | None:
        if np.all(np.isnan(self.scores)):
            return None
        return self.grid[int(np.nanargmax(self.scores))]


def scan(patterns: np.ndarray, method: str) -> Scan:
    """Silhouette over a range of settings."""
    n = len(patterns)
    D = distances(patterns, method)
    if method == "dbscan":
        off = D[np.triu_indices(n, 1)]
        lo, hi = (np.percentile(off, 1), np.percentile(off, 50)) if len(off) else (0.05, 1.0)
        grid = [{"eps": float(e), "min_samples": int(m)}
                for e in np.linspace(max(lo, 1e-3), hi, 12) for m in (2, 3, 5)]
    else:
        grid = [{"n_clusters": c} for c in range(2, max(3, min(20, n - 1) + 1))]
    scores = [silhouette(patterns, cluster(patterns, method, D=D, **g), method, D) for g in grid]
    return Scan(method, grid, np.asarray(scores, float))


@dataclass
class Clusters:
    labels: np.ndarray              # per state; 0 = most time; -1 = DBSCAN noise
    centroids: np.ndarray           # (C, F) mean z-scored pattern per cluster
    n_states: np.ndarray
    time_fraction: np.ndarray       # share of the recording per cluster
    silhouette: float

    @property
    def n(self) -> int:
        return len(self.centroids)


def summarise(patterns: np.ndarray, edges: np.ndarray, labels: np.ndarray,
              method: str) -> Clusters:
    z = zscore_rows(patterns)
    lengths = np.diff(edges).astype(float)
    ids = [c for c in np.unique(labels) if c >= 0]
    order = sorted(ids, key=lambda c: -lengths[labels == c].sum())
    remap = {c: i for i, c in enumerate(order)}
    new = np.array([remap.get(c, -1) for c in labels])
    cents = np.array([z[new == i].mean(0) for i in range(len(order))]).reshape(len(order), -1)
    return Clusters(new, cents,
                    np.array([int(np.sum(new == i)) for i in range(len(order))]),
                    np.array([lengths[new == i].sum() / lengths.sum() for i in range(len(order))]),
                    silhouette(patterns, new, method))


def transitions(labels: np.ndarray, n: int) -> np.ndarray:
    """Counts of moves from one cluster to a different cluster between consecutive states.

    Moves into or out of a DBSCAN noise state are not counted.
    """
    a, b = labels[:-1], labels[1:]
    ok = (a >= 0) & (b >= 0) & (a != b)
    counts = np.zeros((n, n))
    np.add.at(counts, (a[ok], b[ok]), 1)
    return counts
