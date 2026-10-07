"""Clustering the states of one segmentation, on synthetic data with recurring states."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import gsbs_clusters as C

SEQ = [0, 1, 2, 0, 3, 4, 1, 2, 3, 0, 4, 2, 1, 3, 0, 4]     # 5 prototypes, each revisited


@pytest.fixture(scope="module")
def fitted():
    from statesegmentation import fast_GSBS
    rng = np.random.default_rng(3)
    protos = rng.standard_normal((5, 120))
    lens = rng.integers(8, 16, len(SEQ))
    x = np.vstack([np.tile(protos[s], (n, 1)) for s, n in zip(SEQ, lens)])
    x = x + 1.2 * rng.standard_normal(x.shape)
    g = fast_GSBS(kmax=60, x=x)
    g.fit(showProgressBar=False)
    return g


def test_state_patterns(fitted):
    k = fitted.nstates
    patterns, edges = C.state_patterns(fitted, k)
    assert patterns.shape == (k, 120) and edges[0] == 0 and edges[-1] == fitted.x.shape[0]
    np.testing.assert_allclose(patterns[0], fitted.x[edges[0]:edges[1]].mean(0))


@pytest.mark.parametrize("method", C.METHODS)
def test_scan_recovers_recurring_states(fitted, method):
    assert fitted.nstates == len(SEQ)
    patterns, edges = C.state_patterns(fitted, fitted.nstates)
    sc = C.scan(patterns, method)
    labels = C.cluster(patterns, method, **sc.best)
    assert adjusted_rand_score(SEQ, labels) > 0.85


def test_summary(fitted):
    patterns, edges = C.state_patterns(fitted, fitted.nstates)
    cl = C.summarise(patterns, edges, C.cluster(patterns, "hierarchical", n_clusters=5),
                     "hierarchical")
    assert cl.n == 5 and cl.n_states.sum() == len(patterns)
    assert np.all(np.diff(cl.time_fraction) <= 1e-12)          # largest first
    assert np.isclose(cl.time_fraction.sum(), 1.0)


def test_noise_is_not_rewarded(fitted):
    patterns, _ = C.state_patterns(fitted, fitted.nstates)
    labels = np.full(len(patterns), -1)
    labels[:6] = [0, 0, 0, 1, 1, 1]                             # nearly everything noise
    assert C.silhouette(patterns, labels, "dbscan") <= 6 / len(patterns)


def test_transitions():
    counts = C.transitions(np.array([0, 1, 1, 2, -1, 0, 1]), 3)
    assert counts[0, 1] == 2 and counts[1, 2] == 1
    assert counts[1, 1] == 0                                     # same cluster: not a transition
    assert counts.sum() == 3                                     # nothing into / out of noise
