from __future__ import annotations
import numpy as np


def pareto_front(Y: np.ndarray):
    Y = np.asarray(Y, dtype=float)
    keep = np.ones(len(Y), dtype=bool)
    for i in range(len(Y)):
        if not keep[i]:
            continue
        dominated = np.all(Y >= Y[i], axis=1) & np.any(Y > Y[i], axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return Y[keep]


def hvi_single(front: np.ndarray, point: np.ndarray, ref: np.ndarray, tol=1e-12):
    """Exact HVI using inclusion-exclusion; intended for small toy tests only."""
    from itertools import combinations
    F = np.asarray(front, float)
    p = np.asarray(point, float)
    r = np.asarray(ref, float)
    if np.any(p <= r + tol):
        return 0.0
    base = hypervolume(F, r)
    return max(0.0, hypervolume(np.vstack([F, p]), r) - base)


def hypervolume(Y: np.ndarray, ref: np.ndarray):
    """Exact 3D hypervolume by recursive slicing; small-set reference implementation."""
    Y = np.asarray(Y, float)
    ref = np.asarray(ref, float)
    Y = Y[np.all(Y > ref, axis=1)]
    if len(Y) == 0:
        return 0.0
    # Recursion through dimensions; optimized implementation can replace this.
    def hv(points, ref_local):
        if len(points) == 0:
            return 0.0
        dims = points.shape[1]
        if dims == 1:
            return float(np.max(points[:, 0]) - ref_local[0])
        xs = np.unique(np.r_[ref_local[0], points[:, 0]])
        total = 0.0
        for a, b in zip(xs[:-1], xs[1:]):
            if b <= a:
                continue
            active = points[points[:, 0] >= b]
            if len(active):
                total += (b - a) * hv(active[:, 1:], ref_local[1:])
        return total
    return hv(Y, ref)


def qpmhi_scores(
    mu,
    sigma,
    qed,
    sa_inv,
    front,
    ref,
    samples=512,
    seed=7,
    scscore=None,
    steps=None,
    max_steps: int = 4
):
    """Compute qPMHI acquisition probabilities over Pareto front.
    
    Optionally incorporates Coley et al. SCScore and forward synthetic step constraints:
    - Blends Ertl SA with SCScore when scscore is provided.
    - Applies step penalty to candidates requiring > max_steps.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    qed = np.asarray(qed, float)
    sa_inv = np.asarray(sa_inv, float)

    # Composite synthetic accessibility if SCScore is supplied
    if scscore is not None:
        sc = np.asarray(scscore, float)
        # Convert sa_inv back to sa, blend with SCScore*2.0 (mapped to 1-10 scale), and re-invert
        raw_sa = 1.0 / np.maximum(sa_inv, 1e-4)
        composite_synth = 0.5 * raw_sa + 0.5 * (sc * 2.0)
        eff_sa_inv = 1.0 / np.maximum(composite_synth, 1e-4)
    else:
        eff_sa_inv = sa_inv

    # Penalty mask for step constraint violations
    penalty_scale = np.ones(len(mu), dtype=float)
    if steps is not None:
        st = np.asarray(steps, int)
        penalty_scale = np.where(st <= max_steps, 1.0, 0.25)

    base = hypervolume(front, np.asarray(ref, float))
    wins = np.zeros(len(mu), dtype=np.int64)
    for _ in range(samples):
        aff = rng.normal(mu, sigma) * penalty_scale
        pts = np.column_stack([aff, qed, eff_sa_inv])
        best_i, best_hvi = 0, -np.inf
        for i, p in enumerate(pts):
            hvi = hypervolume(np.vstack([front, p]), ref) - base
            if hvi > best_hvi:
                best_hvi = hvi
                best_i = i
            elif np.isclose(hvi, best_hvi, rtol=1e-12, atol=1e-12):
                if rng.random() < 0.5:
                    best_i = i
        wins[best_i] += 1
    return wins / samples

