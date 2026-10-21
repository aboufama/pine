"""Power-law fits with bootstrap intervals, used to pre-register predictions before a run."""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


def law(x, E, A, a):
    return E + A * np.power(x, -a)


def fit(x, y, E_max=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    E_max = float(y.min()) * 0.999 if E_max is None else E_max
    best = None
    for a0 in (0.1, 0.3, 0.6, 1.0):
        for e0 in (0.0, 0.5 * E_max, 0.9 * E_max):
            try:
                A0 = max(1e-6, (y[0] - e0) * x[0] ** a0)
                p, _ = curve_fit(law, x, y, p0=(e0, A0, a0), bounds=([0, 0, 0.01], [E_max, np.inf, 3.0]), maxfev=20000)
                r = float(((np.log(law(x, *p)) - np.log(y)) ** 2).sum())
                if best is None or r < best[1]:
                    best = (p, r)
            except Exception:
                pass
    return best[0]


def bootstrap_prediction(x, per_window_losses, x_new, n=600, seed=0, q=(10, 90)):
    """Refit on bootstrap resamples of each run's validation windows; return the spread at x_new."""
    rng = np.random.default_rng(seed)
    L = [np.asarray(w, float) for w in per_window_losses]
    y0 = np.array([w.mean() for w in L])
    p0 = fit(x, y0)
    preds = []
    for _ in range(n):
        y = np.array([w[rng.integers(0, len(w), len(w))].mean() for w in L])
        try:
            preds.append(law(x_new, *fit(x, y)))
        except Exception:
            pass
    preds = np.array(preds)
    return dict(E=float(p0[0]), A=float(p0[1]), alpha=float(p0[2]), loss=float(law(x_new, *p0)),
                lo=float(np.percentile(preds, q[0])), hi=float(np.percentile(preds, q[1])), n_boot=len(preds))
