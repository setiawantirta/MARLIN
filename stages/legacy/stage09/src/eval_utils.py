"""Metrics, significance tests and calibration for the glioma multi-task benchmark."""
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, brier_score_loss)


# ------------------------------------------------------------------ ranking metrics
def bedroc(y, s, alpha=20.0):
    """Truchon & Bayly (2007) BEDROC."""
    y = np.asarray(y); s = np.asarray(s)
    n, N = int(y.sum()), len(y)
    if n == 0 or n == N:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    ranks = np.where(y[order] == 1)[0] + 1
    Ra = n / N
    sum_exp = np.sum(np.exp(-alpha * ranks / N))
    rie_num = sum_exp / (Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1))
    ra = Ra
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    return (rie_num - rie_min) / (rie_max - rie_min)


def enrichment_factor(y, s, frac=0.01):
    y = np.asarray(y); s = np.asarray(s)
    N, n = len(y), int(y.sum())
    if n == 0:
        return np.nan
    k = max(int(np.ceil(frac * N)), 1)
    order = np.argsort(-s, kind="mergesort")
    return (y[order][:k].sum() / k) / (n / N)


def ece(y, p, bins=10):
    """Expected calibration error (equal-width bins)."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e, N = 0.0, len(y)
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return e


def best_threshold(y, p, grid=None):
    """Threshold maximising MCC on the validation split."""
    if grid is None:
        grid = np.unique(np.round(np.linspace(0.02, 0.98, 49), 3))
    best, bt = -2.0, 0.5
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best:
            best, bt = m, t
    return float(bt), float(best)


def task_metrics(y, p, thr):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    out = {}
    if len(np.unique(y)) < 2:
        return {k: np.nan for k in ["roc_auc", "pr_auc", "mcc", "balanced_acc",
                                    "bedroc20", "ef1", "brier", "ece", "n", "n_pos"]}
    pred = (p >= thr).astype(int)
    out["roc_auc"] = roc_auc_score(y, p)
    out["pr_auc"] = average_precision_score(y, p)
    out["mcc"] = matthews_corrcoef(y, pred)
    out["balanced_acc"] = balanced_accuracy_score(y, pred)
    out["bedroc20"] = bedroc(y, p, 20.0)
    out["ef1"] = enrichment_factor(y, p, 0.01)
    out["brier"] = brier_score_loss(y, p)
    out["ece"] = ece(y, p)
    out["n"] = len(y); out["n_pos"] = int(y.sum())
    return out


# --------------------------------------------------------------------- fast DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, N) with the m positives first. Returns (aucs, cov)."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Two-sided DeLong test for two correlated ROC curves. Returns (auc1, auc2, p)."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    L = np.array([[1, -1]])
    var = float(L @ cov @ L.T)
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


# ----------------------------------------------------------------- paired bootstrap
def paired_bootstrap(y, p1, p2, metric=roc_auc_score, n_boot=2000, seed=0):
    """Two-sided paired bootstrap p-value on metric(p1) - metric(p2)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    obs = metric(y, p1) - metric(y, p2)
    N = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], p1[idx]) - metric(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return obs, 1.0, (np.nan, np.nan)
    centred = diffs - diffs.mean()
    pval = float((np.abs(centred) >= abs(obs)).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(obs), max(pval, 1.0 / (len(diffs) + 1)), (float(lo), float(hi))


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def ci95(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    m = v.mean()
    if len(v) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(m), float(m - h), float(m + h)
