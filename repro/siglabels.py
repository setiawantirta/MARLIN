"""Significance stars for model-comparison figures.

The benchmark compares seven model families, so every panel that shows more than
one model is a multiple-comparison setting. `significance_tests.csv` holds the tests
per evaluation scope with Benjamini-Hochberg corrected q values alongside the
uncorrected p. The number of comparisons is not the same in every scope: the six
per-target scopes and MACRO each carry all 21 pairs, whereas MARINE_PHENOTYPIC
carries 6 - the random forest against each of the others, since that scope documents
the model-selection decision rather than an all-pairs comparison.

Annotate from `q_bootstrap`, not from the raw p. Correction is warranted at either
count: 21 uncorrected tests at a 0.05 threshold expect about one false positive per
panel and 6 expect about one in three, and multiplicity is exactly the criticism a
reviewer raises against a starred benchmark figure.

Default thresholds are 0.05, 0.01 and 0.001 for one, two and three stars, the
declared convention. They are a parameter, not a constant: pass `thresholds=` to use
the more common 0.05 / 0.01 / 0.001 scheme, and state in the caption whichever you
used, together with the test and the correction.

Usage
-----
    from repro.siglabels import stars, vs_reference, star_legend

    stars(0.0042)                      -> '**'
    vs_reference(sig, scope='MACRO')   -> {'GIN': '**', 'GATv2': '**', ...}
    star_legend()                      -> '* q < 0.05, ** q < 0.01, *** q < 0.001 ...'
"""
from __future__ import annotations

import os

import pandas as pd

# Display name -> the identifier used in significance_tests.csv.
MODEL_KEY = {
    "GIN": "GIN",
    "GATv2": "GAT",
    "D-MPNN": "DMPNN",
    "AttentiveFP": "AttentiveFP",
    "GraphGPS": "GraphGPS",
    "Random forest": "RF",
    "XGBoost": "XGB",
}
KEY_MODEL = {v: k for k, v in MODEL_KEY.items()}

# Three tiers, the conventional cut-offs. The earlier two-tier (0.05, 0.005) scale
# put every contrast of Fig. 5 at ** and split the 153 model contrasts of Figs. 3-4
# into 32 at * and 38 at **; three tiers resolve those into 11 / 57 / 2, which is a
# finer reading of the same q values. Fig. 5 saturates under any scale, because its
# largest q is 7.6e-09, so its brackets carry Cohen's d alongside the asterisks.
DEFAULT_THRESHOLDS = (0.05, 0.01, 0.001)


def stars(q, thresholds=DEFAULT_THRESHOLDS, ns=""):
    """Map a (corrected) p value to a star string.

    thresholds is read largest-first and any number of tiers is allowed, so
    (0.05, 0.01, 0.001) gives '*', '**' and '***'. A missing or non-finite q
    returns `ns` rather than silently
    reading as non-significant.
    """
    if q is None:
        return ns
    try:
        q = float(q)
    except (TypeError, ValueError):
        return ns
    if q != q:  # NaN
        return ns
    for i, t in enumerate(sorted(thresholds, reverse=True), start=1):
        if q >= t:
            continue
        # count how many thresholds this q clears
        return "*" * sum(1 for tt in thresholds if q < tt)
    return ns


def load(path=None):
    """Load significance_tests.csv through the module's path resolver when available."""
    if path is None:
        try:
            from repro.config import path as RPATH
            path = RPATH("significance_tests.csv")
        except Exception:
            path = os.environ.get("SIGNIFICANCE_TESTS", "significance_tests.csv")
    return pd.read_csv(path)


def vs_reference(sig, scope, reference="Random forest", q_col="q_bootstrap",
                 thresholds=DEFAULT_THRESHOLDS):
    """Stars for every model compared against one reference, within one scope.

    Returns {display_name: star_string}. The reference itself is absent from the
    mapping - it has no comparison against itself and must not be starred.
    Comparisons are symmetric, so a row is matched in either orientation.
    """
    ref = MODEL_KEY.get(reference, reference)
    g = sig[sig["scope"] == scope]
    if g.empty:
        raise KeyError(f"scope {scope!r} not in significance table; "
                       f"available: {sorted(sig['scope'].unique())}")
    out = {}
    for _, r in g.iterrows():
        a, b = r["model_a"], r["model_b"]
        if ref not in (a, b):
            continue
        other = b if a == ref else a
        out[KEY_MODEL.get(other, other)] = stars(r[q_col], thresholds)
    return out


# --------------------------------------------------------------- effect sizes
# Fig. 5 is a different annotation problem from Figs. 3 and 4. Its eight displayed
# contrasts all reach ** after BH correction, because n is 1,164-6,445 molecules
# rather than a handful of models, so a star carries no information there: it would
# print the same symbol on Cohen's d = 1.55 and on d = -0.17. Annotate those panels
# with the effect size AND the star, so a large and a negligible difference are
# distinguishable at a glance.

def bh(pvals):
    """Benjamini-Hochberg step-up q values, in the input order."""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    run = 1.0
    for k in range(m - 1, -1, -1):
        run = min(run, p[order[k]] * m / (k + 1))
        q[order[k]] = run
    return q


def d_magnitude(d):
    """Cohen's conventional bands, for prose that does not overstate a small effect."""
    a = abs(float(d))
    return ("negligible" if a < 0.2 else "small" if a < 0.5
            else "medium" if a < 0.8 else "large")


def effect_label(d, q, thresholds=DEFAULT_THRESHOLDS, latex=True):
    """'d = 1.53, **' - the pairing to use where every contrast is significant."""
    s = stars(q, thresholds, ns="ns")
    dd = f"$d$ = {d:+.2f}" if latex else f"d = {d:+.2f}"
    return f"{dd}, {s}"


def attribution_tests(gnn_vs_rf, panels):
    """BH-correct the Fig. 5 contrasts as one family and return a per-panel table.

    `gnn_vs_rf` is gnn_vs_rf_attribution.csv; `panels` maps a panel letter to the
    metric-name prefixes it displays. Rows whose Mann-Whitney p is absent are
    dropped rather than counted: the per-target cosine and top-10-overlap rows carry
    no two-group comparison, so they belong in no test family and get no star.
    """
    import pandas as pd
    rows = []
    for pn, prefixes in panels.items():
        for pref in prefixes:
            hit = gnn_vs_rf[gnn_vs_rf["metric"].str.startswith(pref[:48])]
            if hit.empty:
                raise KeyError(f"no row in gnn_vs_rf_attribution.csv starts with {pref[:48]!r}")
            r = hit.iloc[0]
            if not pd.notna(r["mannwhitney_p"]):
                continue
            rows.append(dict(panel=pn, metric=r["metric"], n_a=int(r["n_a"]), n_b=int(r["n_b"]),
                             cohens_d=r["cohens_d"], p=float(r["mannwhitney_p"])))
    out = pd.DataFrame(rows)
    out["q_BH"] = bh(out["p"].values)
    out["stars"] = [stars(q, ns="ns") for q in out["q_BH"]]
    out["magnitude"] = [d_magnitude(d) if pd.notna(d) else "" for d in out["cohens_d"]]
    return out


# Which metric rows each Fig. 5 panel draws. Panel E is deliberately absent: it shows
# six cosine similarities with no comparison group, so there is nothing to test.
FIG5_PANELS = {
    "A": ["RF vs GNN per-molecule atom-attribution Spearman rho"],
    "B": ["IG: share of |attribution| on aromatic atoms",
          "RF-projected SHAP: share on aromatic atoms"],
    "C": ["IG: share of |attribution| on sp3 atoms",
          "RF-projected SHAP: share on sp3 atoms"],
    "D": ["RF: FRACTION of total |SHAP| carried by the 30 global d"],
    "F": ["IG: share of |attribution| on Murcko-scaffold atoms",
          "GNNExplainer: atom-mask share on Murcko-scaffold atoms"],
}


def annotate_heatmap(ax, row_labels, col_labels, sig, reference="Random forest",
                     q_col="q_bootstrap", thresholds=DEFAULT_THRESHOLDS,
                     scope_of_col=None, macro_col="Macro"):
    """Append stars to the numeric text already drawn in heatmap cells.

    Appending to the existing text rather than adding new artists is deliberate:
    the cells are set in 6 pt and a second text object per cell would collide.
    Each column is tested in its own scope, so a cell reads as "this model versus
    the reference, on this target".

    row_labels / col_labels must be in the same order as the plotted matrix.
    Returns the number of stars drawn.
    """
    scope_of_col = scope_of_col or (lambda c: "MACRO" if c == macro_col else c.upper())
    per_col = {}
    for c in col_labels:
        try:
            per_col[c] = vs_reference(sig, scope_of_col(c), reference, q_col, thresholds)
        except KeyError:
            per_col[c] = {}
    texts = [t for t in ax.texts if (t.get_text() or "").strip()]
    nr, nc = len(row_labels), len(col_labels)
    if len(texts) != nr * nc:
        raise ValueError(f"expected {nr}x{nc}={nr*nc} cell texts, found {len(texts)}; "
                         "pass the axes that holds the heatmap, and check the label order")
    n = 0
    for k, t in enumerate(texts):
        i, j = divmod(k, nc)
        s = per_col[col_labels[j]].get(row_labels[i], "")
        if s:
            t.set_text(t.get_text() + s)
            n += 1
    return n


def star_points(ax, labels, star_map, values, horizontal=True, pad=0.004, fontsize=6,
                color="0.15"):
    """Draw a star beside each plotted point in a one-model-per-row panel.

    `labels` is the model order along the categorical axis, `values` the plotted
    statistic, so the star sits just past the marker rather than at a fixed offset.
    """
    n = 0
    for i, lab in enumerate(labels):
        s = star_map.get(lab, "")
        if not s:
            continue
        if horizontal:
            ax.text(values[i] + pad, i, s, va="center", ha="left",
                    fontsize=fontsize, color=color, fontweight="bold")
        else:
            ax.text(i, values[i] + pad, s, va="bottom", ha="center",
                    fontsize=fontsize, color=color, fontweight="bold")
        n += 1
    return n


# Resample count actually used for each scope, read off evaluate.py and confirmed
# against the p-value floor 1/(B+1) observed in significance_tests.csv:
# per-target 1/1001, MACRO 1/501, marine 1/4000. The paired_bootstrap default of
# 2,000 is overridden at every call site, so do not quote it.
SCOPE_RESAMPLES = {
    "EGFR": 1000, "PDGFRA": 1000, "PIK3CA": 1000,
    "KDR": 1000, "CDK6": 1000, "IDH1": 1000,
    "MACRO": 500,
    "MARINE_PHENOTYPIC": 4000,
}


def sig_bracket(ax, x1, x2, q, d=None, level=0, thresholds=DEFAULT_THRESHOLDS,
                tick=0.018, gap=0.095, fontsize=6.0, lw=0.8, color="0.15",
                ns="n.s.", base=0.855):
    """Draw one significance bracket spanning x1..x2 above the data.

    The bracket is placed in AXES fraction on the y axis, so it is unaffected by
    the data scale and two brackets in the same panel can be stacked by passing
    level=0 and level=1. ``tick`` is the length of the downward end ticks and
    ``gap`` the vertical spacing between stacked levels, both as axes fractions.

    The label carries the asterisks AND Cohen's d when d is given. On this figure
    every contrast clears the strictest threshold, so the asterisks alone are the
    same on all eight; d is what distinguishes a large difference from a
    negligible one, and printing both keeps the bracket honest.

    Returns the axes-fraction height of the bar, so a caller can raise the axes
    top limit to make room.
    """
    # gap must exceed the rendered label height or the lower level's text is
    # struck through by the upper level's bar: at 6 pt in a panel roughly 2 in
    # tall the label occupies about 0.04 of the axes, so 0.095 leaves clearance.
    y = base + level * gap
    tr = ax.get_xaxis_transform()        # x in data units, y in axes fraction
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick],
            lw=lw, color=color, transform=tr, clip_on=False, zorder=7,
            solid_joinstyle="miter")
    txt = stars(q, thresholds=thresholds, ns=ns)
    if d is not None:
        txt = f"{txt} $d={d:+.2f}$" if txt else f"$d={d:+.2f}$"
    ax.text((x1 + x2) / 2.0, y + 0.012, txt, transform=tr, ha="center",
            va="bottom", fontsize=fontsize, color=color, clip_on=False, zorder=7)
    return y


# Which pair of x positions each contrast of Fig. 5 spans, keyed by the metric
# name as it appears in gnn_vs_rf_attribution.csv. Panels B, C and F place their
# four boxes/bars at 0, 0.85, 2.4, 3.25; panel A at 0 and 1; panel D at 0, 1, 2.
# In B and C the comparison runs ACROSS the two evaluation sets within one
# attribution method, which is why those brackets span the middle of the panel.
FIG5_BRACKETS = {
    "RF vs GNN per-molecule atom-attribution Spearman rho":            (0.0, 1.0,  0),
    "RF-projected SHAP: share on aromatic atoms":                      (0.0, 2.40, 0),
    "IG: share of |attribution| on aromatic atoms":                    (0.85, 3.25, 1),
    "RF-projected SHAP: share on sp3 atoms":                           (0.0, 2.40, 0),
    "IG: share of |attribution| on sp3 atoms":                         (0.85, 3.25, 1),
    "RF: FRACTION of total |SHAP| carried by the 30 global descriptors": (0.0, 1.0, 0),
    "IG: share of |attribution| on Murcko-scaffold atoms":             (0.0, 0.85, 0),
    "GNNExplainer: atom-mask share on Murcko-scaffold atoms":           (2.40, 3.25, 0),
}


def star_legend(thresholds=DEFAULT_THRESHOLDS, scope=None,
                correction="Benjamini-Hochberg", reference="the random forest"):
    """One sentence naming the thresholds, the test and the correction.

    A starred figure is uninterpretable without all three, so build the caption
    sentence from the same constants the stars were drawn with. Pass `scope` to
    have the resample count filled in from SCOPE_RESAMPLES; the bootstrap is paired
    (the same resampled rows score both models) and is NOT stratified - it draws
    from the pooled rows and skips any draw that leaves one class empty.
    """
    ts = sorted(thresholds, reverse=True)
    parts = ["%s q < %g" % ("*" * (i + 1), t) for i, t in enumerate(ts)]
    if scope is None:
        test = "a paired bootstrap"
    elif isinstance(scope, str):
        test = "a %s-resample paired bootstrap" % f"{SCOPE_RESAMPLES[scope]:,}"
    else:  # several scopes in one panel, e.g. the per-target columns of Fig. 3C
        ns = sorted({SCOPE_RESAMPLES[s] for s in scope})
        test = ("a %s-resample paired bootstrap" % f"{ns[0]:,}" if len(ns) == 1
                else "paired bootstraps of %s resamples respectively"
                     % " and ".join(f"{n:,}" for n in ns))
    return ("%s, versus %s; %s, %s-corrected across the comparisons in each panel."
            % (", ".join(parts), reference, test, correction))
