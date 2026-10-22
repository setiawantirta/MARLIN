"""Publication figures. Executed inside the python kernel with figure-style loaded:
    exec(open('src/figures.py').read())
Expects: apply_figure_style, panel_letter, focal_palette, set_frame in globals().
"""
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from scipy import stats

import mdata
import eval_utils as E
from feat import TASKS

DISPLAY = {"GIN": "GIN", "GAT": "GATv2", "DMPNN": "D-MPNN", "AttentiveFP": "AttentiveFP",
           "GraphGPS": "GraphGPS", "RF": "Random forest", "XGB": "XGBoost"}
ORDER = ["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS", "RF", "XGB"]


# comparator hues deliberately exclude red (reserved for the focal/selected model)
_COMPARATOR = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22"]


def _colors(focal):
    others = [a for a in ORDER if a != focal]
    cmap = {a: _COMPARATOR[i % len(_COMPARATOR)] for i, a in enumerate(others)}
    cmap[focal] = "#c0392b"
    return cmap


def fig_benchmark(ens, Y, mask, split, focal, summary, out="fig_benchmark"):
    """ROC and PR curves per task for every model (3-seed mean probability) plus the
    model x task ROC-AUC heatmap."""
    apply_figure_style(sizes=(8, 7, 6))
    archs = [a for a in ORDER if a in ens]
    C = _colors(focal)
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(2, 6, figure=fig, hspace=0.48, wspace=0.12,
                  left=0.085, right=0.985, top=0.945, bottom=0.475)
    gs2 = GridSpec(1, 1, figure=fig, left=0.155, right=0.945, top=0.355, bottom=0.055)
    sels, row_axes = {}, {}
    for ti, t in enumerate(TASKS):
        sels[t] = np.where((split == "test") & (mask[:, ti] > 0))[0]

    for row, kind in enumerate(["ROC", "PR"]):
        for ti, t in enumerate(TASKS):
            ax = fig.add_subplot(gs[row, ti])
            sel = sels[t]
            y = Y[sel, ti].astype(int)
            for a in archs:
                p = ens[a][sel, ti]
                if not np.isfinite(p).all():
                    continue
                lw = 1.5 if a == focal else 0.8
                z = 5 if a == focal else 2
                al = 1.0 if a == focal else 0.75
                if kind == "ROC":
                    fpr, tpr, _ = roc_curve(y, p)
                    ax.plot(fpr, tpr, color=C[a], lw=lw, zorder=z, alpha=al)
                else:
                    pr, rc, _ = precision_recall_curve(y, p)
                    ax.plot(rc, pr, color=C[a], lw=lw, zorder=z, alpha=al)
            if kind == "ROC":
                ax.plot([0, 1], [0, 1], color="0.7", lw=0.6, ls=":", zorder=1)
                ax.set_title(f"{t}\nn={len(sel)}, {int(y.sum())} active", fontsize=6.5, pad=3)
            else:
                ax.axhline(y.mean(), color="0.7", lw=0.6, ls=":", zorder=1)
            ax.set_xlim(-0.03, 1.03)
            if kind == "ROC":
                ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1])
            else:
                ax.set_ylim(0.58, 1.01)          # shared floor below every prevalence
                ax.set_yticks([0.6, 0.8, 1.0])
            ax.set_xticks([0, 0.5, 1])
            if ti == 0:
                ax.set_ylabel("True positive rate" if kind == "ROC" else "Precision")
            if ti > 0:
                ax.set_yticklabels([])
            set_frame(ax, "open")
            if ti == 0:
                row_axes[kind] = ax

    # one shared axis label per row (labels repeated per panel collide at this width).
    # The offset is -0.036 rather than -0.045 so that the PR row's label clears the
    # model legend below it; the free band between the PR panels and the heatmap has
    # to hold the label and the legend without them touching.
    for kind, lab in [("ROC", "False positive rate"), ("PR", "Recall")]:
        bb = row_axes[kind].get_position()
        fig.text(0.535, bb.y0 - 0.036, lab, ha="center", va="top", fontsize=7)

    # ---- heatmap
    axh = fig.add_subplot(gs2[0, 0])
    cols = list(TASKS) + ["Macro"]
    Hm = np.full((len(archs), len(cols)), np.nan)
    for i, a in enumerate(archs):
        for j, t in enumerate(TASKS):
            sel = sels[t]
            Hm[i, j] = roc_auc_score(Y[sel, TASKS.index(t)].astype(int), ens[a][sel, TASKS.index(t)])
        Hm[i, -1] = np.nanmean(Hm[i, :-1])
    im = axh.imshow(Hm, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    axh.set_xticks(range(len(cols))); axh.set_xticklabels(cols)
    axh.set_yticks(range(len(archs)))
    axh.set_yticklabels([DISPLAY[a] + ("  \u25c0" if a == focal else "") for a in archs])
    for i in range(len(archs)):
        for j in range(len(cols)):
            v = Hm[i, j]
            axh.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                     color="white" if v < 0.82 else "black")
    axh.set_title("Test-set ROC-AUC on the scaffold-disjoint partition "
                  "(higher = better)", fontsize=8, loc="left", pad=4)
    axh.axvline(len(TASKS) - 0.5, color="white", lw=1.5)
    for s in axh.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axh, pad=0.012, fraction=0.02)
    # REVISION ROUND: panel (c) title removed (migrated to the legend); the
    # direction-of-goodness cue it carried moves onto the colourbar label, keeping
    # the cue inside the graphic as figure-style rule 3.6 requires.
    cb.set_label("ROC-AUC (higher = better)", fontsize=6); cb.ax.tick_params(labelsize=6)

    handles = [mpl.lines.Line2D([], [], color=C[a], lw=1.5 if a == focal else 0.9,
                                label=DISPLAY[a]) for a in archs]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.535, 0.399), ncol=7,
               frameon=False, fontsize=6.6, handlelength=1.5, columnspacing=1.2)
    # Panel letters are pinned in FIGURE coordinates, not per-axes: the two GridSpecs
    # have different lefts (0.085 vs 0.155) and the heatmap carries long y-tick labels,
    # so an axes-anchored dx leaves a/b and c at different figure x. The y is taken from
    # each axes' own position with panel_letter's dy=1.02, so vertical placement is
    # unchanged; only x is made common. bbox_inches="tight" then crops to x=0.012 and
    # all three letters finish flush.
    _LETTER_FS = plt.rcParams.get("font.size", 8) + 1
    for letter, ax in [("a", fig.axes[0]), ("b", fig.axes[6]), ("c", axh)]:
        p = ax.get_position()
        fig.text(0.012, p.y0 + 1.02 * p.height, letter, fontsize=_LETTER_FS,
                 fontweight="bold", ha="left", va="bottom")
    # REVISION ROUND: drop the panel (c) title and close the band it occupied. Panels
    # (a) and (b) keep their per-task titles, which carry n and the active count.
    from repro.compact import compact_titles
    compact_titles(fig, skip=tuple(a for a in fig.axes if not a.images))
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    return fig, Hm


def fig_external(enrich, dec_roc, pheno_scatter, pheno_models, fda_rows, holdout_rows,
                 focal, out="fig_external"):
    """Enrichment against ZINC20 decoys, FDA control recovery, and the marine
    phenotypic validation that selects the final model."""
    apply_figure_style(sizes=(8, 7, 6))
    C = _colors(focal)
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.72, wspace=0.52,
                  left=0.085, right=0.90, top=0.92, bottom=0.10)

    # (a) enrichment: BEDROC(20) and EF1% per target
    ax = fig.add_subplot(gs[0, 0])
    en = enrich.set_index("task").reindex(TASKS)
    yy = np.arange(len(TASKS))
    ax.hlines(yy, 0, en["bedroc20"], color="0.75", lw=0.8, zorder=1)
    ax.scatter(en["bedroc20"], yy, s=22, color=C[focal], zorder=3, label="BEDROC ($\\alpha$=20)")
    ax.scatter(en["ef1"] / 10.0, yy, s=18, facecolors="none", edgecolors="#2c3e50",
               lw=0.9, zorder=3, label="EF$_{1\\%}$ / 10")
    ax.set_yticks(yy); ax.set_yticklabels(TASKS)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Enrichment (higher = better)")
    ax.axvline(en["n_actives"].iloc[0] / (en["n_actives"].iloc[0] + en["n_decoys"].iloc[0]),
               color="0.8", lw=0.6, ls=":", zorder=0)
    ax.legend(frameon=False, fontsize=5.8, loc="lower left", handletextpad=0.3,
              borderpad=0.1)
    ax.set_title("Actives vs property-matched\nZINC20 decoys", fontsize=7.2, loc="left", pad=3)
    set_frame(ax, "open")

    # (b) decoy ROC curves per target
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("viridis")
    for k, (t, (y, s)) in enumerate(dec_roc.items()):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.0, color=cmap(k / max(len(dec_roc) - 1, 1)), label=t)
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.6, ls=":")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right", handlelength=1.0,
              labelspacing=0.25, borderpad=0.1)
    ax.set_title("Decoy discrimination by target", fontsize=7.2, loc="left", pad=3)
    set_frame(ax, "open")

    # (c) FDA control recovery: one point per approved drug-target pair, by target
    ax = fig.add_subplot(gs[0, 2])
    f = pd.DataFrame(fda_rows).dropna(subset=["rank_percentile"])
    rng = np.random.default_rng(0)
    for k, t in enumerate(TASKS):
        sub = f[f["gene"] == t]
        if not len(sub):
            continue
        ax.scatter(sub["rank_percentile"], k + rng.uniform(-0.16, 0.16, len(sub)),
                   s=13, color=C[focal], alpha=0.85, lw=0, zorder=3)
        ax.plot([sub["rank_percentile"].median()] * 2, [k - 0.28, k + 0.28],
                color="#2c3e50", lw=1.1, zorder=4)
    ax.axvline(90, color="0.45", lw=0.7, ls="--", zorder=1)
    ax.text(89, -0.75, "top 10%", fontsize=5.4, color="0.35", ha="right", va="bottom")
    ax.set_yticks(range(len(TASKS))); ax.set_yticklabels(TASKS)
    ax.set_ylim(-0.85, len(TASKS) - 0.45)
    ax.set_xlim(0, 104); ax.set_xlabel("Rank percentile within test set")
    ax.set_title(f"FDA-approved drug-target pairs\n({len(f)} pairs, vertical tick = median)",
                 fontsize=7.2, loc="left", pad=3)
    set_frame(ax, "open")

    # (d) marine phenotypic scatter
    ax = fig.add_subplot(gs[1, 0])
    sx, sy = pheno_scatter["score"], pheno_scatter["pact"]
    ax.scatter(sx, sy, s=13, color=C[focal], alpha=0.65, lw=0, zorder=3)
    if len(sx) > 3:
        b, a0 = np.polyfit(sx, sy, 1)
        xs = np.linspace(sx.min(), sx.max(), 20)
        ax.plot(xs, a0 + b * xs, color="#2c3e50", lw=1.0, zorder=4)
    rho, pv = stats.spearmanr(sx, sy)
    ax.axhline(6.0, color="0.75", lw=0.6, ls=":", zorder=1)
    ax.text(0.03, 0.96, f"$\\rho_s$ = {rho:.3f}\np = {pv:.1e}\nn = {len(sx)}",
            transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel("Polypharmacology score ($\\Sigma$ 6 probabilities)", fontsize=6.6)
    ax.set_ylabel("Measured pActivity\n(most sensitive glioma line)")
    ax.set_title("Marine phenotypic validation", fontsize=7.2, loc="left", pad=3)
    ax.margins(0.05)
    set_frame(ax, "open")

    # (e) per-model phenotypic performance (the model-selection panel)
    ax = fig.add_subplot(gs[1, 1])
    pm = pheno_models.copy().sort_values("roc_auc")
    yy = np.arange(len(pm))
    ax.hlines(yy, 0.5, pm["roc_auc"], color="0.8", lw=0.8)
    ax.scatter(pm["roc_auc"], yy, s=26,
               color=[C.get(a, "0.5") for a in pm["arch"]], zorder=3)
    ax.axvline(0.5, color="0.5", lw=0.7, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels([DISPLAY.get(a, a) for a in pm["arch"]], fontsize=6)
    ax.set_xlabel("ROC-AUC (higher = better)")
    ax.set_title("Model selection: pActivity $\\geq$ 6\ndiscrimination on 194 marine NPs",
                 fontsize=7.2, loc="left", pad=3)
    ax.set_xlim(min(0.35, pm["roc_auc"].min() - 0.03),
                max(0.75, pm["roc_auc"].max() + 0.04))
    set_frame(ax, "open")

    # (f) marine holdout predictions
    ax = fig.add_subplot(gs[1, 2])
    h = pd.DataFrame(holdout_rows)
    if len(h):
        h = h.sort_values(["label", "predicted_prob"])
        yy = np.arange(len(h))
        colr = ["#c0392b" if l == 1 else "#2c7fb8" for l in h["label"]]
        ax.scatter(h["predicted_prob"], yy, s=16, color=colr, zorder=3)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{str(c).replace('CMNPD', '')} / {t}"
                            for c, t in zip(h["compound"], h["task"])], fontsize=4.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability")
        hl = [mpl.lines.Line2D([], [], marker="o", ls="", ms=3, color="#c0392b",
                               label="measured active"),
              mpl.lines.Line2D([], [], marker="o", ls="", ms=3, color="#2c7fb8",
                               label="measured inactive")]
        ax.legend(handles=hl, frameon=False, fontsize=5.4, loc="lower right",
                  handletextpad=0.2, borderpad=0.1)
    ax.set_title("CMNPD marine holdout (n=18 compounds)", fontsize=7.2, loc="left", pad=3)
    set_frame(ax, "open")

    for letter, ax in zip("abcdef", fig.axes[:6]):
        panel_letter(ax, letter)
    # REVISION ROUND: all six panel titles removed and their bands closed; each claim
    # they made now lives in the figure legend (see figure_title_migration.md). The
    # titles are still constructed above because compact_titles measures them to learn
    # exactly how much vertical space to reclaim, which keeps every data axes the same
    # physical size as in the published figure.
    from repro.compact import compact_titles
    compact_titles(fig)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    return fig
