"""Figure 2 - chemical space and property profile of the curated marine library.

The uploaded-manuscript version: no in-panel titles, UPPERCASE BOLD SERIF panel
letters A-F, vector PDF plus 300 dpi PNG. Reads two artifacts only - the vendor
property TSV that the original script consumed is not needed, because every
column it supplied is already carried in the curated parquet.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
from repro.figstyle import apply_figure_style, panel_letter, save_figure
_OUT = out_dir("stage11_figures")

"""Figure 2 of the manuscript: chemical-space and property profile of the curated
marine library. Rebuilt to match the uploaded manuscript version -- no in-panel
titles, uppercase bold serif panel letters A-F -- and emitted as vector PDF plus
300 dpi PNG.

Inputs are two artifacts only; the vendor TSV dependency of the original script was
removed because every column it supplied is already carried in the curated parquet.
"""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CURATED  = RPATH("cmnpd_curated.parquet")
CELLLINE = RPATH("cmnpd_glioma_cellline_activity.csv")


cur = pd.read_parquet(CURATED)
gl = pd.read_csv(CELLLINE, dtype={"COMPOUND_ID": str})
glc = gl[gl.pActivity.notna() & (~gl.censored)]
gorder = sorted(set(glc.CELL_LINE))

scaf = cur[cur.murcko_scaffold != ""].murcko_scaffold.value_counts()
cum = np.cumsum(scaf.values) / len(cur) * 100
labs = ["very high", "high", "medium", "low", "undefined"]
vals = [int((cur.cmnpd_blood_brain_barrier_penetration_level == l).sum()) for l in range(5)]

apply_figure_style(sizes=(8, 7, 6))
C = {"main": "#2f6f9f", "accent": "#b8562f", "grey": "#9a9a9a", "dark": "#333333"}
# The median rules of panels A and C are drawn red so they read as an annotation
# over the distribution rather than as another data series; their labels stay black
# because the text sits above the axes where no colour coding is needed.
MEDIAN_RED = "#c0392b"
plt.close("all")
np.random.seed(0)
fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.1))
(a, b, c), (d, e, f) = axes

a.hist(cur.mw, bins=60, color=C["main"], edgecolor="white", linewidth=0.2)
a.axvspan(900, cur.mw.max(), color=C["accent"], alpha=0.12, lw=0)
a.axvline(cur.mw.median(), color=MEDIAN_RED, lw=1, ls="--")
a.set_xlabel("Molecular weight (Da)"); a.set_ylabel("Compounds")
# Centred on the dashed median line and pushed to the very top of the panel, so the
# label identifies the line by position and the histogram below it stays clear. The
# line is red and the label black: the colour marks the line as an annotation rather
# than data, while black keeps the text legible against the bars.
a.text(cur.mw.median(), 1.0, f"Median {cur.mw.median():.0f} Da",
       transform=a.get_xaxis_transform(), ha="center", va="bottom",
       fontsize=6, color="black")
a.annotate(f"> 900 Da flagged,\nkept: n = {int(cur.flag_large_np.sum()):,}", xy=(0.58, 0.52),
           xycoords="axes fraction", fontsize=6, color=C["accent"])
a.margins(x=0.04)

hb = b.hexbin(cur.clogp, cur.tpsa, gridsize=45, bins="log", cmap="Blues", mincnt=1, linewidths=0)
b.add_patch(Rectangle((1, 0), 4, 90, fill=False, ec=C["accent"], lw=1.1))
b.set_xlim(-10, 25); b.set_ylim(-15, 400)
b.set_xlabel("cLogP"); b.set_ylabel("TPSA (Å$^2$)")
b.annotate("cLogP 1–5,\nTPSA ≤ 90 Å$^2$", xy=(5.2, 50), xytext=(9.5, 230), fontsize=6,
           color=C["accent"], arrowprops=dict(arrowstyle="-", lw=0.6, color=C["accent"]))
cb = fig.colorbar(hb, ax=b, pad=0.02, fraction=0.046)
cb.set_label("Compounds (log)", fontsize=6); cb.ax.tick_params(labelsize=6)

c.hist(cur.np_likeness.dropna(), bins=50, color=C["main"], edgecolor="white", linewidth=0.2)
c.axvline(cur.np_likeness.median(), color=MEDIAN_RED, lw=1, ls="--")
c.set_xlabel("NP-likeness score"); c.set_ylabel("Compounds")
c.text(cur.np_likeness.median(), 1.0, f"Median {cur.np_likeness.median():.2f}",
       transform=c.get_xaxis_transform(), ha="center", va="bottom",
       fontsize=6, color="black")
c.margins(x=0.04)

d.bar(range(5), vals, color=[C["accent"], C["accent"], C["main"], C["main"], "white"],
      edgecolor=C["grey"], linewidth=0.6, hatch=["", "", "", "", "///"])
for i, v in enumerate(vals):
    d.text(i, v + 250, f"{v:,}", ha="center", fontsize=6, color=C["dark"])
d.set_xticks(range(5)); d.set_xticklabels(labs, rotation=30, ha="right")
d.set_xlabel("Predicted BBB penetration level"); d.set_ylabel("Compounds")
# The 'outside model applicability' note that sat on the fifth bar has moved to the
# caption; the hatched, unfilled bar already marks that category as the one the
# vendor model does not cover, so the in-panel text was redundant with its styling.
d.set_ylim(0, max(vals) * 1.2)
# rotation_mode='anchor' pins each rotated tick label by its own right edge at the
# tick. Without it matplotlib rotates about the label's bounding-box centre, which
# displaces the text sideways relative to the category it names.
plt.setp(d.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

e.plot(np.arange(1, len(cum) + 1), cum, color=C["main"], lw=1.4)
e.set_xscale("log"); e.set_xlabel("Bemis–Murcko scaffold rank")
e.set_ylabel("Cumulative compounds (%)")
# Anchored to the stretch of the curve it describes, in the idiom of panel B. The
# scaffolds are ranked by the number of compounds they carry, so the singletons are
# the final run: the leader points at the rank where that run begins, which is also
# where the curve's slope becomes constant. Without the anchor the count floated
# over the middle of the curve and read as if it described the whole distribution.
_n_single = int((scaf == 1).sum())
_x_single = max(1, len(cum) - _n_single + 1)
e.annotate(f"{_n_single:,} singleton scaffolds\nof {scaf.size:,} total",
           xy=(_x_single, cum[_x_single - 1]), xytext=(0.06, 0.62),
           textcoords="axes fraction", fontsize=6, color=C["dark"],
           ha="left", va="top",
           arrowprops=dict(arrowstyle="-", lw=0.6, color=C["dark"],
                           shrinkA=1, shrinkB=2))
e.set_ylim(0, 100); e.margins(x=0.04)

vlist = [glc.loc[glc.CELL_LINE == k, "pActivity"].values for k in gorder]
for i, ys in enumerate(vlist):
    jit = (np.random.rand(ys.size) - 0.5) * 2 * 0.24
    f.scatter(np.full(ys.size, i) + jit, ys, s=10, color=C["main"], alpha=0.6,
              linewidths=0, zorder=2)
    f.plot([i - 0.22, i + 0.22], [np.median(ys)] * 2, color="black", lw=1.6, zorder=3)
f.set_xticks(range(len(gorder))); f.set_xticklabels(gorder)
f.set_ylabel("pActivity (−log$_{10}$ M)"); f.set_xlabel("Glioma cell line")
# Nine categories in a narrow panel put three-digit counts 0.5 px apart, so the
# counts are staggered over two rows rather than shrunk below the 6 pt floor.
for i, v in enumerate(vlist):
    f.text(i, 3.00 if i % 2 == 0 else 2.30, f"{v.size}", ha="center", fontsize=6,
           color=C["dark"])
f.text(-0.6, 3.00, "n =", ha="right", fontsize=6, color=C["dark"])
f.set_ylim(2.1, 12.2); f.set_xlim(-0.8, len(gorder) - 0.4)
plt.setp(f.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

for ax, L in zip([a, b, c, d, e, f], "ABCDEF"):
    panel_letter(ax, L)
fig.tight_layout(w_pad=1.9, h_pad=1.6)

_pdf, _png = save_figure(fig, "fig02_marine_library_properties", outdir=_OUT)
print("FIGURE_OK", "fig02_marine_library_properties", _pdf, _png, flush=True)
