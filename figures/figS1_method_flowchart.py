"""Figure S1 - full method flowchart (Additional Material 1).

Pure matplotlib on a 170 x 225 mm canvas; graphviz is NOT required.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
_OUT = out_dir("stage11_figures")

"""Detailed method flowchart (Additional file) for the marine anti-glioma
multi-target machine-learning study.

Explicit three-column portrait layout in matplotlib, drawn in millimetre
coordinates on a 170 x 225 mm canvas (the journal's full-page figure box).
Graphviz `dot` was tried first; its global layout left 60-70% of the page as
whitespace, which forced a 0.37-0.43 downscale and 2.4-2.8 pt type. Laying the
columns out explicitly keeps the type at its design size.

Every count is read from a saved pipeline artifact; provenance is tabulated in
graphics_notes.md.
"""
import textwrap, re, json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
                     "savefig.bbox": None})

W_MM, H_MM = 170.0, 225.0
MARGIN, GUTTER = 2.0, 5.5
FS = 6.2                 # body text, points
FS_NOTE = 5.7            # discarded-branch notes and connector labels
FS_HEAD = 7.2            # stage headers
LINE = 1.12              # line spacing, multiples of the font size
PAD_X, PAD_Y = 1.0, 0.5  # in-box padding, mm
GAP = 1.4                # vertical gap between stacked boxes, mm
HEAD_H = 3.9             # stage header band, mm

COL_W = (W_MM - 2 * MARGIN - 2 * GUTTER) / 3.0
BOX_W = COL_W - 1.6
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(3)]

pt2mm = lambda p: p * 25.4 / 72.0

# ---------------------------------------------------------------- style tokens
STYLE = {
    "keep": dict(fc="white",   ec="#222222", lw=0.7, style="round", tc="#111111",
                 weight="normal", fs=FS, ls="solid"),
    "data": dict(fc="#ececec", ec="#111111", lw=1.4, style="square", tc="#111111",
                 weight="bold", fs=FS, ls="solid"),
    "drop": dict(fc="#f2f2f2", ec="#8d8d8d", lw=0.6, style="square", tc="#4a4a4a",
                 weight="normal", fs=FS_NOTE, ls=(0, (2.2, 1.6))),
    "dec":  dict(fc="#fbe3bf", ec="#9c6500", lw=0.8, style="round", tc="#3d2800",
                 weight="normal", fs=FS, ls="solid"),
    "eval": dict(fc="#dde8f5", ec="#1f4e79", lw=1.1, style="square", tc="#0f2c47",
                 weight="normal", fs=FS, ls=(0, (1.3, 1.3))),
    "sel":  dict(fc="#fceecd", ec="#8a5a00", lw=1.7, style="square", tc="#2f1e00",
                 weight="bold", fs=FS, ls="solid"),
}
ARROW = {
    "keep": dict(color="#222222", lw=0.7, ls="solid"),
    "drop": dict(color="#8d8d8d", lw=0.6, ls=(0, (2.2, 1.6))),
    "eval": dict(color="#1f4e79", lw=0.9, ls=(0, (1.3, 1.3))),
    "sel":  dict(color="#8a5a00", lw=1.6, ls="solid"),
}

# ------------------------------------------------------------------ content
# (node id, kind, text). "|" forces a line break.
STAGES = [
    ("I.  TARGET DISCOVERY", [
        ("t1", "keep", "Open Targets glioma associations: top 180 by score, "
                       "+ 6 added on CIViC evidence = 186 seeds"),
        ("t2", "keep", "STRING PPI, combined score \u2265 0.70: 186 nodes, "
                       "1,421 edges; giant component 176"),
        ("t2x", "drop", "10 seeds had no edge at \u2265 0.70 \u2014 left isolated"),
        ("t3", "keep", "Greedy modularity: 4 modules, Q = 0.380 "
                       "(64/56/43/8 genes; 15 unassigned)"),
        ("t4", "dec", "Druggable hub with enough ChEMBL data?|34 hubs screened"),
        ("t4x", "drop", "4 hubs below 300 ChEMBL compounds"),
        ("t5", "data", "SIX-TARGET PANEL|EGFR 11,158 \u00b7 KDR 10,610|"
                       "PIK3CA 7,688 \u00b7 IDH1 4,300|PDGFRA 937 \u00b7 CDK6 838"),
    ]),
    ("II.  MARINE LIBRARY CURATION", [
        ("m1", "keep", "CMNPD 1.0: 31,561 marine records (property table + "
                       "2D SDF; 0 unparsable)"),
        ("m2", "keep", "Standardisation (same code for all sources): cleanup, "
                       "largest fragment, uncharge, canonical tautomer"),
        ("m3", "keep", "Element/size QC; InChIKey de-duplication"),
        ("m3x", "drop", "42 arsenic-containing \u00b7 294 outside MW 100\u20131500 Da \u00b7 1 with "
                        "< 6 heavy atoms \u00b7 804 duplicate InChIKeys \u00b7 3,666 "
                        "stereo-skeleton collapses"),
        ("m4", "data", "26,754 CURATED MARINE COMPOUNDS|"
                       "272 with glioma cell-line data"),
    ]),
    ("III.  ChEMBL LABEL MATRIX", [
        ("c1", "keep", "ChEMBL activities, two passes: exact 56,733 \u2192 "
                       "56,706; censored 19,633 \u2192 19,218"),
        ("c2", "keep", "Filters: human, null validity, confidence \u2265 8, non-null "
                       "SMILES/value/units; median pChEMBL per pair"),
        ("c3", "dec", "pChEMBL \u2265 6 active, \u2264 5 inactive"),
        ("c3x", "drop", "discarded: 6,479 medians in the 5\u20136 grey zone \u00b7 4,435 "
                        "non-decisive censored bounds (left missing) \u00b7 11 non-molar "
                        "\u00b7 7 contradictory pairs"),
        ("c4", "keep", "Censored recovery, decisive bounds only: 7,848 "
                       "inactives, 6,935 actives; exact evidence overrides 2,152"),
        ("c6", "data", "LABEL MATRIX 32,403 compounds \u00d7 6 tasks|"
                       "194,418 cells, 35,089 filled \u2192 81.95% missing|"
                       "2,277 labelled on \u2265 2 targets"),
        ("c5", "keep", "+1,787 putative ZINC20 inactives (PIK3CA +1,075, "
                       "IDH1 +736; the two tasks still > 85% active) in separate "
                       "label columns, never in test"),
        ("c7", "data", "SCAFFOLD-DISJOINT SPLIT|12,399 Bemis\u2013Murcko scaffolds, "
                       "leakage = 0|ChEMBL-labelled 32,403: train 22,683 / val 3,240 "
                       "/ test 6,480|with putative rows 34,190: train 24,245 / "
                       "val 3,465 / test 6,480"),
    ]),
    ("IV.  CONTROLS (evaluation only)", [
        ("f1", "eval", "ChEMBL mechanism records for the panel, max phase 4: 158 "
                       "records \u2192 134 molecules \u2192 38 approved"),
        ("f1x", "drop", "14 antibody biologics without SMILES dropped"),
        ("f2", "eval", "POSITIVE CONTROLS: 25 FDA drug\u2013target pairs (24 molecules), "
                       "all forced into the test partition"),
        ("z1", "eval", "ZINC20, 121 property tranches read as random byte windows: "
                       "691,000 \u2192 678,886 unique substances"),
        ("z2", "eval", "NEGATIVE CONTROLS: 45,213 property-matched decoys (36,740 "
                       "substances); DUD-E six-descriptor bins, ECFP4 Tanimoto < 0.35 "
                       "to every active; achieved 7.78:1; no training labels"),
    ]),
    ("V.  TRAINING AND BENCHMARKING", [
        ("v1", "keep", "SEVEN ARCHITECTURES|graph networks: GIN, GATv2, D-MPNN, "
                       "AttentiveFP, GraphGPS|fingerprint ensembles: random forest and "
                       "XGBoost on ECFP4 (2048-bit) + 30 RDKit descriptors"),
        ("v2", "keep", "Multi-target head: shared 2-layer 128-unit trunk + 6 "
                       "task-specific heads, masked BCE with unmeasured cells excluded "
                       "from the gradient; 3 seeds per architecture"),
        ("v3", "keep", "Stage 1 \u2014 pre-train on all 24,245 training rows"),
        ("v4", "keep", "Stage 2 \u2014 natural-product fine-tuning (GIN, D-MPNN): hard NP "
                       "subset n = 548, or NP-likeness-weighted full pool n = 24,245"),
        ("v5", "dec", "PRE-DECLARED SELECTION CRITERION: external marine ROC-AUC, "
                      "not internal test"),
    ]),
    ("VI.  EVALUATION AND SELECTION", [
        ("e1", "keep", "Internal scaffold-disjoint test, 6,480 compounds: macro ROC-AUC "
                       "random forest 0.922, graph networks 0.867\u20130.893"),
        ("e2", "eval", "Decoy enrichment, 5,808 test actives vs 45,213 decoys: random "
                       "forest macro ROC-AUC 0.917, macro EF 1% = 9.4"),
        ("e3", "eval", "FDA control recovery, 25 held-out pairs: random forest median "
                       "rank percentile 64.3; 10 of 25 in the top decile"),
        ("e5", "drop", "marine holdout 18 compounds / 20 labels, none for CDK6 or IDH1 "
                       "\u2014 UNDERPOWERED, reported but not used for selection"),
        ("e4", "sel", "EXTERNAL MARINE PHENOTYPIC SET \u2014 SELECTION ARM|"
                      "194 CMNPD compounds with measured glioma cell-line potency, "
                      "51 active (pAct \u2265 6)|random forest \u03c1 = 0.369 "
                      "(p = 1.1 \u00d7 10\u207b\u2077), ROC-AUC 0.685|"
                      "all five graph networks 0.382\u20130.502 = chance"),
        ("e6", "sel", "SELECTED MODEL: random forest (ECFP4 + 30 descriptors), "
                      "per-target 3-seed mean probability"),
    ]),
    ("VII.  INTERPRETATION", [
        ("i1", "keep", "Candidate feature pool: per-target top-40 MDI union top-40 "
                       "mean |SHAP| = 135 of 2,078 features"),
        ("i2", "keep", "Three independent attributions \u2192 consensus rank: exact "
                       "path-dependent TreeSHAP, permutation importance (ROC-AUC drop, "
                       "5 repeats), mean decrease in impurity"),
        ("i3", "data", "CONSENSUS TOP 10 FEATURES, structurally annotated|"
                       "cross-seed consensus Spearman \u03c1 = 0.952\u20130.969"),
        ("i4", "keep", "Graph-network attribution comparison (integrated gradients vs "
                       "RF-projected SHAP): aromatic-atom share of |attribution| 0.471 "
                       "synthetic vs 0.188 marine, Cohen's d = 1.53"),
    ]),
    ("VIII.  MARINE SCREENING CASCADE", [
        ("s0", "data", "26,754 curated marine compounds scored on all six targets by "
                       "the selected random forest"),
        ("sA", "keep", "Step 1 \u2014 active on \u2265 2 targets|"
                       "nominal (MCC-optimal thresholds) 12,950 = 48.4%|"
                       "stringent (nominal or library 95th percentile) 1,390 = 5.2%"),
        ("sB", "keep", "Step 2 \u2014 inside applicability domain, max Tanimoto \u2265 0.30|"
                       "nominal 794  \u00b7  stringent 147"),
        ("sC", "keep", "Step 3 \u2014 blood\u2013brain barrier not excluded|"
                       "nominal 266  \u00b7  stringent 7"),
        ("sD", "keep", "Step 4 \u2014 PAINS-alert-free|nominal 224  \u00b7  stringent 7"),
        ("sx", "drop", "domain-agnostic variant, applicability carried as a per-row "
                       "flag instead: nominal 3,314 \u00b7 stringent 39"),
        ("s9", "sel", "SEVEN STRINGENT SURVIVORS \u2014 0.026% of the library|"
                      "six pyranose-fused indolo[2,3-a]carbazoles and one dibrominated "
                      "tyrosine alkaloid|ceiling of computational triage, not leads: "
                      # Both figures are computed over the 31,267 CMNPD records before
                      # de-duplication, which is the set Fig. S2E-F plots, not the 26,754
                      # curated library named at the head of this stage. Over the curated
                      # library the same quantities are 8.76% and 0.208. The denominator is
                      # stated because the two differ and the box would otherwise be read
                      # against the 26,754.
                      "only 8.23% of the 31,267 CMNPD records lie inside the 0.30 domain "
                      "and their median max Tanimoto to training chemistry is 0.207"),
    ]),
    ("KEY", [
        ("k1", "keep", "retained data or analysis step"),
        ("k2", "data", "principal data object"),
        ("k3", "dec", "decision point"),
        ("k4", "drop", "discarded branch, or a result reported but not used"),
        ("k5", "eval", "evaluation-only path, no training labels"),
        ("k6", "sel", "arm that produced the selected model"),
    ]),
]

# in-column spine: (src, dst, arrow kind, label)
SPINE = [
    ("t1", "t2", "keep", ""), ("t2", "t2x", "drop", ""), ("t2", "t3", "keep", ""),
    ("t3", "t4", "keep", ""), ("t4", "t4x", "drop", ""), ("t4", "t5", "keep", ""),
    ("t5", "m1", "keep", ""),
    ("m1", "m2", "keep", ""), ("m2", "m3", "keep", ""), ("m3", "m3x", "drop", ""),
    ("m3", "m4", "keep", ""),
    ("c1", "c2", "keep", ""), ("c2", "c3", "keep", ""), ("c3", "c3x", "drop", ""),
    ("c3", "c4", "keep", ""), ("c4", "c6", "keep", ""),     ("c6", "c5", "keep", ""), ("c5", "c7", "keep", ""),
    ("f1", "f1x", "drop", ""), ("f1", "f2", "eval", ""), ("z1", "z2", "eval", ""),
    ("v1", "v2", "keep", ""), ("v2", "v3", "keep", ""), ("v3", "v4", "keep", ""),
    ("v4", "v5", "keep", ""),
    ("e1", "e2", "", ""), ("e4", "e6", "sel", ""),
    ("i1", "i2", "keep", ""), ("i2", "i3", "keep", ""), ("i3", "i4", "keep", ""),
    ("s0", "sA", "keep", ""), ("sA", "sB", "keep", ""), ("sB", "sC", "keep", ""),
    ("sC", "sD", "keep", ""), ("sD", "sx", "drop", ""), ("sD", "s9", "sel", ""),
]
# long / cross-column connectors: (src, dst, kind, label, gutter offset mm)
LINKS = [
    ("c7", "f1", "eval", "test partition", 1.2),
    ("c7", "v1", "keep", "train + validation rows", 2.5),
    ("e6", "i1", "keep", "selected model", 1.2),
    ("e6", "s0", "sel", "selected model", 2.5),
]
# one bus from the selection criterion to every evaluation arm
BUS = ("v5", [("e1", "keep"), ("e2", "eval"), ("e3", "eval"),
              ("e5", "drop"), ("e4", "sel")], 4.3)


_MFIG = plt.figure(figsize=(1, 1))
_MREN = _MFIG.canvas.get_renderer()


def text_w(txt, fs, weight):
    """Rendered width of `txt` in mm, measured with the actual font."""
    t = _MFIG.text(0, 0, txt, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(_MREN).width / _MFIG.dpi * 25.4
    t.remove()
    return w


def wrap(txt, width_mm, fs, weight="normal"):
    """Greedy word wrap on measured text width."""
    out = []
    for para in txt.split("|"):
        words, line = re.sub(r"\s+", " ", para).strip().split(), ""
        for wd in words:
            trial = (line + " " + wd).strip()
            if line and text_w(trial, fs, weight) > width_mm:
                out.append(line); line = wd
            else:
                line = trial
        out.append(line)
    return out


def measure():
    """Wrapped lines and height in mm for every node, plus stage heights."""
    nodes, stage_h = {}, []
    for title, items in STAGES:
        h = HEAD_H
        for nid, kind, txt in items:
            fs = STYLE[kind]["fs"]
            lines = wrap(txt, BOX_W - 2 * PAD_X, fs, STYLE[kind]["weight"])
            bh = len(lines) * pt2mm(fs) * LINE + 2 * PAD_Y
            nodes[nid] = dict(lines=lines, kind=kind, h=bh, fs=fs)
            h += bh + GAP
        stage_h.append(h + 1.2)
    return nodes, stage_h


def best_partition(stage_h, ncol=3):
    """Contiguous split of the stage list into `ncol` columns, minimising the
    tallest column."""
    n = len(stage_h)
    best, cuts = None, None
    for a in range(1, n - 1):
        for b in range(a + 1, n):
            hs = [sum(stage_h[:a]), sum(stage_h[a:b]), sum(stage_h[b:])]
            if best is None or max(hs) < best:
                best, cuts = max(hs), (a, b)
    return cuts, best


def layout():
    nodes, stage_h = measure()
    (a, b), tallest = best_partition(stage_h)
    groups = [list(range(0, a)), list(range(a, b)), list(range(b, len(STAGES)))]
    avail = H_MM - 2 * MARGIN
    frames = []
    for ci, gs in enumerate(groups):
        y = H_MM - MARGIN
        for gi in gs:
            title, items = STAGES[gi]
            y0 = y
            y -= HEAD_H
            for nid, kind, txt in items:
                nd = nodes[nid]
                nd["x"] = COL_X[ci] + (COL_W - BOX_W) / 2
                nd["y"] = y - nd["h"]
                nd["col"] = ci
                y -= nd["h"] + GAP
            frames.append((title, COL_X[ci], y0, COL_W, y0 - (y + GAP - 0.8)))
            y -= 1.2
    return nodes, frames, tallest, avail


def draw():
    nodes, frames, tallest, avail = layout()
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM)
    ax.axis("off")

    for title, x, ytop, w, h in frames:
        ax.add_patch(FancyBboxPatch((x, ytop - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="none", ec="#9a9a9a", lw=0.55, zorder=1))
        ax.text(x + 1.4, ytop - HEAD_H / 2 - 0.3, title, ha="left", va="center",
                fontsize=FS_HEAD, fontweight="bold", color="#111111", zorder=3)

    for nid, nd in nodes.items():
        st = STYLE[nd["kind"]]
        bs = ("round,pad=0,rounding_size=0.9" if st["style"] == "round"
              else "square,pad=0")
        ax.add_patch(FancyBboxPatch((nd["x"], nd["y"]), BOX_W, nd["h"], boxstyle=bs,
                                    fc=st["fc"], ec=st["ec"], lw=st["lw"],
                                    linestyle=st["ls"], zorder=2))
        ax.text(nd["x"] + BOX_W / 2, nd["y"] + nd["h"] / 2, "\n".join(nd["lines"]),
                ha="center", va="center", fontsize=nd["fs"], color=st["tc"],
                fontweight=st["weight"], linespacing=LINE, zorder=3)

    def anchor(nd, where):
        if where == "top":    return (nd["x"] + BOX_W / 2, nd["y"] + nd["h"])
        if where == "bottom": return (nd["x"] + BOX_W / 2, nd["y"])
        if where == "left":   return (nd["x"], nd["y"] + nd["h"] / 2)
        return (nd["x"] + BOX_W, nd["y"] + nd["h"] / 2)

    def blocked(p0, p1, skip):
        """True if the straight segment p0-p1 passes through a box other than its ends."""
        for k, nd in nodes.items():
            if k in skip or "x" not in nd:
                continue
            x0, y0, x1, y1 = nd["x"], nd["y"], nd["x"] + BOX_W, nd["y"] + nd["h"]
            for i in range(1, 60):
                t = i / 60.0
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                if x0 + 0.05 < x < x1 - 0.05 and y0 + 0.05 < y < y1 - 0.05:
                    return True
        return False

    def bypass(p0, p1, xr, kind):
        """Route p0 -> p1 around an intervening box, out through the column gutter.

        A stage that discards a branch has two arrows leaving the same box bottom: one
        into the dashed discarded box and one continuing down the spine. Drawn straight,
        the continuing arrow runs through the discarded box, which is the overlap this
        avoids. The detour leaves and rejoins the spine at the midpoint of the vertical
        gaps, so the right angles line up with the box edges above and below.
        """
        ya = p0[1] - GAP * 0.5
        yb = p1[1] + GAP * 0.5
        verts = [p0, (p0[0], ya), (xr, ya), (xr, yb), (p1[0], yb), p1]
        codes = [Path.MOVETO] + [Path.LINETO] * 5
        ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle="-|>",
                                     mutation_scale=4.2, shrinkA=0, shrinkB=0,
                                     joinstyle="miter", zorder=4, **ARROW[kind]))

    # Gutter channel for a bypass, per column. Columns 0 and 1 route into the gutter on
    # their right; the offsets are clear of the cross-column LINKS, which use 1.2 and 2.5
    # from the same edge. Column 2 has only the page margin on its right, so it routes
    # into the gutter on its left.
    BYPASS_OFF = {0: 4.1, 1: 1.6, 2: -1.6}

    for src, dst, kind, lab in SPINE:
        if not kind:
            continue
        s, d = nodes[src], nodes[dst]
        p0, p1 = anchor(s, "bottom"), anchor(d, "top")
        if blocked(p0, p1, {src, dst}):
            off = BYPASS_OFF.get(s.get("col", 0), 4.1)
            xr = s["x"] + BOX_W + off if off > 0 else s["x"] + off
            bypass(p0, p1, xr, kind)
        else:
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                         mutation_scale=4.2, shrinkA=0, shrinkB=0,
                                         zorder=4, **ARROW[kind]))
        if lab:
            ax.text((p0[0] + p1[0]) / 2 + 0.8, (p0[1] + p1[1]) / 2, lab, ha="left",
                    va="center", fontsize=FS_NOTE, color=ARROW[kind]["color"], zorder=4)

    def elbow(p0, xr, p1, kind):
        verts = [p0, (xr, p0[1]), (xr, p1[1]), p1]
        path = Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=4.2,
                                     shrinkA=0, shrinkB=0, joinstyle="miter",
                                     zorder=4, **ARROW[kind]))
        return (xr, (p0[1] + p1[1]) / 2)

    for src, dst, kind, lab, off in LINKS:
        s_, d_ = nodes[src], nodes[dst]
        xr = s_["x"] + BOX_W + off
        mid = elbow(anchor(s_, "right"), xr, anchor(d_, "left"), kind)
        if lab:
            ax.text(mid[0] - 0.1, mid[1], lab, ha="center", va="center", rotation=90,
                    fontsize=5.2, color=ARROW[kind]["color"], zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))

    bsrc, arms, boff = BUS
    b = nodes[bsrc]
    xr = b["x"] - boff
    ys = [nodes[a]["y"] + nodes[a]["h"] / 2 for a, _ in arms]
    ax.plot([xr, xr], [min(ys), b["y"] + b["h"] / 2], color=ARROW["keep"]["color"],
            lw=ARROW["keep"]["lw"], solid_capstyle="butt", zorder=4)
    ax.plot([b["x"], xr], [b["y"] + b["h"] / 2] * 2, color=ARROW["keep"]["color"],
            lw=ARROW["keep"]["lw"], zorder=4)
    for (a, kind), y in zip(arms, ys):
        ax.add_patch(FancyArrowPatch((xr, y), (nodes[a]["x"], y), arrowstyle="-|>",
                                     mutation_scale=4.2, shrinkA=0, shrinkB=0,
                                     zorder=4, **ARROW[kind]))

    return fig, nodes, tallest, avail


if __name__ == "__main__":
    fig, nodes, tallest, avail = draw()
    print("tallest column %.1f mm / %.1f mm available" % (tallest, avail))
    print("nodes %d  body font %.1f pt  canvas %.0f x %.0f mm" %
          (len(nodes), FS, W_MM, H_MM))
    # geometric check: text inside its own box, boxes inside the canvas
    r = fig.canvas.get_renderer()
    bad = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted())
        x0, y0, x1, y1 = bb.x0 * 25.4, bb.y0 * 25.4, bb.x1 * 25.4, bb.y1 * 25.4
        if x0 < 0 or y0 < 0 or x1 > W_MM or y1 > H_MM:
            bad.append(("offcanvas", t.get_text()[:40], round(x0, 1), round(y0, 1),
                        round(x1, 1), round(y1, 1)))
    over = []
    for nid, nd in nodes.items():
        cw = max(text_w(l, nd["fs"], STYLE[nd["kind"]]["weight"]) for l in nd["lines"])
        if cw > BOX_W - 2 * PAD_X + 0.2:
            over.append((nid, round(cw, 1)))
    print("off-canvas text:", bad[:6])
    print("wrap overflow:", over)
    json.dump({nid: dict(h=nd["h"], y=nd["y"], col=nd["col"], nlines=len(nd["lines"]))
               for nid, nd in nodes.items()}, open(_os.path.join(_OUT, "flowchart_layout.json"), "w"), indent=0)

from repro.figstyle import save_figure as _SF
_pdf, _png = _SF(fig, "figS1_method_flowchart", outdir=_OUT, bbox_inches=None)
print("FIGURE_OK", "figS1_method_flowchart", _pdf, _png, flush=True)
