import os as _rsys_os, sys as _rsys
_rsys.path.insert(0, _rsys_os.path.dirname(_rsys_os.path.dirname(_rsys_os.path.abspath(__file__))))
"""Figure 1 - glioma STRING PPI network, revision-round redraw.

Layout/labelling revision only. The graph is rebuilt from the same STRING record that
produced the published figure (186 nodes, 1,421 edges at combined score >= 0.70, giant
component 176) and is also byte-verified against a fresh STRING v12.0 query. No plotted
value changes: same nodes, same edges, same module assignment, same degree encoding.

Changes in this round
  - figure title removed (migrated to the legend; see figure_title_migration.md)
  - key moved from the upper-left corner of the axes to a dedicated band at the bottom
  - module-block layout replaces the single spring hairball: each module is laid out on
    its own induced subgraph and the blocks are placed on a ring, so the four communities
    read as four groups
  - modules now carry a marker shape and a tinted hull in addition to colour, so the
    partition survives greyscale and colour-vision deficiency
  - the degree-to-area mapping is the published one (area = 14 + 5.0 x degree) and is
    now named by a size key; only the red selection ring's extra radius was reduced
    (+210 -> +150 points^2, lw 1.7 -> 1.45) so the four adjacent M1 hubs stay legible
  - edges muted so nodes dominate
  - labels restricted to the six selected targets plus the ten highest composite-hub-rank
    genes (unchanged from the published figure)
"""
import json, math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

MOD_LBL = {"M1": "M1  RTK-PI3K/MAPK & VEGF signalling (n=64)",
           "M2": "M2  Chromatin / epigenetic regulation (n=56)",
           "M3": "M3  TP53-DNA damage & senescence (n=43)",
           "M4": "M4  Ubiquitin-proteasome degradation (n=8)",
           "unassigned": "unassigned (n=5 in component)"}
MOD_COL = {"M1": "#3B6FB6", "M2": "#C98B1B", "M3": "#4B9560",
           "M4": "#8B5FA8", "unassigned": "#BFBFBF"}
MOD_MRK = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "unassigned": "o"}
FINAL = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]


def build(net_json, modules_json, centrality_csv):
    net = json.load(open(net_json))
    mod_of = json.load(open(modules_json))
    cen = pd.read_csv(centrality_csv)
    G = nx.Graph()
    for n in net["nodes"]:
        G.add_node(n["name"])
    for e in net["edges"]:
        G.add_edge(e["a"], e["b"], weight=e["score"])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    Gc = G.subgraph(comps[0]).copy()
    return G, Gc, mod_of, cen, [n for c in comps[1:] for n in c]


def module_block_layout(Gc, mod_of, seed=3):
    """Per-module spring layout, blocks placed on a ring ordered so that the module
    pairs sharing the most edges end up adjacent."""
    mods = ["M1", "M2", "M3", "M4"]
    members = {m: [n for n in Gc if mod_of.get(n, "unassigned") == m] for m in mods}
    unass = [n for n in Gc if mod_of.get(n, "unassigned") not in mods]

    # inter-module edge weight, used only to order the blocks on the ring
    W = {}
    for a, b, d in Gc.edges(data=True):
        ma, mb = mod_of.get(a, "unassigned"), mod_of.get(b, "unassigned")
        if ma in mods and mb in mods and ma != mb:
            W[frozenset((ma, mb))] = W.get(frozenset((ma, mb)), 0) + 1
    best, best_s = None, -1
    for perm in [("M1", "M2", "M3", "M4"), ("M1", "M2", "M4", "M3"), ("M1", "M3", "M2", "M4")]:
        s = sum(W.get(frozenset((perm[i], perm[(i + 1) % 4])), 0) for i in range(4))
        if s > best_s:
            best, best_s = perm, s
    ring = list(best)

    # block radii from membership size; M4 (n=8) gets a floor so it is not a dot
    rad = {m: max(0.42, 0.135 * math.sqrt(len(members[m]))) for m in mods}
    anchor_r = 1.32
    pos = {}
    for i, m in enumerate(ring):
        th = 2 * math.pi * i / 4 + math.pi / 4
        ax_, ay_ = anchor_r * math.cos(th), anchor_r * math.sin(th)
        sub = Gc.subgraph(members[m])
        p = nx.spring_layout(sub, weight="weight", seed=seed,
                             k=1.5 / math.sqrt(max(len(sub), 2)), iterations=900)
        P = np.array([p[n] for n in members[m]], float)
        P -= P.mean(0)
        sc = np.abs(P).max() or 1.0
        P = P / sc * rad[m]
        for n, xy in zip(members[m], P):
            pos[n] = (ax_ + xy[0], ay_ + xy[1])
    # unassigned nodes sit in the centre gap, spread on a small circle
    for i, n in enumerate(sorted(unass)):
        th = 2 * math.pi * i / max(len(unass), 1)
        pos[n] = (0.30 * math.cos(th), 0.30 * math.sin(th))
    return pos, ring, members, unass


#: label stack side per module - each module's labelled genes are stacked just
#: outside its own block, on the side facing away from the figure centre, so leader
#: lines stay short and cannot cross another module
STACK_SIDE = {"M1": "right", "M3": "left", "M2": "left", "M4": "right",
              "unassigned": "right"}


def draw(Gc, pos, mod_of, cen, out="glioma_ppi_network", members=None):
    from repro.figstyle import apply_figure_style
    apply_figure_style(sizes=(8, 7, 6))

    degd = dict(Gc.degree())
    rank = dict(zip(cen.gene_symbol, cen.composite_hub_rank))
    top10 = set(cen.head(10).gene_symbol)

    # --- pre-compute the label stacks and the axes limits so that the figure height can
    #     be matched to the data aspect ratio: with equal aspect an ill-matched height
    #     leaves a white band between the network and the key.
    labels = list(dict.fromkeys(FINAL + sorted(top10, key=lambda g: rank[g])))
    P = np.array([pos[n] for n in Gc])
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    span_y = ymax - ymin
    dyp = 0.050 * span_y
    placed = []
    for m in ["M1", "M2", "M3", "M4", "unassigned"]:
        gl = [n for n in labels if mod_of.get(n, "unassigned") == m]
        if not gl:
            continue
        Q = np.array([pos[n] for n in Gc if mod_of.get(n, "unassigned") == m])
        side = STACK_SIDE[m]
        lx = (Q[:, 0].max() + 0.085 * span_y) if side == "right" else (Q[:, 0].min() - 0.085 * span_y)
        gl = sorted(gl, key=lambda n: -pos[n][1])
        cyb = float(np.mean([pos[n][1] for n in gl]))
        y0 = cyb + dyp * (len(gl) - 1) / 2
        for i, n in enumerate(gl):
            placed.append((n, lx, y0 - i * dyp, side))
    lxs = [p[1] for p in placed]; lys = [p[2] for p in placed]
    padx = 0.055 * (xmax - xmin); pady = 0.045 * span_y
    X0 = min(xmin, min(lxs)) - padx - 0.20 * span_y
    X1 = max(xmax, max(lxs)) + padx + 0.20 * span_y
    Y0 = min(ymin, min(lys)) - pady
    Y1 = max(ymax, max(lys)) + pady

    # LEG_FRAC is the share of the figure given to the legend band. At 0.152 that band
    # was taller than the three rows of keys inside it, and because the save uses the
    # full figure box the surplus was written into the image as about 6.7% dead space
    # at the foot, three times what the other figures carry and visible as a gap
    # between the figure and its caption. 0.098 matches the band to its contents.
    W_IN, LEG_FRAC, HSP, TOP, BOT = 6.69, 0.098, 0.015, 0.996, 0.010
    ax_w_in = W_IN * (0.996 - 0.004)
    ax_h_in = ax_w_in * (Y1 - Y0) / (X1 - X0)
    net_frac = (TOP - BOT - HSP) / (1.0 + LEG_FRAC)
    fig_h = ax_h_in / net_frac

    fig = plt.figure(figsize=(W_IN, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, LEG_FRAC], hspace=HSP,
                          left=0.004, right=0.996, top=TOP, bottom=BOT)
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
    axl = fig.add_subplot(gs[1, 0]); axl.set_axis_off()

    # --- tinted module hulls: a non-hue channel for the partition
    for m in ["M1", "M2", "M3", "M4"]:
        ns = [n for n in Gc if mod_of.get(n, "unassigned") == m]
        P = np.array([pos[n] for n in ns])
        if len(P) >= 3:
            hull = ConvexHull(P)
            V = P[hull.vertices]
            C = V.mean(0)
            V = C + (V - C) * 1.09
            ax.add_patch(Polygon(V, closed=True, facecolor=MOD_COL[m], alpha=0.06,
                                 edgecolor=MOD_COL[m], lw=0.4, zorder=0))

    # --- edges, muted so the nodes dominate; within-module a shade darker
    win, btw = [], []
    for a, b in Gc.edges():
        (win if mod_of.get(a, "u") == mod_of.get(b, "u") else btw).append((a, b))
    nx.draw_networkx_edges(Gc, pos, edgelist=btw, ax=ax, edge_color="#C7CCD1",
                           width=0.26, alpha=0.30)
    nx.draw_networkx_edges(Gc, pos, edgelist=win, ax=ax, edge_color="#9EA6AD",
                           width=0.34, alpha=0.42)

    # --- nodes: area proportional to degree, shape and colour both encode the module
    for m in ["unassigned", "M4", "M2", "M3", "M1"]:
        ns = [n for n in Gc.nodes() if mod_of.get(n, "unassigned") == m]
        if not ns:
            continue
        ax.scatter([pos[n][0] for n in ns], [pos[n][1] for n in ns],
                   s=[14 + 5.0 * degd[n] for n in ns], c=MOD_COL[m],
                   marker=MOD_MRK[m], linewidths=0.3, edgecolors="white",
                   alpha=0.93, zorder=3)
    ax.scatter([pos[n][0] for n in FINAL], [pos[n][1] for n in FINAL],
               s=[14 + 5.0 * degd[n] + 150 for n in FINAL], facecolors="none",
               marker="o", linewidths=1.45, edgecolors="#C0392B", zorder=4)

    # --- labels: the six selected targets plus the ten top composite-hub-rank genes.
    #     Labels are stacked immediately outside their own module block, ordered by the
    #     node's y coordinate, so each leader line is short and stays within the wedge
    #     of its own module.
    for n, lx, ly, side in placed:
        bold = n in FINAL
        ax.annotate(n, xy=pos[n], xytext=(lx, ly), fontsize=7.4 if bold else 6.8,
                    ha="left" if side == "right" else "right", va="center",
                    style="italic", fontweight="bold" if bold else "normal",
                    color="#C0392B" if bold else "#202124", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#8A9096",
                                    shrinkA=1, shrinkB=2.5))
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal", adjustable="box")

    # --- key in its own band at the bottom of the figure
    mh = [Line2D([], [], marker=MOD_MRK[m], ls="", ms=5.2, color=MOD_COL[m],
                 markeredgecolor="white", markeredgewidth=0.3, label=MOD_LBL[m])
          for m in ["M1", "M2", "M3", "M4", "unassigned"]]
    mh.append(Line2D([], [], marker="o", ls="", color="none", markerfacecolor="none",
                     markeredgecolor="#C0392B", markeredgewidth=1.7, ms=8.5,
                     label="selected multi-task panel (n=6)"))
    lg1 = axl.legend(handles=mh, loc="upper left", bbox_to_anchor=(0.005, 1.06),
                     frameon=False, fontsize=6.6, ncol=2, handletextpad=0.5,
                     columnspacing=1.0, labelspacing=0.42, borderpad=0.0)
    axl.add_artist(lg1)
    dref = [10, 50, 94]
    sh = [Line2D([], [], marker="o", ls="", color="#8A9096", markeredgecolor="white",
                 markeredgewidth=0.3, ms=math.sqrt(14 + 5.0 * d) * 0.72,
                 label=f"{d}") for d in dref]
    axl.legend(handles=sh, loc="upper right", bbox_to_anchor=(0.998, 1.06),
               frameon=False, fontsize=6.6, ncol=3, handletextpad=0.45,
               columnspacing=0.9, borderpad=0.0,
               title="node area $\\propto$ STRING degree", title_fontsize=6.6)
    # fixed physical size: 6.69 in = 170.0 mm, the journal's full-page width. No tight
    # bbox, so the saved width is exactly the figsize width. Cropping to the tight
    # bbox instead makes this taller, not shorter: the two legends are anchored above
    # their own axes and so extend past the canvas, which the tight bbox then includes.
    bb = mpl.transforms.Bbox([[0, 0], list(fig.get_size_inches())])
    fig.savefig(out + ".png", dpi=300, bbox_inches=bb)
    fig.savefig(out + ".pdf", bbox_inches=bb)
    return fig, pos, labels
