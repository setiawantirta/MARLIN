"""Figure style for every figure in this module.

Byte-identical to the helper the published figures inlined, with two deliberate
changes required by the uploaded manuscript:

  * :func:`panel_letter` emits UPPERCASE BOLD SERIF letters and ignores any
    ``case="lower"`` argument, so the supplementary figures - which the
    author's own files left lowercase - come out in the main-text convention.
  * :func:`save_figure` always writes a vector PDF next to a 300 dpi PNG. The
    figures embedded in the uploaded .docx are 110-153 dpi; BMC requires 300.

All numeric defaults come from ``config/config.yaml`` under ``figures:``.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- config hook
try:
    from repro.config import CFG as _CFG
    _FIGCFG = _CFG.get("figures", {}) or {}
except Exception:                                   # standalone use
    _FIGCFG = {}
_FIGCFG.setdefault("panel_letter_size", 10)
_FIGCFG.setdefault("panel_letter_weight", "bold")
_FIGCFG.setdefault("panel_letter_family", "serif")
_FIGCFG.setdefault("png_dpi", 300)

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")

    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def set_frame(ax, style="open"):
    show = {"open": (False, False, True, True),
            "boxed": (True, True, True, True),
            "none": (False, False, False, False)}[style]
    for side, vis in zip(("top", "right", "bottom", "left"), show):
        ax.spines[side].set_visible(vis)
        if vis:
            ax.spines[side].set_linewidth(0.6)
    ax.tick_params(direction="out", length=0 if style == "none" else 3, width=0.6)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case=None, fontsize=None):
    """Panel letter in the convention of the uploaded manuscript.

    UPPERCASE, bold, serif. `case` is accepted and ignored so that call sites
    recovered from the pre-revision figures - which passed case="lower" - come
    out in the published convention without being edited. Defaults are read
    from config/config.yaml (figures.panel_letter_*).
    """
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = _FIGCFG.get("panel_letter_size") or (plt.rcParams.get("font.size", 8) + 2)
    ax.text(dx, dy, str(letter).upper(), transform=ax.transAxes,
            fontweight=_FIGCFG.get("panel_letter_weight", "bold"),
            fontfamily=_FIGCFG.get("panel_letter_family", "serif"),
            fontsize=fontsize, va="bottom", ha="left")


def focal_palette(labels, focal, focal_color, other="muted", base_colors=None):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    focal_set = {focal} if isinstance(focal, str) else set(focal)
    n = len(labels)
    if not focal_set & set(labels):
        raise ValueError(f"focal {focal!r} not found in labels")
    if base_colors is None:
        base_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#444444"])
    base_colors = [base_colors[i % len(base_colors)] for i in range(n)]
    if other == "grey":
        rest = ["#BCBCBC"] * n
    elif other == "ordinal":
        nf = max(1, n - len(focal_set))
        ramp = [mcolors.to_hex((v, v, v)) for v in
                ([0.55] if nf == 1 else [0.80 - 0.35 * i / (nf - 1) for i in range(nf)])]
        rest, k = [], 0
        for l in labels:
            rest.append(ramp[min(k, nf - 1)]); k += (l not in focal_set)
    else:
        def mute(c):
            r, g, b = mcolors.to_rgb(c)
            m = (r + g + b) / 3
            return mcolors.to_hex((0.3 * r + 0.7 * m, 0.3 * g + 0.7 * m, 0.3 * b + 0.7 * m))
        rest = [mute(c) for c in base_colors]
    return [focal_color if l in focal_set else rest[i] for i, l in enumerate(labels)]


def panel_crops(fig, dpi=None, pad_px=6, bbox_inches=None, pad_inches=None):
    import matplotlib as mpl
    import matplotlib.text
    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", fig.dpi)
        if dpi == "figure":
            dpi = fig.dpi
    dpi = float(dpi)
    if bbox_inches is None:
        bbox_inches = mpl.rcParams.get("savefig.bbox")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    if bbox_inches == "tight":
        if pad_inches is None:
            pad_inches = mpl.rcParams.get("savefig.pad_inches", 0.1)
        tb = fig.get_tightbbox(r).padded(pad_inches)
        ox_in, oy_in = tb.x0, tb.y0
        W_in, H_in = tb.width, tb.height
    elif isinstance(bbox_inches, mpl.transforms.BboxBase):
        ox_in, oy_in = bbox_inches.x0, bbox_inches.y0
        W_in, H_in = bbox_inches.width, bbox_inches.height
    else:
        ox_in, oy_in = 0.0, 0.0
        W_in, H_in = fig.get_size_inches()
    W_px, H_px = int(round(W_in * dpi)), int(round(H_in * dpi))
    lettered = {}
    for ax in fig.axes:
        for t in ax.findobj(matplotlib.text.Text):
            s = (t.get_text() or "").strip()
            if len(s) == 1 and s.isalpha() and t.get_fontweight() in ("bold", 700):
                lettered[ax] = s
                break

    if not lettered:
        lettered = {ax: str(i) for i, ax in enumerate(fig.axes)}
    out = {}
    for ax, letter in lettered.items():
        bbs = [ax.get_tightbbox(r)]

        ss = ax.get_subplotspec()
        for sib in fig.axes:
            if sib is ax or sib in lettered:
                continue
            ssib = sib.get_subplotspec()
            same_row = ss is None or ssib is None or ss.rowspan == ssib.rowspan
            same_col = ss is None or ssib is None or ss.colspan == ssib.colspan
            if ((ax.get_shared_y_axes().joined(ax, sib) and same_row)
                    or (ax.get_shared_x_axes().joined(ax, sib) and same_col)):
                bbs.append(sib.get_tightbbox(r))
        bb = mpl.transforms.Bbox.union(bbs)

        bx0 = (bb.x0 / fig.dpi - ox_in) * dpi
        bx1 = (bb.x1 / fig.dpi - ox_in) * dpi
        by0 = H_px - (bb.y1 / fig.dpi - oy_in) * dpi
        by1 = H_px - (bb.y0 / fig.dpi - oy_in) * dpi
        out[letter] = (
            max(int(bx0) - pad_px, 0),
            max(int(by0) - pad_px, 0),
            min(int(bx1) + pad_px, W_px),
            min(int(by1) + pad_px, H_px),
        )
    return out




# --------------------------------------------------------------- output helper
def save_figure(fig, stem, outdir=None, dpi=None, bbox_inches="tight"):
    """Write <stem>.pdf (vector) and <stem>.png (300 dpi) and return both paths.

    This is the only sanctioned way for a stage to emit a figure; it guarantees
    the publication-quality pair the journal requires.
    """
    import os
    if outdir is None:
        try:
            from repro.config import out_dir
            outdir = out_dir("stage11_figures")
        except Exception:
            outdir = "."
    os.makedirs(outdir, exist_ok=True)
    capitalise_axis_labels(fig)
    dpi = int(dpi or _FIGCFG.get("png_dpi", 300))
    pdf = os.path.join(outdir, f"{stem}.pdf")
    png = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf, bbox_inches=bbox_inches)
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches)
    return pdf, png


def capitalise_axis_labels(fig):
    """Capitalise the first letter of every x and y axis label on `fig`.

    Applied centrally in save_figure so the convention holds for every figure the
    module emits, rather than being patched panel by panel. Only the first
    character is touched, so an initial LaTeX group or symbol is left alone:
    a label opening with '$' (maths), a digit, or a backslash is skipped, and a
    label already starting with a capital is unchanged. The rest of the string
    keeps its case, which matters for units and for terms such as 'pActivity',
    'sp$^3$ atoms' and 'cosine(synthetic, marine)'.

    Returns the number of labels it changed.
    """
    n = 0
    for ax in fig.axes:
        for getter, setter in ((ax.get_xlabel, ax.set_xlabel),
                               (ax.get_ylabel, ax.set_ylabel)):
            s = getter() or ""
            if not s or not s[0].isalpha() or s[0].isupper():
                continue
            # A lowercase letter followed immediately by a capital is a scientific
            # symbol whose lowercase prefix carries meaning, not a sentence that
            # happens to start small: pActivity, pChEMBL and cLogP all appear as
            # axis labels here, and 'PActivity' or 'CLogP' would be wrong. The
            # pattern catches the whole family (pKa, mRNA, nM) without a list.
            if len(s) > 1 and s[1].isupper():
                continue
            setter(s[0].upper() + s[1:])
            n += 1
    return n


def uppercase_existing_panel_letters(fig):
    """Rewrite any single-letter bold label already drawn on `fig` to the
    manuscript convention. Used on figures whose generator inlined its own
    lowercase panel letters."""
    import matplotlib.text
    n = 0
    for ax in fig.axes:
        for t in ax.findobj(matplotlib.text.Text):
            s = (t.get_text() or "").strip()
            if len(s) == 1 and s.isalpha() and t.get_fontweight() in ("bold", 700):
                t.set_text(s.upper())
                t.set_fontfamily(_FIGCFG.get("panel_letter_family", "serif"))
                t.set_fontweight(_FIGCFG.get("panel_letter_weight", "bold"))
                n += 1
    for t in fig.texts:
        s = (t.get_text() or "").strip()
        if len(s) == 1 and s.isalpha() and t.get_fontweight() in ("bold", 700):
            t.set_text(s.upper())
            t.set_fontfamily(_FIGCFG.get("panel_letter_family", "serif"))
            n += 1
    return n
