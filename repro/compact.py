"""Remove panel titles from an already-laid-out figure and close the band they
occupied, WITHOUT resizing any data axes.

The published figures were laid out either by tight_layout or by a GridSpec with
explicit fractional rects. In the first case simply deleting the titles makes the axes
grow into the freed space; in the second it leaves a white band. Neither is what the
revision asks for, which is: same axes, shorter figure.

compact_titles() therefore works on the finished figure:

  1. measures every axes position in INCHES and every panel title's height plus pad,
  2. deletes the titles,
  3. groups axes into rows, and for each row takes the band to remove as the tallest
     title in that row,
  4. shrinks the figure height by the sum of the bands and re-pins every axes to its
     original inch size, shifted down by the bands that lie below it,
  5. shifts figure-level artists (fig.text, fig.legend) by the same rule.

The result has byte-identical data geometry: every axes keeps its width, height and
data limits, so no plotted value can move.
"""
import numpy as np


def _axes_pos_in(ax, W, H):
    p = ax.get_position()
    return [p.x0 * W, p.y0 * H, p.width * W, p.height * H]


def compact_titles(fig, skip=(), extra_pad_in=0.0):
    """Delete panel titles and close the bands. `skip` holds axes whose title is kept."""
    W, H = fig.get_size_inches()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    pos_in = {ax: _axes_pos_in(ax, W, H) for ax in fig.axes}

    def _titles(ax):
        """matplotlib keeps loc='left'/'right' titles in separate Text artists."""
        return [t for t in (ax.title, getattr(ax, "_left_title", None),
                            getattr(ax, "_right_title", None)) if t is not None]

    band_of = {}
    for ax in fig.axes:
        if ax in skip:
            continue
        live = [t for t in _titles(ax) if (t.get_text() or "").strip() and t.get_visible()]
        if not live:
            continue
        # The reclaimable band is the part of the strip above the axes that ONLY the
        # title occupies. Panel letters usually sit on the same line as the title, so
        # that part of the strip is still needed and must not be reclaimed.
        top_title = max(z.get_window_extent(r).y1 for z in live)
        keep = [ax.get_window_extent(r).y1]
        for o in ax.texts:
            if o.get_visible() and (o.get_text() or "").strip():
                keep.append(o.get_window_extent(r).y1)
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            keep.append(lg.get_window_extent(r).y1)
        band_of[ax] = max(0.0, (top_title - max(keep)) / fig.dpi)

    for ax in list(band_of):
        for t in _titles(ax):
            t.set_text("")
    band_of = {a: b for a, b in band_of.items() if b > 1e-6}

    if not band_of:
        return fig, 0.0

    # --- group axes into rows by their top edge
    tops = sorted({round(pos_in[ax][1] + pos_in[ax][3], 3) for ax in fig.axes})
    rows = []
    for t in tops:
        if rows and abs(t - rows[-1][0]) < 0.06:
            rows[-1][1].append(t)
        else:
            rows.append([t, [t]])
    row_top = [np.mean(g) for _, g in rows]

    def row_of(ax):
        top = pos_in[ax][1] + pos_in[ax][3]
        return int(np.argmin([abs(top - rt) for rt in row_top]))

    band_row = np.zeros(len(row_top))
    for ax, b in band_of.items():
        i = row_of(ax)
        band_row[i] = max(band_row[i], b + extra_pad_in)

    # band i sits immediately above row i, whose top is at row_top[i]
    D = float(band_row.sum())
    H2 = H - D

    def shift_for(y_in):
        """Downward shift for content at height y_in: every band below it is gone."""
        return float(sum(b for rt, b in zip(row_top, band_row) if b > 0 and rt < y_in - 1e-6))

    # figure-level artists first, while the old geometry is still readable
    fig_artists = []
    for t in list(fig.texts):
        x, y = t.get_position()
        fig_artists.append((t, x, y * H))
    dpi = fig.dpi
    for lg in list(fig.legends):
        bb = lg.get_bbox_to_anchor()
        fig_artists.append((lg, None, None))
        # anchor box in figure INCHES, so it survives the height change
        lg._compact_anchor = (bb.x0 / dpi, bb.y0 / dpi, bb.width / dpi, bb.height / dpi)

    fig.set_size_inches(W, H2)

    for ax in fig.axes:
        x0, y0, w, h = pos_in[ax]
        top = y0 + h
        y0n = y0 - shift_for(top)
        ax.set_position([x0 / W, y0n / H2, w / W, h / H2])

    for obj, x, y_in in fig_artists:
        if x is None:
            ax0, ay0, aw, ah = obj._compact_anchor
            ay0n = ay0 - shift_for(ay0 + ah)
            obj.set_bbox_to_anchor((ax0 / W, ay0n / H2, aw / W, ah / H2),
                                   transform=obj.figure.transFigure)
        else:
            obj.set_position((x, (y_in - shift_for(y_in)) / H2))
    return fig, D


def free_band(fig, upper_axes, lower_axes):
    """Vertical figure-fraction band between two rows of axes that nothing occupies.
    Uses rendered tight bboxes, so panel letters, tick labels and axis labels count."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    top = min(a.get_tightbbox(r).y0 for a in upper_axes) / H
    bot = max(a.get_tightbbox(r).y1 for a in lower_axes) / H
    return bot, top


def place_legend_in_band(fig, legend, upper_axes, lower_axes, x_center=0.5):
    """Re-anchor a figure-level legend into the free band below `upper_axes`."""
    bot, top = free_band(fig, upper_axes, lower_axes)
    y = (bot + top) / 2.0
    legend.set_bbox_to_anchor((x_center, y), transform=fig.transFigure)
    legend._loc = 10  # 'center'
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    lb = legend.get_window_extent(r)
    return dict(band=(round(bot, 4), round(top, 4)),
                legend_y=(round(lb.y0 / H, 4), round(lb.y1 / H, 4)),
                fits=bool(lb.y0 / H >= bot - 1e-4 and lb.y1 / H <= top + 1e-4))
