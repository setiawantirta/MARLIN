"""Digest every plotted value in a matplotlib figure, so a before/after pair of renders
can be compared numerically. Used to prove that removing panel titles changed no data."""
import hashlib
import json
import numpy as np


def _h(a):
    a = np.asarray(a, dtype=object) if a is None else np.asarray(a)
    if a.dtype == object:
        return "obj:" + hashlib.sha1(repr(a.tolist()).encode()).hexdigest()[:16]
    a = np.ascontiguousarray(np.asarray(a, float))
    a = np.where(np.isfinite(a), a, -1.2345e38)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16] + f"|shape{a.shape}"


def digest(fig):
    out = []
    for i, ax in enumerate(fig.axes):
        d = {"ax": i, "lines": [], "coll": [], "patches": [], "images": [],
             "xlim": list(np.round(ax.get_xlim(), 10)),
             "ylim": list(np.round(ax.get_ylim(), 10)),
             "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
             "xticks": _h(ax.get_xticks()), "yticks": _h(ax.get_yticks()),
             "xticklab": [t.get_text() for t in ax.get_xticklabels()],
             "yticklab": [t.get_text() for t in ax.get_yticklabels()],
             "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel()}
        for ln in ax.lines:
            d["lines"].append([_h(ln.get_xdata()), _h(ln.get_ydata())])
        for c in ax.collections:
            e = {}
            try:
                e["off"] = _h(c.get_offsets())
            except Exception:
                pass
            try:
                arr = c.get_array()
                if arr is not None:
                    e["arr"] = _h(np.asarray(arr))
            except Exception:
                pass
            try:
                sz = c.get_sizes()
                if sz is not None and len(sz):
                    e["sz"] = _h(np.asarray(sz))
            except Exception:
                pass
            try:
                segs = c.get_segments()
                e["segs"] = _h(np.concatenate([np.asarray(s) for s in segs])) if segs else "none"
            except Exception:
                pass
            try:
                paths = c.get_paths()
                if paths:
                    e["paths"] = _h(np.concatenate([p.vertices for p in paths]))
            except Exception:
                pass
            d["coll"].append(e)
        for p in ax.patches:
            try:
                d["patches"].append(_h(p.get_path().transformed(p.get_patch_transform()).vertices))
            except Exception:
                d["patches"].append("na")
        for im in ax.images:
            d["images"].append(_h(np.asarray(im.get_array(), float)))
        out.append(d)
    return out


def annotations(fig):
    """Every visible text string in the figure, for a text-level before/after diff."""
    import matplotlib.text
    return sorted(t.get_text() for t in fig.findobj(matplotlib.text.Text)
                  if t.get_visible() and (t.get_text() or "").strip())


def save(fig, path):
    json.dump({"digest": digest(fig), "texts": annotations(fig)},
              open(path, "w"), indent=0)


DATA_KEYS = ("lines", "coll", "patches", "images", "xlim", "ylim", "xscale", "yscale",
             "xticks", "yticks", "xticklab", "yticklab")


def compare(pa, pb):
    """(data identical?, texts removed, texts added, non-data field diffs)"""
    A = json.load(open(pa)); B = json.load(open(pb))
    da, db = A["digest"], B["digest"]
    data_same = (len(da) == len(db)) and all(
        all(x.get(k) == y.get(k) for k in DATA_KEYS) for x, y in zip(da, db))
    other = []
    for x, y in zip(da, db):
        for k in ("xlabel", "ylabel"):
            if x.get(k) != y.get(k):
                other.append((x["ax"], k, x.get(k), y.get(k)))
    gone = [t for t in A["texts"] if t not in B["texts"]]
    added = [t for t in B["texts"] if t not in A["texts"]]
    return data_same, gone, added, other
