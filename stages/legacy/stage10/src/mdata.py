"""Dataset assembly: parquet -> PyG graph objects, split loaders, task weighting."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as P, CFG, out_dir
import os, pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from feat import TASKS, featurise_list, mol_to_graph

# Resolved LAZILY, on first attribute access. These were resolved at import time,
# which meant that merely importing mdata - and therefore importing train_gnn, which
# imports it - raised MissingInput unless all seven study data products were already
# on disk. That made the stage-09 synthetic smoke test, which reads none of them,
# impossible to run before a 173 MB download. Behaviour with the data present is
# unchanged: the same repro.config.path() call, the same MissingInput if it is absent.
_LAZY_INPUTS = {
    "MATRIX": "chembl_multitask_matrix.parquet",
    "DECOYS": "decoy_set.parquet",
    "FDA": "fda_positive_controls.csv",
    "MARINE_HOLDOUT": "marine_holdout.csv",
    "CMNPD": "cmnpd_curated.parquet",
    "CELLLINE": "cmnpd_glioma_cellline_activity.csv",
    "AD": "applicability_domain.csv",
}


def _lazy(name):
    """Resolve and memoise one declared input path."""
    if name not in globals():
        globals()[name] = P(_LAZY_INPUTS[name])
    return globals()[name]


def __getattr__(name):            # PEP 562: mdata.MATRIX still works, now lazily
    if name in _LAZY_INPUTS:
        return _lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
CACHE = _os.environ.get("MARINE_REPRO_CACHE") or _os.path.join(out_dir(), "cache")


class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "rev":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


def to_data(g, y=None, mask=None, sw=1.0, idx=0):
    d = MolData(x=torch.from_numpy(g["x"]),
                edge_index=torch.from_numpy(g["edge_index"]),
                edge_attr=torch.from_numpy(g["edge_attr"]),
                rev=torch.from_numpy(g["rev"]),
                pe=torch.from_numpy(g["pe"]))
    d.y = torch.zeros(1, len(TASKS)) if y is None else torch.from_numpy(y).float().view(1, -1)
    d.mask = torch.zeros(1, len(TASKS)) if mask is None else torch.from_numpy(mask).float().view(1, -1)
    d.sw = torch.tensor([sw], dtype=torch.float)
    d.idx = torch.tensor([idx], dtype=torch.long)
    return d


def _cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return obj


def load_matrix():
    df = pd.read_parquet(_lazy("MATRIX"))
    Y = np.stack([df[f"label_{t}"].to_numpy(dtype="float64") for t in TASKS], axis=1)
    mask = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y, nan=0.0).astype(np.float32)
    return df, Yf, mask


def build_chembl_graphs():
    df, _, _ = load_matrix()
    graphs, ok = featurise_list(df["canonical_smiles"].tolist(), tag=":chembl")
    return graphs, ok


def chembl_dataset():
    df, Y, mask = load_matrix()
    graphs, ok = _cached("chembl_graphs.pkl", build_chembl_graphs)
    ds = {}
    for i in range(len(df)):
        if not ok[i]:
            continue
        ds[i] = (graphs[i], Y[i], mask[i])
    return df, ds


def pos_weight_from(Y, mask, sel, cap=10.0):
    pw = []
    for t in range(len(TASKS)):
        m = mask[sel, t] > 0
        pos = float((Y[sel, t][m] > 0.5).sum()); neg = float((Y[sel, t][m] <= 0.5).sum())
        pw.append(min(max(neg / max(pos, 1.0), 0.1), cap))
    return torch.tensor(pw, dtype=torch.float)


def make_loader(df, ds, rows, batch=128, shuffle=False, sw_col=None, drop_last=False):
    items = []
    for i in rows:
        if i not in ds:
            continue
        g, y, m = ds[i]
        sw = 1.0 if sw_col is None else float(df[sw_col].iloc[i])
        items.append(to_data(g, y, m, sw, i))
    return DataLoader(items, batch_size=batch, shuffle=shuffle, drop_last=drop_last), \
        np.asarray([int(it.idx.item()) for it in items])


def eval_loader_from_smiles(smiles, batch=256, with_pe=True, tag=""):
    graphs, ok = featurise_list(smiles, with_pe=with_pe, tag=tag)
    items, keep = [], []
    for i, g in enumerate(graphs):
        if g is None:
            continue
        items.append(to_data(g, idx=i))
        keep.append(i)
    return DataLoader(items, batch_size=batch, shuffle=False), np.asarray(keep)


# ----------------------------------------------------------- memory-mapped flat store
class FlatStore:
    """Memory-mapped flat graph arrays; concurrent processes share one physical copy."""

    def __init__(self, tag):
        d = os.path.join(CACHE, f"flat_{tag}")
        self.d = d
        self.x = np.load(os.path.join(d, "x.npy"), mmap_mode="r")
        self.pe = np.load(os.path.join(d, "pe.npy"), mmap_mode="r")
        self.ei = np.load(os.path.join(d, "edge_index.npy"), mmap_mode="r")
        self.ea = np.load(os.path.join(d, "edge_attr.npy"), mmap_mode="r")
        self.rv = np.load(os.path.join(d, "rev.npy"), mmap_mode="r")
        self.np_ = np.load(os.path.join(d, "node_ptr.npy"))
        self.ep = np.load(os.path.join(d, "edge_ptr.npy"))
        self.ok = np.load(os.path.join(d, "ok.npy"))
        self.n = len(self.ok)

    def get(self, i, y=None, mask=None, sw=1.0):
        a, b = int(self.np_[i]), int(self.np_[i + 1])
        c, e = int(self.ep[i]), int(self.ep[i + 1])
        d = MolData(x=torch.from_numpy(np.array(self.x[a:b])),
                    edge_index=torch.from_numpy(np.array(self.ei[:, c:e]).astype(np.int64)),
                    edge_attr=torch.from_numpy(np.array(self.ea[c:e])),
                    rev=torch.from_numpy(np.array(self.rv[c:e]).astype(np.int64)),
                    pe=torch.from_numpy(np.array(self.pe[a:b])))
        d.y = torch.zeros(1, len(TASKS)) if y is None else torch.from_numpy(y).float().view(1, -1)
        d.mask = torch.zeros(1, len(TASKS)) if mask is None else torch.from_numpy(mask).float().view(1, -1)
        d.sw = torch.tensor([sw], dtype=torch.float)
        d.idx = torch.tensor([i], dtype=torch.long)
        return d


class FlatSubset(torch.utils.data.Dataset):
    def __init__(self, store, rows, Y=None, mask=None, sw=None):
        self.s, self.rows, self.Y, self.mask, self.sw = store, np.asarray(rows), Y, mask, sw

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = int(self.rows[k])
        return self.s.get(i, None if self.Y is None else self.Y[i],
                          None if self.mask is None else self.mask[i],
                          1.0 if self.sw is None else float(self.sw[i]))


def flat_loader(store, rows, batch=256, shuffle=False, Y=None, mask=None, sw=None,
                drop_last=False):
    ds = FlatSubset(store, rows, Y, mask, sw)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last)
