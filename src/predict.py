"""Reusable multi-target predictor for the marine anti-glioma model.

    from predict import predict_smiles
    P = predict_smiles(["CCO", "c1ccccc1"])          # (n, 6) probability matrix
    P, cols = predict_smiles(smis, return_columns=True)

`model_config.json` records which model was selected, its weight files and the six task
columns. Two model families are supported:

  * "fingerprint" - the selected Random Forest on ECFP4(2048, r=2) + 30 RDKit
    descriptors, one classifier per target (this is the model the paper selects).
  * "gnn"         - any of the five graph architectures; when several seed
    checkpoints are listed the returned probability is their mean.
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from feat import mol_to_graph, TASKS           # noqa: E402
import models as _models                       # noqa: E402
from mdata import MolData                      # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

_CACHE = {}


def _load(config_path=None):
    cfg_path = config_path or os.path.join(_HERE, "model_config.json")
    if cfg_path in _CACHE:
        return _CACHE[cfg_path]
    cfg = json.load(open(cfg_path))
    nets = []
    if cfg["family"] == "fingerprint":
        import joblib
        for t in cfg["tasks"]:
            p = cfg["checkpoints"][t]
            nets.append(joblib.load(p if os.path.isabs(p) else os.path.join(_HERE, p)))
    else:
        for ck in cfg["checkpoints"]:
            p = ck if os.path.isabs(ck) else os.path.join(_HERE, ck)
            m = _models.build(cfg["architecture"])
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            m.eval()
            nets.append(m)
    _CACHE[cfg_path] = (cfg, nets)
    return cfg, nets


def _predict_fingerprint(smiles, cfg, nets):
    """ECFP4 + RDKit descriptor featurisation, then one Random Forest per target."""
    from fp_features import featurise
    FP, D, ok = featurise(list(smiles))
    med = np.asarray(cfg["desc_median"], dtype=np.float32)
    mu = np.asarray(cfg["desc_mean"], dtype=np.float32)
    sd = np.asarray(cfg["desc_std"], dtype=np.float32)
    D = np.where(np.isnan(D), med, D)
    Xf = np.hstack([FP.astype(np.float32), ((D - mu) / sd).astype(np.float32)])
    out = np.full((len(smiles), len(cfg["tasks"])), np.nan)
    if ok.any():
        for ti, clf in enumerate(nets):
            out[ok, ti] = clf.predict_proba(Xf[ok])[:, 1]
    return out


def predict_smiles(smiles, batch_size=256, config_path=None, return_columns=False,
                   threads=None):
    """Probability of activity on each of the six glioma targets.

    smiles : list[str]
    returns: (n, 6) float array; rows for unparsable SMILES are NaN.
    """
    if threads:
        torch.set_num_threads(threads)
    cfg, nets = _load(config_path)
    tasks = cfg["tasks"]
    if cfg["family"] == "fingerprint":
        out = _predict_fingerprint(smiles, cfg, nets)
        return (out, tasks) if return_columns else out
    out = np.full((len(smiles), len(tasks)), np.nan, dtype=np.float64)
    items, rows = [], []
    for i, s in enumerate(smiles):
        g = mol_to_graph(s) if isinstance(s, str) else None
        if g is None:
            continue
        d = MolData(x=torch.from_numpy(g["x"]),
                    edge_index=torch.from_numpy(g["edge_index"]),
                    edge_attr=torch.from_numpy(g["edge_attr"]),
                    rev=torch.from_numpy(g["rev"]),
                    pe=torch.from_numpy(g["pe"]))
        d.idx = torch.tensor([i], dtype=torch.long)
        items.append(d)
        rows.append(i)
    if items:
        preds, idxs = [], []
        with torch.no_grad():
            for b in DataLoader(items, batch_size=batch_size):
                p = np.mean([torch.sigmoid(m(b)).numpy() for m in nets], axis=0)
                preds.append(p)
                idxs.append(b.idx.numpy())
        out[np.concatenate(idxs)] = np.concatenate(preds)
    if return_columns:
        return out, tasks
    return out


def polypharmacology(P, threshold=0.5):
    """(max probability, summed probability, number of targets above threshold)."""
    P = np.asarray(P, dtype=float)
    return (np.nanmax(P, axis=1), np.nansum(P, axis=1),
            np.nansum(P >= threshold, axis=1).astype(int))


if __name__ == "__main__":
    smis = sys.argv[1:] or ["CC(=O)Oc1ccccc1C(=O)O"]
    P, cols = predict_smiles(smis, return_columns=True)
    for s, row in zip(smis, P):
        print(s, dict(zip(cols, np.round(row, 4))))
