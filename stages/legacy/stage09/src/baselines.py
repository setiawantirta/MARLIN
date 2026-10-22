"""Per-target Random Forest and XGBoost baselines on ECFP4 + RDKit descriptors.

Each target is a separate binary model (the conventional baseline formulation), tuned by
randomised search scored on the validation split, then refit on train and scored on test.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata
from feat import TASKS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RF_GRID = {"n_estimators": [300, 500], "max_depth": [None, 20, 40],
           "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.05, 0.1]}
XGB_GRID = {"n_estimators": [300, 600], "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1], "subsample": [0.7, 1.0],
            "colsample_bytree": [0.3, 0.7], "min_child_weight": [1, 5]}


def load_X():
    z = np.load(os.path.join(mdata.CACHE, "chembl_fp.npz"), allow_pickle=True)
    FP, D, ok = z["FP"], z["D"], z["ok"]
    med = np.nanmedian(D, axis=0)
    D = np.where(np.isnan(D), med, D)
    mu, sd = D.mean(0), D.std(0) + 1e-8
    X = np.hstack([FP.astype(np.float32), ((D - mu) / sd).astype(np.float32)])
    return X, ok, {"desc_mean": mu.tolist(), "desc_std": sd.tolist(), "desc_median": med.tolist()}


def sample_grid(grid, n, rng):
    keys = list(grid)
    seen, out = set(), []
    while len(out) < n and len(seen) < np.prod([len(grid[k]) for k in keys]):
        c = tuple(grid[k][rng.integers(len(grid[k]))] for k in keys)
        if c in seen:
            continue
        seen.add(c); out.append(dict(zip(keys, c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["RF", "XGB"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_search", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--params", default="", help="reuse hyperparameters chosen on seed 0")
    ap.add_argument("--predict_cmnpd", action="store_true")
    ap.add_argument("--predict_decoy", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    X, ok, norm = load_X()
    split = df["split"].to_numpy()
    name = f"{a.model}_s{a.seed}_baseline"

    fitted = {}
    Pv = np.full((len(df), len(TASKS)), np.nan, dtype=np.float32)
    Pt = np.full((len(df), len(TASKS)), np.nan, dtype=np.float32)
    chosen = {}
    for ti, task in enumerate(TASKS):
        obs = (mask[:, ti] > 0) & ok
        tr = np.where(obs & (split == "train"))[0]
        va = np.where(obs & (split == "val"))[0]
        te = np.where(obs & (split == "test"))[0]
        ytr, yva = Y[tr, ti].astype(int), Y[va, ti].astype(int)
        pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
        if a.params:
            cands = [json.load(open(a.params))["chosen"][task]["params"]]
        else:
            cands = sample_grid(RF_GRID if a.model == "RF" else XGB_GRID, a.n_search, rng)
        best, bp, bm = -1.0, None, None
        for c in cands:
            if a.model == "RF":
                m = RandomForestClassifier(random_state=a.seed, n_jobs=a.jobs,
                                           class_weight="balanced_subsample", **c)
            else:
                m = XGBClassifier(random_state=a.seed, n_jobs=a.jobs, tree_method="hist",
                                  eval_metric="logloss",
                                  scale_pos_weight=max(neg / max(pos, 1), 1.0), **c)
            m.fit(X[tr], ytr)
            auc = roc_auc_score(yva, m.predict_proba(X[va])[:, 1]) if len(set(yva)) > 1 else np.nan
            if np.isfinite(auc) and auc > best:
                best, bp, bm = auc, c, m
        Pv[va, ti] = bm.predict_proba(X[va])[:, 1]
        Pt[te, ti] = bm.predict_proba(X[te])[:, 1]
        fitted[task] = bm
        chosen[task] = {"params": {k: (None if v is None else v) for k, v in bp.items()},
                        "val_roc_auc": float(best), "n_train": int(len(tr)),
                        "n_pos_train": pos, "n_val": int(len(va)), "n_test": int(len(te))}
        print(f"[{name}] {task} val_auc={best:.4f} {bp} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs("metrics", exist_ok=True)
    np.savez_compressed(f"metrics/{name}_preds.npz", Pv=Pv, Pt=Pt)
    for flag, tag, smi_n in [(a.predict_cmnpd, "cmnpd_fp.npz", "cmnpd"),
                             (a.predict_decoy, "decoy_fp.npz", "decoy")]:
        if not flag:
            continue
        outp = f"metrics/{smi_n}_pred_{name}.npz"
        if os.path.exists(outp):
            continue
        z = np.load(os.path.join(mdata.CACHE, tag), allow_pickle=True)
        FPz, Dz, okz = z["FP"], z["D"], z["ok"]
        med = np.array(norm["desc_median"], dtype=np.float32)
        mu = np.array(norm["desc_mean"], dtype=np.float32)
        sd = np.array(norm["desc_std"], dtype=np.float32)
        nrow = FPz.shape[0]
        P = np.full((nrow, len(TASKS)), np.nan, dtype=np.float32)
        CH = 8000
        for st in range(0, nrow, CH):
            en = min(st + CH, nrow)
            Dc = np.where(np.isnan(Dz[st:en]), med, Dz[st:en])
            Xc = np.hstack([FPz[st:en].astype(np.float32), ((Dc - mu) / sd).astype(np.float32)])
            for ti, task in enumerate(TASKS):
                P[st:en, ti] = fitted[task].predict_proba(Xc)[:, 1]
            del Xc
            print(f"[{name}] {smi_n} {en}/{nrow}", flush=True)
        np.savez_compressed(outp, P=P, I=np.arange(nrow), ok=okz)
        del z, FPz, Dz
        import gc; gc.collect()
    json.dump({"name": name, "arch": a.model, "seed": a.seed, "stage": "baseline",
               "chosen": chosen, "n_search": a.n_search, "wall_s": round(time.time() - t0, 1),
               "feature_dim": int(X.shape[1])},
              open(f"metrics/{name}_run.json", "w"), indent=1)
    print(f"[{name}] DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
