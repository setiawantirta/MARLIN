"""Run this FIRST. It measures how long one fit takes on this machine and projects
the cost of the full grid, so you can decide between --grid full and --grid reduced
before committing a session to it.

It also verifies that the feature matrix builds, that xgboost sees a GPU if one is
attached, and that every target has enough positives for stratified k-fold.
"""
import os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, _HERE)
os.chdir(_HERE)
import mdata
from feat import TASKS
from baselines import load_X
from tune_baselines import (RF_FULL, RF_REDUCED, XGB_FULL, XGB_REDUCED, expand,
                            make_rf, XGB_N_MAX, XGB_EARLY)
from sklearn.metrics import roc_auc_score


def gpu_available():
    """xgboost >= 2.1 falls back to CPU with a warning instead of raising when no GPU
    is present, so a successful device='cuda' train proves nothing. Ask the driver."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and "GPU" in r.stdout:
            return True, r.stdout.strip().splitlines()[0][:70]
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"torch sees {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return False, "no CUDA device visible to the driver"


def main():
    folds = 5
    t0 = time.time()
    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    # the ChEMBL fingerprint cache is the only one needed for timing; the external
    # sets are featurised at the end of the real run, not here
    import fp_features
    os.makedirs(mdata.CACHE, exist_ok=True)
    if not os.path.exists(os.path.join(mdata.CACHE, "chembl_fp.npz")):
        print("building ChEMBL fingerprint cache (one time, a few minutes)...", flush=True)
        fp_features.build("chembl_fp.npz", df["canonical_smiles"].tolist())
    X, ok, norm = load_X()
    split = df["split"].to_numpy()
    print(f"feature matrix {X.shape} built in {time.time()-t0:.0f}s")

    has_gpu, info = gpu_available()
    print(f"xgboost GPU: {'YES' if has_gpu else 'no'}  ({info})")
    print(f"cpu cores visible: {os.cpu_count()}")

    print(f"\n{'target':10s} {'n_train':>8s} {'n_pos':>7s} {'min fold pos':>13s}")
    sizes = {}
    for ti, t in enumerate(TASKS):
        obs = (mask[:, ti] > 0) & ok
        tr = np.where(obs & (split == "train"))[0]
        npos = int(Y[tr, ti].sum())
        sizes[t] = len(tr)
        flag = "" if npos >= folds else "  <-- TOO FEW FOR 5-FOLD"
        print(f"{t:10s} {len(tr):8d} {npos:7d} {npos//folds:13d}{flag}")

    # time one fit on the largest and the smallest target, at a mid-cost configuration
    big = max(sizes, key=sizes.get)
    small = min(sizes, key=sizes.get)
    per_fit = {}
    for t in (big, small):
        ti = list(TASKS).index(t)
        obs = (mask[:, ti] > 0) & ok
        tr = np.where(obs & (split == "train"))[0]
        ytr = Y[tr, ti].astype(int)
        cut = int(0.8 * len(tr))
        cfg = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
               "max_features": "sqrt"}
        s = time.time()
        m = make_rf(cfg, 0, -1)
        m.fit(X[tr][:cut], ytr[:cut])
        roc_auc_score(ytr[cut:], m.predict_proba(X[tr][cut:])[:, 1]) if len(set(ytr[cut:])) > 1 else None
        per_fit[("RF", t)] = time.time() - s
        print(f"\nRF  one fit on {t} (n={cut}, 300 trees): {per_fit[('RF', t)]:.1f}s")

        try:
            from xgboost import XGBClassifier
            pos, neg = int(ytr[:cut].sum()), int(cut - ytr[:cut].sum())
            kw = dict(random_state=0, n_jobs=-1, tree_method="hist", eval_metric="logloss",
                      scale_pos_weight=max(neg / max(pos, 1), 1.0),
                      n_estimators=XGB_N_MAX, early_stopping_rounds=XGB_EARLY,
                      max_depth=6, learning_rate=0.05, subsample=0.7,
                      colsample_bytree=0.3, min_child_weight=1)
            if has_gpu:
                kw["device"] = "cuda"
            s = time.time()
            xm = XGBClassifier(**kw)
            n_es = max(int(0.1 * cut), 50)
            xm.fit(X[tr][n_es:cut], ytr[n_es:cut],
                   eval_set=[(X[tr][:n_es], ytr[:n_es])], verbose=False)
            per_fit[("XGB", t)] = time.time() - s
            print(f"XGB one fit on {t} (n={cut-n_es}, stopped at iter "
                  f"{getattr(xm, 'best_iteration', '?')}): {per_fit[('XGB', t)]:.1f}s")
        except Exception as e:
            print(f"XGB fit FAILED: {type(e).__name__}: {str(e)[:120]}")

    # project: scale the big-target time by each target's row count
    print("\n" + "=" * 68)
    print("PROJECTED WALL TIME FOR THE FULL RUN (one seed, all six targets)")
    print(f"measured on: random forest CPU ({os.cpu_count()} cores), "
          f"XGBoost {'GPU' if has_gpu else 'CPU'}")
    print("=" * 68)
    for model, gfull, gred in [("RF", RF_FULL, RF_REDUCED), ("XGB", XGB_FULL, XGB_REDUCED)]:
        if (model, big) not in per_fit:
            continue
        # cost of a fit scales roughly linearly in rows at fixed configuration
        unit = per_fit[(model, big)] / sizes[big]
        # average over the grid: RF cost scales with n_estimators, so weight it
        if model == "RF":
            wf = np.mean([c["n_estimators"] / 300 for c in expand(gfull)])
            wr = np.mean([c["n_estimators"] / 300 for c in expand(gred)])
        else:
            wf = wr = 1.0
        for label, g, w in [("full", gfull, wf), ("reduced", gred, wr)]:
            n_cfg = len(expand(g))
            secs = sum(unit * sizes[t] * 0.8 for t in TASKS) * n_cfg * folds * w
            print(f"  {model:4s} --grid {label:8s} {n_cfg:4d} configs x {folds} folds "
                  f"x 6 targets = {n_cfg*folds*6:6d} fits  ~ {secs/3600:6.1f} h")
    print("\nAdd ~5 min once for the feature cache, and ~3 min for the external-set")
    print("predictions at the end. Kaggle allows 9 h per session, Colab less on the")
    print("free tier; if a projection exceeds your budget use --grid reduced, or run")
    print("one model per session (they write separate output files).")


if __name__ == "__main__":
    main()
