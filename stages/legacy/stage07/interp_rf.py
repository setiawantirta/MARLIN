import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from repro.config import path as RPATH, CFG, out_dir
"""Stage A - feature attribution on the SELECTED model (per-target Random Forest on
ECFP4(2048, r=2) + 30 z-scored RDKit descriptors).

Three independent rankings per target:
  (1) TreeSHAP  - exact path-dependent Shapley values, scaffold-disjoint test partition
  (2) permutation importance - ROC-AUC drop under column shuffling, same partition
  (3) MDI       - RandomForest mean decrease in Gini impurity (training-set quantity)

TreeSHAP is additionally computed on the 194-compound CMNPD marine phenotypic set, which
is where transfer actually matters.

Outputs (workspace, npz/csv): shap_test_{target}.npz, shap_marine_{target}.npz,
perm_{target}.npz, mdi.npz, attrib_long.csv
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

MATRIX = RPATH("chembl_multitask_matrix.parquet")
CMNPD = RPATH("cmnpd_curated.parquet")
CELLLINE = RPATH("cmnpd_glioma_cellline_activity.csv")
MODELS = "bundle/glioma_models/models/rf_final"
TASKS = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
OUT = "attrib"
os.makedirs(OUT, exist_ok=True)

N_SHAP_TEST = 2000      # cap on test molecules entering TreeSHAP, per target
TOP_POOL = 40           # per target, per method, feeds the permutation-importance pool
PERM_REPEATS = 5
SEED = 0


# ------------------------------------------------------------------ feature assembly
def load_feature_matrix(cache_file, desc_mean, desc_std, desc_median):
    z = np.load(cache_file, allow_pickle=True)
    FP, D, ok = z["FP"], z["D"], z["ok"]
    D = np.where(np.isnan(D), desc_median, D)
    X = np.hstack([FP.astype(np.float32), ((D - desc_mean) / desc_std).astype(np.float32)])
    return X, ok, FP, D


def main():
    cfg = json.load(open("bundle/glioma_models/model_config.json"))
    dmean = np.array(cfg["desc_mean"], dtype=np.float32)
    dstd = np.array(cfg["desc_std"], dtype=np.float32)
    dmed = np.array(cfg["desc_median"], dtype=np.float32)
    DESC_NAMES = cfg["featurisation"]["descriptors"]
    FEATNAMES = [f"ECFP4_bit_{i}" for i in range(2048)] + list(DESC_NAMES)
    np.save(f"{OUT}/featnames.npy", np.array(FEATNAMES))

    df = pd.read_parquet(MATRIX)
    split = df["split"].to_numpy()
    Y = np.stack([df[f"label_{t}"].to_numpy(dtype="float64") for t in TASKS], axis=1)
    mask = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y, nan=0.0).astype(np.float32)

    X, ok, FP_chembl, D_chembl = load_feature_matrix("cache/chembl_fp.npz", dmean, dstd, dmed)
    print("chembl X", X.shape, "ok", int(ok.sum()), flush=True)
    np.savez_compressed(f"{OUT}/chembl_raw.npz", FP=FP_chembl, D=D_chembl, ok=ok,
                        split=split.astype("U8"), Y=Yf, mask=mask)

    # ---------------------------------------------------------- marine phenotypic set
    cm = pd.read_parquet(CMNPD, columns=["COMPOUND_ID", "inchikey", "smiles_std"])
    cl = pd.read_csv(CELLLINE)
    c = cl[(~cl["censored"].astype(bool)) & cl["pActivity"].notna()].copy()
    idx = {v: i for i, v in enumerate(cm["COMPOUND_ID"].tolist())}
    c["row"] = c["COMPOUND_ID"].map(idx)
    c = c[c["row"].notna()]
    c["row"] = c["row"].astype(int)
    g = c.groupby("row").agg(pact_max=("pActivity", "max"), pact_med=("pActivity", "median"),
                             n_rec=("pActivity", "size"),
                             n_cell=("CELL_LINE", "nunique")).reset_index()
    Xm_all, okm, FP_m, D_m = load_feature_matrix("cache/cmnpd_fp.npz", dmean, dstd, dmed)
    g = g[okm[g["row"].to_numpy()]]
    mrows = g["row"].to_numpy()
    print("marine phenotypic set n =", len(mrows), flush=True)
    np.savez_compressed(f"{OUT}/marine_pheno.npz", rows=mrows,
                        pact_max=g["pact_max"].to_numpy(), pact_med=g["pact_med"].to_numpy())
    np.savez_compressed(f"{OUT}/cmnpd_raw.npz", FP=FP_m, D=D_m, ok=okm)
    Xm = Xm_all[mrows]

    # memory-map the two feature matrices so the per-target worker processes share one
    # physical copy instead of each holding a private 280 MB array
    np.save(f"{OUT}/X_chembl.npy", X)
    np.save(f"{OUT}/X_marine.npy", Xm)
    del X, Xm, Xm_all
    import gc; gc.collect()
    print("wrote memmap matrices", flush=True)


def worker(args):
    ti, t = args
    import shap
    X = np.load(f"{OUT}/X_chembl.npy", mmap_mode="r")
    Xm = np.load(f"{OUT}/X_marine.npy", mmap_mode="r")
    raw = np.load(f"{OUT}/chembl_raw.npz")
    ok, split, Yf, mask = raw["ok"], raw["split"], raw["Y"], raw["mask"]
    mrows = np.load(f"{OUT}/marine_pheno.npz")["rows"]
    rng = np.random.default_rng(SEED)
    if True:
        t0 = time.time()
        m = joblib.load(f"{MODELS}/rf_{t}_s0.joblib")
        assert m.n_features_in_ == X.shape[1], (m.n_features_in_, X.shape[1])
        mdi = m.feature_importances_
        np.save(f"{OUT}/mdi_{t}.npy", mdi)

        obs = (mask[:, ti] > 0) & ok
        te = np.where(obs & (split == "test"))[0]
        yte = Yf[te, ti].astype(int)
        # stratified subsample for TreeSHAP
        if len(te) > N_SHAP_TEST:
            pos = te[yte == 1]; neg = te[yte == 0]
            k_pos = min(len(pos), max(1, int(round(N_SHAP_TEST * len(pos) / len(te)))))
            k_neg = N_SHAP_TEST - k_pos
            sel = np.concatenate([rng.choice(pos, k_pos, replace=False),
                                  rng.choice(neg, min(k_neg, len(neg)), replace=False)])
        else:
            sel = te
        sel = np.sort(sel)
        ysel = Yf[sel, ti].astype(int)

        ex = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
        sv = ex.shap_values(X[sel], check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:                       # (n, feat, class) or (class, n, feat)
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
        sv = sv.astype(np.float32)
        assert sv.shape == (len(sel), X.shape[1]), sv.shape
        np.savez_compressed(f"{OUT}/shap_test_{t}.npz", sv=sv, rows=sel, y=ysel)
        print(f"[{t}] SHAP test n={len(sel)} ({time.time()-t0:.0f}s)", flush=True)

        svm = np.asarray(ex.shap_values(Xm, check_additivity=False))
        if svm.ndim == 3:
            svm = svm[..., 1] if svm.shape[-1] == 2 else svm[1]
        np.savez_compressed(f"{OUT}/shap_marine_{t}.npz", sv=svm.astype(np.float32), rows=mrows)
        print(f"[{t}] SHAP marine n={len(mrows)} ({time.time()-t0:.0f}s)", flush=True)

        # ------------------------------------------------- permutation importance pool
        mshap = np.abs(sv).mean(0)
        pool = set(np.argsort(-mdi)[:TOP_POOL]) | set(np.argsort(-mshap)[:TOP_POOL])
        pool = np.array(sorted(pool))
        Xp = X[sel].copy()
        base = roc_auc_score(ysel, m.predict_proba(Xp)[:, 1])
        drops = np.zeros((len(pool), PERM_REPEATS))
        prng = np.random.default_rng(SEED)
        for k, fi in enumerate(pool):
            col = Xp[:, fi].copy()
            for r in range(PERM_REPEATS):
                Xp[:, fi] = prng.permutation(col)
                drops[k, r] = base - roc_auc_score(ysel, m.predict_proba(Xp)[:, 1])
            Xp[:, fi] = col
        np.savez_compressed(f"{OUT}/perm_{t}.npz", pool=pool, drops=drops, base=base)
        print(f"[{t}] perm |pool|={len(pool)} base_auc={base:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        del m, ex, sv, svm, Xp
        import gc; gc.collect()
        return t


if __name__ == "__main__":
    from multiprocessing import Pool
    if not os.path.exists(f"{OUT}/X_chembl.npy"):
        main()
    # heaviest targets first (deepest forests) so the pool drains evenly
    order = ["KDR", "PIK3CA", "EGFR", "IDH1", "PDGFRA", "CDK6"]
    jobs = [(TASKS.index(t), t) for t in order]
    with Pool(3, maxtasksperchild=1) as pool:
        for done in pool.imap_unordered(worker, jobs):
            print("finished", done, flush=True)
    MDI = np.stack([np.load(f"{OUT}/mdi_{t}.npy") for t in TASKS])
    np.savez_compressed(f"{OUT}/mdi.npz", MDI=MDI, tasks=np.array(TASKS))
    print("STAGE A DONE", flush=True)
