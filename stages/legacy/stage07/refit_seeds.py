"""Rank stability across the three benchmark seeds.

The shipped model bundle contains only the seed-0 Random Forests, so seeds 1 and 2 are
refitted here with the RECORDED hyperparameters (model_config.json), the same training
rows and the same class_weight='balanced_subsample' as src/fit_final_rf.py, then MDI and
TreeSHAP are recomputed so the consensus ranking can be re-derived per seed.

TreeSHAP for the stability check uses a smaller test subsample (N_SHAP_STAB) than the
main analysis, because it is only used to rank features, not to quantify contributions.
"""
import json, os, time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

TASKS = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
OUT = "attrib"
SEEDS = [1, 2]
N_SHAP_STAB = 500
JOBS = 3


def worker(args):
    seed, t, ti = args
    import shap
    cfg = json.load(open("bundle/glioma_models/model_config.json"))
    p = dict(cfg["hyperparameters_per_task"][t])
    X = np.load(f"{OUT}/X_chembl.npy", mmap_mode="r")
    raw = np.load(f"{OUT}/chembl_raw.npz")
    ok, split, Y, mask = raw["ok"], raw["split"], raw["Y"], raw["mask"]
    tr = np.where((mask[:, ti] > 0) & ok & (split == "train"))[0]
    t0 = time.time()
    m = RandomForestClassifier(random_state=seed, n_jobs=2,
                               class_weight="balanced_subsample", **p)
    m.fit(np.asarray(X[tr]), Y[tr, ti].astype(int))
    np.save(f"{OUT}/mdi_{t}_s{seed}.npy", m.feature_importances_)
    rows = np.load(f"{OUT}/shap_test_{t}.npz")["rows"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(rows, min(N_SHAP_STAB, len(rows)), replace=False))
    ex = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
    sv = np.asarray(ex.shap_values(np.asarray(X[sel]), check_additivity=False))
    if sv.ndim == 3:
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
    np.savez_compressed(f"{OUT}/shap_test_{t}_s{seed}.npz",
                        sv=sv.astype(np.float32), rows=sel)
    print(f"[s{seed}/{t}] n_train={len(tr)} {time.time()-t0:.0f}s", flush=True)
    return f"s{seed}_{t}"


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(s, t, TASKS.index(t)) for s in SEEDS
            for t in ["KDR", "PIK3CA", "EGFR", "IDH1", "PDGFRA", "CDK6"]]
    with Pool(JOBS, maxtasksperchild=1) as pool:
        for d in pool.imap_unordered(worker, jobs):
            print("done", d, flush=True)
    print("SEED REFIT DONE", flush=True)
