"""Exhaustive grid search with stratified k-fold cross-validation for the two
fingerprint baselines, replacing the 6-candidate randomised search on a single
validation split used in the published run.

What changes relative to the published protocol
-----------------------------------------------
published : 6 configurations sampled at random from the grid, scored once on the
            validation split; the best-scoring fitted model kept as-is.
here      : every configuration in the grid, scored by the mean ROC-AUC over
            StratifiedKFold(k) on the TRAIN rows only, then refit on all TRAIN
            rows with the winning configuration. The validation and test splits
            are never seen during selection.

Everything else is held fixed: the same scaffold-disjoint split column, the same
ECFP4(2048, r=2) + 30-descriptor feature matrix, the same per-target binary
formulation, the same class re-weighting, the same seeds.

Outputs, per model and seed, into metrics/
  tuned_<MODEL>_s<seed>.json          chosen configuration and CV score per target
  tuned_<MODEL>_s<seed>_cvtable.csv   every configuration x target with its CV score
  tuned_<MODEL>_s<seed>_preds.npz     Pv/Pt/Yv/Yt/Mv/Mt  (same layout as the GNN runs)
  tuned_<MODEL>_s<seed>_cmnpd.npz     probabilities over the curated marine library
  tuned_<MODEL>_s<seed>_decoy.npz     probabilities over the ZINC20 decoy set
  tuned_<MODEL>_s<seed>_fda.npz       probabilities over the FDA positive controls

Crash and session-expiry behaviour
----------------------------------
Every configuration's cross-validation score is appended to a JSONL file and
flushed to disk as soon as it is computed, and every finished target writes an
NPZ holding its prediction columns. A restart re-reads both and continues at the
first configuration it has no record of. Nothing already computed is repeated.

Point --ckpt_dir at storage that outlives the virtual machine (a mounted Google
Drive folder on Colab, or /kaggle/working with periodic "Save version") or the
checkpoints die with the session and the resume is worthless. See the README.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, _HERE)
os.chdir(_HERE)
import mdata
from feat import TASKS
from baselines import load_X
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- grids
# full: the complete cross-product. reduced: the same three shape axes explored
# at a cheap forest size, with the winner refit at the largest size. The reduced
# variant exists only because a CPU-only session may not finish the full grid;
# it is not the default and the README states the difference.
RF_FULL = {"n_estimators": [300, 700, 1200], "max_depth": [None, 20, 40],
           "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.05, 0.1]}
RF_REDUCED = {"n_estimators": [300], "max_depth": [None, 20, 40],
              "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.05, 0.1]}

# XGBoost: n_estimators is not a grid axis. It is fixed high and chosen per fit by
# early stopping on an inner 10% holdout carved out of the fold's training part, so
# the outer fold stays untouched for scoring. This is the conventional protocol and
# it removes an axis that would otherwise multiply the grid by three.
XGB_FULL = {"max_depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 1.0], "colsample_bytree": [0.3, 0.7],
            "min_child_weight": [1, 5]}
XGB_REDUCED = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1],
               "subsample": [0.7, 1.0], "colsample_bytree": [0.3, 0.7],
               "min_child_weight": [1, 5]}

XGB_N_MAX = 2000
XGB_EARLY = 50


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def make_rf(cfg, seed, jobs):
    return RandomForestClassifier(random_state=seed, n_jobs=jobs,
                                  class_weight="balanced_subsample", **cfg)


def make_xgb(cfg, seed, jobs, spw, device, n_estimators=XGB_N_MAX):
    from xgboost import XGBClassifier
    kw = dict(random_state=seed, n_jobs=jobs, tree_method="hist",
              eval_metric="logloss", scale_pos_weight=spw,
              n_estimators=n_estimators, early_stopping_rounds=XGB_EARLY)
    if device == "cuda":
        kw["device"] = "cuda"
    return XGBClassifier(**kw, **cfg)


def fit_score(model_kind, cfg, Xtr, ytr, Xva, yva, seed, jobs, device, rng):
    """One fit on (Xtr, ytr), scored on (Xva, yva). Returns (auc, best_iteration)."""
    if model_kind == "RF":
        m = make_rf(cfg, seed, jobs)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        return roc_auc_score(yva, p), None
    pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    spw = max(neg / max(pos, 1), 1.0)
    # inner holdout for early stopping, carved out of the training part only
    idx = rng.permutation(len(ytr))
    cut = max(int(0.1 * len(idx)), 50)
    ies, itr = idx[:cut], idx[cut:]
    m = make_xgb(cfg, seed, jobs, spw, device)
    m.fit(Xtr[itr], ytr[itr], eval_set=[(Xtr[ies], ytr[ies])], verbose=False)
    p = m.predict_proba(Xva)[:, 1]
    return roc_auc_score(yva, p), int(getattr(m, "best_iteration", XGB_N_MAX) or XGB_N_MAX)


# ---------------------------------------------------------------- checkpointing
def cand_signature(cands):
    """Config indices in a checkpoint are only meaningful against the candidate list
    that produced them. Fingerprinting that list into the filename makes a changed
    grid start fresh instead of silently resuming against the wrong configurations."""
    import hashlib
    return hashlib.md5(json.dumps(cands, sort_keys=True, default=str).encode()).hexdigest()[:8]


def ck_paths(ckdir, name, task, sig=""):
    tag = f"{name}__{task}" + (f"__{sig}" if sig else "")
    return (os.path.join(ckdir, f"{tag}__cv.jsonl"),
            os.path.join(ckdir, f"{tag}__done.npz"),
            os.path.join(ckdir, f"{tag}__done.json"))


def append_jsonl(path, row):
    """Append one record and force it to stable storage. A power cut between two
    configurations must cost at most that one configuration."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from an abrupt kill
            out[int(r["config_index"])] = r
    return out


def external_columns(model, norm, tag, n_rows, chunk=20000):
    """Score one external set for one target, streaming in row chunks so the
    full 45k x 2078 matrix is never resident alongside a fitted forest."""
    z = np.load(os.path.join(mdata.CACHE, f"{tag}_fp.npz"), allow_pickle=True)
    FPe, De, oke = z["FP"], z["D"], z["ok"]
    med = np.array(norm["desc_median"], dtype=np.float32)
    mu = np.array(norm["desc_mean"], dtype=np.float32)
    sd = np.array(norm["desc_std"], dtype=np.float32)
    col = np.full(n_rows, np.nan, dtype=np.float32)
    rows = np.where(oke)[0]
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        d = np.where(np.isnan(De[r]), med, De[r])
        Xc = np.hstack([FPe[r].astype(np.float32), ((d - mu) / sd).astype(np.float32)])
        col[r] = model.predict_proba(Xc)[:, 1]
    return col, oke


def build_train_cache(df):
    """The ChEMBL fingerprint matrix. Built here rather than assumed, so this script
    runs standalone without the smoke test having been run first."""
    import fp_features
    os.makedirs(mdata.CACHE, exist_ok=True)
    if not os.path.exists(os.path.join(mdata.CACHE, "chembl_fp.npz")):
        print("building ChEMBL fingerprint cache (one time, a few minutes)...", flush=True)
        fp_features.build("chembl_fp.npz", df["canonical_smiles"].tolist())


def build_external_caches():
    """Featurise the three evaluation sets once. The cache lives under
    GLIOMA_CACHE, so putting that on persistent storage means a restart does not
    repeat it."""
    import fp_features
    os.makedirs(mdata.CACHE, exist_ok=True)
    sets = {}
    for tag, src, col in [("cmnpd", mdata.CMNPD, "smiles_std"),
                          ("decoy", mdata.DECOYS, "std_smiles"),
                          ("fda", mdata.FDA, None)]:
        if tag == "fda":
            ext = pd.read_csv(mdata.FDA)
            col = [c for c in ext.columns if "smiles" in c.lower()][0]
        else:
            ext = pd.read_parquet(src)
        smi = ext[col].tolist()
        fp_features.build(f"{tag}_fp.npz", smi)
        sets[tag] = len(smi)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["RF", "XGB"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", choices=["full", "reduced"], default="full")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--only_task", default="", help="debug: restrict to one target")
    ap.add_argument("--no_external", action="store_true")
    ap.add_argument("--ckpt_dir", default=os.environ.get("GLIOMA_CKPT", "checkpoints"),
                    help="MUST outlive the VM (mounted Drive, or saved Kaggle output)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and start over")
    ap.add_argument("--candidates_from", default="",
                    help="JSON from screen_rf_cuml.py; replaces the built-in grid with "
                         "the GPU-screened shortlist, refit here in scikit-learn")
    a = ap.parse_args()

    ckdir = os.path.abspath(a.ckpt_dir)
    os.makedirs(ckdir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    name = f"tuned_{a.model}_s{a.seed}"
    t0 = time.time()

    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    build_train_cache(df)
    X, ok, norm = load_X()
    split = df["split"].to_numpy()
    screen_src, screen_score = None, {}
    if a.candidates_from:
        if a.model != "RF":
            sys.exit("--candidates_from applies to the random forest only; XGBoost is "
                     "tuned on its own grid, on GPU, with no proxy stage.")
        screen_src = json.load(open(a.candidates_from))
        cands_by_task, screen_role = {}, {}
        for t, lst in screen_src["candidates"].items():
            cands_by_task[t] = [c["params"] for c in lst]
            screen_score[t] = [c.get("screen_roc_auc_mean") for c in lst]
            screen_role[t] = [c.get("role", "top") for c in lst]
        cands = None
    else:
        grid = (RF_FULL if a.grid == "full" else RF_REDUCED) if a.model == "RF" else \
               (XGB_FULL if a.grid == "full" else XGB_REDUCED)
        cands = expand(grid)
        cands_by_task = {t: cands for t in TASKS}
    tasks = [t for t in TASKS if (not a.only_task or t == a.only_task)]
    ext_sizes = {} if a.no_external else build_external_caches()

    if screen_src:
        n = sum(len(cands_by_task[t]) for t in tasks)
        print(f"[{a.model}] candidates from {a.candidates_from}: {n} across {len(tasks)} "
              f"targets, screened from {screen_src['n_configurations']} by "
              f"{screen_src['backend']} -> {n*a.folds} fits", flush=True)
    else:
        print(f"[{a.model}] grid={a.grid} configurations={len(cands)} folds={a.folds} "
              f"targets={len(tasks)} -> {len(cands)*a.folds*len(tasks)} fits", flush=True)
    print(f"[ckpt] {ckdir}", flush=True)

    Pv = np.full((len(df), len(TASKS)), np.nan, dtype=np.float32)
    Pt = np.full((len(df), len(TASKS)), np.nan, dtype=np.float32)
    EXT = {tag: np.full((n, len(TASKS)), np.nan, dtype=np.float32)
           for tag, n in ext_sizes.items()}
    EXT_OK = {}
    chosen, cvrows = {}, []

    for ti, task in enumerate(TASKS):
        if task not in tasks:
            continue
        cands = cands_by_task[task]
        cvp, donep, donej = ck_paths(ckdir, name, task, cand_signature(cands))
        if a.fresh:
            for f in (cvp, donep, donej):
                if os.path.exists(f):
                    os.remove(f)

        # ---- target already finished in an earlier session: reload, do not refit
        if os.path.exists(donep) and os.path.exists(donej):
            d = np.load(donep, allow_pickle=True)
            meta = json.load(open(donej))
            Pv[:, ti] = d["Pv_col"]
            Pt[:, ti] = d["Pt_col"]
            for tag in EXT:
                if f"{tag}_col" in d:
                    EXT[tag][:, ti] = d[f"{tag}_col"]
                    EXT_OK[tag] = d[f"{tag}_ok"]
            chosen[task] = meta["chosen"]
            cvrows.extend(read_jsonl(cvp).values())
            print(f"[{task}] restored from checkpoint, cv={meta['chosen']['cv_roc_auc_mean']:.4f}",
                  flush=True)
            continue

        obs = (mask[:, ti] > 0) & ok
        tr = np.where(obs & (split == "train"))[0]
        va = np.where(obs & (split == "val"))[0]
        te = np.where(obs & (split == "test"))[0]
        ytr = Y[tr, ti].astype(int)
        skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        folds = list(skf.split(X[tr], ytr))

        # ---- resume the grid at the first configuration with no record
        done = read_jsonl(cvp)
        if done:
            print(f"[{task}] resuming: {len(done)}/{len(cands)} configurations already scored",
                  flush=True)
        best = (-1.0, None, None)
        for ci, r in done.items():
            if r["cv_roc_auc_mean"] > best[0]:
                best = (r["cv_roc_auc_mean"],
                        {k: r[k] for k in cands[ci]}, r.get("median_best_iteration"))

        for ci, cfg in enumerate(cands):
            if ci in done:
                continue
            # the fold seed is derived from (seed, config) so a resumed run
            # reproduces exactly what an uninterrupted one would have done
            rng = np.random.default_rng([a.seed, ci])
            aucs, iters = [], []
            for i_in, i_out in folds:
                auc, bi = fit_score(a.model, cfg, X[tr][i_in], ytr[i_in],
                                    X[tr][i_out], ytr[i_out], a.seed, a.jobs, a.device, rng)
                aucs.append(auc)
                if bi is not None:
                    iters.append(bi)
            mu_, sd_ = float(np.mean(aucs)), float(np.std(aucs))
            med_it = int(np.median(iters)) if iters else None
            row = {"task": task, "config_index": ci, **cfg,
                   "cv_roc_auc_mean": mu_, "cv_roc_auc_sd": sd_,
                   "median_best_iteration": med_it}
            if screen_score:
                row["screen_roc_auc_mean"] = screen_score[task][ci]
                row["role"] = screen_role[task][ci]
            append_jsonl(cvp, row)
            if mu_ > best[0]:
                best = (mu_, cfg, med_it)
            print(f"  [{task}] {ci+1}/{len(cands)} cv={mu_:.4f}+-{sd_:.4f} "
                  f"best={best[0]:.4f} ({time.time()-t0:.0f}s)", flush=True)

        mu_, cfg, med_iter = best
        # ---- refit the winner on all train rows
        if a.model == "RF":
            m = make_rf(cfg, a.seed, a.jobs)
            m.fit(X[tr], ytr)
        else:
            pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
            from xgboost import XGBClassifier
            kw = dict(random_state=a.seed, n_jobs=a.jobs, tree_method="hist",
                      eval_metric="logloss",
                      scale_pos_weight=max(neg / max(pos, 1), 1.0),
                      n_estimators=max(med_iter or 300, 50))
            if a.device == "cuda":
                kw["device"] = "cuda"
            m = XGBClassifier(**kw, **cfg)
            m.fit(X[tr], ytr, verbose=False)

        Pv[va, ti] = m.predict_proba(X[va])[:, 1]
        Pt[te, ti] = m.predict_proba(X[te])[:, 1]
        chosen[task] = {"params": {k: (None if v is None else v) for k, v in cfg.items()},
                        "cv_roc_auc_mean": mu_, "n_configurations": len(cands),
                        "median_best_iteration": med_iter,
                        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                        "n_pos_train": int(ytr.sum())}

        # ---- score the evaluation sets now, while this model is in memory, so a
        # crash later never costs a refit
        payload = {"Pv_col": Pv[:, ti], "Pt_col": Pt[:, ti]}
        for tag, n in ext_sizes.items():
            col, oke = external_columns(m, norm, tag, n)
            EXT[tag][:, ti] = col
            EXT_OK[tag] = oke
            payload[f"{tag}_col"] = col
            payload[f"{tag}_ok"] = oke
            print(f"  [{task}] scored {tag} ({time.time()-t0:.0f}s)", flush=True)
        del m

        tmp = donep + ".tmp.npz"
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, donep)          # atomic: a reader never sees a half file
        json.dump({"task": task, "chosen": chosen[task],
                   "model": a.model, "seed": a.seed, "grid": a.grid, "folds": a.folds},
                  open(donej + ".tmp", "w"), indent=1)
        os.replace(donej + ".tmp", donej)
        cvrows.extend(read_jsonl(cvp).values())
        print(f"[{task}] checkpointed. chosen {cfg} cv={mu_:.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- did the screening buy anything, and did its ordering survive refitting?
    # Both questions are answered against the control configurations, which were
    # sampled from across the whole screened grid. A correlation computed on the
    # shortlist alone would be range-restricted: the top candidates are near-tied, so
    # fold noise dominates and the coefficient carries no information about ranking
    # quality. The controls restore the spread that makes it interpretable.
    agreement = {}
    if screen_score:
        from scipy import stats as _st
        by_task = {}
        for r in cvrows:
            if r.get("screen_roc_auc_mean") is None:
                continue          # the depth arms have no screen score by construction
            d = by_task.setdefault(r["task"], {"sc": [], "cv": [], "top": [], "ctl": []})
            d["sc"].append(r["screen_roc_auc_mean"])
            d["cv"].append(r["cv_roc_auc_mean"])
            (d["top"] if r.get("role") == "top" else
             d["ctl"] if r.get("role") == "control" else []).append(r["cv_roc_auc_mean"])
        for t, d in by_task.items():
            e = {"n_scored": len(d["sc"])}
            if len(d["sc"]) >= 4:
                rho, pv = _st.spearmanr(d["sc"], d["cv"])
                e.update(spearman_spanning=float(rho), p=float(pv))
            if d["top"] and d["ctl"]:
                e.update(best_top=max(d["top"]), best_control=max(d["ctl"]),
                         screening_gain=max(d["top"]) - max(d["ctl"]),
                         n_top=len(d["top"]), n_control=len(d["ctl"]))
            agreement[t] = e
        print("[proxy] did the GPU screening help? best shortlisted minus best "
              "randomly drawn control, on the scikit-learn score:", flush=True)
        for t, e in agreement.items():
            g = e.get("screening_gain")
            rho = e.get("spearman_spanning")
            print(f"    {t:10s} gain={'n/a' if g is None else format(g, '+.4f')}"
                  f"  (top {e.get('n_top','?')} vs control {e.get('n_control','?')})"
                  f"  rank agreement rho={'n/a' if rho is None else format(rho, '+.3f')}",
                  flush=True)
        print("    A gain at or below zero means the shortlist was no better than "
              "configurations drawn at random from\n    the same grid: the screening added "
              "nothing, so report the result as an unscreened search or rerun with\n"
              "    RF_MODE=grid. Negative rank agreement with a positive gain usually means "
              "the candidates are near-tied.", flush=True)

    # ---- assemble the deliverables from the per-target checkpoints
    json.dump({"model": a.model, "seed": a.seed, "folds": a.folds, "grid": a.grid,
               "n_configurations": len(cands), "tasks": list(TASKS), "chosen": chosen,
               "device": a.device, "wall_seconds": round(time.time() - t0, 1),
               "checkpoint_dir": ckdir,
               "candidates_from": a.candidates_from or None,
               "screened_from_n_configurations": (screen_src or {}).get("n_configurations"),
               "screen_backend": (screen_src or {}).get("backend"),
               "screen_diagnostics": agreement or None,
               "protocol": ("GPU-screened shortlist refit in scikit-learn with "
                            "class_weight='balanced_subsample', StratifiedKFold on train rows"
                            if screen_src else
                            "exhaustive grid, StratifiedKFold on train rows, refit on train")},
              open(f"metrics/{name}.json", "w"), indent=1)
    pd.DataFrame(cvrows).to_csv(f"metrics/{name}_cvtable.csv", index=False)
    np.savez_compressed(f"metrics/{name}_preds.npz",
                        Pv=Pv, Pt=Pt, Yv=Y, Yt=Y,
                        Mv=(mask * (split == "val")[:, None]).astype(np.int8),
                        Mt=(mask * (split == "test")[:, None]).astype(np.int8))
    for tag in EXT:
        np.savez_compressed(f"metrics/{name}_{tag}.npz", P=EXT[tag], ok=EXT_OK[tag])
    print(f"DONE {name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
