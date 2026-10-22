"""GPU screening pass over a large random-forest grid, to be confirmed on CPU.

Why two stages
--------------
cuML's forest is fast but it is not scikit-learn's forest. Three differences matter
for this study and none of them is negligible:

1. `class_weight='balanced_subsample'`, which the published protocol uses, is not
   supported by cuML and raises ValueError. The screen uses `'balanced'`, which
   reweights once on the full label vector instead of per bootstrap sample.
2. cuML's own documentation states that the algorithm for tree node splits differs
   from scikit-learn's, and cuML quantises each feature into `n_bins` candidate
   thresholds rather than considering every observed value.
3. The interpretability chapter loads joblib-pickled scikit-learn forests into
   `shap.TreeExplainer` and reads `feature_importances_`. A cuML forest cannot be
   substituted there.

So cuML is used here only as a RANKING DEVICE over a search space far larger than a
CPU grid could cover. Whatever it likes is then refit in scikit-learn, with
`balanced_subsample` restored and the same folds, and the winner is chosen on the
scikit-learn score. Every number that reaches the manuscript comes from a
scikit-learn forest.

The proxy is imperfect by construction, so two safeguards are built in. The
confirmation stage receives the top k rather than the single best, and each candidate
is also offered with `max_depth=None` — an arm cuML cannot represent, since it needs a
finite depth. The rank agreement between the screen and the confirmation is computed
afterwards and reported, so the proxy's quality is measured rather than assumed.

Usage
-----
    python screen_rf_cuml.py --seed 0 --grid medium --topk 24          # GPU
    python screen_rf_cuml.py --seed 0 --grid tiny --backend sklearn    # logic check

Writes metrics/screen_RF_s<seed>_topk.json, which tune_baselines.py consumes with
--candidates_from.
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
from tune_baselines import (append_jsonl, read_jsonl, build_train_cache)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Screening grids. These are deliberately far larger than the CPU grid: the point of
# the GPU pass is search space, not speed on a space that already fits.
#   tiny   : for verifying the plumbing, not for science
#   small  : 96 configurations
#   medium : 320 configurations
#   large  : 960 configurations
# max_depth carries no None arm because cuML requires a finite depth; the
# confirmation stage adds that arm back on CPU.
SCREEN = {
    "tiny":   {"n_estimators": [300], "max_depth": [16], "min_samples_leaf": [1, 2],
               "max_features": ["sqrt", 0.05], "min_samples_split": [2]},
    "small":  {"n_estimators": [300, 800], "max_depth": [8, 16, 24, 40],
               "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.05],
               "min_samples_split": [2, 16]},
    "medium": {"n_estimators": [300, 800], "max_depth": [8, 16, 24, 40],
               "min_samples_leaf": [1, 2, 4, 8],
               "max_features": ["sqrt", 0.02, 0.05, 0.1, 0.2],
               "min_samples_split": [2, 16]},
    "large":  {"n_estimators": [300, 800, 1500], "max_depth": [8, 12, 16, 24, 32, 40],
               "min_samples_leaf": [1, 2, 4, 8],
               "max_features": ["sqrt", 0.02, 0.05, 0.1, 0.2],
               "min_samples_split": [2, 8, 32]},
}


def expand(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def weight_strategy(backend, seed, forced="auto"):
    """Decide how the screen reweights classes, from what this build actually accepts.

    cuML's support for class weighting is version-dependent. Some builds accept
    class_weight but reject 'balanced_subsample' with a ValueError; others do not
    recognise the argument at all and raise TypeError, and in those the parameter
    cannot be used even with a supported value. Probing with two-tree fits is cheap
    and removes the guesswork; the alternative is discovering it hours in.

    Returns (kind, note) with kind in {'class_weight', 'sample_weight', 'none'}.
    """
    if forced != "auto":
        return forced, f"forced by --weight_mode {forced}"
    Xp = np.asarray(np.random.default_rng(0).random((64, 4)), dtype=np.float32)
    yp = (np.arange(64) % 2).astype(np.int32)

    def build(**kw):
        if backend == "cuml":
            from cuml.ensemble import RandomForestClassifier as cuRF
            return cuRF(n_estimators=2, max_depth=4, random_state=seed, n_streams=1, **kw)
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=2, max_depth=4, random_state=seed, **kw)

    try:
        build(class_weight="balanced").fit(Xp, yp)
        return "class_weight", "class_weight='balanced' accepted"
    except Exception as e1:
        first = f"{type(e1).__name__}: {str(e1)[:70]}"
    try:
        build().fit(Xp, yp, sample_weight=np.ones(64, dtype=np.float32))
        return "sample_weight", ("class_weight unusable (%s); reweighting through "
                                 "sample_weight at fit time instead" % first)
    except Exception as e2:
        return "none", ("neither class_weight (%s) nor sample_weight (%s: %s) is usable; "
                        "THE SCREEN RUNS UNWEIGHTED and this is recorded in the output"
                        % (first, type(e2).__name__, str(e2)[:60]))


def balanced_weights(y):
    """sklearn's 'balanced' rule: n / (n_classes * count(class)), per sample."""
    cls, cnt = np.unique(y, return_counts=True)
    w = {int(c): len(y) / (len(cls) * int(n)) for c, n in zip(cls, cnt)}
    return np.asarray([w[int(v)] for v in y], dtype=np.float32)


def make_screen_model(cfg, seed, backend, kind):
    kw = dict(cfg)
    if kind == "class_weight":
        kw["class_weight"] = "balanced"
    if backend == "cuml":
        from cuml.ensemble import RandomForestClassifier as cuRF
        # n_streams=1 is required for a reproducible forest
        return cuRF(random_state=seed, n_streams=1, **kw)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **kw)


def check_backend(backend):
    if backend == "sklearn":
        import sklearn
        return f"sklearn {sklearn.__version__} (CPU; logic check only)"
    try:
        import cuml
    except ImportError as e:
        sys.exit("cuML is not installed in this session. On Kaggle attach a GPU and use "
                 "the RAPIDS image; on Colab install cuml-cu12 (see the README). "
                 f"Import error: {e}")
    import subprocess
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if r.returncode != 0 or "GPU" not in r.stdout:
        sys.exit("cuML is installed but no GPU is visible to the driver. Attach a GPU "
                 "runtime, or run with --backend sklearn to check the plumbing only.")
    return f"cuml {cuml.__version__} | {r.stdout.strip().splitlines()[0][:60]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", choices=list(SCREEN), default="medium")
    ap.add_argument("--topk", type=int, default=24,
                    help="candidates handed to the scikit-learn confirmation stage")
    ap.add_argument("--n_control", type=int, default=12,
                    help="configurations drawn at random from the whole screened grid and "
                         "refit alongside the shortlist, as a control on whether the "
                         "screening added anything")
    ap.add_argument("--backend", choices=["cuml", "sklearn"], default="cuml")
    ap.add_argument("--weight_mode", choices=["auto", "class_weight", "sample_weight", "none"],
                    default="auto",
                    help="how the screen reweights classes; 'auto' probes what this build accepts")
    ap.add_argument("--only_task", default="")
    ap.add_argument("--ckpt_dir", default=os.environ.get("GLIOMA_CKPT", "checkpoints"))
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    print("[backend]", check_backend(a.backend), flush=True)
    wkind, wnote = weight_strategy(a.backend, a.seed, a.weight_mode)
    print(f"[weights] {wkind} -- {wnote}", flush=True)
    if wkind == "none":
        print("[weights] The screen ranking is then further from the scikit-learn model "
              "than intended.\n          It remains usable as a ranking device, but read "
              "screening_gain in the\n          confirmation output before trusting it.",
              flush=True)
    ckdir = os.path.abspath(a.ckpt_dir)
    os.makedirs(ckdir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    t0 = time.time()

    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    build_train_cache(df)
    X, ok, norm = load_X()
    split = df["split"].to_numpy()
    cands = expand(SCREEN[a.grid])
    tasks = [t for t in TASKS if (not a.only_task or t == a.only_task)]
    print(f"[screen] grid={a.grid} configurations={len(cands)} folds={a.folds} "
          f"targets={len(tasks)} -> {len(cands)*a.folds*len(tasks)} fits", flush=True)

    topk, screen_rows = {}, []
    for ti, task in enumerate(TASKS):
        if task not in tasks:
            continue
        sp = os.path.join(ckdir, f"screen_RF_s{a.seed}__{task}__screen.jsonl")
        if a.fresh and os.path.exists(sp):
            os.remove(sp)
        obs = (mask[:, ti] > 0) & ok
        tr = np.where(obs & (split == "train"))[0]
        ytr = Y[tr, ti].astype(int)
        Xtr = np.ascontiguousarray(X[tr], dtype=np.float32)
        folds = list(StratifiedKFold(n_splits=a.folds, shuffle=True,
                                     random_state=a.seed).split(Xtr, ytr))

        done = read_jsonl(sp)
        if done:
            print(f"[{task}] resuming screen: {len(done)}/{len(cands)} scored", flush=True)
        for ci, cfg in enumerate(cands):
            if ci in done:
                continue
            aucs = []
            for i_in, i_out in folds:
                m = make_screen_model(cfg, a.seed, a.backend, wkind)
                yin = ytr[i_in].astype(np.int32)
                if wkind == "sample_weight":
                    m.fit(Xtr[i_in], yin, sample_weight=balanced_weights(yin))
                else:
                    m.fit(Xtr[i_in], yin)
                p = m.predict_proba(Xtr[i_out])
                p = np.asarray(p.to_numpy() if hasattr(p, "to_numpy") else p)[:, 1]
                aucs.append(roc_auc_score(ytr[i_out], p))
                del m
            row = {"task": task, "config_index": ci, **cfg,
                   "screen_roc_auc_mean": float(np.mean(aucs)),
                   "screen_roc_auc_sd": float(np.std(aucs))}
            append_jsonl(sp, row)
            if (ci + 1) % 10 == 0 or ci + 1 == len(cands):
                print(f"  [{task}] {ci+1}/{len(cands)} ({time.time()-t0:.0f}s)", flush=True)

        rows = sorted(read_jsonl(sp).values(), key=lambda r: -r["screen_roc_auc_mean"])
        screen_rows.extend(rows)
        keys = list(SCREEN[a.grid])
        picked, seen = [], set()

        def add(r, role, score):
            cfg = {k: r[k] for k in keys}
            sig = tuple(sorted((k, str(v)) for k, v in cfg.items()))
            if sig in seen:
                return
            seen.add(sig)
            picked.append({"params": cfg, "screen_roc_auc_mean": score, "role": role})

        for r in rows[:a.topk]:
            add(r, "top", r["screen_roc_auc_mean"])
            # the arm cuML cannot represent: unlimited depth
            deep = dict({k: r[k] for k in keys}, max_depth=None)
            sig2 = tuple(sorted((k, str(v)) for k, v in deep.items()))
            if sig2 not in seen:
                seen.add(sig2)
                picked.append({"params": deep, "screen_roc_auc_mean": None,
                               "role": "depth_arm"})

        # Controls: configurations sampled uniformly from the WHOLE screened grid,
        # refit on CPU alongside the shortlist. Two things become measurable that a
        # top-k-only comparison cannot show. First, whether the screening actually
        # bought anything: the best shortlisted candidate should beat the best
        # control. Second, a rank agreement that spans the score range instead of the
        # range-restricted top slice, where near-ties make a correlation meaningless.
        rest = rows[a.topk:]
        if rest and a.n_control > 0:
            rng = np.random.default_rng(a.seed)
            for i in rng.choice(len(rest), size=min(a.n_control, len(rest)), replace=False):
                add(rest[int(i)], "control", rest[int(i)]["screen_roc_auc_mean"])
        topk[task] = picked
        nrole = {r: sum(1 for x in picked if x["role"] == r)
                 for r in ("top", "depth_arm", "control")}
        print(f"[{task}] screen best={rows[0]['screen_roc_auc_mean']:.4f} -> "
              f"{len(picked)} candidates for the CPU stage "
              f"({nrole['top']} top, {nrole['depth_arm']} depth arms, "
              f"{nrole['control']} controls) ({time.time()-t0:.0f}s)", flush=True)

    out = f"metrics/screen_RF_s{a.seed}_topk.json"
    json.dump({"seed": a.seed, "folds": a.folds, "grid": a.grid,
               "n_configurations": len(cands), "topk": a.topk, "backend": a.backend,
               "weight_strategy": wkind, "weight_note": wnote,
               "note": ("GPU screen reweighted by %s; the confirmation stage refits in "
                        "scikit-learn with class_weight='balanced_subsample'" % wkind),
               "wall_seconds": round(time.time() - t0, 1), "candidates": topk},
              open(out, "w"), indent=1)
    pd.DataFrame(screen_rows).to_csv(f"metrics/screen_RF_s{a.seed}_table.csv", index=False)
    print(f"wrote {out}", flush=True)
    print("next: python tune_baselines.py --model RF --seed %d --candidates_from %s"
          % (a.seed, out), flush=True)


if __name__ == "__main__":
    main()
