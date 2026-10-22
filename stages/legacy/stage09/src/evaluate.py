"""Internal test-set evaluation: per-task and macro metrics, calibration, significance."""
import glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata, eval_utils as E
from sklearn.metrics import roc_auc_score
from feat import TASKS

STAGES = ("pretrain", "baseline")
METRICS = ["roc_auc", "pr_auc", "mcc", "balanced_acc", "bedroc20", "ef1", "brier", "ece"]


def load_runs(stages=STAGES):
    runs = []
    for p in sorted(glob.glob("metrics/*_run.json")):
        r = json.load(open(p))
        if r.get("stage") not in stages:
            continue
        npz = p.replace("_run.json", "_preds.npz")
        if not os.path.exists(npz):
            continue
        r["_npz"] = npz
        runs.append(r)
    return runs


def full_preds(run, n):
    """Return (Pval, Ptest) as (n, 6) arrays of NaN outside the respective split."""
    z = np.load(run["_npz"])
    if run["stage"] == "baseline":
        return z["Pv"].astype(float), z["Pt"].astype(float)
    Pv = np.full((n, len(TASKS)), np.nan); Pt = np.full((n, len(TASKS)), np.nan)
    Pv[z["Iv"]] = z["Pv"]; Pt[z["It"]] = z["Pt"]
    return Pv, Pt


def main():
    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    split = df["split"].to_numpy()
    n = len(df)
    runs = load_runs()
    print("runs:", [r["name"] for r in runs], flush=True)

    rows, cal_rows, thresholds, store = [], [], {}, {}
    for r in runs:
        Pv, Pt = full_preds(r, n)
        store[r["name"]] = (r, Pt)
        per_task = {}
        for ti, t in enumerate(TASKS):
            vsel = np.where((split == "val") & (mask[:, ti] > 0) & np.isfinite(Pv[:, ti]))[0]
            tsel = np.where((split == "test") & (mask[:, ti] > 0) & np.isfinite(Pt[:, ti]))[0]
            thr, _ = E.best_threshold(Y[vsel, ti].astype(int), Pv[vsel, ti])
            m = E.task_metrics(Y[tsel, ti].astype(int), Pt[tsel, ti], thr)
            per_task[t] = m
            thresholds[(r["name"], t)] = thr
            for k in METRICS:
                rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"],
                                 stage=r["stage"], task=t, metric=k, value=m[k]))
            rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"], stage=r["stage"],
                             task=t, metric="threshold", value=thr))
            rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"], stage=r["stage"],
                             task=t, metric="n_test", value=m["n"]))
            rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"], stage=r["stage"],
                             task=t, metric="n_test_pos", value=m["n_pos"]))
            cal_rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"], task=t,
                                 brier=m["brier"], ece=m["ece"], n=m["n"], n_pos=m["n_pos"]))
        for k in METRICS:
            vals = [per_task[t][k] for t in TASKS]
            rows.append(dict(model=r["name"], arch=r["arch"], seed=r["seed"], stage=r["stage"],
                             task="MACRO", metric=k, value=float(np.nanmean(vals))))
    long = pd.DataFrame(rows)
    long.to_csv("benchmark_metrics.csv", index=False)
    pd.DataFrame(cal_rows).to_csv("calibration.csv", index=False)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in thresholds.items()},
              open("metrics/thresholds.json", "w"), indent=1)

    # ---------------- publication summary: mean +/- 95% CI over seeds
    srows = []
    for (arch, stage), g in long.groupby(["arch", "stage"]):
        for task in list(TASKS) + ["MACRO"]:
            for k in METRICS:
                v = g[(g.task == task) & (g.metric == k)]["value"].to_numpy()
                m, lo, hi = E.ci95(v)
                srows.append(dict(arch=arch, stage=stage, task=task, metric=k, n_seeds=len(v),
                                  mean=m, ci95_lo=lo, ci95_hi=hi))
    pd.DataFrame(srows).to_csv("benchmark_summary.csv", index=False)

    # ---------------- seed-averaged probabilities per architecture (for significance)
    ens = {}
    for (arch, stage), g in long.groupby(["arch", "stage"]):
        names = sorted(g["model"].unique())
        Ps = [store[nm][1] for nm in names]
        ens[arch] = np.nanmean(np.stack(Ps), axis=0)
    json.dump(sorted(ens), open("metrics/arch_list.json", "w"))
    np.savez_compressed("metrics/ens_preds.npz", **{a: p for a, p in ens.items()})

    # ---------------- pairwise significance
    archs = sorted(ens)
    sig = []
    for ti, t in enumerate(TASKS):
        tsel = np.where((split == "test") & (mask[:, ti] > 0))[0]
        yt = Y[tsel, ti].astype(int)
        for i in range(len(archs)):
            for j in range(i + 1, len(archs)):
                a, b = archs[i], archs[j]
                pa, pb = ens[a][tsel, ti], ens[b][tsel, ti]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                auc_a, auc_b, pdl = E.delong_test(yt, pa, pb)
                d, pbt, ci = E.paired_bootstrap(yt, pa, pb, n_boot=1000, seed=7)
                sig.append(dict(scope=t, model_a=a, model_b=b, metric="roc_auc",
                                value_a=auc_a, value_b=auc_b, delta=d,
                                ci95_lo=ci[0], ci95_hi=ci[1],
                                p_delong=pdl, p_bootstrap=pbt, n=len(yt)))
        print("sig done", t, flush=True)

    # macro-level paired bootstrap on macro ROC-AUC
    rng = np.random.default_rng(11)
    tsels = [np.where((split == "test") & (mask[:, ti] > 0))[0] for ti in range(len(TASKS))]

    def macro_auc(P, boot_idx=None):
        vals = []
        for ti, sel in enumerate(tsels):
            y = Y[sel, ti].astype(int); p = P[sel, ti]
            if boot_idx is not None:
                bi = boot_idx[ti]
                y, p = y[bi], p[bi]
            if len(np.unique(y)) < 2:
                continue
            vals.append(roc_auc_score(y, p))
        return float(np.mean(vals))

    boots = [[rng.integers(0, len(s), len(s)) for s in tsels] for _ in range(500)]
    macro_vals = {a: macro_auc(ens[a]) for a in archs}
    macro_boot = {a: np.array([macro_auc(ens[a], b) for b in boots]) for a in archs}
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            a, b = archs[i], archs[j]
            d = macro_vals[a] - macro_vals[b]
            diffs = macro_boot[a] - macro_boot[b]
            centred = diffs - diffs.mean()
            pval = float(max((np.abs(centred) >= abs(d)).mean(), 1 / (len(diffs) + 1)))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            sig.append(dict(scope="MACRO", model_a=a, model_b=b, metric="macro_roc_auc",
                            value_a=macro_vals[a], value_b=macro_vals[b], delta=d,
                            ci95_lo=float(lo), ci95_hi=float(hi),
                            p_delong=np.nan, p_bootstrap=pval, n=int(sum(len(s) for s in tsels))))
    S = pd.DataFrame(sig)
    for col, out in [("p_bootstrap", "q_bootstrap"), ("p_delong", "q_delong")]:
        ok = S[col].notna()
        S.loc[ok, out] = E.bh(S.loc[ok, col].to_numpy())
    S["significant_q05"] = (S["q_bootstrap"] < 0.05)
    S.to_csv("significance_tests.csv", index=False)
    print("MACRO ROC-AUC (3-seed mean probability):",
          {a: round(v, 4) for a, v in sorted(macro_vals.items(), key=lambda x: -x[1])}, flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
