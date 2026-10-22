"""Test-set metrics for the fine-tuning and putative-inactive ablation arms.

Validation loss is NOT comparable across arms (each arm recomputes its positive-class
weights from its own training rows), so every ablation comparison here uses test metrics.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata, evaluate as EV, eval_utils as E
from feat import TASKS

STAGES = ("pretrain", "ft_np", "ft_npw", "noputative")


def main():
    df = pd.read_parquet(mdata.MATRIX)
    _, Y, mask = mdata.load_matrix()
    split = df["split"].to_numpy()
    n = len(df)
    runs = EV.load_runs(STAGES)
    rows = []
    for r in runs:
        Pv, Pt = EV.full_preds(r, n)
        pt = {}
        for ti, t in enumerate(TASKS):
            vsel = np.where((split == "val") & (mask[:, ti] > 0) & np.isfinite(Pv[:, ti]))[0]
            tsel = np.where((split == "test") & (mask[:, ti] > 0) & np.isfinite(Pt[:, ti]))[0]
            thr, _ = E.best_threshold(Y[vsel, ti].astype(int), Pv[vsel, ti])
            m = E.task_metrics(Y[tsel, ti].astype(int), Pt[tsel, ti], thr)
            pt[t] = m
            for k in EV.METRICS:
                rows.append(dict(arch=r["arch"], seed=r["seed"], stage=r["stage"],
                                 n_train=r["n_train"], task=t, metric=k, value=m[k]))
        for k in EV.METRICS:
            rows.append(dict(arch=r["arch"], seed=r["seed"], stage=r["stage"],
                             n_train=r["n_train"], task="MACRO", metric=k,
                             value=float(np.nanmean([pt[t][k] for t in TASKS]))))
    long = pd.DataFrame(rows)
    long.to_csv("ablation_metrics.csv", index=False)
    srows = []
    for (arch, stage, task, metric), g in long.groupby(["arch", "stage", "task", "metric"]):
        m, lo, hi = E.ci95(g["value"].to_numpy())
        srows.append(dict(arch=arch, stage=stage, task=task, metric=metric,
                          n_seeds=len(g), mean=m, ci95_lo=lo, ci95_hi=hi,
                          n_train=int(g["n_train"].iloc[0])))
    S = pd.DataFrame(srows)
    S.to_csv("ablation_summary.csv", index=False)
    piv = S[(S.task == "MACRO") & S.metric.isin(["roc_auc", "pr_auc", "mcc", "brier"])]
    print(piv.pivot_table(index=["arch", "stage"], columns="metric", values="mean").round(4).to_string())
    print("ABLATION EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
