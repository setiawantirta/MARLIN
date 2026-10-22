"""Run the four external validations, select the final model on marine performance,
and emit every external table."""
import glob, json, os, subprocess, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import external as X
import evaluate as EV
import mdata
from feat import TASKS

PY = sys.executable
# The child script was addressed relative to the CALLER's working directory, which the
# stage runner sets to the stage output directory. Resolve it from this file.
_SRC = os.path.dirname(os.path.abspath(__file__))
GNN = ["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS"]


def main():
    df, cm, cl, ad = X.load_core()
    _, Y, mask = mdata.load_matrix()
    split = df["split"].to_numpy()
    n_cm = len(cm)

    # ---- 1. per-architecture seed-mean CMNPD probabilities
    files = X.cmnpd_pred_files()
    by_arch = {}
    for name, p in files.items():
        by_arch.setdefault(X.arch_of(name), []).append(p)
    Pcm = {a: X.seed_mean(sorted(ps), n_cm) for a, ps in by_arch.items()}
    print("CMNPD-scored architectures:", {a: len(v) for a, v in by_arch.items()}, flush=True)

    # ---- 2. marine phenotypic validation (the model-selection criterion)
    g, crec = X.phenotypic_table(cm, cl, "exclude")
    g_all, _ = X.phenotypic_table(cm, cl, "include")
    rows = []
    for a, P in Pcm.items():
        for r in X.phenotypic_metrics(P, g, a):
            r["censoring"] = "censored_excluded"; rows.append(r)
        for r in X.phenotypic_metrics(P, g_all, a):
            r["censoring"] = "censored_included"; rows.append(r)
    # per-seed spread
    for a, ps in by_arch.items():
        for p in sorted(ps):
            nm = os.path.basename(p)[len("cmnpd_pred_"):-len(".npz")]
            for r in X.phenotypic_metrics(X.cmnpd_matrix(p, n_cm), g, nm):
                r["censoring"] = "censored_excluded_per_seed"; rows.append(r)
    ph = pd.DataFrame(rows)
    ph["arch"] = ph["model"].map(lambda s: X.arch_of(s))
    ph.to_csv("external_phenotypic_validation.csv", index=False)

    # per-cell-line where n permits
    cls_rows = []
    for a, P in Pcm.items():
        for cell, sub in crec.groupby("CELL_LINE"):
            gg = sub.groupby("row").agg(pact_max=("pActivity", "max"),
                                        pact_med=("pActivity", "median")).reset_index()
            if len(gg) < 20:
                continue
            for r in X.phenotypic_metrics(P, gg, a):
                r["cell_line"] = cell; r["n_compounds"] = len(gg); cls_rows.append(r)
    pd.DataFrame(cls_rows).to_csv("external_phenotypic_by_cellline.csv", index=False)

    # ---- 3. selection rule (declared a priori): ROC-AUC of the polypharmacology score
    #         against measured pActivity >= 6, censored records excluded.
    sel = ph[(ph.censoring == "censored_excluded") & (ph.score == "polypharm_score") &
             (ph.potency.str.startswith("AUC"))].copy()
    sel = sel.sort_values("roc_auc", ascending=False)
    print(sel[["arch", "roc_auc", "n", "n_pos"]].to_string(index=False), flush=True)
    best_arch = sel.iloc[0]["arch"]
    print("SELECTED:", best_arch, flush=True)

    # ---- 4. decoy predictions for the selected architecture (GNNs only need a pass)
    if best_arch in GNN:
        for s in [0, 1, 2]:
            out = f"metrics/decoy_pred_{best_arch}_s{s}_pretrain.npz"
            ck = f"models/{best_arch}_s{s}_pretrain.pt"
            if os.path.exists(out) or not os.path.exists(ck):
                continue
            subprocess.run([PY, os.path.join(_SRC, "predict_sets.py"), "--arch", best_arch, "--ckpt", ck,
                            "--set", "decoy", "--out", out], check=True,
                           env=dict(os.environ, OMP_NUM_THREADS="2"))

    dec = pd.read_parquet(mdata.DECOYS)
    runs = EV.load_runs()
    ens = dict(np.load("metrics/ens_preds.npz"))

    dpaths = sorted(glob.glob(f"metrics/decoy_pred_{best_arch}_s*.npz"))
    Ps = []
    for p in dpaths:
        z = np.load(p)
        M = np.full((len(dec), len(TASKS)), np.nan)
        M[z["I"]] = z["P"]
        Ps.append(M)
    Pdec = np.nanmean(np.stack(Ps), axis=0)
    enr = X.enrichment(df, dec, ens[best_arch], Pdec, mask, Y, split)
    pd.DataFrame(enr).to_csv("enrichment_results.csv", index=False)
    print(pd.DataFrame(enr).to_string(index=False), flush=True)

    # decoy ROC inputs for the figure
    droc = {}
    for ti, t in enumerate(TASKS):
        act = np.where((split == "test") & (mask[:, ti] > 0) & (Y[:, ti] > 0.5) &
                       np.isfinite(ens[best_arch][:, ti]))[0]
        ds = np.where((dec["gene"].to_numpy() == t) & np.isfinite(Pdec[:, ti]))[0]
        if len(act) and len(ds):
            droc[t] = (np.concatenate([np.ones(len(act)), np.zeros(len(ds))]).astype(int),
                       np.concatenate([ens[best_arch][act, ti], Pdec[ds, ti]]))
    np.savez_compressed("metrics/decoy_roc_inputs.npz",
                        **{f"{t}_y": v[0] for t, v in droc.items()},
                        **{f"{t}_s": v[1] for t, v in droc.items()})

    # ---- 5. FDA recovery and marine holdout for every architecture
    fda_all, mh_all = [], []
    for a, P in ens.items():
        fda_all += X.fda_recovery(df, P, mask, Y, split, a)
        mh_all += X.marine_holdout(df, P, a)
    pd.DataFrame(fda_all).to_csv("fda_control_recovery.csv", index=False)
    pd.DataFrame(mh_all).to_csv("marine_holdout_predictions.csv", index=False)

    json.dump({"selected_arch": best_arch,
               "selection_metric": "ROC-AUC of polypharmacology score vs measured "
                                   "pActivity>=6 on the CMNPD glioma cell-line set, "
                                   "censored records excluded",
               "n_phenotypic_compounds": int(len(g)),
               "n_phenotypic_compounds_censored_included": int(len(g_all)),
               "n_phenotypic_records": int(len(crec))},
              open("metrics/selection.json", "w"), indent=1)
    print("EXTERNAL DONE", flush=True)


if __name__ == "__main__":
    main()
