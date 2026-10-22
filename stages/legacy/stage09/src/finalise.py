"""Score the full curated CMNPD library with the selected model and write the
screening table the interpretation and screening tracks consume."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata
from feat import TASKS


def build(selected_arch, seeds=(0, 1, 2), stage="pretrain", thresholds=None, out="cmnpd_predictions.parquet"):
    cm = pd.read_parquet(mdata.CMNPD)
    n = len(cm)
    Ps = []
    used = []
    for s in seeds:
        p = f"metrics/cmnpd_pred_{selected_arch}_s{s}_{stage}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        M = np.full((n, len(TASKS)), np.nan)
        M[z["I"]] = z["P"]
        Ps.append(M); used.append(p)
    if not Ps:
        raise SystemExit(f"no CMNPD predictions for {selected_arch}")
    P = np.nanmean(np.stack(Ps), axis=0)

    ad = pd.read_csv(mdata.AD)[["COMPOUND_ID", "max_tanimoto_to_train",
                                "inside_AD_datadriven", "inside_AD_0.30"]]
    out_df = pd.DataFrame({"cmnpd_id": cm["COMPOUND_ID"], "inchikey": cm["inchikey"],
                           "smiles_std": cm["smiles_std"]})
    for ti, t in enumerate(TASKS):
        out_df[f"prob_{t}"] = P[:, ti]
    thr = thresholds or {t: 0.5 for t in TASKS}
    out_df["max_prob"] = np.nanmax(P, axis=1)
    out_df["polypharm_score"] = np.nansum(P, axis=1)
    above = np.zeros(n, dtype=int)
    for ti, t in enumerate(TASKS):
        above += (P[:, ti] >= thr[t]).astype(int)
    out_df["n_targets_above_threshold"] = above
    out_df["top_target"] = [TASKS[int(i)] if np.isfinite(P[k]).any() else None
                            for k, i in enumerate(np.nanargmax(np.where(np.isfinite(P), P, -1), axis=1))]
    out_df = out_df.merge(ad, left_on="cmnpd_id", right_on="COMPOUND_ID", how="left")
    out_df = out_df.rename(columns={"max_tanimoto_to_train": "ad_max_tanimoto",
                                    "inside_AD_0.30": "in_applicability_domain",
                                    "inside_AD_datadriven": "in_applicability_domain_datadriven"})
    out_df = out_df.drop(columns=["COMPOUND_ID"])
    for c in ["has_glioma_activity", "is_indonesia", "cmnpd_blood_brain_barrier_penetration_level",
              "np_likeness", "mw", "clogp", "is_macrocycle", "peptide_like"]:
        if c in cm.columns:
            out_df[c] = cm[c].to_numpy()
    out_df.to_parquet(out, index=False)
    meta = {"selected_architecture": selected_arch, "stage": stage,
            "checkpoints_used": used, "tasks": list(TASKS),
            "thresholds": {k: float(v) for k, v in thr.items()},
            "n_scored": int(np.isfinite(P).all(axis=1).sum()), "n_rows": int(n)}
    print(json.dumps(meta, indent=1))
    return out_df, meta


if __name__ == "__main__":
    arch = sys.argv[1]
    thr = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    build(arch, thresholds=thr)
