"""Rebuild the multi-task label matrix using exact AND recovered censored evidence.

Label precedence (as specified): the uncensored '=' median always decides a (compound, target)
pair when exact evidence exists - including when it lands in the discarded 5-6 grey zone, in
which case the pair stays MISSING even if censored evidence would have called it. Pairs with
only censored evidence take the censored call; pairs whose censored evidence is internally
contradictory (both a decisive active and a decisive inactive bound) are DROPPED and counted.
"""
import sys, os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.argv = ["x"]
exec(open("standardize.py").read().split("if __name__")[0])

def main():
    GENES = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
    ACTIVE_T, INACTIVE_T = 6.0, 5.0

    agg = pd.read_csv("work/agg_per_compound_target.csv.gz")
    cal = pd.read_csv("work/censored_calls.csv.gz")

    # ---- standardise compounds that only the censored records introduce --------------------
    std = pd.read_parquet("work/std_chembl.parquet").rename(columns={"src_id": "molecule_chembl_id"})
    need = set(cal.molecule_chembl_id) - set(std.molecule_chembl_id)
    print(f"new compounds introduced by censored records: {len(need)}", flush=True)
    if need:
        sub = cal[cal.molecule_chembl_id.isin(need)].drop_duplicates("molecule_chembl_id")
        args = list(zip(sub.molecule_chembl_id.astype(str), sub.canonical_smiles.astype(str)))
        with Pool(7, initializer=_init, initargs=(True,)) as pool:
            res = list(pool.imap_unordered(_one, args, chunksize=100))
        new = pd.DataFrame(res)
        okn = new[new.fail.isna()].drop(columns=["fail"]).rename(columns={"src_id": "molecule_chembl_id"})
        bad = new[new.fail.notna()]
        print(f"  standardised {len(okn)}, failed {len(bad)}: "
              f"{bad.fail.str.replace(r'mw_\\d+','mw_window',regex=True).value_counts().to_dict()}",
              flush=True)
        std = pd.concat([std, okn], ignore_index=True)
    std.to_parquet("work/std_chembl_v2.parquet", index=False)
    key = std.drop_duplicates("molecule_chembl_id").set_index("molecule_chembl_id")

    # ---- exact evidence, collapsed on InChIKey --------------------------------------------
    agg["inchikey"] = agg.molecule_chembl_id.map(key.inchikey)
    agg = agg[agg.inchikey.notna()]
    ex = (agg.groupby(["inchikey", "gene"])
             .agg(pchembl=("pchembl_median", "median"), n_meas=("n_measurements", "sum"),
                  n_chembl_ids=("molecule_chembl_id", "nunique"),
                  chembl_ids=("molecule_chembl_id", lambda s: ";".join(sorted(set(s)))))
             .reset_index())

    # ---- censored evidence, collapsed on InChIKey -----------------------------------------
    cal["inchikey"] = cal.molecule_chembl_id.map(key.inchikey)
    cal = cal[cal.inchikey.notna()]
    cens = (cal.groupby(["inchikey", "gene"])
               .agg(n_cens_active=("call", lambda s: int((s == "active").sum())),
                    n_cens_inactive=("call", lambda s: int((s == "inactive").sum())),
                    cens_bound_min=("bound", "min"), cens_bound_max=("bound", "max"),
                    cens_ids=("molecule_chembl_id", lambda s: ";".join(sorted(set(s)))))
               .reset_index())
    print(f"censored (compound,target) pairs after InChIKey collapse: {len(cens)}", flush=True)

    m = ex.merge(cens, on=["inchikey", "gene"], how="outer")
    m["has_exact"] = m.pchembl.notna()
    m["n_cens_active"] = m.n_cens_active.fillna(0).astype(int)
    m["n_cens_inactive"] = m.n_cens_inactive.fillna(0).astype(int)

    def decide(r):
        if r.has_exact:
            if r.pchembl >= ACTIVE_T: return 1.0, "exact"
            if r.pchembl <= INACTIVE_T: return 0.0, "exact"
            return np.nan, "exact_greyzone"
        if r.n_cens_active and r.n_cens_inactive: return np.nan, "censored_conflict"
        if r.n_cens_active: return 1.0, "censored_active"
        if r.n_cens_inactive: return 0.0, "censored_inactive"
        return np.nan, "none"

    dec = m.apply(decide, axis=1, result_type="expand")
    m["label"], m["evidence"] = dec[0], dec[1]

    acc = {
     "pairs_total": len(m),
     "pairs_exact": int(m.has_exact.sum()),
     "pairs_censored_only": int((~m.has_exact).sum()),
     "exact_greyzone_discarded": int((m.evidence == "exact_greyzone").sum()),
     "censored_conflict_dropped": int((m.evidence == "censored_conflict").sum()),
     "censored_active_pairs": int((m.evidence == "censored_active").sum()),
     "censored_inactive_pairs": int((m.evidence == "censored_inactive").sum()),
     "censored_calls_overridden_by_exact": int((m.has_exact &
            ((m.n_cens_active > 0) | (m.n_cens_inactive > 0))).sum()),
     "censored_overridden_and_contradicted": int((m.has_exact &
            (((m.label == 1) & (m.n_cens_inactive > 0)) |
             ((m.label == 0) & (m.n_cens_active > 0)))).sum()),
    }
    print("\naccounting:", json.dumps(acc, indent=1), flush=True)

    m.to_csv("work/pairs_v2.csv.gz", index=False)
    lab = m[m.label.notna()]
    Y = lab.pivot(index="inchikey", columns="gene", values="label").reindex(columns=GENES)
    P = m.pivot(index="inchikey", columns="gene", values="pchembl").reindex(
            columns=GENES).reindex(Y.index)
    EV = lab.pivot(index="inchikey", columns="gene", values="evidence").reindex(columns=GENES)
    Y.to_parquet("work/Y_v2.parquet"); P.to_parquet("work/P_v2.parquet")
    EV.to_parquet("work/EV_v2.parquet")

    print(f"\ncompounds with >=1 usable label: {len(Y)}", flush=True)
    cells = Y.size; filled = int(Y.notna().sum().sum())
    print(f"matrix {Y.shape[0]} x 6 = {cells} cells; filled {filled}; "
          f"sparsity {100*(1-filled/cells):.2f}%", flush=True)
    s = pd.DataFrame({"n_labelled": Y.notna().sum(), "n_active": (Y == 1).sum(),
                      "n_inactive": (Y == 0).sum()})
    s["pct_active"] = (100 * s.n_active / s.n_labelled).round(1)
    s["n_inactive_from_censored"] = [(EV[g] == "censored_inactive").sum() for g in GENES]
    s["n_active_from_censored"] = [(EV[g] == "censored_active").sum() for g in GENES]
    print(s.to_string(), flush=True)
    nl = Y.notna().sum(axis=1)
    print(f"\nlabels per compound: {nl.value_counts().sort_index().to_dict()}", flush=True)
    print(f"compounds labelled on >=2 targets: {int((nl>=2).sum())}; "
          f"active on >=2: {int(((Y==1).sum(axis=1)>=2).sum())}", flush=True)
    print(f"tasks still above 85% active: "
          f"{[g for g in GENES if s.loc[g,'pct_active']>85]}", flush=True)
    json.dump(acc, open("work/acc_v2.json", "w"))
    print("BUILD v2 DONE", flush=True)


if __name__ == "__main__":
    main()
