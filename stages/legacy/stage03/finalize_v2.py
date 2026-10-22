"""Assemble the v2 deliverable matrix: splits, NP weight, putative-inactive column, summary."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import json, os
import numpy as np
import pandas as pd

GENES = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
TARGETS = {"train": 0.70, "val": 0.10, "test": 0.20}


def main():
    Y = pd.read_parquet("work/Y_v2.parquet")
    P = pd.read_parquet("work/P_v2.parquet").reindex(Y.index)
    EV = pd.read_parquet("work/EV_v2.parquet").reindex(Y.index)
    pairs = pd.read_csv("work/pairs_v2.csv.gz")
    std = pd.read_parquet("work/std_chembl_v2.parquet")
    base = std.drop_duplicates("inchikey").set_index("inchikey").reindex(Y.index)

    # ---- FDA controls and marine holdout, recomputed against the LARGER pool -----------
    fda = pd.read_csv(RPATH("fda_positive_controls.csv"))
    fda_ik = set(fda.inchikey) & set(Y.index)
    # CMNPD 1.0 vendor property table. Not redistributable with this module -
    # download it from cmnpd.org and place it in the data directory; see
    # config/inputs_manifest.tsv and data/README.md.
    cm_raw = pd.read_csv(RPATH("CMNPD_1.0_calc_prop.tsv"),
                         sep="\t", encoding="utf-8-sig", low_memory=False)
    cm_raw["ik14_raw"] = cm_raw.STANDARD_INCHI_KEY.str.split("-").str[0]
    cm_std = pd.read_parquet("work/std_cmnpd.parquet").rename(columns={"src_id": "COMPOUND_ID"})
    pool_full, pool14 = set(Y.index), {k.split("-")[0] for k in Y.index}
    RAW_F = set(cm_raw.STANDARD_INCHI_KEY.dropna())
    RAW_14 = set(cm_raw.ik14_raw.dropna())
    STD_F = set(cm_std.inchikey.dropna())
    STD_14 = set(cm_std.inchikey14.dropna())
    sch = {
        "full_key_cmnpd_native": ({k for k in pool_full if k in RAW_F},
                                  cm_raw[cm_raw.STANDARD_INCHI_KEY.isin(pool_full)].COMPOUND_ID.nunique()),
        "full_key_restandardised": ({k for k in pool_full if k in STD_F},
                                    cm_std[cm_std.inchikey.isin(pool_full)].COMPOUND_ID.nunique()),
        "skeleton14_cmnpd_native": ({k for k in pool_full if k.split("-")[0] in RAW_14},
                                    cm_raw[cm_raw.ik14_raw.isin(pool14)].COMPOUND_ID.nunique()),
        "skeleton14_restandardised": ({k for k in pool_full if k.split("-")[0] in STD_14},
                                      cm_std[cm_std.inchikey14.isin(pool14)].COMPOUND_ID.nunique()),
    }
    ov = pd.DataFrame([{"scheme": k, "n_chembl_pool_compounds": len(v[0]),
                        "n_cmnpd_compounds": v[1]} for k, v in sch.items()])
    hold = set().union(*[v[0] for v in sch.values()])
    full_any = sch["full_key_cmnpd_native"][0] | sch["full_key_restandardised"][0]
    k14_any = sch["skeleton14_cmnpd_native"][0] | sch["skeleton14_restandardised"][0]
    for nm, s in [("ANY full key (union)", full_any), ("ANY 14-char skeleton (union)", k14_any),
                  ("MARINE HOLDOUT (union of all schemes)", hold)]:
        ov.loc[len(ov)] = {"scheme": nm, "n_chembl_pool_compounds": len(s),
                           "n_cmnpd_compounds": None}
    ov.to_csv("marine_overlap_schemes.csv", index=False)
    print(ov.to_string(index=False), flush=True)

    # ---- scaffold split over the real compounds ---------------------------------------
    scaf = base.scaffold.fillna("").copy()
    n_empty = int((scaf == "").sum())
    grp = scaf.copy()
    grp[grp == ""] = [f"__acyclic_{i}" for i in range(n_empty)]
    force = (fda_ik | hold)
    forced_groups = set(grp.loc[sorted(force)])
    gsz = grp.value_counts(); tot_sz = gsz.sum()
    ina = (Y == 0).astype(int); act = (Y == 1).astype(int)
    gina = ina.groupby(grp).sum(); gact = act.groupby(grp).sum()
    tot_ina, tot_act = ina.sum(), act.sum()
    rng = np.random.default_rng(11)
    assign = {g: "test" for g in forced_groups}
    cur = {k: {"n": 0} for k in TARGETS}
    for g in forced_groups:
        cur["test"]["n"] += gsz[g]

    def spread(groups, w):
        """Distribute scaffold groups 70/10/20 weighted by how many members of the SCARCE
        class each group carries, so the scarce class itself is stratified (weighting by
        total group size lets a large group with one scarce member dominate a quota)."""
        groups = [g for g in groups if g not in assign]
        rng.shuffle(groups)
        quota = [("val", 0.10), ("test", 0.20), ("train", 0.70)]
        tot = sum(w.get(g, 0) for g in groups); acc = {k: 0 for k in TARGETS}
        for g in groups:
            wg = max(w.get(g, 0), 1)
            best = min(quota, key=lambda kv: (acc[kv[0]] + wg) / (kv[1] * max(tot, 1)))[0]
            assign[g] = best; acc[best] += wg; cur[best]["n"] += gsz[g]

    for task in tot_ina.sort_values().index:
        w = gina[task].to_dict()
        spread(sorted(set(grp.loc[Y.index[(Y[task] == 0).values]])), w)
    for task in tot_act.sort_values().index[:2]:
        w = gact[task].to_dict()
        spread(sorted(set(grp.loc[Y.index[(Y[task] == 1).values]])), w)
    for g in [g for g in gsz.sort_values(ascending=False).index if g not in assign]:
        best = min(TARGETS, key=lambda p: (cur[p]["n"] + gsz[g]) / (TARGETS[p] * tot_sz))
        assign[g] = best; cur[best]["n"] += gsz[g]
    split = grp.map(assign)

    # ---- matrix ------------------------------------------------------------------------
    mat = pd.DataFrame({"inchikey": Y.index})
    for c in ["inchikey14", "std_smiles", "scaffold", "mw", "clogp", "hbd", "hba", "rotb",
              "charge", "tpsa", "np_score", "fsp3", "heavy_atoms"]:
        mat[c] = base[c].values
    mat = mat.rename(columns={"std_smiles": "canonical_smiles"})
    for g in GENES:
        mat[f"label_{g}"] = Y[g].values
        mat[f"pchembl_{g}"] = P[g].values
        mat[f"evidence_{g}"] = EV[g].values
    mat["n_labels"] = Y.notna().sum(axis=1).values
    mat["n_active"] = (Y == 1).sum(axis=1).values
    mat["split"] = split.values
    mat["is_fda_control"] = mat.inchikey.isin(fda_ik)
    mat["is_marine_holdout"] = mat.inchikey.isin(hold)
    mat["is_putative_decoy"] = False
    ids = pairs.dropna(subset=["inchikey"]).groupby("inchikey").apply(
        lambda d: ";".join(sorted({x for v in
                                   list(d.chembl_ids.dropna()) + list(d.cens_ids.dropna())
                                   for x in str(v).split(";") if x})), include_groups=False)
    mat["chembl_ids"] = ids.reindex(Y.index).values

    # ---- NP flags and the continuous NP weight ----------------------------------------
    import gzip as _gz
    _rows = []
    with _gz.open("raw/molecules.jsonl.gz", "rt") as _fh:
        for _l in _fh:
            _o = json.loads(_l)
            _rows.append((_o["molecule_chembl_id"], _o.get("natural_product")))
    mol = pd.DataFrame(_rows, columns=["molecule_chembl_id", "natural_product"]
                       ).drop_duplicates("molecule_chembl_id")
    npmap = (std[["molecule_chembl_id", "inchikey"]].merge(mol, on="molecule_chembl_id", how="left")
             .groupby("inchikey").natural_product.max())
    mat["np_chembl_flag"] = (npmap.reindex(Y.index).values == 1)
    mat["is_np"] = mat.np_chembl_flag | (mat.np_score >= 0)
    tr = mat.split == "train"
    lo, hi = float(mat.loc[tr, "np_score"].min()), float(mat.loc[tr, "np_score"].max())
    mat["np_weight"] = ((mat.np_score - lo) / (hi - lo)).clip(0, 1)
    print(f"\nnp_weight: min-max rescaling of np_score over the TRAINING pool "
          f"[{lo:.4f}, {hi:.4f}], clipped to [0,1]", flush=True)
    print(f"  np_weight over training: mean {mat.loc[tr,'np_weight'].mean():.3f}, "
          f"median {mat.loc[tr,'np_weight'].median():.3f}, "
          f"90th pct {mat.loc[tr,'np_weight'].quantile(0.9):.3f}", flush=True)

    # ---- putative inactives ------------------------------------------------------------
    for g in GENES:
        mat[f"label_{g}_putative"] = mat[f"label_{g}"]
    if os.path.exists("work/putative_decoys.parquet"):
        put = pd.read_parquet("work/putative_decoys.parquet")
        sc2sp = dict(zip(mat.scaffold, mat.split))
        keep, dropped_test_scaffold = [], 0
        acc = {"train": 0, "val": 0}
        for _, r in put.iterrows():
            s = sc2sp.get(r.scaffold)
            if s == "test":
                dropped_test_scaffold += 1
                continue
            if s in ("train", "val"):
                part = s
            else:
                part = "train" if acc["train"] / 0.875 <= acc["val"] / 0.125 else "val"
                # record it so any later putative decoy on the SAME new scaffold lands in the
                # same partition - otherwise two putatives sharing a scaffold would leak
                sc2sp[r.scaffold] = part
            acc[part] = acc.get(part, 0) + 1
            keep.append((r, part))
        print(f"\nputative decoys: {len(put)} generated, {dropped_test_scaffold} dropped for "
              f"sharing a scaffold with a test compound, {len(keep)} kept", flush=True)
        rows = []
        for r, part in keep:
            d = {"inchikey": r.inchikey, "inchikey14": r.inchikey14,
                 "canonical_smiles": r.std_smiles, "scaffold": r.scaffold,
                 "mw": r.mw, "clogp": r.clogp, "hbd": r.hbd, "hba": r.hba, "rotb": r.rotb,
                 "charge": r.charge, "tpsa": r.tpsa, "np_score": r.np_score, "fsp3": r.fsp3,
                 "heavy_atoms": r.heavy_atoms, "n_labels": 0, "n_active": 0, "split": part,
                 "is_fda_control": False, "is_marine_holdout": False, "is_putative_decoy": True,
                 "chembl_ids": None, "np_chembl_flag": False,
                 "is_np": bool(r.np_score >= 0),
                 "np_weight": float(min(max((r.np_score - lo) / (hi - lo), 0), 1))}
            for g in GENES:
                d[f"label_{g}"] = np.nan; d[f"pchembl_{g}"] = np.nan
                d[f"evidence_{g}"] = None
                d[f"label_{g}_putative"] = 0.0 if g == r.gene else np.nan
            rows.append(d)
        pm = pd.DataFrame(rows)
        pm = pm.groupby("inchikey", as_index=False).first()
        mat = pd.concat([mat, pm], ignore_index=True)

    mat.to_parquet("chembl_multitask_matrix.parquet", index=False)
    real = mat[~mat.is_putative_decoy]
    print(f"\nmatrix rows: {len(mat)} ({len(real)} ChEMBL-labelled + "
          f"{int(mat.is_putative_decoy.sum())} putative)", flush=True)
    print("splits (all rows):", mat.split.value_counts().to_dict(), flush=True)
    print("splits (ChEMBL only):", real.split.value_counts().to_dict(),
          {k: f"{100*v/len(real):.1f}%" for k, v in real.split.value_counts().items()}, flush=True)

    A = pd.DataFrame({p: [(real.loc[real.split == p, f"label_{g}"] == 1).sum() for g in GENES]
                      for p in ["train", "val", "test"]}, index=GENES)
    I = pd.DataFrame({p: [(real.loc[real.split == p, f"label_{g}"] == 0).sum() for g in GENES]
                      for p in ["train", "val", "test"]}, index=GENES)
    print("\nACTIVES per task per partition:\n", A.to_string(), flush=True)
    print("\nINACTIVES per task per partition:\n", I.to_string(), flush=True)
    print("\nempty cells - actives:", int((A == 0).sum().sum()),
          "inactives:", int((I == 0).sum().sum()), flush=True)
    leak = (mat[["scaffold", "split"]].drop_duplicates().groupby("scaffold").split.nunique() > 1).sum()
    print("scaffold leakage across partitions (incl. putative):", int(leak), flush=True)
    print("all FDA in test:", bool((real[real.is_fda_control].split == "test").all()),
          "| all marine in test:", bool((real[real.is_marine_holdout].split == "test").all()),
          "| putative never in test:",
          bool((mat[mat.is_putative_decoy].split != "test").all()), flush=True)
    print("FINALIZE DONE", flush=True)


if __name__ == "__main__":
    main()
