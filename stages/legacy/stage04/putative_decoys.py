"""Top up the negative class of tasks that remain >85% active after censored recovery.

These are PUTATIVE inactives: ZINC20 compounds with no recorded activity anywhere, property-matched
to the task's actives and ECFP4-dissimilar (Tanimoto < 0.35) to every active of that task. They are
drawn from a pool made DISJOINT from the enrichment decoy set (decoy_set.parquet) so the held-out
enrichment evaluation can never see a compound the model was trained on, and they are written to a
separate column (label_<gene>_putative) so the modelling track can train with and without them.

Target: bring each affected task to a 3:1 active:inactive ratio (75% active), in line with the four
tasks that censored recovery already brought to 77-83%.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import decoys as D

TARGET_ACTIVE_FRAC = 0.75
GENES = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]


def main():
    Y = pd.read_parquet("work/Y_v2.parquet")
    std = pd.read_parquet("work/std_chembl_v2.parquet")
    base = std.drop_duplicates("inchikey").set_index("inchikey")
    enrich = pd.read_parquet("decoy_set.parquet")
    used = set(enrich.zinc_id.astype(str))

    z = pd.read_parquet("work/std_zinc.parquet").rename(columns={"src_id": "zinc_id"})
    z = z.drop_duplicates("inchikey")
    z["zinc_id"] = z.zinc_id.astype(str)
    z = z[~z.zinc_id.isin(used)].reset_index(drop=True)
    print(f"ZINC pool disjoint from the enrichment decoys: {len(z)} "
          f"(removed {len(used)} enrichment substances)", flush=True)
    zbins = D.bins_of(z)

    rows, stats = [], []
    for gene in GENES:
        n_act = int((Y[gene] == 1).sum()); n_ina = int((Y[gene] == 0).sum())
        frac = n_act / (n_act + n_ina)
        if frac <= 0.85:
            print(f"{gene}: {100*frac:.1f}% active - no top-up needed", flush=True)
            continue
        need = int(round(n_act * (1 - TARGET_ACTIVE_FRAC) / TARGET_ACTIVE_FRAC)) - n_ina
        print(f"{gene}: {100*frac:.1f}% active ({n_act} act / {n_ina} ina) -> "
              f"adding {need} putative inactives", flush=True)
        act_ik = Y.index[(Y[gene] == 1).values]
        adf = base.reindex(act_ik).dropna(subset=["std_smiles"])
        act_bits = []
        for smi in adf.std_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                act_bits.append(D.MFPGEN.GetFingerprint(mol).ToBitString())
        abins = D.bins_of(adf)
        order = np.random.default_rng(5).permutation(len(adf))
        args = [(int(i), abins[i]) for i in order]
        got, seen = [], set()
        with Pool(7, initializer=D._init,
                  initargs=(z.std_smiles.tolist(), z.zinc_id.tolist(), zbins, act_bits)) as pool:
            for ai, lvl, keep in pool.imap_unordered(D._one, args, chunksize=8):
                for ci in keep[:2]:
                    if ci in seen:
                        continue
                    seen.add(ci)
                    got.append({"gene": gene, "zinc_id": z.zinc_id[ci],
                                "inchikey": z.inchikey[ci], "inchikey14": z.inchikey14[ci],
                                "std_smiles": z.std_smiles[ci], "scaffold": z.scaffold[ci],
                                "mw": z.mw[ci], "clogp": z.clogp[ci], "hbd": z.hbd[ci],
                                "hba": z.hba[ci], "rotb": z.rotb[ci], "charge": z.charge[ci],
                                "tpsa": z.tpsa[ci], "np_score": z.np_score[ci],
                                "fsp3": z.fsp3[ci], "heavy_atoms": z.heavy_atoms[ci],
                                "matched_active_inchikey": adf.index[ai],
                                "match_relaxation_level": lvl})
                if len(got) >= need:
                    break
                pool  # keep pool alive
        got = got[:need]
        rows.extend(got)
        stats.append({"gene": gene, "n_active": n_act, "n_inactive_real": n_ina,
                      "pct_active_before": round(100 * frac, 1), "n_putative_added": len(got),
                      "pct_active_after": round(100 * n_act / (n_act + n_ina + len(got)), 1)})
        print(f"  added {len(got)}", flush=True)

    put = pd.DataFrame(rows)
    if len(put):
        put.to_parquet("work/putative_decoys.parquet", index=False)
        pd.DataFrame(stats).to_csv("work/putative_stats.csv", index=False)
        print(f"\ntotal putative inactives: {len(put)} over "
              f"{put.zinc_id.nunique()} unique ZINC substances", flush=True)
        print(f"overlap with enrichment decoy set: "
              f"{len(set(put.zinc_id) & used)} (must be 0)", flush=True)
        print(pd.DataFrame(stats).to_string(index=False), flush=True)
    print("PUTATIVE DONE", flush=True)


if __name__ == "__main__":
    main()
