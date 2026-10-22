"""Property-matched ZINC20 decoys under the DUD-E criterion.

For every held-out-test active of each target we draw decoys from the ZINC20 pool that are
matched on six physicochemical descriptors (MW, cLogP, HBD, HBA, rotatable bonds, net formal
charge) while being TOPOLOGICALLY DISSIMILAR to *every* known active of that target
(ECFP4 / Morgan r=2, 2048 bit, Tanimoto < 0.35). Matching uses binned descriptor space with
a documented relaxation ladder; the level actually used is recorded per decoy.

Decoys accompany the TEST actives because enrichment is evaluated on the held-out partition;
they are written to a separate table and never carry training labels.
"""
import sys, os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")

TANIMOTO_MAX = 0.35
DECOYS_PER_ACTIVE = 10
CAND_SAMPLE = 80
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# (mw_bins, logp_bins, dhbd, dhba, drotb, exact_charge)
LADDER = [(0, 0, 0, 0, 0, True), (1, 1, 0, 1, 1, True), (2, 2, 1, 2, 2, True),
          (3, 3, 1, 3, 3, True), (5, 4, 2, 4, 4, True), (7, 6, 3, 6, 6, False)]

_W = {}


def bins_of(df):
    return np.stack([(df.mw.values // 25).astype(int),
                     np.floor(df.clogp.values / 0.5).astype(int),
                     np.minimum(df.hbd.values, 6).astype(int),
                     np.minimum(df.hba.values, 12).astype(int),
                     np.minimum(df.rotb.values, 12).astype(int),
                     np.clip(df.charge.values, -2, 2).astype(int)], axis=1)


def _init(zsmiles, zids, zbins, act_fp_bits):
    _W["zsmiles"] = zsmiles
    _W["zids"] = zids
    idx = {}
    for i, b in enumerate(zbins):
        idx.setdefault(tuple(b), []).append(i)
    _W["idx"] = idx
    fps = []
    for bs in act_fp_bits:
        fps.append(DataStructs.CreateFromBitString(bs))
    _W["act_fps"] = fps
    _W["rng"] = np.random.default_rng(1234)


def _one(arg):
    ai, abin = arg
    idx, rng = _W["idx"], _W["rng"]
    for lvl, (dm, dl, dd, da, dr, exact_q) in enumerate(LADDER):
        cands = []
        for m in range(abin[0] - dm, abin[0] + dm + 1):
            for l in range(abin[1] - dl, abin[1] + dl + 1):
                for d in range(abin[2] - dd, abin[2] + dd + 1):
                    for a in range(abin[3] - da, abin[3] + da + 1):
                        for r in range(abin[4] - dr, abin[4] + dr + 1):
                            qs = [abin[5]] if exact_q else range(abin[5] - 1, abin[5] + 2)
                            for q in qs:
                                cands.extend(idx.get((m, l, d, a, r, q), ()))
        if len(cands) >= DECOYS_PER_ACTIVE * 3 or lvl == len(LADDER) - 1:
            if not cands:
                return ai, lvl, []
            cands = np.asarray(cands)
            if len(cands) > CAND_SAMPLE:
                cands = rng.choice(cands, CAND_SAMPLE, replace=False)
            keep = []
            for ci in cands:
                mol = Chem.MolFromSmiles(_W["zsmiles"][ci])
                if mol is None:
                    continue
                fp = MFPGEN.GetFingerprint(mol)
                if max(DataStructs.BulkTanimotoSimilarity(fp, _W["act_fps"])) < TANIMOTO_MAX:
                    keep.append(int(ci))
                if len(keep) >= DECOYS_PER_ACTIVE:
                    break
            return ai, lvl, keep
    return ai, len(LADDER) - 1, []


if __name__ == "__main__":
    z = pd.read_parquet("work/std_zinc.parquet").rename(columns={"src_id": "zinc_id"})
    z = z.drop_duplicates("inchikey").reset_index(drop=True)
    z["zinc_id"] = z.zinc_id.astype(str)
    # keep the enrichment decoys DISJOINT from the putative training inactives, so a compound
    # the model may have trained on can never appear in the held-out enrichment background
    if os.path.exists("work/putative_decoys.parquet"):
        pz = set(pd.read_parquet("work/putative_decoys.parquet").zinc_id.astype(str))
        z = z[~z.zinc_id.isin(pz)].reset_index(drop=True)
        print(f"excluded {len(pz)} putative-inactive substances from the enrichment pool",
              flush=True)
    print(f"ZINC pool after standardisation + InChIKey dedup: {len(z)}", flush=True)
    zbins = bins_of(z)
    acts = pd.read_parquet("work/actives_for_decoys.parquet")

    all_rows, stats = [], []
    for gene, sub in acts.groupby("gene"):
        test = sub[sub.is_test].reset_index(drop=True)
        act_bits = []
        for smi in sub.std_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                act_bits.append(MFPGEN.GetFingerprint(mol).ToBitString())
        abins = bins_of(test)
        args = [(i, abins[i]) for i in range(len(test))]
        used, rows, lvls, short = set(), [], [], 0
        with Pool(7, initializer=_init,
                  initargs=(z.std_smiles.tolist(), z.zinc_id.tolist(), zbins, act_bits)) as pool:
            for ai, lvl, keep in pool.imap_unordered(_one, args, chunksize=8):
                lvls.append(lvl)
                n = 0
                for ci in keep:
                    if ci in used:
                        continue
                    used.add(ci); n += 1
                    rows.append({"gene": gene, "zinc_id": z.zinc_id[ci],
                                 "inchikey": z.inchikey[ci], "std_smiles": z.std_smiles[ci],
                                 "mw": z.mw[ci], "clogp": z.clogp[ci], "hbd": z.hbd[ci],
                                 "hba": z.hba[ci], "rotb": z.rotb[ci], "charge": z.charge[ci],
                                 "np_score": z.np_score[ci],
                                 "matched_active_inchikey": test.inchikey[ai],
                                 "match_relaxation_level": lvl})
                if n < DECOYS_PER_ACTIVE:
                    short += 1
        st = {"gene": gene, "n_actives_total": len(sub), "n_test_actives": len(test),
              "n_decoys": len(rows), "ratio": round(len(rows) / max(len(test), 1), 2),
              "actives_short_of_10": short,
              "median_relaxation_level": float(np.median(lvls)) if lvls else None,
              "level0_frac": round(float(np.mean(np.array(lvls) == 0)), 3) if lvls else None}
        stats.append(st); print(st, flush=True)
        all_rows.extend(rows)

    dec = pd.DataFrame(all_rows)
    dec.to_parquet("decoy_set.parquet", index=False)
    pd.DataFrame(stats).to_csv("work/decoy_stats.csv", index=False)
    print(f"\nTOTAL decoys: {len(dec)} | unique ZINC substances: {dec.zinc_id.nunique()}",
          flush=True)
    print("DECOYS DONE", flush=True)
