"""Applicability domain of the training pool with respect to the full CMNPD marine library,
plus a marine-vs-synthetic chemical-space projection.

AD metric: each marine compound's MAXIMUM ECFP4 (Morgan r=2, 2048 bit) Tanimoto similarity to
any training-set compound. Two thresholds are reported:
  (a) a data-driven threshold - the 5th percentile of the training set's own leave-one-out
      nearest-neighbour similarity distribution, i.e. a marine compound is inside the domain
      if it is at least as close to the training set as the least-typical 5% of the training
      compounds themselves are to each other (Tropsha-style distance-to-model cutoff);
  (b) a grid of fixed conventional cutoffs (0.20-0.45) for comparability with the literature.
"""
import numpy as np, pandas as pd, os
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool
RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_S = {}


def fps_of(smiles, nproc=7):
    with Pool(nproc) as p:
        return [DataStructs.CreateFromBitString(b) for b in
                p.map(_fp_bin, smiles, chunksize=500) if b is not None]


def _fp_bin(smi):
    m = Chem.MolFromSmiles(smi)
    return GEN.GetFingerprint(m).ToBitString() if m is not None else None


def _init(train_bits):
    _S["tr"] = [DataStructs.CreateFromBitString(b) for b in train_bits]


def _maxsim(bits):
    fp = DataStructs.CreateFromBitString(bits)
    return float(max(DataStructs.BulkTanimotoSimilarity(fp, _S["tr"])))


def _loo(i):
    fp = _S["tr"][i]
    s = DataStructs.BulkTanimotoSimilarity(fp, _S["tr"])
    s[i] = -1.0
    return float(max(s))


if __name__ == "__main__":
    mat = pd.read_parquet("chembl_multitask_matrix.parquet")
    if "is_putative_decoy" in mat.columns:
        # the applicability domain is defined by the REAL labelled training chemistry; the
        # putative ZINC inactives are a training aid and are reported separately below
        putv = mat[(mat.split == "train") & (mat.is_putative_decoy)]
        tr = mat[(mat.split == "train") & (~mat.is_putative_decoy)]
    else:
        putv, tr = mat.iloc[:0], mat[mat.split == "train"]
    cm = pd.read_parquet("work/std_cmnpd.parquet").rename(columns={"src_id": "COMPOUND_ID"})
    print(f"training compounds: {len(tr)} | marine compounds: {len(cm)}", flush=True)

    with Pool(7) as p:
        tr_bits = p.map(_fp_bin, tr.canonical_smiles.tolist(), chunksize=200)
        cm_bits = p.map(_fp_bin, cm.std_smiles.tolist(), chunksize=200)
    tr_bits = [b for b in tr_bits if b is not None]
    ok = [i for i, b in enumerate(cm_bits) if b is not None]
    cm = cm.iloc[ok].reset_index(drop=True)
    cm_bits = [cm_bits[i] for i in ok]
    print(f"fingerprints: train {len(tr_bits)}, marine {len(cm_bits)}", flush=True)

    with Pool(7, initializer=_init, initargs=(tr_bits,)) as p:
        loo = np.array(p.map(_loo, range(len(tr_bits)), chunksize=100))
        ms = np.array(p.map(_maxsim, cm_bits, chunksize=100))
    thr = float(np.percentile(loo, 5))
    print(f"training LOO nearest-neighbour similarity: mean {loo.mean():.3f}, "
          f"median {np.median(loo):.3f}, 5th pct {thr:.3f}", flush=True)

    cm["max_tanimoto_to_train"] = ms
    cm["inside_AD_datadriven"] = ms >= thr
    grid = {}
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        grid[t] = float((ms >= t).mean())
        cm[f"inside_AD_{t:.2f}"] = ms >= t
    print("\nmarine max-Tanimoto-to-training distribution:", flush=True)
    q = np.percentile(ms, [0, 5, 25, 50, 75, 95, 100])
    print("  min/5/25/50/75/95/max:", np.round(q, 3).tolist(),
          f"| mean {ms.mean():.3f}", flush=True)
    print(f"\ndata-driven AD (>= {thr:.3f}): {int((ms>=thr).sum())} / {len(ms)} "
          f"({100*(ms>=thr).mean():.2f}%) of the marine library inside the domain", flush=True)
    for t, f in grid.items():
        print(f"  fixed cutoff {t:.2f}: {100*f:.2f}% inside ({int(f*len(ms))} compounds)",
              flush=True)
    # how much the domain moves if the putative ZINC inactives are also trained on
    if len(putv):
        with Pool(7) as p:
            pv_bits = [b for b in p.map(_fp_bin, putv.canonical_smiles.tolist(),
                                        chunksize=200) if b is not None]
        with Pool(7, initializer=_init, initargs=(tr_bits + pv_bits,)) as p:
            ms_aug = np.array(p.map(_maxsim, cm_bits, chunksize=100))
        cm["max_tanimoto_to_train_with_putative"] = ms_aug
        print(f"\nwith the {len(pv_bits)} putative ZINC inactives added to the training pool: "
              f"marine median max-Tanimoto {np.median(ms_aug):.3f} "
              f"(was {np.median(ms):.3f}); inside AD at 0.30: "
              f"{100*(ms_aug>=0.30).mean():.2f}% (was {100*(ms>=0.30).mean():.2f}%)", flush=True)

    cm.to_csv("applicability_domain.csv", index=False)
    np.save("work/ad_loo_train.npy", loo)
    json_thr = {"threshold_datadriven": thr, "grid": grid,
                "n_marine": int(len(ms)), "n_train": int(len(tr_bits))}
    pd.Series(json_thr).to_json("work/ad_meta.json")

    # ---- chemical-space projection -----------------------------------------
    from sklearn.decomposition import TruncatedSVD
    import scipy.sparse as sp

    def to_mat(bitlist):
        rows = []
        for b in bitlist:
            arr = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(DataStructs.CreateFromBitString(b), arr)
            rows.append(sp.csr_matrix(arr))
        return sp.vstack(rows)

    X = sp.vstack([to_mat(tr_bits), to_mat(cm_bits)])
    src = np.array(["ChEMBL (synthetic training pool)"] * len(tr_bits) +
                   ["CMNPD (marine natural products)"] * len(cm_bits))
    svd = TruncatedSVD(n_components=50, random_state=0)
    Z = svd.fit_transform(X.astype(np.float32))
    print(f"\nSVD-50 explained variance: {svd.explained_variance_ratio_.sum():.3f}", flush=True)
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="euclidean",
                        random_state=42).fit_transform(Z)
        method = "UMAP (on 50-component SVD of ECFP4)"
    except Exception as ex:
        print("UMAP failed, PCA fallback:", ex, flush=True)
        emb = Z[:, :2]; method = "PCA/SVD (UMAP unavailable)"
    out = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "source": src})
    out["max_tanimoto_to_train"] = np.concatenate([loo, ms])
    out.to_parquet("work/chemspace.parquet", index=False)
    print("projection method:", method, flush=True)
    print("AD DONE", flush=True)
