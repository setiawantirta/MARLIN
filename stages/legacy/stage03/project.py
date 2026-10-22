"""Marine-vs-synthetic chemical-space projection (UMAP on a 50-component SVD of ECFP4)."""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs("/tmp/numba_cache", exist_ok=True)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from rdkit import DataStructs
from sklearn.decomposition import TruncatedSVD
import ad_umap as A

if __name__ == "__main__":
    mat = pd.read_parquet("chembl_multitask_matrix.parquet")
    tr = mat[(mat.split == "train") & (~mat.get("is_putative_decoy", False))]
    cm = pd.read_parquet("applicability_domain.csv.parquet") if False else \
        pd.read_csv("applicability_domain.csv")
    with Pool(7) as p:
        tb = [b for b in p.map(A._fp_bin, tr.canonical_smiles.tolist(), chunksize=200) if b]
        cb = [b for b in p.map(A._fp_bin, cm.std_smiles.tolist(), chunksize=200) if b]
    print(f"fps: train {len(tb)}, marine {len(cb)}", flush=True)

    def to_mat(bl):
        X = np.zeros((len(bl), 2048), dtype=np.uint8)
        for i, b in enumerate(bl):
            DataStructs.ConvertToNumpyArray(DataStructs.CreateFromBitString(b), X[i])
        return sp.csr_matrix(X)

    X = sp.vstack([to_mat(tb), to_mat(cb)])
    svd = TruncatedSVD(n_components=50, random_state=0)
    Z = svd.fit_transform(X.astype(np.float32))
    print("SVD-50 explained variance: %.3f" % svd.explained_variance_ratio_.sum(), flush=True)
    import umap
    emb = umap.UMAP(n_neighbors=25, min_dist=0.1, metric="euclidean",
                    random_state=42, verbose=False).fit_transform(Z)
    src = np.array(["ChEMBL training pool"] * len(tb) + ["CMNPD marine"] * len(cb))
    out = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "source": src})
    out["max_tanimoto_to_train"] = np.concatenate(
        [np.load("work/ad_loo_train.npy"), cm.max_tanimoto_to_train.values])
    out.to_parquet("work/chemspace.parquet", index=False)
    print("PROJECTION DONE (UMAP)", flush=True)
