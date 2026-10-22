import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from repro.config import path as RPATH, CFG, out_dir
"""Rebuild the ECFP4(2048,r=2)+30-RDKit-descriptor cache for the interpretability track.

Uses the EXACT featurisation of code/src/fp_features.py (same DESC list, same Morgan
generator) but parallelised over processes, because the interpretability track needs both
the ChEMBL matrix (34,190) and the full curated CMNPD library (26,754).
"""
import os, sys, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

# fp_features.py lives in the package's own src/ directory. This was written as
# os.path.abspath("code/src"), a path relative to the working directory, which
# existed in the original project workspace but not in the packaged module: the
# stage runner uses the stage's output directory as cwd, so the import failed with
# ModuleNotFoundError and took Figure 5 and Figure 8 down with it. Derive it from
# __file__ instead, the way the repo root is derived two lines above.
SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "src")
if not os.path.isdir(SRC):                      # tolerate the original layout too
    SRC = os.path.abspath("code/src")
sys.path.insert(0, SRC)

# fp_features imports mdata only for the CACHE constant; mdata pulls in torch, which is
# not needed here. Stub it with the same absolute artifact paths mdata.py declares.
import types                        # noqa: E402
_m = types.ModuleType("mdata")
_m.CACHE = "cache"
_m.MATRIX = RPATH("chembl_multitask_matrix.parquet")
_m.CMNPD = RPATH("cmnpd_curated.parquet")
sys.modules["mdata"] = _m
mdata = _m
import fp_features as FPF          # noqa: E402  (DESC, FP_GEN, featurise)

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)


def _chunk(smiles):
    return FPF.featurise(smiles)


def build(name, smiles, nproc=7, chunk=1500):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        print("exists", name, flush=True)
        return
    t0 = time.time()
    parts = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    with Pool(nproc) as pool:
        res = pool.map(_chunk, parts)
    FP = np.vstack([r[0] for r in res])
    D = np.vstack([r[1] for r in res])
    ok = np.concatenate([r[2] for r in res])
    assert FP.shape == (len(smiles), 2048), FP.shape
    assert D.shape == (len(smiles), 30), D.shape
    np.savez_compressed(p, FP=FP, D=D, ok=ok,
                        desc_names=np.array(FPF.DESC_NAMES))
    print("saved", name, FP.shape, "ok=%d" % ok.sum(), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    df = pd.read_parquet(mdata.MATRIX)
    build("chembl_fp.npz", df["canonical_smiles"].tolist())
    cm = pd.read_parquet(mdata.CMNPD, columns=["smiles_std"])
    build("cmnpd_fp.npz", cm["smiles_std"].tolist())
    print("FEATURE CACHE DONE", flush=True)
