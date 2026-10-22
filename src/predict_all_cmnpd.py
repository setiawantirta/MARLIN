"""Score every trained GNN checkpoint over the curated CMNPD library."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_sets import predict

stages = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pretrain"]
for p in sorted(glob.glob("metrics/*_run.json")):
    r = json.load(open(p))
    if r.get("stage") not in stages or r["arch"] in ("RF", "XGB"):
        continue
    out = f"metrics/cmnpd_pred_{r['name']}.npz"
    if os.path.exists(out):
        continue
    ck = f"models/{r['name']}.pt"
    if not os.path.exists(ck):
        continue
    P, I = predict(r["arch"], ck, "cmnpd", threads=2)
    np.savez_compressed(out, P=P, I=I)
    print("wrote", out, P.shape, flush=True)
print("CMNPD PRED DONE", flush=True)
