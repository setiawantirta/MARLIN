"""Refit the selected Random Forest with the recorded hyperparameters and persist it,
so `predict.py` can score arbitrary SMILES without re-running the benchmark."""
import json, os, sys, time
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata
from baselines import load_X
from feat import TASKS
from sklearn.ensemble import RandomForestClassifier

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cfg = json.load(open(f"metrics/RF_s0_baseline_run.json"))
df = pd.read_parquet(mdata.MATRIX)
_, Y, mask = mdata.load_matrix()
X, ok, norm = load_X()
split = df["split"].to_numpy()
os.makedirs("models/rf_final", exist_ok=True)
sizes = {}
t0 = time.time()
for ti, t in enumerate(TASKS):
    tr = np.where((mask[:, ti] > 0) & ok & (split == "train"))[0]
    p = dict(cfg["chosen"][t]["params"])
    m = RandomForestClassifier(random_state=SEED, n_jobs=6,
                               class_weight="balanced_subsample", **p)
    m.fit(X[tr], Y[tr, ti].astype(int))
    fp = f"models/rf_final/rf_{t}_s{SEED}.joblib"
    joblib.dump(m, fp, compress=3)
    sizes[t] = round(os.path.getsize(fp) / 1e6, 1)
    print(f"{t}: n={len(tr)} params={p} {sizes[t]} MB ({time.time()-t0:.0f}s)", flush=True)
json.dump({"seed": SEED, "tasks": list(TASKS), "sizes_mb": sizes,
           "desc_mean": norm["desc_mean"], "desc_std": norm["desc_std"],
           "desc_median": norm["desc_median"],
           "params": {t: cfg["chosen"][t]["params"] for t in TASKS}},
          open("models/rf_final/rf_meta.json", "w"), indent=1)
print("TOTAL MB", round(sum(sizes.values()), 1), flush=True)
