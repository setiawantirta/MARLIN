"""Pre-compute and cache molecular graphs for every evaluation set."""
import os, pickle, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata
from feat import featurise_list

os.makedirs(mdata.CACHE, exist_ok=True)


def save(name, obj):
    with open(os.path.join(mdata.CACHE, name), "wb") as f:
        pickle.dump(obj, f, protocol=4)


def need(name):
    return not os.path.exists(os.path.join(mdata.CACHE, name))


if need("chembl_graphs.pkl"):
    df = pd.read_parquet(mdata.MATRIX)
    save("chembl_graphs.pkl", featurise_list(df["canonical_smiles"].tolist(), tag=":chembl"))
    print("chembl done", flush=True)

if need("cmnpd_graphs.pkl"):
    cm = pd.read_parquet(mdata.CMNPD)
    col = "standardized_smiles" if "standardized_smiles" in cm.columns else \
          ("smiles_std" if "smiles_std" in cm.columns else "canonical_smiles")
    print("cmnpd smiles column:", col, cm.shape, flush=True)
    save("cmnpd_graphs.pkl", featurise_list(cm[col].tolist(), tag=":cmnpd"))
    print("cmnpd done", flush=True)

if need("decoy_graphs.pkl"):
    dc = pd.read_parquet(mdata.DECOYS)
    print("decoy cols:", list(dc.columns)[:15], dc.shape, flush=True)
    scol = [c for c in dc.columns if "smiles" in c.lower()][0]
    save("decoy_graphs.pkl", featurise_list(dc[scol].tolist(), tag=":decoy"))
    print("decoy done", flush=True)
print("ALL CACHE DONE", flush=True)
