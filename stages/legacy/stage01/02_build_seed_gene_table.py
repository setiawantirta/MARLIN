import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import json
import pandas as pd
import numpy as np

raw = json.load(open(RPATH("step1_raw.json")))
civ = json.load(open(RPATH("civic_genes.json")))

rows = []
for did, blk in raw["ot"].items():
    for r in blk["rows"]:
        t = r["target"]
        dts = {d["id"]: d["score"] for d in r["datatypeScores"]}
        rows.append(dict(disease_id=did, disease=blk["name"], gene_symbol=t["approvedSymbol"],
                         ensembl_id=t["id"], protein_name=t["approvedName"], biotype=t["biotype"],
                         score=r["score"], **{f"dt_{k}": v for k, v in dts.items()}))
ot = pd.DataFrame(rows)
print(ot.shape, ot.gene_symbol.nunique())
piv = ot.pivot_table(index="gene_symbol", columns="disease", values="score", aggfunc="max")
agg = (ot.groupby(["gene_symbol", "ensembl_id", "protein_name", "biotype"])
         .agg(ot_max_score=("score", "max"), ot_mean_score=("score", "mean"), n_diseases=("disease", "nunique"))
         .reset_index())
dtcols = [c for c in ot.columns if c.startswith("dt_")]
dtmax = ot.groupby("gene_symbol")[dtcols].max()
agg = agg.merge(dtmax, on="gene_symbol", how="left")
agg["civic_evidence_items"] = agg.gene_symbol.map(civ["counts"]).fillna(0).astype(int)
agg["sources"] = np.where(agg.civic_evidence_items > 0, "OpenTargets;CIViC", "OpenTargets")
extra = [g for g in civ["counts"] if g not in set(agg.gene_symbol)]
print("civic-only genes:", extra)
agg = agg.sort_values(["ot_max_score", "n_diseases"], ascending=False).reset_index(drop=True)
print(agg.shape)
print(agg.head(25)[["gene_symbol", "ot_max_score", "n_diseases", "civic_evidence_items", "biotype"]].to_string())

seed = agg[agg.biotype == "protein_coding"].copy()
civic_only = set(civ["counts"])
top = seed.head(180).copy()
add = seed[(~seed.gene_symbol.isin(top.gene_symbol)) & (seed.gene_symbol.isin(civic_only))]
seed_sel = pd.concat([top, add]).reset_index(drop=True)
seed_sel["in_seed_reason"] = np.where(seed_sel.index < len(top), "top180_by_OT_score", "CIViC_clinical_evidence")
print("n_seed_genes =", len(seed_sel), "| civic-augmented:", len(add), list(add.gene_symbol))
seed_sel.to_csv("glioma_seed_genes.csv", index=False)
print(seed_sel.columns.tolist())
print("score range:", seed_sel.ot_max_score.min().round(4), seed_sel.ot_max_score.max().round(4))

extra_rows = pd.DataFrame([dict(gene_symbol=g, ensembl_id="", protein_name="", biotype="protein_coding",
    ot_max_score=np.nan, ot_mean_score=np.nan, n_diseases=0,
    civic_evidence_items=civ["counts"][g], sources="CIViC", in_seed_reason="CIViC_clinical_evidence")
    for g in ["SDHB", "MYB", "MYBL1"]])
seed_sel = pd.concat([seed_sel, extra_rows], ignore_index=True)
seed_sel.to_csv("glioma_seed_genes.csv", index=False)
symbols = seed_sel.gene_symbol.tolist()
print("n_seed_genes =", len(seed_sel))