import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import json
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

react = json.load(open(RPATH("reactome_raw.json")))
go = json.load(open(RPATH("go_raw.json")))
cen = pd.read_csv(RPATH("centrality.csv"))

# --- Reactome ---
rrows = []
for m, r in react.items():
    n_found = r["batch_summary"]["identifiers_found_count"]
    for p in r["batch_summary"]["pathways"]:
        rrows.append(dict(module=m, source="Reactome_v"+str(r["reactome_version"]), term_id=p["stId"],
            term_name=p["name"], in_disease=p["inDisease"], entities_found=p["entities_found"],
            entities_total=p["entities_total"], module_genes_mapped=n_found,
            pvalue=p["entities_pvalue"], fdr=p["entities_fdr"]))
rea = pd.DataFrame(rrows)
rea = rea[(rea.fdr < 0.05) & (rea.entities_found >= 3)].sort_values(["module", "fdr"]).reset_index(drop=True)

# --- GO BP/MF ---
g2t = {g: {(a["aspect"], a["id"], a["label"]) for a in v} for g, v in go.items()}
allg = [g for g in cen.gene_symbol if g in g2t]
N = len(allg)
term_genes = {}
for g in allg:
    for t in g2t[g]:
        term_genes.setdefault(t, set()).add(g)

grows = []
for m in ["M1", "M2", "M3", "M4"]:
    mg = [g for g in cen.loc[cen.module == m, "gene_symbol"] if g in g2t]
    n = len(mg)
    for t, gs in term_genes.items():
        K = len(gs)
        k = len(gs & set(mg))
        if k < 3 or K < 3:
            continue
        p = hypergeom.sf(k - 1, N, K, n)
        grows.append(dict(module=m, source="GO_" + t[0], term_id=t[1], term_name=t[2],
                          entities_found=k, entities_total=K, module_genes_mapped=n, pvalue=p,
                          genes=";".join(sorted(gs & set(mg)))))
gof = pd.DataFrame(grows)
gof["fdr"] = np.nan
for m, idx in gof.groupby("module").groups.items():
    gof.loc[idx, "fdr"] = multipletests(gof.loc[idx, "pvalue"], method="fdr_bh")[1]
gof_sig = gof[gof.fdr < 0.05].sort_values(["module", "fdr"]).reset_index(drop=True)

enr = pd.concat([rea.assign(background="Reactome genome-wide (AnalysisService)"),
                 gof_sig.assign(background=f"within-network ({N} seed proteins)")], ignore_index=True)
enr.to_csv("glioma_pathway_enrichment.csv", index=False)