import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import json
import pandas as pd
import numpy as np

counts = json.load(open(RPATH("chembl_counts.json")))
cen = pd.read_csv(RPATH("centrality.csv"))

cand = pd.DataFrame([dict(gene_symbol=g, **{k:v for k,v in d.items() if k!="fda_drugs"},
                          fda_drugs="; ".join([x for x in d.get("fda_drugs",[]) if x][:8]))
                     for g,d in counts.items() if "err" not in d])
cand = cand.merge(cen[["gene_symbol","degree","betweenness","closeness","eigenvector","module",
                       "composite_hub_rank","composite_hub_score","ot_max_score","civic_evidence_items"]],
                  on="gene_symbol", how="left")
cand["passes_300"] = cand.n_compounds >= 300
cand = cand.sort_values("composite_hub_rank")

FINAL = ["EGFR","PDGFRA","PIK3CA","KDR","CDK6","IDH1"]

axis_map = {
 "EGFR": "RTK–PI3K/RAS-MAPK axis (Reactome: Signaling by EGFR in Cancer FDR=3.4e-4; Constitutive Signaling by EGFRvIII FDR=6.3e-5)",
 "PDGFRA":"RTK–PDGF axis (Reactome: Signaling by PDGFR in disease FDR=1.0e-7; Signaling by PDGF FDR=2.1e-5)",
 "PIK3CA":"PI3K-AKT-mTOR axis (Reactome: PI3K/AKT Signaling in Cancer FDR=1.1e-14)",
 "KDR":  "VEGF/angiogenesis axis (Reactome: VEGFA-VEGFR2 Pathway FDR=1.5e-3; GO-BP angiogenesis FDR=2.2e-2)",
 "CDK6": "CDKN2A–RB1 cell-cycle G1/S axis (module M3; Reactome Oncogene Induced Senescence FDR=1.1e-5)",
 "IDH1": "IDH-mutant 2-hydroxyglutarate metabolic axis (module M3; axis from gene-level OT/CIViC evidence, no module-level metabolic Reactome term at FDR<0.05)",
}
rationale = {
 "EGFR":"Highest-degree druggable RTK hub (degree 44, composite rank 7); EGFR amplification/EGFRvIII is the canonical classical-GBM lesion and Reactome flags EGFRvIII constitutive signalling inside module M1. Largest labelled chemical space of the panel (11,158 compounds) and 20 approved drugs for positive controls.",
 "PDGFRA":"Proneural-GBM amplification target and second RTK arm of module M1 (degree 22, rank 31); keeps the panel able to learn RTK selectivity rather than a single-receptor signature. 937 compounds, 4 approved drugs.",
 "PIK3CA":"Top-ranked druggable hub overall (degree 51, composite rank 5) and the statistically strongest axis in the network (PI3K/AKT Signaling in Cancer, 34/132 entities, FDR=1.1e-14). 7,688 compounds, 4 approved drugs.",
 "KDR":"Sole representative of the VEGF/angiogenesis axis, which is enriched in module M1 and is the therapeutic rationale for anti-angiogenic therapy in GBM. Very dense labels (10,610 compounds) support multi-task transfer; 4 approved drugs.",
 "CDK6":"Represents the third canonical GBM core pathway (CDKN2A deletion - RB1 inactivation); CNS-penetrant CDK4/6 inhibitors (abemaciclib) are in glioma trials. Lower centrality (degree 9, rank 134) accepted deliberately for axis coverage; 838 compounds, 5 approved drugs.",
 "IDH1":"Molecularly defining lesion of WHO grade 2-3 IDH-mutant glioma and the highest-scoring CIViC-evidenced gene in the seed set (17 evidence items, OT score 0.842); inhibitor chemistry is mature (4,300 compounds; ivosidenib, olutasidenib approved). Adds a non-kinase, metabolic-enzyme target that forces the graph models to generalise beyond ATP-site chemotypes.",
}
panel = cand[cand.gene_symbol.isin(FINAL)].copy()
panel["pathway_axis"]=panel.gene_symbol.map(axis_map)
panel["selection_rationale"]=panel.gene_symbol.map(rationale)
panel = panel.rename(columns={"pref_name":"protein_name","uniprot":"uniprot_id",
                              "n_compounds":"n_compounds_chembl","n_fda":"n_fda_approved"})
cols=["gene_symbol","protein_name","uniprot_id","chembl_target_id","module","degree","betweenness",
      "closeness","eigenvector","composite_hub_rank","composite_hub_score","ot_max_score",
      "civic_evidence_items","n_activities","n_compounds_chembl","n_fda_approved","fda_drugs",
      "pathway_axis","selection_rationale"]
panel = panel[cols].rename(columns={"ot_max_score":"disease_association_score"}).sort_values("composite_hub_rank")
panel.to_csv("glioma_target_panel.csv", index=False)