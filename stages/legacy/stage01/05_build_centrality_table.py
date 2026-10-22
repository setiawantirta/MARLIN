import json, os, collections, re
import pandas as pd
import numpy as np
import networkx as nx

os.makedirs("handoff", exist_ok=True)

# Load raw data
raw = json.load(open("handoff/step1_raw.json"))
civ = json.load(open("handoff/civic_genes.json"))

# Build OpenTargets dataframe
rows = []
for did, blk in raw["ot"].items():
    for r in blk["rows"]:
        t = r["target"]
        dts = {d["id"]: d["score"] for d in r["datatypeScores"]}
        rows.append(dict(disease_id=did, disease=blk["name"], gene_symbol=t["approvedSymbol"],
                         ensembl_id=t["id"], protein_name=t["approvedName"], biotype=t["biotype"],
                         score=r["score"], **{f"dt_{k}": v for k, v in dts.items()}))
ot = pd.DataFrame(rows)

dtcols = [c for c in ot.columns if c.startswith("dt_")]
agg = (ot.groupby(["gene_symbol", "ensembl_id", "protein_name", "biotype"])
         .agg(ot_max_score=("score", "max"), ot_mean_score=("score", "mean"), n_diseases=("disease", "nunique"))
         .reset_index())
dtmax = ot.groupby("gene_symbol")[dtcols].max()
agg = agg.merge(dtmax, on="gene_symbol", how="left")
agg["civic_evidence_items"] = agg.gene_symbol.map(civ["counts"]).fillna(0).astype(int)
agg["sources"] = np.where(agg.civic_evidence_items > 0, "OpenTargets;CIViC", "OpenTargets")
agg = agg.sort_values(["ot_max_score", "n_diseases"], ascending=False).reset_index(drop=True)

# Select seed genes
seed = agg[agg.biotype == "protein_coding"].copy()
civic_only = set(civ["counts"])
top = seed.head(180).copy()
add = seed[(~seed.gene_symbol.isin(top.gene_symbol)) & (seed.gene_symbol.isin(civic_only))]
seed_sel = pd.concat([top, add]).reset_index(drop=True)
seed_sel["in_seed_reason"] = np.where(seed_sel.index < len(top), "top180_by_OT_score", "CIViC_clinical_evidence")

# Add extra CIViC-only genes
extra_rows = pd.DataFrame([dict(gene_symbol=g, ensembl_id="", protein_name="", biotype="protein_coding",
    ot_max_score=np.nan, ot_mean_score=np.nan, n_diseases=0,
    civic_evidence_items=civ["counts"][g], sources="CIViC", in_seed_reason="CIViC_clinical_evidence")
    for g in ["SDHB", "MYB", "MYBL1"]])
seed_sel = pd.concat([seed_sel, extra_rows], ignore_index=True)

# Build STRING network
net = json.load(open("handoff/string_net.json"))
G = nx.Graph()
for n in net["nodes"]:
    G.add_node(n["name"])
for e in net["edges"]:
    G.add_edge(e["a"], e["b"], weight=e["score"])

# Compute centrality measures
deg = dict(G.degree())
btw = nx.betweenness_centrality(G, weight=None, normalized=True)
clo = nx.closeness_centrality(G)
try:
    eig = nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
except Exception as e:
    eig = nx.eigenvector_centrality_numpy(G, weight="weight")

comm = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
mod_q = nx.algorithms.community.modularity(G, comm, weight="weight")

mod_of = {}
for i, c in enumerate(sorted(comm, key=len, reverse=True), start=1):
    lab = f"M{i}" if len(c) >= 5 else "unassigned"
    for g in c:
        mod_of[g] = lab

cen = pd.DataFrame({"gene_symbol": list(G.nodes())})
cen["degree"] = cen.gene_symbol.map(deg)
cen["betweenness"] = cen.gene_symbol.map(btw)
cen["closeness"] = cen.gene_symbol.map(clo)
cen["eigenvector"] = cen.gene_symbol.map(eig)
cen["module"] = cen.gene_symbol.map(mod_of)
cen = cen.merge(seed_sel[["gene_symbol", "protein_name", "ot_max_score", "civic_evidence_items", "sources", "ensembl_id"]], on="gene_symbol", how="left")

def z(s):
    s = s.astype(float)
    return (s - s.mean()) / s.std(ddof=0)

cen["composite_hub_score"] = (z(cen.degree) + z(cen.betweenness) + z(cen.closeness) + z(cen.eigenvector)
                              + z(cen.ot_max_score.fillna(cen.ot_max_score.min()))) / 5
cen = cen.sort_values("composite_hub_score", ascending=False).reset_index(drop=True)
cen["composite_hub_rank"] = cen.index + 1

cen.to_csv("handoff/centrality.csv", index=False)