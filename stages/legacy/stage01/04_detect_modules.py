import json, os, collections, re, numpy as np, pandas as pd
import networkx as nx

os.makedirs("handoff", exist_ok=True)

# Step 1: Load raw data
raw = json.load(open("handoff/step1_raw.json"))
civ = json.load(open("handoff/civic_genes.json"))

# Step 2: Build OT dataframe
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
dtmax = ot.groupby("gene_symbol")[dtcols].max()
agg = (ot.groupby(["gene_symbol", "ensembl_id", "protein_name", "biotype"])
         .agg(ot_max_score=("score", "max"), ot_mean_score=("score", "mean"), n_diseases=("disease", "nunique"))
         .reset_index())
agg = agg.merge(dtmax, on="gene_symbol", how="left")
agg["civic_evidence_items"] = agg.gene_symbol.map(civ["counts"]).fillna(0).astype(int)
agg["sources"] = np.where(agg.civic_evidence_items > 0, "OpenTargets;CIViC", "OpenTargets")
agg = agg.sort_values(["ot_max_score", "n_diseases"], ascending=False).reset_index(drop=True)

# Step 3: Build seed genes
seed = agg[agg.biotype == "protein_coding"].copy()
civic_only = set(civ["counts"])
top = seed.head(180).copy()
add = seed[(~seed.gene_symbol.isin(top.gene_symbol)) & (seed.gene_symbol.isin(civic_only))]
seed_sel = pd.concat([top, add]).reset_index(drop=True)
seed_sel["in_seed_reason"] = np.where(seed_sel.index < len(top), "top180_by_OT_score", "CIViC_clinical_evidence")

extra_rows = pd.DataFrame([dict(gene_symbol=g, ensembl_id="", protein_name="", biotype="protein_coding",
    ot_max_score=np.nan, ot_mean_score=np.nan, n_diseases=0,
    civic_evidence_items=civ["counts"][g], sources="CIViC", in_seed_reason="CIViC_clinical_evidence")
    for g in ["SDHB", "MYB", "MYBL1"]])
seed_sel = pd.concat([seed_sel, extra_rows], ignore_index=True)

# Step 4: Build network and compute modules
net = json.load(open("handoff/string_net.json"))
G = nx.Graph()
for n in net["nodes"]:
    G.add_node(n["name"])
for e in net["edges"]:
    G.add_edge(e["a"], e["b"], weight=e["score"])

comm = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))

mod_of = {}
for i, c in enumerate(sorted(comm, key=len, reverse=True), start=1):
    lab = f"M{i}" if len(c) >= 5 else "unassigned"
    for g in c:
        mod_of[g] = lab

json.dump({g: mod_of.get(g, "unassigned") for g in G.nodes()}, open("handoff/modules.json", "w"))