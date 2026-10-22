import json, os, re, collections
import pandas as pd
import numpy as np

os.makedirs("handoff", exist_ok=True)

DIS = {
 "MONDO_0018177":"glioblastoma",
 "MONDO_0021042":"glioma",
 "MONDO_0021637":"low grade glioma",
 "MONDO_0100342":"malignant glioma",
 "MONDO_0016695":"oligodendroglioma",
 "MONDO_0019781":"astrocytoma (excluding glioblastoma)",
}
q = """
query($id: String!, $n: Int!){
  disease(efoId: $id){
    id name
    associatedTargets(page:{index:0, size:$n}){
      count
      rows{
        score
        datatypeScores{ id score }
        target{ id approvedSymbol approvedName biotype }
      }
    }
  }
}
"""
ot = {}
for did in DIS:
    try:
        r = host.mcp("clinical-genomics","open_targets_graphql", query=q, variables={"id":did,"n":200})
        d = r["data"]["disease"]
        ot[did] = {"name": d["name"], "count": d["associatedTargets"]["count"], "rows": d["associatedTargets"]["rows"]}
        print(did, d["name"], "total_assoc=", d["associatedTargets"]["count"], "retrieved=", len(d["associatedTargets"]["rows"]))
    except Exception as e:
        print("ERR", did, e)

civic = {}
for dn in ["Glioblastoma","Glioma","Astrocytoma","Oligodendroglioma","Brain Cancer"]:
    try:
        r = host.mcp("clinical-genomics","civic_search_evidence", disease_name=dn, status="ACCEPTED")
        civic[dn] = r
        print("CIViC", dn, "total=", r.get("total_count"), "recs=", len(r.get("records",[])))
    except Exception as e:
        print("ERR civic", dn, repr(e)[:200])

json.dump({"ot":ot,"civic":civic,"dis":DIS}, open("handoff/step1_raw.json","w"))

raw = json.load(open("handoff/step1_raw.json"))
civ = collections.Counter(); civ_dis = collections.defaultdict(set); lvl = collections.defaultdict(list)
for dn, blk in raw["civic"].items():
    for r in blk.get("records", []):
        name = r.get("molecularProfile",{}).get("name","")
        for part in re.split(r"\s+(?:AND|OR)\s+", name):
            tok = part.strip().lstrip("(").split()
            if not tok: continue
            g = tok[0].strip("()")
            if re.fullmatch(r"[A-Z][A-Z0-9orf\-]{1,9}", g):
                civ[g] += 1; civ_dis[g].add(dn); lvl[g].append(r.get("evidenceLevel"))
json.dump({"counts":dict(civ), "dis":{k:sorted(v) for k,v in civ_dis.items()}, "levels":{k:v for k,v in lvl.items()}},
          open("handoff/civic_genes.json","w"))

raw = json.load(open("handoff/step1_raw.json"))
civ = json.load(open("handoff/civic_genes.json"))
rows=[]
for did, blk in raw["ot"].items():
    for r in blk["rows"]:
        t=r["target"]
        dts={d["id"]:d["score"] for d in r["datatypeScores"]}
        rows.append(dict(disease_id=did, disease=blk["name"], gene_symbol=t["approvedSymbol"],
                         ensembl_id=t["id"], protein_name=t["approvedName"], biotype=t["biotype"],
                         score=r["score"], **{f"dt_{k}":v for k,v in dts.items()}))
ot = pd.DataFrame(rows)
piv = ot.pivot_table(index="gene_symbol", columns="disease", values="score", aggfunc="max")
agg = (ot.groupby(["gene_symbol","ensembl_id","protein_name","biotype"])
         .agg(ot_max_score=("score","max"), ot_mean_score=("score","mean"), n_diseases=("disease","nunique"))
         .reset_index())
dtcols=[c for c in ot.columns if c.startswith("dt_")]
dtmax = ot.groupby("gene_symbol")[dtcols].max()
agg = agg.merge(dtmax, on="gene_symbol", how="left")
agg["civic_evidence_items"]=agg.gene_symbol.map(civ["counts"]).fillna(0).astype(int)
agg["sources"]=np.where(agg.civic_evidence_items>0,"OpenTargets;CIViC","OpenTargets")
agg=agg.sort_values(["ot_max_score","n_diseases"],ascending=False).reset_index(drop=True)

seed = agg[agg.biotype=="protein_coding"].copy()
civic_only = set(civ["counts"])
top = seed.head(180).copy()
add = seed[(~seed.gene_symbol.isin(top.gene_symbol)) & (seed.gene_symbol.isin(civic_only))]
seed_sel = pd.concat([top, add]).reset_index(drop=True)
seed_sel["in_seed_reason"] = np.where(seed_sel.index < len(top), "top180_by_OT_score", "CIViC_clinical_evidence")

extra_rows = pd.DataFrame([dict(gene_symbol=g, ensembl_id="", protein_name="", biotype="protein_coding",
    ot_max_score=np.nan, ot_mean_score=np.nan, n_diseases=0,
    civic_evidence_items=civ["counts"][g], sources="CIViC", in_seed_reason="CIViC_clinical_evidence")
    for g in ["SDHB","MYB","MYBL1"]])
seed_sel = pd.concat([seed_sel, extra_rows], ignore_index=True)
seed_sel.to_csv("glioma_seed_genes.csv", index=False)
symbols = seed_sel.gene_symbol.tolist()
json.dump(symbols, open("handoff/seed_symbols.json","w"))

syms = json.load(open("handoff/seed_symbols.json"))
net = host.mcp("protein-annotation","get_string_network", symbols=syms, species=9606, required_score=700)
print(json.dumps(net["summary"]))
print("unmapped:", net.get("unmapped"))
json.dump(net, open("handoff/string_net.json","w"))
print("node keys:", list(net["nodes"][0].keys()), "| edge keys:", list(net["edges"][0].keys()))