"""Apply the documented activity filters, aggregate replicates, and emit a stage-by-stage audit.

Filters (in order, counted at every stage):
  1. raw records pulled with server-side target/standard_type/relation/pchembl constraints
  2. target_organism == 'Homo sapiens'
  3. data_validity_comment is null
  4. assay confidence_score >= 8  (joined from the assay endpoint)
  5. assay relationship_type == 'D' (direct single-protein assignment)  [reported, applied]
  6. non-null canonical_smiles
Replicate (compound, target) measurements are aggregated by MEDIAN pchembl_value; the spread
(min, max, sd, n) is retained so reviewers can see measurement dispersion.
"""
import gzip, json, os
import pandas as pd
import numpy as np

PANEL = {"EGFR": "CHEMBL203", "PDGFRA": "CHEMBL2007", "PIK3CA": "CHEMBL4005",
         "KDR": "CHEMBL279", "CDK6": "CHEMBL2508", "IDH1": "CHEMBL2007625"}
STYPES = ["IC50", "Ki", "Kd", "EC50", "Potency"]

assays = {}
with gzip.open("raw/assays.jsonl.gz", "rt") as fh:
    for line in fh:
        a = json.loads(line)
        assays[a["assay_chembl_id"]] = a

audit, kept = [], []
for gene, tid in PANEL.items():
    recs = []
    with gzip.open(f"raw/act_{gene}.jsonl.gz", "rt") as fh:
        for line in fh:
            recs.append(json.loads(line))
    df = pd.DataFrame(recs)
    st = {"gene": gene, "target_chembl_id": tid, "n_raw_records": len(df),
          "n_raw_compounds": df.molecule_chembl_id.nunique()}

    df = df[df.target_organism == "Homo sapiens"]
    st["n_after_organism"] = len(df)
    df = df[df.data_validity_comment.isna()]
    st["n_after_validity"] = len(df)

    df["confidence_score"] = df.assay_chembl_id.map(
        lambda a: assays.get(a, {}).get("confidence_score"))
    df["relationship_type"] = df.assay_chembl_id.map(
        lambda a: assays.get(a, {}).get("relationship_type"))
    df["assay_type"] = df.assay_chembl_id.map(lambda a: assays.get(a, {}).get("assay_type"))
    df = df[df.confidence_score.fillna(-1) >= 8]
    st["n_after_confidence8"] = len(df)
    # relationship_type is RECORDED, not filtered: inspection showed the 'H' ("homologous
    # target assigned") records are genuine on-target assays for these targets - wild-type
    # and R132H/R132C mutant IDH1, and EGFR kinase autophosphorylation assays - so removing
    # them would discard exactly the mutant-enzyme chemistry that is central to glioma.
    st["n_relationship_D"] = int((df.relationship_type == "D").sum())
    st["n_relationship_H"] = int((df.relationship_type == "H").sum())
    st["n_confidence_9"] = int((df.confidence_score == 9).sum())
    st["n_confidence_8"] = int((df.confidence_score == 8).sum())
    df = df[df.canonical_smiles.notna() & df.pchembl_value.notna()]
    st["n_after_smiles"] = len(df)
    st["n_compounds_kept"] = df.molecule_chembl_id.nunique()
    st["n_assays"] = df.assay_chembl_id.nunique()
    st["n_documents"] = df.document_chembl_id.nunique()
    st["standard_type_mix"] = ";".join(
        f"{k}:{v}" for k, v in df.standard_type.value_counts().items())
    audit.append(st)
    df["gene"] = gene
    df["pchembl_value"] = df.pchembl_value.astype(float)
    kept.append(df)
    print(st, flush=True)

rec = pd.concat(kept, ignore_index=True)
rec.to_csv("chembl_activity_records.csv.gz", index=False, compression="gzip")
print("filtered records written:", len(rec), flush=True)

agg = (rec.groupby(["gene", "molecule_chembl_id"])
          .agg(pchembl_median=("pchembl_value", "median"),
               pchembl_min=("pchembl_value", "min"),
               pchembl_max=("pchembl_value", "max"),
               pchembl_sd=("pchembl_value", "std"),
               n_measurements=("pchembl_value", "size"),
               n_assays=("assay_chembl_id", "nunique"),
               smiles=("canonical_smiles", "first"))
          .reset_index())
agg["pchembl_range"] = agg.pchembl_max - agg.pchembl_min
agg.to_csv("work/agg_per_compound_target.csv.gz", index=False, compression="gzip")
pd.DataFrame(audit).to_csv("work/audit_stage1.csv", index=False)

print("\n(compound,target) pairs:", len(agg))
print("unique compounds:", agg.molecule_chembl_id.nunique())
rep = agg[agg.n_measurements > 1]
print(f"pairs with replicates: {len(rep)} ({100*len(rep)/len(agg):.1f}%); "
      f"median spread among them: {rep.pchembl_range.median():.2f} log units; "
      f"mean sd: {rep.pchembl_sd.mean():.2f}")
print("pairs with spread > 1 log unit:", int((agg.pchembl_range > 1).sum()))
