"""Recover the negative class from right-censored ChEMBL measurements.

ChEMBL assigns a pchembl_value only to exact ('=') measurements, so censored records - the
'>10 uM'-style results that carry most of the genuine inactivity - were absent from the first
build. Here they are converted by hand and turned into label calls only where the censoring
bound is decisive.

Conversion: standard_value + standard_units -> molar concentration -> a BOUND on pActivity,
    bound = -log10(conc_M)
Decision rules (thresholds identical to the exact-value labels: active >= 6.0, inactive <= 5.0):
    '>'  / '>=' : pAct < bound.  Decisive INACTIVE only if bound <= 5.0 (i.e. >= 10 uM).
                  Example: 'IC50 > 10000 nM' -> bound 5.0 -> pAct < 5 -> inactive.
                           'IC50 > 100 nM'   -> bound 7.0 -> pAct anywhere below 7 -> UNDECIDABLE.
    '<'  / '<=' : pAct > bound.  Decisive ACTIVE only if bound >= 6.0 (i.e. <= 1 uM).
Only molar-convertible units are used; mass-per-volume units (ug.mL-1 etc.) would need a
per-compound MW conversion and are counted and discarded instead.
"""
import gzip, json, os
import numpy as np
import pandas as pd

PANEL = {"EGFR": "CHEMBL203", "PDGFRA": "CHEMBL2007", "PIK3CA": "CHEMBL4005",
         "KDR": "CHEMBL279", "CDK6": "CHEMBL2508", "IDH1": "CHEMBL2007625"}
ACTIVE_T, INACTIVE_T = 6.0, 5.0
UNIT_M = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "\u00b5M": 1e-6, "nM": 1e-9,
          "pM": 1e-12, "fM": 1e-15}

assays = {}
with gzip.open("raw/assays.jsonl.gz", "rt") as fh:
    for line in fh:
        a = json.loads(line)
        assays[a["assay_chembl_id"]] = a

audit, calls = [], []
for gene in PANEL:
    recs = [json.loads(l) for l in gzip.open(f"raw/cens_{gene}.jsonl.gz", "rt")]
    df = pd.DataFrame(recs)
    st = {"gene": gene, "n_censored_raw": len(df)}
    df = df[df.target_organism == "Homo sapiens"]
    df = df[df.data_validity_comment.isna()]
    st["n_after_organism_validity"] = len(df)
    df["confidence_score"] = df.assay_chembl_id.map(
        lambda a: assays.get(a, {}).get("confidence_score"))
    df["relationship_type"] = df.assay_chembl_id.map(
        lambda a: assays.get(a, {}).get("relationship_type"))
    df = df[df.confidence_score.fillna(-1) >= 8]
    st["n_after_confidence8"] = len(df)
    df = df[df.canonical_smiles.notna() & df.standard_value.notna() & df.standard_units.notna()]
    st["n_with_value_and_units"] = len(df)

    st["unit_mix"] = ";".join(f"{k}:{v}" for k, v in df.standard_units.value_counts().items())
    df["factor"] = df.standard_units.map(UNIT_M)
    st["n_non_molar_units_dropped"] = int(df.factor.isna().sum())
    df = df[df.factor.notna()]
    df["standard_value"] = pd.to_numeric(df.standard_value, errors="coerce")
    df = df[df.standard_value > 0]
    st["n_molar_convertible"] = len(df)

    df["conc_M"] = df.standard_value * df.factor
    df["bound"] = -np.log10(df.conc_M)
    right = df.standard_relation.isin([">", ">="])
    left = df.standard_relation.isin(["<", "<="])
    df["call"] = None
    df.loc[right & (df.bound <= INACTIVE_T), "call"] = "inactive"
    df.loc[left & (df.bound >= ACTIVE_T), "call"] = "active"
    st["n_right_censored"] = int(right.sum())
    st["n_left_censored"] = int(left.sum())
    st["n_decisive_inactive_records"] = int((df.call == "inactive").sum())
    st["n_decisive_active_records"] = int((df.call == "active").sum())
    st["n_undecidable_records"] = int(df.call.isna().sum())
    st["n_undecidable_right"] = int((right & df.call.isna()).sum())
    st["n_undecidable_left"] = int((left & df.call.isna()).sum())
    audit.append(st)
    print(st, flush=True)

    d = df[df.call.notna()].copy()
    d["gene"] = gene
    calls.append(d[["gene", "molecule_chembl_id", "canonical_smiles", "call", "bound",
                    "standard_relation", "standard_type", "standard_value", "standard_units",
                    "conc_M", "assay_chembl_id", "confidence_score", "relationship_type",
                    "document_chembl_id"]])
    u = df[df.call.isna()].copy(); u["gene"] = gene
    u[["gene", "molecule_chembl_id", "standard_relation", "standard_value",
       "standard_units", "bound"]].to_csv(f"work/undecidable_{gene}.csv.gz", index=False)

cal = pd.concat(calls, ignore_index=True)
cal.to_csv("work/censored_calls.csv.gz", index=False)
pd.DataFrame(audit).to_csv("work/audit_censored.csv", index=False)
print(f"\ndecisive censored records: {len(cal)}", flush=True)
print(cal.groupby(["gene", "call"]).size().unstack(fill_value=0).to_string(), flush=True)
print(f"\ndistinct (compound,target) pairs with a decisive censored call: "
      f"{len(cal.drop_duplicates(['gene','molecule_chembl_id']))}", flush=True)
print("RECOVERY DONE", flush=True)
