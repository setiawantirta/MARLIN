"""Pull the CENSORED activity records ('>', '>=', '<', '<=') for the glioma panel.

The original activity pull constrained standard_relation='=' AND pchembl_value non-null, so the
censored records - which ChEMBL never assigns a pchembl_value to - were never downloaded. They are
recovered here so the negative class can be reconstructed from real experimental inactivity.
"""
import requests, json, gzip, os, time

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PANEL = {"EGFR": "CHEMBL203", "PDGFRA": "CHEMBL2007", "PIK3CA": "CHEMBL4005",
         "KDR": "CHEMBL279", "CDK6": "CHEMBL2508", "IDH1": "CHEMBL2007625"}
STYPES = ["IC50", "Ki", "Kd", "EC50", "Potency"]
RELATIONS = [">", ">=", "<", "<="]
FIELDS = ["activity_id", "assay_chembl_id", "assay_description", "assay_type",
          "molecule_chembl_id", "canonical_smiles", "target_chembl_id", "target_organism",
          "standard_type", "standard_relation", "standard_value", "standard_units",
          "pchembl_value", "data_validity_comment", "activity_comment",
          "document_chembl_id", "document_year", "bao_label", "src_id", "standard_flag"]
S = requests.Session()
S.headers.update({"Accept": "application/json"})


def get(url, params=None, tries=6):
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(4 * (k + 1))
    raise RuntimeError(url)


if __name__ == "__main__":
    os.makedirs("raw", exist_ok=True)
    aids = set()
    for gene, tid in PANEL.items():
        path = f"raw/cens_{gene}.jsonl.gz"
        if os.path.exists(path):
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    aids.add(json.loads(line)["assay_chembl_id"])
            print(f"{gene}: cached", flush=True)
            continue
        n = 0
        with gzip.open(path + ".tmp", "wt") as fh:
            for rel in RELATIONS:
                url, params = f"{BASE}/activity.json", {
                    "target_chembl_id": tid, "standard_type__in": ",".join(STYPES),
                    "standard_relation": rel, "limit": 1000, "only": ",".join(FIELDS)}
                while True:
                    d = get(url, params); params = None
                    for a in d["activities"]:
                        fh.write(json.dumps(a) + "\n"); n += 1
                        aids.add(a["assay_chembl_id"])
                    nxt = d["page_meta"].get("next")
                    if not nxt:
                        break
                    url = "https://www.ebi.ac.uk" + nxt
        os.replace(path + ".tmp", path)
        print(f"{gene}: {n} censored records", flush=True)

    have = set()
    if os.path.exists("raw/assays.jsonl.gz"):
        with gzip.open("raw/assays.jsonl.gz", "rt") as fh:
            for line in fh:
                have.add(json.loads(line)["assay_chembl_id"])
    todo = sorted(aids - have)
    print(f"new assays to fetch: {len(todo)}", flush=True)
    if todo:
        with gzip.open("raw/assays.jsonl.gz", "at") as fh:
            for i in range(0, len(todo), 50):
                d = get(f"{BASE}/assay.json",
                        {"assay_chembl_id__in": ",".join(todo[i:i + 50]), "limit": 1000,
                         "only": "assay_chembl_id,confidence_score,assay_type,assay_organism,"
                                 "relationship_type,target_chembl_id,description"})
                for a in d["assays"]:
                    fh.write(json.dumps(a) + "\n")
    print("CENSORED FETCH DONE", flush=True)
