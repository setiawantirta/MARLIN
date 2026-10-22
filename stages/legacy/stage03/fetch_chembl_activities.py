"""Stream ChEMBL activity + assay records for the 6-target glioma panel to disk (jsonl.gz)."""
import requests, json, gzip, os, sys, time

BASE = "https://www.ebi.ac.uk/chembl/api/data"
PANEL = {"EGFR":"CHEMBL203","PDGFRA":"CHEMBL2007","PIK3CA":"CHEMBL4005",
         "KDR":"CHEMBL279","CDK6":"CHEMBL2508","IDH1":"CHEMBL2007625"}
STYPES = ["IC50","Ki","Kd","EC50","Potency"]
OUT = "raw"
os.makedirs(OUT, exist_ok=True)

S = requests.Session()
S.headers.update({"Accept": "application/json", "User-Agent": "python-requests"})

ACT_FIELDS = ["activity_id","assay_chembl_id","assay_description","assay_type",
              "molecule_chembl_id","canonical_smiles","target_chembl_id","target_organism",
              "target_pref_name","standard_type","standard_relation","standard_value",
              "standard_units","pchembl_value","data_validity_comment","activity_comment",
              "document_chembl_id","document_year","bao_label","potential_duplicate",
              "src_id","standard_flag"]


def get(url, params=None, tries=6):
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (k + 1)); continue
            r.raise_for_status()
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed {url}")


def pull_activities(gene, tid):
    path = f"{OUT}/act_{gene}.jsonl.gz"
    if os.path.exists(path):
        print(f"{gene}: cached", flush=True); return path
    params = {"target_chembl_id": tid,
              "standard_type__in": ",".join(STYPES),
              "standard_relation": "=",
              "pchembl_value__isnull": "false",
              "limit": 1000,
              "only": ",".join(ACT_FIELDS)}
    url = f"{BASE}/activity.json"
    n = 0
    with gzip.open(path + ".tmp", "wt") as fh:
        while True:
            d = get(url, params)
            params = None
            for a in d["activities"]:
                fh.write(json.dumps(a) + "\n"); n += 1
            pm = d["page_meta"]
            if pm.get("next"):
                url = "https://www.ebi.ac.uk" + pm["next"]
            else:
                break
            print(f"  {gene}: {n}/{pm.get('total_count')}", flush=True)
    os.replace(path + ".tmp", path)
    print(f"{gene}: {n} raw activity records -> {path}", flush=True)
    return path


def pull_assays(assay_ids):
    path = f"{OUT}/assays.jsonl.gz"
    have = set()
    if os.path.exists(path):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                have.add(json.loads(line)["assay_chembl_id"])
    todo = sorted(set(assay_ids) - have)
    print(f"assays: {len(have)} cached, {len(todo)} to fetch", flush=True)
    if not todo:
        return path
    with gzip.open(path, "at") as fh:
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            d = get(f"{BASE}/assay.json",
                    {"assay_chembl_id__in": ",".join(chunk), "limit": 1000,
                     "only": "assay_chembl_id,confidence_score,assay_type,"
                             "assay_organism,relationship_type,target_chembl_id,"
                             "assay_category,bao_format,description"})
            for a in d["assays"]:
                fh.write(json.dumps(a) + "\n")
            if i % 500 == 0:
                print(f"  assays {i}/{len(todo)}", flush=True)
    return path


if __name__ == "__main__":
    aids = set()
    for gene, tid in PANEL.items():
        p = pull_activities(gene, tid)
        with gzip.open(p, "rt") as fh:
            for line in fh:
                aids.add(json.loads(line)["assay_chembl_id"])
    print(f"distinct assays across panel: {len(aids)}", flush=True)
    pull_assays(aids)
    print("DONE", flush=True)
