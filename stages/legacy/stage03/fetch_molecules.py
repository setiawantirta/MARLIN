"""Fetch ChEMBL molecule records (natural_product flag, max_phase, properties) for every
compound in the filtered activity pool."""
import requests, json, gzip, os, time
import pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = "raw/molecules.jsonl.gz"
FIELDS = ("molecule_chembl_id,pref_name,max_phase,molecule_type,natural_product,"
          "structure_type,first_approval,chirality,prodrug,oral,parenteral,topical,"
          "molecule_properties,molecule_structures")

S = requests.Session()
S.headers.update({"Accept": "application/json"})


def get(u, p=None, tries=6):
    for k in range(tries):
        try:
            r = S.get(u, params=p, timeout=180)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(4 * (k + 1))
    raise RuntimeError(u)


agg = pd.read_csv("work/agg_per_compound_target.csv.gz")
ids = sorted(agg.molecule_chembl_id.unique())
have = set()
if os.path.exists(OUT):
    with gzip.open(OUT, "rt") as fh:
        for line in fh:
            have.add(json.loads(line)["molecule_chembl_id"])
todo = [i for i in ids if i not in have]
print(f"{len(ids)} compounds, {len(have)} cached, {len(todo)} to fetch", flush=True)

with gzip.open(OUT, "at") as fh:
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        d = get(f"{BASE}/molecule.json",
                {"molecule_chembl_id__in": ",".join(chunk), "limit": 1000, "only": FIELDS})
        for m in d["molecules"]:
            fh.write(json.dumps(m) + "\n")
        if (i // 100) % 20 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
print("MOLECULES DONE", flush=True)
