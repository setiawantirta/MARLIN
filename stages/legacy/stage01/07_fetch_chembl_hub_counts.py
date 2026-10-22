import requests, json

tmap = json.load(open("handoff/chembl_targets.json"))
tid = {g: v[0]["id"] for g,v in tmap.items() if isinstance(v,list) and v}
acc = {g: (v[0]["acc"] or [None])[0] for g,v in tmap.items() if isinstance(v,list) and v}
pref = {g: v[0]["pref"] for g,v in tmap.items() if isinstance(v,list) and v}
B="https://www.ebi.ac.uk/chembl/api/data"; S=requests.Session()

def count_compounds(t):
    mols=set()
    r=S.get(f"{B}/activity.json", params=dict(target_chembl_id=t, standard_type__in="IC50,Ki,Kd",
            pchembl_value__isnull="false", limit=1000, only="molecule_chembl_id"), timeout=180)
    r.raise_for_status(); d=r.json(); total=d["page_meta"]["total_count"]
    while True:
        mols.update(a["molecule_chembl_id"] for a in d["activities"] if a.get("molecule_chembl_id"))
        nxt=d["page_meta"].get("next")
        if not nxt: break
        d=S.get("https://www.ebi.ac.uk"+nxt, timeout=180).json()
    return total, mols

def fda_count(t):
    r=S.get(f"{B}/mechanism.json", params=dict(target_chembl_id=t, limit=1000), timeout=60); r.raise_for_status()
    ms=sorted({m["molecule_chembl_id"] for m in r.json()["mechanisms"] if m.get("molecule_chembl_id")})
    appr=[]
    for i in range(0,len(ms),50):
        rr=S.get(f"{B}/molecule.json", params={"molecule_chembl_id__in":",".join(ms[i:i+50]),
                 "max_phase":4,"limit":1000,"only":"molecule_chembl_id,pref_name"}, timeout=60)
        rr.raise_for_status(); appr += [m.get("pref_name") for m in rr.json()["molecules"]]
    return len(appr), appr

res={}
for g,t in tid.items():
    try:
        tot, mols = count_compounds(t); nf, appr = fda_count(t)
        res[g]=dict(chembl_target_id=t, uniprot=acc[g], pref_name=pref[g], n_activities=tot,
                    n_compounds=len(mols), n_fda=nf, fda_drugs=appr[:15])
        print(f"{g:8s} {t:16s} act={tot:7d} cpd={len(mols):6d} fda={nf}")
    except Exception as e:
        res[g]=dict(err=repr(e)[:200]); print(g,"ERR",repr(e)[:120])
json.dump(res, open("handoff/chembl_counts.json","w"))