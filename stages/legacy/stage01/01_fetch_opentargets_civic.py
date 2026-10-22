import json, os, collections
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
print("keys civic sample:", list(civic.get("Glioblastoma",{}).get("records",[{}])[0].keys())[:30])