"""Portable retrieval of the three network sources, against their public endpoints.

Why this module exists
----------------------
Two scripts in stage 01 as originally run reach their sources through a hosted
connector rather than over plain HTTP:

    stages/legacy/stage01/01_fetch_opentargets_civic.py   host.mcp("clinical-genomics", ...)
    stages/legacy/stage01/03_fetch_string_network.py      host.mcp("protein-annotation", ...)

That `host` object exists only inside the environment the study was run in, so those
two steps cannot execute on a reader's laptop, on Colab or on Kaggle. Every other
retrieval step in the module already speaks HTTP directly: the ChEMBL steps of
stages 01 and 03 call https://www.ebi.ac.uk/chembl/api/data and the ZINC step of
stage 04 calls https://files.docking.org.

The functions here replace the three connector calls with requests to the services'
own public APIs, writing byte-compatible files so the downstream steps
(02_build_seed_gene_table, 04_detect_modules, 05_build_centrality_table,
06_pathway_enrichment) run unmodified.

Endpoints, and what each supplies
---------------------------------
    Open Targets  https://api.platform.opentargets.org/api/v4/graphql
                  disease-to-target association scores for six MONDO glioma terms
    CIViC         https://civicdb.org/api/graphql
                  accepted clinical evidence items, for genes with curated glioma
                  evidence but a sub-threshold association score
    STRING        https://string-db.org/api/json/network
                  the protein-protein interaction edges of the seed set
    Reactome      https://reactome.org/AnalysisService
                  pathway over-representation per detected module

Reproducibility note
--------------------
These are live databases. Re-running retrieval today will not return the study's
exact seed set: Open Targets re-scores associations at every release and CIViC gains
curated evidence continuously. The stored tables are the record of what the paper
used; this module is how a reader re-derives the pipeline, and how you refresh the
panel if a referee asks for a current snapshot. Each function records the service
version or release it saw, so a refreshed run is self-documenting rather than silently
different.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
CIVIC = "https://civicdb.org/api/graphql"
STRING = "https://string-db.org/api"
REACTOME = "https://reactome.org/AnalysisService"

# The six MONDO terms the study treated as glioma, kept here so a refreshed run uses
# the same disease scope rather than a re-chosen one.
GLIOMA_DISEASES = {
    "MONDO_0018177": "glioblastoma",
    "MONDO_0021042": "glioma",
    "MONDO_0021637": "low grade glioma",
    "MONDO_0100342": "malignant glioma",
    "MONDO_0016695": "oligodendroglioma",
    "MONDO_0019781": "astrocytoma (excluding glioblastoma)",
}
CIVIC_DISEASES = ["Glioblastoma", "Glioma", "Astrocytoma", "Oligodendroglioma", "Brain Cancer"]

OT_QUERY = """
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


def _post(url, payload, timeout=120, retries=4, pause=2.0):
    """POST JSON with bounded retries. Public APIs rate-limit; a 429 or a 5xx is
    transient and worth retrying, a 4xx is a bad request and is not."""
    body = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json", "Accept": "application/json"}
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=hdr)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(pause * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"{url} failed after {retries} attempts: {last}")


def _get(url, timeout=120, retries=4, pause=2.0):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            return urllib.request.urlopen(req, timeout=timeout).read().decode()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(pause * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"{url} failed after {retries} attempts: {last}")


# ------------------------------------------------------------------ Open Targets
def fetch_open_targets(diseases=None, n=200, verbose=True):
    """Association scores for each disease term.

    Returns {disease_id: {"name", "count", "rows"}} - the same structure the
    connector returned, so handoff/step1_raw.json keeps its shape. `count` is the
    service's total association count and `rows` the top `n` of them, which is why a
    retrieved row count below `count` is expected rather than a truncation bug.
    """
    diseases = diseases or GLIOMA_DISEASES
    out = {}
    for did in diseases:
        r = _post(OPEN_TARGETS, {"query": OT_QUERY, "variables": {"id": did, "n": n}})
        if "errors" in r:
            raise RuntimeError(f"Open Targets rejected {did}: {r['errors']}")
        d = r["data"]["disease"]
        if d is None:
            raise RuntimeError(f"Open Targets returned no disease for {did}; the term may "
                               "have been retired or merged in the current release")
        out[did] = {"name": d["name"],
                    "count": d["associatedTargets"]["count"],
                    "rows": d["associatedTargets"]["rows"]}
        if verbose:
            print(f"  {did}  {d['name']:<38s} associations={d['associatedTargets']['count']:>6}  "
                  f"retrieved={len(d['associatedTargets']['rows'])}", flush=True)
    return out


# -------------------------------------------------------------------------- CIViC
CIVIC_QUERY = """
query($disease: String!, $after: String){
  evidenceItems(diseaseName: $disease, status: ACCEPTED, first: 100, after: $after){
    totalCount
    pageInfo{ hasNextPage endCursor }
    nodes{ id evidenceLevel evidenceType significance status
           molecularProfile{ name } disease{ name } }
  }
}
"""


def fetch_civic(disease_names=None, verbose=True):
    """Accepted evidence items per disease name, following the cursor to the end.

    Returns {disease_name: {"total_count", "records"}}. `records` carries
    molecularProfile.name and evidenceLevel, which is what the gene-symbol parser in
    02_build_seed_gene_table reads; a short page would silently shrink the seed set,
    so the cursor is followed rather than taking only the first page.
    """
    disease_names = disease_names or CIVIC_DISEASES
    out = {}
    for dn in disease_names:
        recs, after, total = [], None, None
        while True:
            r = _post(CIVIC, {"query": CIVIC_QUERY, "variables": {"disease": dn, "after": after}})
            if "errors" in r:
                raise RuntimeError(f"CIViC rejected {dn}: {r['errors']}")
            blk = r["data"]["evidenceItems"]
            total = blk["totalCount"]
            recs.extend(blk["nodes"])
            if not blk["pageInfo"]["hasNextPage"]:
                break
            after = blk["pageInfo"]["endCursor"]
        out[dn] = {"total_count": total, "records": recs}
        if verbose:
            print(f"  CIViC  {dn:<18s} total={total:>5}  retrieved={len(recs)}", flush=True)
        if total != len(recs):
            raise RuntimeError(f"CIViC pagination incomplete for {dn}: {len(recs)} of {total}")
    return out


# ------------------------------------------------------------------------- STRING
def fetch_string(symbols, species=9606, required_score=700, verbose=True):
    """PPI edges among `symbols`, as {"summary", "nodes", "edges", "unmapped"}.

    Node and edge keys match what 04_detect_modules and 05_build_centrality_table
    read: nodes carry "name", edges carry "a", "b" and a 0-1 "score". STRING returns
    only the pairs it has evidence for, so a symbol absent from every edge is
    reported in "unmapped" - those are the isolated seeds the paper discusses, and
    losing them silently would change the giant-component count.
    """
    ids = "%0d".join(urllib.parse.quote(s) for s in symbols)
    common = f"species={species}&caller_identity=marine_antiglioma_repro"

    # Resolve the requested symbols first. STRING answers the network call with ITS
    # preferred symbol, which is not always the one asked for: a requested alias comes
    # back under a different name, which would add a node absent from the seed table
    # and leave the alias looking isolated. Mapping stringId back to the symbol that
    # was queried keeps node identity aligned with glioma_seed_genes.csv, which is what
    # the centrality table is later joined on.
    idmap = json.loads(_get(f"{STRING}/json/get_string_ids?identifiers={ids}&{common}"))
    to_query = {}
    for rec in idmap:
        q = rec.get("queryItem") or rec.get("queryIndex")
        if isinstance(q, int):
            q = symbols[q]
        to_query[rec["stringId"]] = q
    renamed = {rec["preferredName"]: to_query[rec["stringId"]] for rec in idmap
               if rec["preferredName"] != to_query.get(rec["stringId"])}

    url = (f"{STRING}/json/network?identifiers={ids}&{common}"
           f"&required_score={required_score}")
    rows = json.loads(_get(url))
    edges, seen = [], set()
    for e in rows:
        a = to_query.get(e["stringId_A"], e["preferredName_A"])
        b = to_query.get(e["stringId_B"], e["preferredName_B"])
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"a": a, "b": b, "score": float(e["score"])})
    # Every requested symbol becomes a node, including those STRING returns no edge
    # for. Returning only the connected ones would silently drop the isolated seeds,
    # and the paper's network is 186 nodes with a 176-node giant component - that
    # distinction disappears if the isolates never enter the graph that
    # 04_detect_modules builds.
    present = {s for e in edges for s in (e["a"], e["b"])}
    nodes = [{"name": s} for s in symbols]
    isolated = sorted(set(symbols) - present)
    summary = {"requested": len(symbols), "nodes": len(nodes), "edges": len(edges),
               "connected": len(present), "isolated": len(isolated),
               "species": species, "required_score": required_score,
               "source": "STRING REST /api/json/network"}
    if verbose:
        print(f"  STRING  requested={len(symbols)}  nodes={len(nodes)}  edges={len(edges)}  "
              f"connected={len(present)}  isolated={len(isolated)}", flush=True)
    # "unmapped" is the key the connector used and the downstream step prints.
    return {"summary": summary, "nodes": nodes, "edges": edges,
            "unmapped": isolated, "isolated": isolated}


# ------------------------------------------------- Reactome and GO (OPTIONAL)
# These feed 06_pathway_enrichment.py, which writes the descriptive pathway table.
# They are NOT inputs to target selection: 08_select_target_panel.py reads only
# chembl_counts.json and centrality.csv. A failure here costs the pathway table and
# nothing else, which is why the notebook treats this section as skippable.

def fetch_reactome(module_genes, verbose=True):
    """Per-module pathway over-representation, shaped for 06_pathway_enrichment.py.

    Returns {module: {"reactome_version", "batch_summary": {"identifiers_found_count",
    "pathways": [...]}}} - the structure that script unwraps, not a convenience shape.

    NETWORK NOTE: reactome.org sits behind Cloudflare and answers some automated
    clients with error 1010 regardless of the request being well formed; it does so
    from the environment this module was prepared in. Do not work around it by
    disguising the client. Either run this section from a network Reactome accepts
    (Colab often is one), or use the Reactome web analysis page for each module and
    save the JSON yourself. The error is raised with that explanation attached.
    """
    try:
        ver = _get(f"{REACTOME}/database/version").strip().strip('"')
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                "Reactome returned HTTP 403 (Cloudflare error 1010): this network's client "
                "is refused. The pathway enrichment is optional - target selection does not "
                "use it - so skip this section, or run it from a different network, or paste "
                "each module's gene list into https://reactome.org/PathwayBrowser/#TOOL=AT "
                "and save the JSON as reactome_raw.json.") from e
        raise
    out = {}
    for mod, genes in module_genes.items():
        req = urllib.request.Request(
            f"{REACTOME}/identifiers/projection?interactors=false&pageSize=200&page=1"
            "&sortBy=ENTITIES_PVALUE&order=ASC",
            data="\n".join(genes).encode(),
            headers={"Content-Type": "text/plain", "Accept": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        pathways = [{"stId": p["stId"], "name": p["name"],
                     "inDisease": p.get("inDisease", False),
                     "entities_found": p["entities"]["found"],
                     "entities_total": p["entities"]["total"],
                     "entities_pvalue": p["entities"]["pValue"],
                     "entities_fdr": p["entities"]["fdr"]}
                    for p in r.get("pathways", [])]
        found = len(genes) - len(r.get("identifiersNotFound", []) or [])
        out[str(mod)] = {"reactome_version": ver,
                         "batch_summary": {"identifiers_found_count": found,
                                           "pathways": pathways}}
        if verbose:
            print(f"  Reactome  module {mod}: {len(genes)} genes, {found} mapped "
                  f"-> {len(pathways)} pathways", flush=True)
    return out


UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
QUICKGO = "https://www.ebi.ac.uk/QuickGO/services"


def fetch_go(symbols, verbose=True):
    """GO biological-process and molecular-function annotations per gene symbol.

    Returns {symbol: [{"aspect", "id", "label"}, ...]}, which is what the GO block of
    06_pathway_enrichment.py reads. QuickGO is keyed on gene-product accessions rather
    than symbols, so each symbol is resolved through UniProt first; a symbol with no
    reviewed human entry is returned with an empty list rather than dropped, so the
    caller can see the coverage.
    """
    acc = {}
    for s in symbols:
        q = (f"{UNIPROT}?query=gene:{urllib.parse.quote(s)}+AND+organism_id:9606"
             "+AND+reviewed:true&fields=accession&format=json&size=1")
        try:
            r = json.loads(_get(q))
            if r.get("results"):
                acc[s] = r["results"][0]["primaryAccession"]
        except Exception:
            pass
    if verbose:
        print(f"  UniProt  resolved {len(acc)} of {len(symbols)} symbols", flush=True)

    out = {s: [] for s in symbols}
    labels = {}
    for s, a in acc.items():
        url = (f"{QUICKGO}/annotation/search?geneProductId={a}&taxonId=9606"
               "&aspect=biological_process,molecular_function&limit=200")
        try:
            r = json.loads(_get(url))
        except Exception:
            continue
        seen = set()
        for rec in r.get("results", []):
            gid = rec["goId"]
            if gid in seen:
                continue
            seen.add(gid)
            out[s].append({"aspect": rec["goAspect"], "id": gid, "label": rec.get("goName")})
            labels[gid] = labels.get(gid) or rec.get("goName")

    # QuickGO's annotation search leaves goName null; fetch the term labels in batches.
    missing = sorted({t["id"] for v in out.values() for t in v if not t["label"]})
    for i in range(0, len(missing), 100):
        chunk = ",".join(missing[i:i + 100])
        try:
            r = json.loads(_get(f"{QUICKGO}/ontology/go/terms/{chunk}"))
            for term in r.get("results", []):
                labels[term["id"]] = term.get("name")
        except Exception:
            continue
    for v in out.values():
        for t in v:
            if not t["label"]:
                t["label"] = labels.get(t["id"])
    if verbose:
        n = sum(len(v) for v in out.values())
        print(f"  QuickGO  {n} annotations over {sum(1 for v in out.values() if v)} genes",
              flush=True)
    return out


CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_chembl_targets(symbols, verbose=True):
    """Map gene symbols to human SINGLE PROTEIN ChEMBL targets.

    Returns {symbol: [{"id", "acc", "pref"}, ...]} - the shape
    07_fetch_chembl_hub_counts.py reads from handoff/chembl_targets.json. That file is
    read by the step but written by no script in the module; it was produced outside
    the shipped pipeline in the original run, so step 07 cannot execute without this.

    Only reviewed human single-protein targets are kept. A protein complex or a protein
    family would make the per-hub compound count mean something different from what the
    panel-selection threshold assumes, so those are excluded rather than merged.
    """
    out = {}
    for s in symbols:
        url = (f"{CHEMBL}/target.json?target_synonym__iexact={urllib.parse.quote(s)}"
               "&target_type=SINGLE%20PROTEIN&organism=Homo%20sapiens&limit=20")
        try:
            r = json.loads(_get(url))
        except Exception:
            out[s] = []
            continue
        recs = []
        for t in r.get("targets", []):
            accs = [c.get("accession") for c in t.get("target_components", [])
                    if c.get("accession")]
            recs.append({"id": t["target_chembl_id"], "acc": accs,
                         "pref": t.get("pref_name")})
        out[s] = recs
    if verbose:
        hit = sum(1 for v in out.values() if v)
        print(f"  ChEMBL targets  {hit} of {len(symbols)} symbols mapped to a human "
              f"single-protein target", flush=True)
    return out
