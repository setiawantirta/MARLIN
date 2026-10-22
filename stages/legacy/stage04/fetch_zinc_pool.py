"""Sample a property-diverse ZINC20 candidate pool for decoy matching.

ZINC20 2D tranches are binned by MW (1st letter) x logP (2nd letter); the 3rd letter is
reactivity (C = clean, PAINS ok) and the 4th purchasability (A/B in stock, C in stock via
agent). We take the 'clean' reactivity level and the most inclusive in-stock purchasability
available, then read RANDOM BYTE WINDOWS from each tranche file rather than the head of the
file, so the sample is not biased by ZINC's internal file ordering.
"""
import requests, re, os, random, sys, time, gzip, subprocess

BASE = "https://files.docking.org/2D"
# files.docking.org's Apache returns 403 to the requests-library default User-Agent but
# serves curl normally, so curl is used as the HTTP client here (its own default UA;
# no browser impersonation).


def curl(url, extra=()):
    p = subprocess.run(["curl", "-sS", "-m", "180", *extra, url],
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else ""


def curl_len(url):
    out = curl(url, ("-I",))
    m = re.search(r"(?i)content-length:\s*(\d+)", out)
    return int(m.group(1)) if m else 0
OUT = os.environ.get("ZOUT", "raw/zinc_pool.smi.gz")
WINDOW = 100_000          # bytes per window
WINDOWS_PER_TRANCHE = 6
REACT_PREF = ["C", "B", "A", "E"]      # clean -> bother -> anodyne -> mild
PURCH_PREF = ["D", "C", "B", "A"]      # most inclusive for-sale level first (D = make-on-demand)
random.seed(20240912)

S = requests.Session()
S.headers.update({"User-Agent": "python-requests"})
MW = "ABCDEFGHIJK"
LOGP = "ABCDEFGHIJK"


def listing(tr):
    for k in range(4):
        out = curl(f"{BASE}/{tr}/")
        if out and "Index of" in out:
            return out
        time.sleep(3 * (k + 1))
    return ""


def pick_file(tr, html):
    files = dict(re.findall(r'href="(' + tr + r'([A-Z][A-Z])\.smi)"', html))
    # sizes from the apache listing rows
    for react in REACT_PREF:
        for purch in PURCH_PREF:
            fn = f"{tr}{react}{purch}.smi"
            if fn in files:
                return fn
    return None


def sample_file(tr, fn, fh_out, seen):
    url = f"{BASE}/{tr}/{fn}"
    total = curl_len(url)
    if total == 0:
        return 0
    n = 0
    if total <= WINDOW * WINDOWS_PER_TRANCHE:
        spans = [(0, total - 1)]
    else:
        starts = sorted(random.sample(range(0, total - WINDOW), WINDOWS_PER_TRANCHE))
        spans = [(s, s + WINDOW - 1) for s in starts]
    for s, e in spans:
        txt = curl(url, ("-r", f"{s}-{e}"))
        if not txt:
            continue
        lines = txt.splitlines()
        if s > 0:
            lines = lines[1:]          # drop partial first line
        if e < total - 1:
            lines = lines[:-1]         # drop partial last line
        for ln in lines:
            p = ln.split()
            if len(p) != 2 or p[0] == "smiles":
                continue
            smi, zid = p
            if zid in seen:
                continue
            seen.add(zid)
            fh_out.write(f"{smi}\t{zid}\t{tr}\n")
            n += 1
    return n


if __name__ == "__main__":
    seen = set()
    total = 0
    with gzip.open(OUT + ".tmp", "wt") as fh:
        fh.write("smiles\tzinc_id\ttranche\n")
        for a in MW:
            for b in LOGP:
                tr = a + b
                html = listing(tr)
                if not html:
                    print(f"{tr}: no listing", flush=True); continue
                fn = pick_file(tr, html)
                if not fn:
                    print(f"{tr}: no suitable file", flush=True); continue
                got = sample_file(tr, fn, fh, seen)
                total += got
                print(f"{tr}: {fn} -> {got} (total {total})", flush=True)
    os.replace(OUT + ".tmp", OUT)
    print(f"POOL DONE {total} unique ZINC substances", flush=True)
