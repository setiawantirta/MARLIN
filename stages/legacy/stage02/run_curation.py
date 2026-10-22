"""Driver: stream CMNPD 2D SDF for fallback molblocks, curate all compounds in parallel."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import os, sys, time, pickle
import pandas as pd
from rdkit import Chem, RDLogger
import multiprocessing as mp
sys.path.append(os.getcwd())
import curate_worker

RDLogger.DisableLog('rdApp.*')
# The two CMNPD vendor files are resolved by their own basenames, not by first
# resolving the folder that happened to hold them in the original workspace. The old
# form asked for a directory named 'cmnpd_screening' and then joined 'data semua
# senyawa' onto it, which only ever worked on the machine the study was run on:
# repro.config.path() returns files, never directories, so the lookup could not
# succeed anywhere else and failed with a misleading "intermediate written by an
# earlier stage". Resolving each file by basename means the pair can sit in any
# folder under any data root, whatever it is called.
PROP_TSV = RPATH("CMNPD_1.0_calc_prop.tsv")

# Vendor releases have shipped the 2D structures under more than one name. Try each
# and report all of them if none is present, rather than failing on one spelling.
_SDF_NAMES = ("CMNPD_1.0_2d.sdf", "CMNPD_1.0.sdf", "CMNPD_1.0_2D.sdf")
SDF_FILE = None
for _n in _SDF_NAMES:
    try:
        SDF_FILE = RPATH(_n)
        break
    except Exception:
        continue


def main():
    prop = pd.read_csv(PROP_TSV, sep="\t",
                       encoding="utf-8-sig", dtype=str, low_memory=False)
    bad = {cid for cid, s in zip(prop.COMPOUND_ID, prop.SMILES)
           if Chem.MolFromSmiles(s) is None}
    print("smiles_parse_failures", len(bad), flush=True)
    mbs = {}
    if bad:
        # The SDF is only needed to recover molblocks for the SMILES that RDKit could
        # not parse. If no SMILES failed, it is never opened, so a missing SDF is only
        # fatal when there is actually something to recover.
        if SDF_FILE is None:
            raise SystemExit(
                f"{len(bad)} SMILES failed to parse and need molblocks from the CMNPD "
                f"2D SDF, but none of {_SDF_NAMES} was found under any data root.\n"
                "Place the vendor SDF anywhere under $MARINE_REPRO_DATA or a configured "
                "data root and re-run; see data/README.md.")
        supp = Chem.ForwardSDMolSupplier(open(SDF_FILE, 'rb'),
                                         sanitize=False)
        for m in supp:
            if m is None:
                continue
            cid = m.GetProp("COMPOUND_ID") if m.HasProp("COMPOUND_ID") else None
            if cid in bad:
                mbs[cid] = Chem.MolToMolBlock(m, kekulize=False)
    recs = [(cid, s, mbs.get(cid)) for cid, s in zip(prop.COMPOUND_ID, prop.SMILES)]
    t0 = time.time()
    out = []
    # A bar over the compounds, so the longest step in the pipeline reports its own
    # progress and an ETA instead of a count every four thousand records. tqdm is
    # optional: without it the loop falls back to the periodic print it always had.
    try:
        from tqdm.auto import tqdm
        it = tqdm(total=len(recs), desc="curating CMNPD", unit="cmpd", smoothing=0.05)
    except ImportError:
        it = None
    with mp.Pool(7) as pool:
        for i, r in enumerate(pool.imap(curate_worker.curate, recs, chunksize=100), 1):
            out.append(r)
            if it is not None:
                it.update(1)
            elif i % 4000 == 0:
                print(i, round(time.time() - t0), flush=True)
    if it is not None:
        it.close()
    print("done", len(out), round(time.time() - t0), flush=True)

    # Write to the output directory, not back into a read-only data root. RPATH
    # resolves an EXISTING file; on a first run curation_raw.pkl does not exist yet,
    # so asking for it by name would raise MissingInput before anything is written.
    dest = os.path.join(out_dir(), "curation_raw.pkl")
    with open(dest, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
