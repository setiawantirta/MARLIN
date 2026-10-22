"""Uniform structure standardisation + descriptors for every chemical source in this dataset.

Pipeline (identical for ChEMBL, CMNPD and ZINC unless --no-tautomer is given):
    1. parse + sanitize (Chem.MolFromSmiles)
    2. rdMolStandardize.Cleanup  - normalize functional groups, disconnect metals, reionize
    3. LargestFragmentChooser(preferOrganic=True)  - strip salts/counter-ions
    4. Uncharger  - neutralise
    5. TautomerEnumerator().Canonicalize  - canonical tautomer
    6. key on the full InChIKey; MW window 100-1500 Da
Descriptors emitted for property matching and applicability-domain work: MW, cLogP (Crippen),
HBD, HBA, rotatable bonds, net formal charge, TPSA, aromatic rings, heavy atoms, fraction
Csp3, and the Ertl/Schuffenhauer natural-product-likeness score.

Usage: standardize.py IN.(tsv|csv|smi[.gz]) OUT.parquet ID_COL SMILES_COL [--no-tautomer]
"""
import sys, os, gzip
import pandas as pd
import numpy as np
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, RDConfig
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "NP_Score"))
import npscorer

_G = {}


def _init(use_taut):
    _G["lfc"] = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    _G["unc"] = rdMolStandardize.Uncharger()
    te_params = rdMolStandardize.CleanupParameters()
    te_params.maxTautomers = 200
    te_params.maxTransforms = 200
    _G["te"] = rdMolStandardize.TautomerEnumerator(te_params)
    _G["np"] = npscorer.readNPModel()
    _G["taut"] = use_taut


def _one(arg):
    cid, smi = arg
    out = {"src_id": cid, "fail": None}
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out["fail"] = "parse"; return out
        mol = rdMolStandardize.Cleanup(mol)
        mol = _G["lfc"].choose(mol)
        mol = _G["unc"].uncharge(mol)
        if _G["taut"]:
            mol = _G["te"].Canonicalize(mol)
        Chem.SanitizeMol(mol)
        if mol.GetNumHeavyAtoms() == 0:
            out["fail"] = "empty"; return out
        mw = Descriptors.MolWt(mol)
        if not (100.0 <= mw <= 1500.0):
            out["fail"] = f"mw_{mw:.0f}"; return out
        ik = Chem.MolToInchiKey(mol)
        if not ik:
            out["fail"] = "inchi"; return out
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        out.update({
            "std_smiles": Chem.MolToSmiles(mol),
            "inchikey": ik,
            "inchikey14": ik.split("-")[0],
            "scaffold": scaf,
            "mw": mw,
            "clogp": Crippen.MolLogP(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "charge": Chem.GetFormalCharge(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "n_arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "np_score": npscorer.scoreMol(mol, _G["np"]),
        })
    except Exception as ex:
        out["fail"] = f"err_{type(ex).__name__}"
    return out


def read_any(path, id_col, smi_col):
    op = gzip.open if path.endswith(".gz") else open
    sep = "\t" if (".tsv" in path or ".smi" in path) else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    df = df[[id_col, smi_col]].dropna()
    df.columns = ["id", "smiles"]
    return df.drop_duplicates("id")


if __name__ == "__main__":
    inp, outp, id_col, smi_col = sys.argv[1:5]
    use_taut = "--no-tautomer" not in sys.argv
    df = read_any(inp, id_col, smi_col)
    print(f"{inp}: {len(df)} rows, tautomer_canonicalisation={use_taut}", flush=True)
    args = list(zip(df.id.astype(str), df.smiles.astype(str)))
    res = []
    with Pool(7, initializer=_init, initargs=(use_taut,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, args, chunksize=200)):
            res.append(r)
            if i % 20000 == 0:
                print(f"  {i}/{len(args)}", flush=True)
    out = pd.DataFrame(res)
    ok = out[out.fail.isna()].drop(columns=["fail"])
    bad = out[out.fail.notna()]
    print(f"standardised OK: {len(ok)}; failed: {len(bad)}", flush=True)
    if len(bad):
        print(bad.fail.str.replace(r"mw_\d+", "mw_window", regex=True)
                 .value_counts().to_dict(), flush=True)
    ok.to_parquet(outp, index=False)
    bad.to_csv(outp.replace(".parquet", "_failed.csv"), index=False)
    print("WROTE", outp, flush=True)
