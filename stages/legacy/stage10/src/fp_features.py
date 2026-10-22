"""ECFP4 (2048 bit, r=2) + a fixed RDKit descriptor block for the fingerprint baselines.

Computed with the same code for ChEMBL, CMNPD and ZINC decoys so the baseline sees
exactly the same chemistry as the GNNs.
"""
import os, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata

DESC = [
    ("MolWt", Descriptors.MolWt), ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA), ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms()),
    ("NHOHCount", Descriptors.NHOHCount), ("NOCount", Descriptors.NOCount),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("MolMR", Crippen.MolMR), ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT), ("Chi0v", rdMolDescriptors.CalcChi0v),
    ("Chi1v", rdMolDescriptors.CalcChi1v), ("Chi2v", rdMolDescriptors.CalcChi2v),
    ("Chi3v", rdMolDescriptors.CalcChi3v), ("Chi4v", rdMolDescriptors.CalcChi4v),
    ("Kappa1", rdMolDescriptors.CalcKappa1), ("Kappa2", rdMolDescriptors.CalcKappa2),
    ("Kappa3", rdMolDescriptors.CalcKappa3), ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("HallKierAlpha", rdMolDescriptors.CalcHallKierAlpha),
    ("NumSpiroAtoms", rdMolDescriptors.CalcNumSpiroAtoms),
    ("NumBridgeheadAtoms", rdMolDescriptors.CalcNumBridgeheadAtoms),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("QED", QED.qed),
]
DESC_NAMES = [d[0] for d in DESC]
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurise(smiles):
    n = len(smiles)
    FP = np.zeros((n, 2048), dtype=np.uint8)
    D = np.full((n, len(DESC)), np.nan, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        FP[i] = np.frombuffer(FP_GEN.GetFingerprintAsNumPy(m).astype(np.uint8), dtype=np.uint8)
        for j, (_, fn) in enumerate(DESC):
            try:
                D[i, j] = float(fn(m))
            except Exception:
                D[i, j] = np.nan
        ok[i] = True
        if (i + 1) % 10000 == 0:
            print(f"[fp] {i+1}/{n}", flush=True)
    return FP, D, ok


def build(name, smiles):
    p = os.path.join(mdata.CACHE, name)
    if os.path.exists(p):
        return
    FP, D, ok = featurise(smiles)
    np.savez_compressed(p, FP=FP, D=D, ok=ok, desc_names=np.array(DESC_NAMES))
    print("saved", name, FP.shape, flush=True)


if __name__ == "__main__":
    os.makedirs(mdata.CACHE, exist_ok=True)
    df = pd.read_parquet(mdata.MATRIX)
    build("chembl_fp.npz", df["canonical_smiles"].tolist())
    cm = pd.read_parquet(mdata.CMNPD)
    build("cmnpd_fp.npz", cm["smiles_std"].tolist())
    dc = pd.read_parquet(mdata.DECOYS)
    build("decoy_fp.npz", dc["std_smiles"].tolist())
    print("FP CACHE DONE", flush=True)
