"""Shared molecular graph featurisation for the anti-glioma multi-task benchmark.

A single atom/bond feature specification is used by every GNN architecture so that
the benchmark comparison is architectural, not featurisation-driven.
"""
import os, pickle
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

TASKS = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]

ATOMS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Se", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUMHS = [0, 1, 2, 3, 4]
HYBS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2]
CHIRAL = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
          Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
BONDTYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
STEREOS = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
           Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
           Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS]

ATOM_DIM = (len(ATOMS) + 1) + (len(DEGREES) + 1) + (len(CHARGES) + 1) + \
           (len(NUMHS) + 1) + (len(HYBS) + 1) + (len(CHIRAL) + 1) + 3
BOND_DIM = len(BONDTYPES) + (len(STEREOS) + 1) + 2
RWSE_K = 16


def _oh(v, allowed):
    x = [0.0] * (len(allowed) + 1)
    try:
        x[allowed.index(v)] = 1.0
    except ValueError:
        x[-1] = 1.0
    return x


def atom_feats(a):
    return (_oh(a.GetSymbol(), ATOMS) + _oh(a.GetTotalDegree(), DEGREES) +
            _oh(a.GetFormalCharge(), CHARGES) + _oh(a.GetTotalNumHs(), NUMHS) +
            _oh(a.GetHybridization(), HYBS) + _oh(a.GetChiralTag(), CHIRAL) +
            [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() / 100.0])


def bond_feats(b):
    return (_oh(b.GetBondType(), BONDTYPES)[:len(BONDTYPES)] + _oh(b.GetStereo(), STEREOS) +
            [float(b.GetIsConjugated()), float(b.IsInRing())])


def rwse(edge_index, n, k=RWSE_K):
    """Random-walk structural encoding: diagonal of P^1..P^k (GraphGPS RWSE)."""
    A = np.zeros((n, n), dtype=np.float32)
    if edge_index.shape[1] > 0:
        A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(1, keepdims=True)
    deg[deg == 0] = 1.0
    P = A / deg
    out = np.zeros((n, k), dtype=np.float32)
    M = np.eye(n, dtype=np.float32)
    for i in range(k):
        M = M @ P
        out[:, i] = np.diag(M)
    return out


def mol_to_graph(smiles, with_pe=True):
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return None
    x = np.asarray([atom_feats(a) for a in m.GetAtoms()], dtype=np.float32)
    src, dst, ef = [], [], []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_feats(b)
        src += [i, j]; dst += [j, i]; ef += [f, f]
    if len(src) == 0:                      # single-atom molecule: paired self loops
        src, dst = [0, 0], [0, 0]
        ef = [[0.0] * BOND_DIM, [0.0] * BOND_DIM]
    ei = np.asarray([src, dst], dtype=np.int64)
    ea = np.asarray(ef, dtype=np.float32)
    # directed bonds are stored in consecutive (i->j, j->i) pairs, so the reverse edge
    # of edge k is k^1 -- needed by the D-MPNN message exclusion term
    rev = (np.arange(ei.shape[1], dtype=np.int64) ^ 1)
    g = {"x": x, "edge_index": ei, "edge_attr": ea, "rev": rev, "n": m.GetNumAtoms()}
    if with_pe:
        g["pe"] = rwse(ei, m.GetNumAtoms())
    return g


def featurise_list(smiles_list, with_pe=True, log_every=5000, tag=""):
    graphs, ok = [], []
    for i, s in enumerate(smiles_list):
        g = mol_to_graph(s, with_pe=with_pe)
        graphs.append(g)
        ok.append(g is not None)
        if log_every and (i + 1) % log_every == 0:
            print(f"[feat{tag}] {i+1}/{len(smiles_list)}", flush=True)
    return graphs, np.asarray(ok)


def cache(path, fn):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return obj
