import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from repro.config import path as RPATH, CFG, out_dir
"""Stage B - attribution on the best graph model (D-MPNN, seed 0 = best validation loss)
for CONTRAST with the Random Forest.

Integrated Gradients over the 45-dimensional atom feature matrix (baseline = all-zero
atom features, 32 Riemann-midpoint steps), per task logit, on
  (i)  synthetic ChEMBL test molecules (scaffold-disjoint partition)
  (ii) the 194-compound CMNPD marine phenotypic set.

To put the RF and the GNN on ONE comparable footing, per-atom RF attribution is derived by
projecting each ECFP4 bit's SHAP value onto the atoms of the circular environment that set
that bit (RDKit Morgan bitInfo), so both models yield an atom-resolved attribution vector
for the same molecule. Then we quantify:
  - Spearman correlation between RF- and GNN-derived atom attributions, per molecule
  - normalised attribution entropy (how diffuse the attribution is)
  - fraction of |attribution| carried by Bemis-Murcko scaffold atoms vs side-chain atoms
  - fraction of |attribution| carried by aromatic atoms and by sp3 atoms
  - cosine similarity of the mean 45-dim atom-feature attribution between synthetic and
    marine molecules (attribution-domain shift)
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats

RDLogger.DisableLog("rdApp.*")
SRC = os.path.abspath("code/src")
sys.path.insert(0, SRC)
import types
_m = types.ModuleType("mdata"); _m.CACHE = "cache"
sys.modules.setdefault("mdata", _m)
from feat import mol_to_graph, ATOM_DIM, BOND_DIM, TASKS          # noqa: E402
from models import DMPNNNet                                        # noqa: E402

MATRIX = RPATH("chembl_multitask_matrix.parquet")
CMNPD = RPATH("cmnpd_curated.parquet")
CKPT = "bundle/glioma_models/models/DMPNN_best.pt"
OUT = "attrib"
IG_STEPS = 32
N_PER_TASK = 200      # synthetic (molecule, target) pairs sampled per target
SEED = 0
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ------------------------------------------------------------------------ model plumbing
class Bag:
    """Minimal stand-in for a PyG Batch holding a single molecule."""
    def __init__(self, x, edge_index, edge_attr, rev, batch):
        self.x, self.edge_index, self.edge_attr = x, edge_index, edge_attr
        self.rev, self.batch = rev, batch


def load_model():
    m = DMPNNNet(hidden=128, depth=3, dropout=0.2, head_hidden=128)
    sd = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m


def replicate(g, k):
    """k independent copies of one molecule in a single batch (for the IG path)."""
    n, e = g["n"], g["edge_index"].shape[1]
    ei = np.concatenate([g["edge_index"] + i * n for i in range(k)], axis=1)
    rv = np.concatenate([g["rev"] + i * e for i in range(k)])
    ea = np.tile(g["edge_attr"], (k, 1))
    bt = np.repeat(np.arange(k), n)
    return (torch.from_numpy(ei), torch.from_numpy(ea), torch.from_numpy(rv),
            torch.from_numpy(bt))


def integrated_gradients(model, g, task_idx, steps=IG_STEPS):
    """IG on atom features for one task logit. Returns (n_atoms, ATOM_DIM) attribution."""
    x0 = torch.from_numpy(g["x"])
    ei, ea, rv, bt = replicate(g, steps)
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    xpath = torch.cat([x0 * a for a in alphas], dim=0).requires_grad_(True)
    d = Bag(xpath, ei, ea, rv, bt)
    logits = model(d)[:, task_idx].sum()
    grad, = torch.autograd.grad(logits, xpath)
    n = g["n"]
    avg_grad = grad.view(steps, n, ATOM_DIM).mean(0)
    return (x0 * avg_grad).detach().numpy()


# ------------------------------------------------------------- RF -> per-atom projection
def rf_atom_attribution(mol, shap_vec):
    """Project ECFP4-bit SHAP values onto atoms via the Morgan bit environments.

    Each set bit b has one or more (centre, radius) environments; its SHAP value is split
    equally over the atoms of each environment (and over environments, if several).
    Descriptor SHAP is global (no atom locus) and is therefore excluded from the atom map;
    the descriptor share is returned separately.
    """
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    FP_GEN.GetFingerprint(mol, additionalOutput=ao)
    bi = ao.GetBitInfoMap()
    n = mol.GetNumAtoms()
    atom_attr = np.zeros(n, dtype=np.float64)
    for b, envs in bi.items():
        v = float(shap_vec[b])
        if v == 0.0:
            continue
        per_env = v / len(envs)
        for (centre, rad) in envs:
            if rad == 0:
                idxs = [centre]
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, centre)
                amap = {}
                Chem.PathToSubmol(mol, env, atomMap=amap)
                idxs = list(amap.keys()) if amap else [centre]
            for a in idxs:
                atom_attr[a] += per_env / len(idxs)
    desc_share = float(np.abs(shap_vec[2048:]).sum())
    return atom_attr, desc_share


# ------------------------------------------------------------------------ atom partitions
def scaffold_atoms(mol):
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return set()
        match = mol.GetSubstructMatch(core)
        if match:
            return set(match)
    except Exception:
        pass
    return {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}


def norm_entropy(v):
    w = np.abs(np.asarray(v, dtype=np.float64))
    s = w.sum()
    if s <= 0 or len(w) < 2:
        return np.nan
    p = w / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(w)))


def frac_on(v, idxs):
    w = np.abs(np.asarray(v, dtype=np.float64))
    s = w.sum()
    if s <= 0:
        return np.nan
    return float(w[list(idxs)].sum() / s) if len(idxs) else 0.0


# --------------------------------------------------------------------------------- main
def main():
    model = load_model()
    rng = np.random.default_rng(SEED)

    df = pd.read_parquet(MATRIX)
    split = df["split"].to_numpy()
    Y = np.stack([df[f"label_{t}"].to_numpy(dtype="float64") for t in TASKS], axis=1)
    mask = (~np.isnan(Y)).astype(np.float32)

    # SHAP already computed on the test subsamples and on the marine set (stage A)
    shap_test = {t: np.load(f"{OUT}/shap_test_{t}.npz") for t in TASKS}
    shap_mar = {t: np.load(f"{OUT}/shap_marine_{t}.npz") for t in TASKS}
    mrows = np.load(f"{OUT}/marine_pheno.npz")["rows"]

    cm = pd.read_parquet(CMNPD, columns=["COMPOUND_ID", "smiles_std", "inchikey"])
    pact = np.load(f"{OUT}/marine_pheno.npz")["pact_max"]
    pact_of = {int(r): float(p) for r, p in zip(mrows, pact)}

    # (molecule, target) PAIRS, built so that an RF SHAP vector always exists for the pair:
    # synthetic pairs are sampled from each target's own TreeSHAP subsample, marine pairs
    # cover all 194 compounds x 6 targets.
    pairs = []
    for ti, t in enumerate(TASKS):
        rws = shap_test[t]["rows"]
        ys = shap_test[t]["y"]
        take = rng.choice(len(rws), min(N_PER_TASK, len(rws)), replace=False)
        for k in take:
            pairs.append(("synthetic_test", int(rws[k]), ti, t, int(ys[k]),
                          df["canonical_smiles"].iloc[int(rws[k])]))
        for k, rw in enumerate(mrows):
            pairs.append(("marine", int(rw), ti, t, -1, cm["smiles_std"].iloc[int(rw)]))
    print("pairs", len(pairs), flush=True)

    shap_lookup = {"synthetic_test": {t: {int(r): k for k, r in enumerate(shap_test[t]["rows"])}
                                      for t in TASKS},
                   "marine": {t: {int(r): k for k, r in enumerate(shap_mar[t]["rows"])}
                              for t in TASKS}}
    shap_mat = {"synthetic_test": {t: shap_test[t]["sv"] for t in TASKS},
                "marine": {t: shap_mar[t]["sv"] for t in TASKS}}

    rows = []
    featattr, featattr_n = {}, {}
    molcache = {}
    t0 = time.time()
    for n, (sname, ridx, ti, t, ylab, smi) in enumerate(pairs):
        key = (sname, ridx)
        if key not in molcache:
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None or mol.GetNumAtoms() < 2:
                molcache[key] = None
            else:
                g = mol_to_graph(smi, with_pe=False)
                if g is None:
                    molcache[key] = None
                else:
                    scaf = scaffold_atoms(mol)
                    molcache[key] = (mol, g, scaf,
                                     set(range(mol.GetNumAtoms())) - scaf,
                                     {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()},
                                     {a.GetIdx() for a in mol.GetAtoms()
                                      if a.GetHybridization() == Chem.rdchem.HybridizationType.SP3})
        if molcache[key] is None:
            continue
        mol, g, scaf, side, arom, sp3 = molcache[key]
        ig = integrated_gradients(model, g, ti)
        gnn_atom = ig.sum(1)
        kk = (sname, t)
        featattr[kk] = featattr.get(kk, np.zeros(ATOM_DIM)) + np.abs(ig).sum(0)
        featattr_n[kk] = featattr_n.get(kk, 0) + 1
        rec = dict(set=sname, row=ridx, task=t, label=ylab, n_atoms=mol.GetNumAtoms(),
                   pact_max=pact_of.get(ridx) if sname == "marine" else None,
                   gnn_entropy=norm_entropy(gnn_atom),
                   gnn_frac_scaffold=frac_on(gnn_atom, scaf),
                   gnn_frac_sidechain=frac_on(gnn_atom, side),
                   gnn_frac_aromatic=frac_on(gnn_atom, arom),
                   gnn_frac_sp3=frac_on(gnn_atom, sp3),
                   gnn_abs_total=float(np.abs(gnn_atom).sum()))
        kidx = shap_lookup[sname][t].get(ridx)
        if kidx is not None:
            rf_atom, dshare = rf_atom_attribution(mol, shap_mat[sname][t][kidx])
            rec.update(rf_entropy=norm_entropy(rf_atom),
                       rf_frac_scaffold=frac_on(rf_atom, scaf),
                       rf_frac_sidechain=frac_on(rf_atom, side),
                       rf_frac_aromatic=frac_on(rf_atom, arom),
                       rf_frac_sp3=frac_on(rf_atom, sp3),
                       rf_desc_share=dshare,
                       rf_abs_total=float(np.abs(rf_atom).sum()))
            if mol.GetNumAtoms() >= 5 and np.abs(rf_atom).sum() > 0 and \
               np.abs(gnn_atom).sum() > 0:
                rho, pv = stats.spearmanr(rf_atom, gnn_atom)
                rec.update(rf_gnn_atom_rho=float(rho), rf_gnn_atom_p=float(pv))
        rows.append(rec)
        if (n + 1) % 400 == 0:
            print(f"  {n+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT}/gnn_rf_atom_attrib.csv", index=False)
    np.savez_compressed(f"{OUT}/gnn_featattr.npz",
                        **{f"{s}__{t}": v / max(featattr_n[(s, t)], 1)
                           for (s, t), v in featattr.items()})
    print("STAGE B DONE", len(rows), f"{time.time()-t0:.0f}s", flush=True)
    gnnexplainer_arm(model, df, cm, shap_test, mrows, rng)


# ---------------------------------------------------------------- GNNExplainer arm
def gnnexplainer_arm(model, df, cm, shap_test, mrows, rng, n_each=60, epochs=60):
    """Second, independent graph attribution: PyG GNNExplainer node/edge masks.

    Learns a soft mask over atoms and bonds that preserves the D-MPNN task logit. Reported
    per molecule: normalised entropy of the atom mask, and the share of atom-mask weight on
    Bemis-Murcko scaffold atoms and on aromatic atoms - the same statistics computed for
    Integrated Gradients, so the two attribution methods can be compared directly.
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        from torch_geometric.explain.config import ModelConfig
    except Exception as e:
        print("GNNExplainer unavailable:", e, flush=True)
        return

    class Wrap(torch.nn.Module):
        def __init__(self, m, ti):
            super().__init__()
            self.m, self.ti = m, ti

        def forward(self, x, edge_index, edge_attr=None, rev=None, batch=None):
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long)
            d = Bag(x, edge_index, edge_attr, rev, batch)
            return self.m(d)[:, self.ti].view(-1, 1)

    items = []
    for ti, t in enumerate(TASKS):
        rws = shap_test[t]["rows"]
        for k in rng.choice(len(rws), min(n_each // len(TASKS) + 1, len(rws)), replace=False):
            items.append(("synthetic_test", int(rws[k]), ti, t,
                          df["canonical_smiles"].iloc[int(rws[k])]))
        for rw in rng.choice(mrows, min(n_each // len(TASKS) + 1, len(mrows)), replace=False):
            items.append(("marine", int(rw), ti, t, cm["smiles_std"].iloc[int(rw)]))

    rows, t0 = [], time.time()
    for n, (sname, ridx, ti, t, smi) in enumerate(items):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None or mol.GetNumAtoms() < 3:
            continue
        g = mol_to_graph(smi, with_pe=False)
        if g is None:
            continue
        try:
            expl = Explainer(model=Wrap(model, ti), algorithm=GNNExplainer(epochs=epochs),
                             explanation_type="model", node_mask_type="attributes",
                             # the D-MPNN aggregates directed-bond messages with an explicit
                             # index_add_ rather than a PyG MessagePassing layer, so PyG
                             # cannot attach an edge mask; the atom mask is what we report
                             edge_mask_type=None,
                             model_config=ModelConfig(mode="binary_classification",
                                                      task_level="graph",
                                                      return_type="raw"))
            e = expl(torch.from_numpy(g["x"]), torch.from_numpy(g["edge_index"]),
                     edge_attr=torch.from_numpy(g["edge_attr"]),
                     rev=torch.from_numpy(g["rev"]),
                     batch=torch.zeros(g["n"], dtype=torch.long))
        except Exception as ex:
            if n < 3:
                print("GNNExplainer failed:", type(ex).__name__, ex, flush=True)
            continue
        nm = e.node_mask.detach().numpy().sum(1)
        scaf = scaffold_atoms(mol)
        arom = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
        rows.append(dict(set=sname, row=ridx, task=t, n_atoms=mol.GetNumAtoms(),
                         gnnexp_entropy=norm_entropy(nm),
                         gnnexp_frac_scaffold=frac_on(nm, scaf),
                         gnnexp_frac_aromatic=frac_on(nm, arom)))
        if (n + 1) % 60 == 0:
            print(f"  [gnnexp] {n+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(f"{OUT}/gnnexplainer_atom_attrib.csv", index=False)
        print("GNNEXPLAINER DONE", len(rows), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
