import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from repro.config import path as RPATH, CFG, out_dir
"""Stage C - consensus feature ranking, substructure recovery and statistics.

AGGREGATION RULE (primary - score-level fusion):
  1. Candidate pool = union over the six targets of the top-40 features by MDI and the
     top-40 by mean|SHAP|; this is exactly the set for which permutation importance was
     evaluated. A feature outside the pool is absent from the top 40 of two of the three
     methods for every target, so it cannot enter the consensus top ten.
  2. For each target t and each method m in {TreeSHAP mean|SHAP| on the scaffold-disjoint
     test partition, permutation ROC-AUC drop (negative drops clipped to 0, features
     outside that target's pool set to 0), RF mean-decrease-in-impurity}, the pool
     importances are scaled by their maximum, giving u_m,t(f) in [0, 1].
  3. Method-level global score U_m(f) = MEAN over the six targets of u_m,t(f).
  4. Consensus score C(f) = mean of the three U_m(f); HIGHER is better. Ties broken by
     pooled mean|SHAP|.

  The mean (not median) over targets is deliberate: in a multi-target model different
  targets are driven by different chemistry, and a median-rank rule implicitly requires a
  feature to matter for ALL six targets before it can rank highly. Applied here that rule
  returns an all-descriptor top ten and excludes every substructure feature, because each
  ECFP4 bit is important for only one or two targets. That variant is computed as a
  SENSITIVITY ANALYSIS (columns rank_median_*), reported, and discussed as a property of
  the aggregation rather than of the chemistry.

Outputs: top10_features.csv, feature_importance_full.csv, ecfp4_bit_dictionary.csv
"""
import json, os, sys
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats

RDLogger.DisableLog("rdApp.*")
MATRIX = RPATH("chembl_multitask_matrix.parquet")
CMNPD = RPATH("cmnpd_curated.parquet")
TASKS = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
OUT = "attrib"
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
N_ENV_SAMPLE = 400      # molecules inspected per bit when recovering its environment

DESC_DEF = {
 "MolWt": "Average molecular weight (Da); RDKit Descriptors.MolWt",
 "MolLogP": "Wildman-Crippen octanol/water partition coefficient; RDKit Crippen.MolLogP",
 "TPSA": "Topological polar surface area (A^2), Ertl fragment method; CalcTPSA",
 "NumHDonors": "Count of hydrogen-bond donors (Lipinski definition); CalcNumHBD",
 "NumHAcceptors": "Count of hydrogen-bond acceptors (Lipinski definition); CalcNumHBA",
 "NumRotatableBonds": "Count of rotatable single bonds; CalcNumRotatableBonds",
 "RingCount": "Number of rings (SSSR); CalcNumRings",
 "NumAromaticRings": "Number of aromatic rings; CalcNumAromaticRings",
 "FractionCSP3": "Fraction of carbon atoms that are sp3-hybridised; CalcFractionCSP3",
 "HeavyAtomCount": "Number of non-hydrogen atoms",
 "NHOHCount": "Number of NH and OH groups; Descriptors.NHOHCount",
 "NOCount": "Number of nitrogen and oxygen atoms; Descriptors.NOCount",
 "NumHeteroatoms": "Number of non-carbon, non-hydrogen atoms; CalcNumHeteroatoms",
 "MolMR": "Wildman-Crippen molar refractivity; Crippen.MolMR",
 "BalabanJ": "Balaban's J topological connectivity index",
 "BertzCT": "Bertz complexity index (graph-theoretical molecular complexity)",
 "Chi0v": "Valence-corrected zeroth-order molecular connectivity index (Kier-Hall)",
 "Chi1v": "Valence-corrected first-order molecular connectivity index (Kier-Hall)",
 "Chi2v": "Valence-corrected second-order molecular connectivity index (Kier-Hall)",
 "Chi3v": "Valence-corrected third-order molecular connectivity index (Kier-Hall)",
 "Chi4v": "Valence-corrected fourth-order molecular connectivity index (Kier-Hall)",
 "Kappa1": "First-order Kier-Hall kappa shape index (atom count / linearity)",
 "Kappa2": "Second-order Kier-Hall kappa shape index (spatial density / branching)",
 "Kappa3": "Third-order Kier-Hall kappa shape index (centrality of branching)",
 "LabuteASA": "Labute's approximate accessible surface area (A^2)",
 "HallKierAlpha": "Hall-Kier alpha: sum of atom-type polarisability corrections",
 "NumSpiroAtoms": "Number of spiro atoms (single-atom ring fusions)",
 "NumBridgeheadAtoms": "Number of bridgehead atoms (shared by fused rings)",
 "NumSaturatedRings": "Number of fully saturated rings",
 "QED": "Quantitative estimate of drug-likeness (Bickerton et al.), 0-1",
}


# ------------------------------------------------------------------- bit -> environment
def bit_environments(bit, mols, limit=N_ENV_SAMPLE):
    """Collect the distinct circular environments that set `bit` across `mols`.

    Returns (records, n_mols_inspected). Each record: fragment SMILES (rooted at the
    Morgan centre), SMARTS of the environment submol, radius, centre element, count.
    """
    frags = Counter()
    detail = {}
    seen = 0
    for mol in mols:
        if seen >= limit:
            break
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        FP_GEN.GetFingerprint(mol, additionalOutput=ao)
        bi = ao.GetBitInfoMap()
        if bit not in bi:
            continue
        seen += 1
        for (centre, rad) in bi[bit]:
            if rad == 0:
                atoms = [centre]
                bonds = []
            else:
                bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, rad, centre))
                amap = {}
                Chem.PathToSubmol(mol, bonds, atomMap=amap)
                atoms = list(amap.keys())
            try:
                smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, bondsToUse=bonds,
                                               rootedAtAtom=centre, canonical=True,
                                               allBondsExplicit=False)
            except Exception:
                continue
            key = (smi, rad)
            frags[key] += 1
            if key not in detail:
                sub = Chem.PathToSubmol(mol, bonds) if bonds else None
                sm = None
                if sub is not None:
                    try:
                        sm = Chem.MolToSmarts(sub)
                    except Exception:
                        sm = None
                detail[key] = dict(smarts=sm,
                                   centre_symbol=mol.GetAtomWithIdx(centre).GetSymbol(),
                                   centre_aromatic=bool(mol.GetAtomWithIdx(centre).GetIsAromatic()),
                                   n_atoms=len(atoms))
    recs = []
    for (smi, rad), n in frags.most_common():
        d = dict(detail[(smi, rad)]); d.update(fragment_smiles=smi, radius=rad, count=n)
        recs.append(d)
    return recs, seen


# ------------------------------------------------------------------------ main
def main():
    featnames = list(np.load(f"{OUT}/featnames.npy"))
    nfeat = len(featnames)
    MDI = np.load(f"{OUT}/mdi.npz")["MDI"]
    raw = np.load(f"{OUT}/chembl_raw.npz")
    FPc, Dc, okc, split, Y, mask = (raw["FP"], raw["D"], raw["ok"], raw["split"],
                                    raw["Y"], raw["mask"])
    craw = np.load(f"{OUT}/cmnpd_raw.npz")
    FPm, Dm, okm = craw["FP"], craw["D"], craw["ok"]

    shap_test = {t: np.load(f"{OUT}/shap_test_{t}.npz") for t in TASKS}
    shap_mar = {t: np.load(f"{OUT}/shap_marine_{t}.npz") for t in TASKS}
    perm = {t: np.load(f"{OUT}/perm_{t}.npz") for t in TASKS}

    # ------------------------------------------------ per-target per-method importances
    imp = {}                                   # imp[method][target] -> (nfeat,) array
    imp["shap"] = {t: np.abs(shap_test[t]["sv"]).mean(0) for t in TASKS}
    imp["shap_marine"] = {t: np.abs(shap_mar[t]["sv"]).mean(0) for t in TASKS}
    imp["mdi"] = {t: MDI[ti] for ti, t in enumerate(TASKS)}
    imp["perm"] = {}
    pool_all = set()
    for t in TASKS:
        v = np.full(nfeat, np.nan)
        v[perm[t]["pool"]] = perm[t]["drops"].mean(1)
        imp["perm"][t] = v
        pool_all |= set(perm[t]["pool"].tolist())
    pool = np.array(sorted(pool_all))
    print("candidate pool size", len(pool), flush=True)

    # ------------------------------------------------------------- consensus (rule above)
    def ranks_in_pool(vals):
        v = np.asarray(vals)[pool]
        v = np.where(np.isnan(v), -np.inf, v)
        order = np.argsort(-v, kind="stable")
        r = np.empty(len(pool)); r[order] = np.arange(1, len(pool) + 1)
        return r

    def scores_in_pool(vals):
        v = np.asarray(vals, dtype=np.float64)[pool]
        v = np.nan_to_num(v, nan=0.0)
        v = np.clip(v, 0.0, None)
        mx = v.max()
        return v / mx if mx > 0 else v

    meth_score, meth_rank = {}, {}
    for m in ["shap", "perm", "mdi"]:
        meth_score[m] = np.mean(np.stack([scores_in_pool(imp[m][t]) for t in TASKS]), axis=0)
        meth_rank[m] = np.median(np.stack([ranks_in_pool(imp[m][t]) for t in TASKS]), axis=0)
    consensus = np.mean(np.stack([meth_score[m] for m in ["shap", "perm", "mdi"]]), axis=0)
    sens_rank = np.mean(np.stack([meth_rank[m] for m in ["shap", "perm", "mdi"]]), axis=0)
    pooled_shap = np.mean(np.stack([imp["shap"][t][pool] for t in TASKS]), axis=0)
    order = np.lexsort((-pooled_shap, -consensus))
    ranked = pool[order]

    # ---------------------------------------------------------------- full-table export
    rows = []
    for k, fi in enumerate(pool):
        r = dict(feature=featnames[fi], feature_index=int(fi),
                 type="ECFP4_bit" if fi < 2048 else "RDKit_descriptor",
                 consensus_score=float(consensus[k]),
                 score_shap=float(meth_score["shap"][k]),
                 score_perm=float(meth_score["perm"][k]),
                 score_mdi=float(meth_score["mdi"][k]),
                 rank_median_shap=float(meth_rank["shap"][k]),
                 rank_median_perm=float(meth_rank["perm"][k]),
                 rank_median_mdi=float(meth_rank["mdi"][k]),
                 sensitivity_median_rank_score=float(sens_rank[k]),
                 mean_abs_shap_pooled=float(pooled_shap[k]))
        for t in TASKS:
            r[f"meanabsshap_{t}"] = float(imp["shap"][t][fi])
            r[f"meanabsshap_marine_{t}"] = float(imp["shap_marine"][t][fi])
            r[f"permdrop_{t}"] = float(imp["perm"][t][fi])
            r[f"mdi_{t}"] = float(imp["mdi"][t][fi])
        rows.append(r)
    full = pd.DataFrame(rows).sort_values(
        ["consensus_score", "mean_abs_shap_pooled"], ascending=False).reset_index(drop=True)
    full.insert(0, "consensus_rank", np.arange(1, len(full) + 1))
    full["sensitivity_median_rank_position"] = (
        full["sensitivity_median_rank_score"].rank(method="first").astype(int))
    full.to_csv("feature_importance_full.csv", index=False)
    print("wrote feature_importance_full.csv", full.shape, flush=True)
    print("sensitivity (median-rank) top 10:",
          list(full.sort_values("sensitivity_median_rank_score").head(10)["feature"]), flush=True)

    top = full.head(10).copy()
    json.dump(dict(pool=[int(x) for x in pool],
                   ranked=[int(x) for x in ranked],
                   top10_index=[int(x) for x in top["feature_index"]]),
              open(f"{OUT}/consensus.json", "w"))
    print("TOP 10:", list(top["feature"]), flush=True)


if __name__ == "__main__":
    main()
