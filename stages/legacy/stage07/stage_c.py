import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from repro.config import path as RPATH, CFG, out_dir
"""Stage C - consensus top ten, substructure recovery, and per-feature statistics.

PRIMARY AGGREGATION RULE
  1. Candidate pool = union over the six targets of the top-40 features by MDI and the
     top-40 by mean|SHAP|; this is exactly the set for which permutation importance was
     evaluated, so a feature outside it is absent from the top 40 of two of three methods
     for every target and cannot reach the top ten.
  2. For each target t and method m in {TreeSHAP mean|SHAP| on the scaffold-disjoint test
     partition, permutation ROC-AUC drop (negative drops clipped to 0; features outside
     that target's pool set to 0), RF mean-decrease-in-impurity}, pool importances are
     divided by their maximum, giving u_m,t(f) in [0, 1].
  3. Method-level score U_m(f) = MAX over the six targets of u_m,t(f).
  4. Consensus C(f) = mean of the three U_m(f); higher is better. Ties broken by pooled
     mean|SHAP|.

  The MAX over targets is the question-appropriate statistic because the deployed model is
  a per-target ensemble of six independent Random Forests: a feature that is the single
  dominant driver of one target's predictions IS one of the model's most important
  features. Two alternative aggregations are computed and reported as sensitivity
  analyses - MEAN over targets, and the median-of-per-target-ranks Borda rule - both of
  which return an all-descriptor top ten, because every ECFP4 bit is important for only
  one or two targets whereas the 30 descriptors are informative for all six. That
  divergence is a property of the aggregation and of the dense-vs-sparse asymmetry of the
  two feature bases (30 dense descriptors vs 2048 sparse bits), not of the chemistry, and
  is discussed as such.
"""
import json, os, sys
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Crippen, rdMolDescriptors
from scipy import stats

RDLogger.DisableLog("rdApp.*")
MATRIX = RPATH("chembl_multitask_matrix.parquet")
CMNPD = RPATH("cmnpd_curated.parquet")
TASKS = ["EGFR", "PDGFRA", "PIK3CA", "KDR", "CDK6", "IDH1"]
OUT = "attrib"
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
N_ENV_SAMPLE = 500
SEEDS_STAB = [0, 1, 2]

DESC_DEF = {
 "MolWt": "Average molecular weight in Da (RDKit Descriptors.MolWt)",
 "MolLogP": "Wildman-Crippen atom-contribution estimate of log P(octanol/water) (Crippen.MolLogP)",
 "TPSA": "Topological polar surface area in A^2, Ertl fragment method (CalcTPSA)",
 "NumHDonors": "Number of hydrogen-bond donors, Lipinski definition (CalcNumHBD)",
 "NumHAcceptors": "Number of hydrogen-bond acceptors, Lipinski definition (CalcNumHBA)",
 "NumRotatableBonds": "Number of rotatable single bonds (CalcNumRotatableBonds)",
 "RingCount": "Number of rings in the smallest set of smallest rings (CalcNumRings)",
 "NumAromaticRings": "Number of aromatic rings (CalcNumAromaticRings)",
 "FractionCSP3": "Fraction of carbon atoms that are sp3-hybridised (CalcFractionCSP3)",
 "HeavyAtomCount": "Number of non-hydrogen atoms",
 "NHOHCount": "Number of NH and OH groups (Descriptors.NHOHCount)",
 "NOCount": "Number of nitrogen plus oxygen atoms (Descriptors.NOCount)",
 "NumHeteroatoms": "Number of non-carbon non-hydrogen atoms (CalcNumHeteroatoms)",
 "MolMR": "Wildman-Crippen molar refractivity, an atom-contribution polarisability/volume term (Crippen.MolMR)",
 "BalabanJ": "Balaban J index: average distance-sum connectivity of the molecular graph",
 "BertzCT": "Bertz CT graph-theoretical index of molecular complexity (bond and atom-type entropy)",
 "Chi0v": "Zeroth-order valence-corrected Kier-Hall molecular connectivity index",
 "Chi1v": "First-order valence-corrected Kier-Hall molecular connectivity index",
 "Chi2v": "Second-order valence-corrected Kier-Hall molecular connectivity index",
 "Chi3v": "Third-order valence-corrected Kier-Hall molecular connectivity index",
 "Chi4v": "Fourth-order valence-corrected Kier-Hall molecular connectivity index",
 "Kappa1": "First-order Kier-Hall kappa shape index (size/linearity of the graph)",
 "Kappa2": "Second-order Kier-Hall kappa shape index (spatial density, degree of branching)",
 "Kappa3": "Third-order Kier-Hall kappa shape index (centrality of branching)",
 "LabuteASA": "Labute approximate solvent-accessible surface area in A^2",
 "HallKierAlpha": "Hall-Kier alpha: sum of per-atom polarisability corrections relative to sp3 carbon; becomes more negative with more sp3/small atoms and less negative with heavy, polarisable or unsaturated atoms",
 "NumSpiroAtoms": "Number of spiro atoms (single shared atom between two rings)",
 "NumBridgeheadAtoms": "Number of bridgehead atoms shared by fused rings",
 "NumSaturatedRings": "Number of fully saturated rings",
 "QED": "Quantitative estimate of drug-likeness, 0-1 (Bickerton et al. 2012)",
}
# descriptors with a per-atom decomposition that can be drawn on a structure
ATOM_DECOMPOSABLE = {"MolLogP", "MolMR", "HallKierAlpha", "NumHAcceptors", "NumHDonors",
                     "FractionCSP3", "TPSA", "NumHeteroatoms", "NOCount", "NHOHCount"}


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def bit_environments(bit, mols, limit=N_ENV_SAMPLE):
    """Distinct circular environments that set `bit`, over up to `limit` carrier molecules."""
    frags, detail, seen = Counter(), {}, 0
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
                atoms, bonds = [centre], []
            else:
                bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, rad, centre))
                amap = {}
                Chem.PathToSubmol(mol, bonds, atomMap=amap)
                atoms = list(amap.keys())
            try:
                smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, bondsToUse=bonds,
                                               rootedAtAtom=centre, canonical=True)
            except Exception:
                continue
            key = (smi, rad)
            frags[key] += 1
            if key not in detail:
                sm = None
                if bonds:
                    try:
                        sm = Chem.MolToSmarts(Chem.PathToSubmol(mol, bonds))
                    except Exception:
                        sm = None
                a = mol.GetAtomWithIdx(centre)
                detail[key] = dict(smarts=sm, centre_symbol=a.GetSymbol(),
                                   centre_aromatic=bool(a.GetIsAromatic()),
                                   centre_nH=int(a.GetTotalNumHs()),
                                   centre_degree=int(a.GetTotalDegree()),
                                   n_atoms=len(atoms))
    recs = []
    for (smi, rad), n in frags.most_common():
        d = dict(detail[(smi, rad)])
        d.update(fragment_smiles=smi, radius=rad, count=n)
        recs.append(d)
    return recs, seen


def main():
    featnames = list(np.load(f"{OUT}/featnames.npy"))
    nfeat = len(featnames)
    MDI = np.load(f"{OUT}/mdi.npz")["MDI"]
    raw = np.load(f"{OUT}/chembl_raw.npz")
    FPc, Dc, okc, split, Y, mask = (raw["FP"], raw["D"], raw["ok"], raw["split"],
                                    raw["Y"], raw["mask"])
    cr = np.load(f"{OUT}/cmnpd_raw.npz")
    FPm, Dm, okm = cr["FP"], cr["D"], cr["ok"]
    st = {t: np.load(f"{OUT}/shap_test_{t}.npz") for t in TASKS}
    sm = {t: np.load(f"{OUT}/shap_marine_{t}.npz") for t in TASKS}
    pm = {t: np.load(f"{OUT}/perm_{t}.npz") for t in TASKS}
    Xc = np.load(f"{OUT}/X_chembl.npy", mmap_mode="r")

    imp = {"shap": {t: np.abs(st[t]["sv"]).mean(0) for t in TASKS},
           "shap_marine": {t: np.abs(sm[t]["sv"]).mean(0) for t in TASKS},
           "mdi": {t: MDI[i] for i, t in enumerate(TASKS)}, "perm": {}}
    poolset = set()
    for t in TASKS:
        v = np.zeros(nfeat)
        v[pm[t]["pool"]] = np.clip(pm[t]["drops"].mean(1), 0, None)
        imp["perm"][t] = v
        poolset |= set(pm[t]["pool"].tolist())
    pool = np.array(sorted(poolset))

    def scaled(vals):
        v = np.clip(np.nan_to_num(np.asarray(vals, float)[pool]), 0, None)
        mx = v.max()
        return v / mx if mx > 0 else v

    def ranks(vals):
        v = np.nan_to_num(np.asarray(vals, float)[pool], nan=-np.inf)
        o = np.argsort(-v, kind="stable")
        r = np.empty(len(pool)); r[o] = np.arange(1, len(pool) + 1)
        return r

    per = {m: np.stack([scaled(imp[m][t]) for t in TASKS]) for m in ["shap", "perm", "mdi"]}
    rnk = {m: np.stack([ranks(imp[m][t]) for t in TASKS]) for m in ["shap", "perm", "mdi"]}
    C_primary = np.mean(np.stack([per[m].max(0) for m in per]), axis=0)
    C_mean = np.mean(np.stack([per[m].mean(0) for m in per]), axis=0)
    C_medrank = np.mean(np.stack([np.median(rnk[m], axis=0) for m in rnk]), axis=0)
    pooled_shap = np.mean(np.stack([imp["shap"][t][pool] for t in TASKS]), axis=0)

    rows = []
    for k, fi in enumerate(pool):
        r = dict(feature=featnames[fi], feature_index=int(fi),
                 type="ECFP4_bit" if fi < 2048 else "RDKit_descriptor",
                 consensus_score_primary_maxovertargets=float(C_primary[k]),
                 score_shap_max=float(per["shap"].max(0)[k]),
                 score_perm_max=float(per["perm"].max(0)[k]),
                 score_mdi_max=float(per["mdi"].max(0)[k]),
                 sensitivity_score_meanovertargets=float(C_mean[k]),
                 sensitivity_median_rank_borda=float(C_medrank[k]),
                 mean_abs_shap_pooled=float(pooled_shap[k]))
        for ti, t in enumerate(TASKS):
            r[f"meanabsshap_{t}"] = float(imp["shap"][t][fi])
            r[f"meanabsshap_marine_{t}"] = float(imp["shap_marine"][t][fi])
            r[f"permdrop_{t}"] = float(imp["perm"][t][fi])
            r[f"mdi_{t}"] = float(imp["mdi"][t][fi])
            r[f"rank_shap_{t}"] = int(rnk["shap"][ti, k])
        rows.append(r)
    full = pd.DataFrame(rows).sort_values(
        ["consensus_score_primary_maxovertargets", "mean_abs_shap_pooled"],
        ascending=False).reset_index(drop=True)
    full.insert(0, "consensus_rank", np.arange(1, len(full) + 1))
    full["rank_sensitivity_meanovertargets"] = (
        full["sensitivity_score_meanovertargets"].rank(ascending=False, method="first").astype(int))
    full["rank_sensitivity_median_borda"] = (
        full["sensitivity_median_rank_borda"].rank(ascending=True, method="first").astype(int))
    full.to_csv("feature_importance_full.csv", index=False)
    print("pool", len(pool), "| wrote feature_importance_full.csv", full.shape, flush=True)

    top = full.head(10).reset_index(drop=True)
    print("FINAL TEN:", list(zip(top["feature"], top["type"])), flush=True)

    # ------------------------------------------------------------------ rank stability
    stab_scores = {}
    for s in SEEDS_STAB:
        ok_all = True
        sh, md = {}, {}
        for t in TASKS:
            fs = f"{OUT}/shap_test_{t}.npz" if s == 0 else f"{OUT}/shap_test_{t}_s{s}.npz"
            fm = f"{OUT}/mdi_{t}.npy" if s == 0 else f"{OUT}/mdi_{t}_s{s}.npy"
            if not (os.path.exists(fs) and os.path.exists(fm)):
                ok_all = False; break
            sh[t] = np.abs(np.load(fs)["sv"]).mean(0)
            md[t] = np.load(fm)
        if not ok_all:
            print(f"seed {s} incomplete - skipped in stability", flush=True)
            continue
        a = np.stack([scaled(sh[t]) for t in TASKS]).max(0)
        b = np.stack([scaled(md[t]) for t in TASKS]).max(0)
        stab_scores[s] = (a + b) / 2.0        # SHAP+MDI only; perm exists for seed 0 only
    stab = {}
    if len(stab_scores) > 1:
        seeds = sorted(stab_scores)
        rk = {s: pd.Series(-stab_scores[s]).rank(method="first").to_numpy() for s in seeds}
        pairs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                rho = stats.spearmanr(stab_scores[seeds[i]], stab_scores[seeds[j]]).statistic
                pairs.append((seeds[i], seeds[j], float(rho)))
        stab = dict(seeds=seeds, pairwise_spearman=pairs,
                    per_feature_rank={featnames[pool[k]]: {int(s): int(rk[s][k]) for s in seeds}
                                      for k in range(len(pool))})
        json.dump(stab, open(f"{OUT}/stability.json", "w"), indent=1)
        print("stability pairwise spearman:", pairs, flush=True)

    # ------------------------------------------------------------- per-feature statistics
    dfm = pd.read_parquet(MATRIX, columns=["canonical_smiles", "inchikey"])
    cmdf = pd.read_parquet(CMNPD, columns=["COMPOUND_ID", "inchikey", "smiles_std",
                                           "murcko_scaffold", "np_likeness", "is_indonesia"])
    DESC_NAMES = featnames[2048:]
    out_rows, bitdict_rows = [], []
    for _, rr in top.iterrows():
        fi = int(rr["feature_index"])
        name = rr["feature"]
        is_bit = fi < 2048
        rec = dict(rank=int(rr["consensus_rank"]), feature=name,
                   type="ECFP4 bit (Morgan r=2, 2048-bit)" if is_bit else "RDKit descriptor",
                   consensus_score=float(rr["consensus_score_primary_maxovertargets"]),
                   mean_abs_shap_pooled=float(rr["mean_abs_shap_pooled"]))
        # per-target mean|SHAP|, direction, and driven targets
        drivers, dirs = [], []
        for ti, t in enumerate(TASKS):
            rec[f"mean_abs_shap_{t}"] = float(rr[f"meanabsshap_{t}"])
            rec[f"mean_abs_shap_marine_{t}"] = float(rr[f"meanabsshap_marine_{t}"])
            rec[f"rank_shap_{t}"] = int(rr[f"rank_shap_{t}"])
            sv = st[t]["sv"][:, fi]
            xv = np.asarray(Xc[st[t]["rows"], fi])
            if is_bit:
                pres = xv > 0.5
                d = float(sv[pres].mean()) if pres.sum() >= 5 else np.nan
            else:
                d = float(stats.spearmanr(xv, sv).statistic) if np.std(xv) > 0 else np.nan
            rec[f"direction_{t}"] = d
            if int(rr[f"rank_shap_{t}"]) <= 10:
                drivers.append(t)
                if np.isfinite(d):
                    dirs.append(d)
        rec["targets_driven"] = ";".join(drivers) if drivers else "none in top-10"
        pos = [t for ti, t in enumerate(TASKS) if np.isfinite(rec[f"direction_{t}"])
               and rec[f"direction_{t}"] > 0]
        neg = [t for ti, t in enumerate(TASKS) if np.isfinite(rec[f"direction_{t}"])
               and rec[f"direction_{t}"] < 0]
        rec["direction"] = ("raises predicted activity" if len(pos) > len(neg)
                            else "lowers predicted activity" if len(neg) > len(pos)
                            else "target-dependent")
        rec["direction_detail"] = f"raises for {len(pos)}/6, lowers for {len(neg)}/6 targets"

        # actives vs inactives, pooled over targets (labelled training+val+test rows)
        if is_bit:
            x = FPc[:, fi] > 0
            a_pres = a_abs = i_pres = i_abs = 0
            for ti, t in enumerate(TASKS):
                obs = (mask[:, ti] > 0) & okc
                act = obs & (Y[:, ti] > 0.5); ina = obs & (Y[:, ti] <= 0.5)
                a_pres += int((x & act).sum()); a_abs += int((~x & act).sum())
                i_pres += int((x & ina).sum()); i_abs += int((~x & ina).sum())
            tab = [[a_pres, a_abs], [i_pres, i_abs]]
            orr, pv = stats.fisher_exact(tab)
            rec.update(freq_in_actives=a_pres / max(a_pres + a_abs, 1),
                       freq_in_inactives=i_pres / max(i_pres + i_abs, 1),
                       n_actives=a_pres + a_abs, n_inactives=i_pres + i_abs,
                       effect_size=float(orr), effect_size_kind="odds ratio",
                       p_value=float(pv), test="Fisher exact (2x2, label pairs pooled)")
            rec["n_marine_carrying"] = int((FPm[:, fi] > 0).sum())
            rec["prevalence_marine"] = float((FPm[:, fi] > 0).mean())
            rec["prevalence_chembl_pool"] = float(x.mean())
            # environment recovery from carrier molecules
            carriers = np.where(x & okc)[0]
            mols = []
            for i in carriers[:4000]:
                m = Chem.MolFromSmiles(dfm["canonical_smiles"].iloc[int(i)])
                if m is not None:
                    mols.append(m)
                if len(mols) >= N_ENV_SAMPLE:
                    break
            envs, seen = bit_environments(fi, mols)
            rec["n_distinct_environments"] = len(envs)
            rec["smarts_or_definition"] = envs[0]["smarts"] if envs else None
            rec["modal_fragment_smiles"] = envs[0]["fragment_smiles"] if envs else None
            rec["modal_radius"] = envs[0]["radius"] if envs else None
            rec["modal_env_share"] = (envs[0]["count"] / sum(e["count"] for e in envs)
                                      if envs else np.nan)
            for e in envs[:12]:
                bitdict_rows.append(dict(bit=fi, **e, n_carriers_inspected=seen))
        else:
            di = DESC_NAMES.index(name)
            vc = Dc[:, di].astype(float)
            av, iv = [], []
            for ti, t in enumerate(TASKS):
                obs = (mask[:, ti] > 0) & okc
                av.append(vc[obs & (Y[:, ti] > 0.5)])
                iv.append(vc[obs & (Y[:, ti] <= 0.5)])
            av, iv = np.concatenate(av), np.concatenate(iv)
            u, pv = stats.mannwhitneyu(av[np.isfinite(av)], iv[np.isfinite(iv)],
                                       alternative="two-sided")
            rec.update(mean_in_actives=float(np.nanmean(av)), mean_in_inactives=float(np.nanmean(iv)),
                       sd_in_actives=float(np.nanstd(av, ddof=1)),
                       sd_in_inactives=float(np.nanstd(iv, ddof=1)),
                       n_actives=int(np.isfinite(av).sum()), n_inactives=int(np.isfinite(iv).sum()),
                       effect_size=cohens_d(av, iv), effect_size_kind="Cohen d",
                       p_value=float(pv), test="Mann-Whitney U (label pairs pooled)")
            vm = Dm[:, di].astype(float)
            rec.update(marine_mean=float(np.nanmean(vm)), marine_median=float(np.nanmedian(vm)),
                       marine_sd=float(np.nanstd(vm, ddof=1)),
                       chembl_mean=float(np.nanmean(vc)), chembl_median=float(np.nanmedian(vc)),
                       chembl_sd=float(np.nanstd(vc, ddof=1)),
                       marine_vs_chembl_cohens_d=cohens_d(vm, vc),
                       marine_vs_chembl_p=float(stats.mannwhitneyu(
                           vm[np.isfinite(vm)], vc[np.isfinite(vc)],
                           alternative="two-sided").pvalue))
            # "carrying" for a continuous descriptor: on the activity-favouring side of the
            # median of the ChEMBL ACTIVE compounds
            med_act = float(np.nanmedian(av))
            favour_high = np.nanmean(av) > np.nanmean(iv)
            rec["n_marine_carrying"] = int(np.nansum(vm >= med_act) if favour_high
                                           else np.nansum(vm <= med_act))
            rec["n_marine_carrying_definition"] = (
                f"marine compounds with {name} {'>=' if favour_high else '<='} "
                f"{med_act:.4g} (median of ChEMBL actives; actives skew "
                f"{'high' if favour_high else 'low'})")
            rec["smarts_or_definition"] = DESC_DEF.get(name, "")
            rec["atom_decomposable"] = name in ATOM_DECOMPOSABLE
        if stab:
            rec["rank_stability_per_seed"] = json.dumps(
                stab["per_feature_rank"].get(name, {}))
        out_rows.append(rec)

    t10 = pd.DataFrame(out_rows)
    t10.to_csv("top10_features.csv", index=False)
    pd.DataFrame(bitdict_rows).to_csv("ecfp4_bit_dictionary.csv", index=False)
    print("wrote top10_features.csv", t10.shape,
          "| ecfp4_bit_dictionary.csv", len(bitdict_rows), flush=True)


if __name__ == "__main__":
    main()
