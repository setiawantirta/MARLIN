import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import pandas as pd
import numpy as np
import json
import tarfile
import joblib
import os
import time
from itertools import combinations
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED
RDLogger.DisableLog('rdApp.*')

pred = pd.read_parquet(RPATH("cmnpd_predictions.parquet"))
cur = pd.read_parquet(RPATH("cmnpd_curated.parquet"))

X_fp = sp.load_npz(RPATH("cmnpd_ecfp4.npz")).tocsr()

DESC_NAMES = ["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds","RingCount",
 "NumAromaticRings","FractionCSP3","HeavyAtomCount","NHOHCount","NOCount","NumHeteroatoms","MolMR",
 "BalabanJ","BertzCT","Chi0v","Chi1v","Chi2v","Chi3v","Chi4v","Kappa1","Kappa2","Kappa3","LabuteASA",
 "HallKierAlpha","NumSpiroAtoms","NumBridgeheadAtoms","NumSaturatedRings","QED"]
fns = dict(Descriptors.descList)
getters = [(n, (QED.qed if n=="QED" else fns[n])) for n in DESC_NAMES]

def desc_row(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return [np.nan]*30
    out=[]
    for n,f in getters:
        try: out.append(float(f(m)))
        except Exception: out.append(np.nan)
    return out

smis = pred['smiles_std'].tolist()
t0=time.time()
rows=[]
for i,s in enumerate(smis):
    rows.append(desc_row(s))
    if i and i%8000==0: print(i, round(time.time()-t0,1), flush=True)
D = np.array(rows, dtype=np.float64)
print(D.shape, "nan cells:", int(np.isnan(D).sum()), "rows any nan:", int(np.isnan(D).any(1).sum()), round(time.time()-t0,1))

T = "bundle_models.tar.gz"
# We need the bundle - re-extract if needed
# The trace used a local path; we need to handle this
# Actually the bundle was extracted to "bundle/" directory in the trace
# We need to re-extract from the tar - but this artifact isn't mapped
# The trace extracted it locally; let's just load what we need

# Load meta from bundle
os.makedirs("bundle", exist_ok=True)

# Since we can't reference the tar artifact (not in dependency mappings),
# we need to work with what's available. Let's check if bundle exists.
# The trace extracted to "bundle/" - we'll try to use it directly.

meta = json.load(open("bundle/glioma_models/models/rf_final/rf_meta.json"))

desc_mean=np.array(meta['desc_mean']); desc_std=np.array(meta['desc_std']); desc_med=np.array(meta['desc_median'])
Dz = (np.where(np.isnan(D), desc_med, D) - desc_mean)/desc_std
X = np.hstack([X_fp.toarray().astype(np.float32), Dz.astype(np.float32)])
print(X.shape, X.dtype, X.nbytes/1e6)

TARGETS=["EGFR","PDGFRA","PIK3CA","KDR","CDK6","IDH1"]
THR = {"EGFR":0.6266666666666666,"PDGFRA":0.49333333333333335,"PIK3CA":0.6733333333333333,
       "KDR":0.62,"CDK6":0.28,"IDH1":0.5266666666666667}

res={}
for t in TARGETS:
    m = joblib.load(f"bundle/glioma_models/models/rf_final/rf_{t}_s0.joblib")
    pos = list(m.classes_).index(1)
    tp = np.empty((X.shape[0], len(m.estimators_)), dtype=np.float32)
    for j,est in enumerate(m.estimators_):
        tp[:,j] = est.predict_proba(X)[:,pos]
    res[t] = (tp.mean(1), tp.std(1))
    print(t, "seed0-mean vs stored corr:", round(float(np.corrcoef(res[t][0], pred['prob_'+t])[0,1]),4),
          "| mean tree-SD", round(float(res[t][1].mean()),4), flush=True)
    del m, tp

J = pred.merge(cur.drop(columns=[c for c in cur.columns if c in pred.columns and c!='inchikey']),
               left_on='inchikey', right_on='inchikey', how='left', validate='1:1')
print("joined:", J.shape)
assert (J['cmnpd_id'].values==pred['cmnpd_id'].values).all()

for t in TARGETS:
    J[f'treeSD_{t}'] = res[t][1]
J['mean_treeSD'] = J[[f'treeSD_{t}' for t in TARGETS]].mean(1)

p95 = {t: float(pred[f'prob_{t}'].quantile(0.95)) for t in TARGETS}
STRICT = {t: max(THR[t], p95[t]) for t in TARGETS}
print("nominal vs library-p95 vs strict threshold:")
for t in TARGETS: print(f"  {t}: nominal {THR[t]:.3f}  p95 {p95[t]:.3f}  strict {STRICT[t]:.3f}  n_strict {(pred['prob_'+t]>=STRICT[t]).sum()}")
hit_nom = pd.DataFrame({t: (J[f'prob_{t}']>=THR[t]) for t in TARGETS})
hit_str = pd.DataFrame({t: (J[f'prob_{t}']>=STRICT[t]) for t in TARGETS})
J['n_targets_strict'] = hit_str.sum(1)
J['n_targets_nominal'] = hit_nom.sum(1)

def combo_table(hits, minn=2):
    lab = hits.apply(lambda r: "+".join([t for t in TARGETS if r[t]]), axis=1)
    sub = lab[hits.sum(axis=1)>=minn]
    return sub.value_counts()
cs = combo_table(hit_str); cn = combo_table(hit_nom)
print("STRINGENT exact combinations (>=2 targets), top 15:"); print(cs.head(15).to_string())
print("\nNOMINAL exact combinations (>=2), top 10:"); print(cn.head(10).to_string())
pair_rows=[]
for a,b in combinations(TARGETS,2):
    n_s=int((hit_str[a]&hit_str[b]).sum()); n_n=int((hit_nom[a]&hit_nom[b]).sum())
    exp_s = float(hit_str[a].mean()*hit_str[b].mean()*len(J))
    pair_rows.append(dict(pair=f"{a}+{b}", n_stringent=n_s, n_nominal=n_n,
                          expected_if_independent_stringent=round(exp_s,1),
                          enrichment_stringent=round(n_s/exp_s,2) if exp_s>0 else np.nan))
pairs=pd.DataFrame(pair_rows).sort_values('n_stringent',ascending=False)
print("\npairwise co-prediction (stringent):"); print(pairs.to_string(index=False))

cr=[]
for lbl,cc in [('stringent',cs),('nominal',cn)]:
    for combo,n in cc.items():
        cr.append(dict(definition=lbl, kind='exact_combination', combination=combo,
                       n_targets=combo.count('+')+1, n_compounds=int(n)))
for _,r in pairs.iterrows():
    cr.append(dict(definition='stringent', kind='pairwise', combination=r['pair'], n_targets=2,
                   n_compounds=int(r['n_stringent']), expected_if_independent=r['expected_if_independent_stringent'],
                   enrichment_over_independence=r['enrichment_stringent'], n_compounds_nominal=int(r['n_nominal'])))
comb=pd.DataFrame(cr); comb.to_csv("polypharmacology_combinations.csv", index=False)
print("\ncombinations table:", comb.shape)
print(comb[comb.kind=='exact_combination'].query("definition=='stringent'").head(8).to_string(index=False))