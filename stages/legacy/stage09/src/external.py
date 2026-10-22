"""External validation: marine phenotypic correlation, ZINC decoy enrichment,
FDA control recovery and the CMNPD marine holdout."""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata, eval_utils as E
from feat import TASKS


# --------------------------------------------------------------------------- loading
def load_core():
    df = pd.read_parquet(mdata.MATRIX)
    cm = pd.read_parquet(mdata.CMNPD)
    cl = pd.read_csv(mdata.CELLLINE)
    ad = pd.read_csv(mdata.AD)
    return df, cm, cl, ad


def cmnpd_pred_files():
    out = {}
    for p in sorted(glob.glob("metrics/cmnpd_pred_*.npz")):
        out[os.path.basename(p)[len("cmnpd_pred_"):-len(".npz")]] = p
    return out


def cmnpd_matrix(path, n):
    z = np.load(path)
    P = np.full((n, len(TASKS)), np.nan, dtype=np.float64)
    P[z["I"]] = z["P"]
    return P


def arch_of(name):
    return name.split("_s")[0]


def seed_mean(paths, n):
    Ps = [cmnpd_matrix(p, n) for p in paths]
    return np.nanmean(np.stack(Ps), axis=0)


# ------------------------------------------------------- (d) phenotypic validation
def phenotypic_table(cm, cl, censored_mode="exclude"):
    """Per-compound measured anti-glioma potency from the CMNPD cell-line records."""
    c = cl.copy()
    if censored_mode == "exclude":
        c = c[~c["censored"].astype(bool)]
    c = c[c["pActivity"].notna()]
    idx = {v: i for i, v in enumerate(cm["COMPOUND_ID"].tolist())}
    c["row"] = c["COMPOUND_ID"].map(idx)
    c = c[c["row"].notna()]
    c["row"] = c["row"].astype(int)
    g = c.groupby("row").agg(pact_max=("pActivity", "max"), pact_med=("pActivity", "median"),
                             n_rec=("pActivity", "size"),
                             n_cell=("CELL_LINE", "nunique")).reset_index()
    return g, c


def phenotypic_metrics(P, g, label="", cutoff=6.0):
    rows = []
    sel = g[np.isfinite(P[g["row"].to_numpy()]).all(axis=1)]
    r = sel["row"].to_numpy()
    maxp = np.nanmax(P[r], axis=1)
    poly = np.nansum(P[r], axis=1)
    y = (sel["pact_max"].to_numpy() >= cutoff).astype(int)
    for score_name, s in [("max_prob", maxp), ("polypharm_score", poly)]:
        for pot_name, pot in [("pact_max", sel["pact_max"].to_numpy()),
                              ("pact_median", sel["pact_med"].to_numpy())]:
            rho, pv = stats.spearmanr(s, pot)
            rows.append(dict(model=label, score=score_name, potency=pot_name,
                             spearman_rho=float(rho), spearman_p=float(pv), n=int(len(s))))
        if len(np.unique(y)) > 1:
            rows.append(dict(model=label, score=score_name, potency=f"AUC(pAct>={cutoff})",
                             spearman_rho=np.nan, spearman_p=np.nan, n=int(len(s)),
                             roc_auc=float(roc_auc_score(y, s)), n_pos=int(y.sum())))
    return rows


# ------------------------------------------------------------- (a) decoy enrichment
def enrichment(df, dec, Ptest, Pdecoy, mask, Y, split):
    rows = []
    for ti, t in enumerate(TASKS):
        act = np.where((split == "test") & (mask[:, ti] > 0) & (Y[:, ti] > 0.5) &
                       np.isfinite(Ptest[:, ti]))[0]
        dsel = np.where((dec["gene"].to_numpy() == t) & np.isfinite(Pdecoy[:, ti]))[0]
        if len(act) == 0 or len(dsel) == 0:
            continue
        s = np.concatenate([Ptest[act, ti], Pdecoy[dsel, ti]])
        y = np.concatenate([np.ones(len(act)), np.zeros(len(dsel))]).astype(int)
        rows.append(dict(task=t, n_actives=len(act), n_decoys=len(dsel),
                         ratio=round(len(dsel) / len(act), 2),
                         roc_auc=float(roc_auc_score(y, s)),
                         bedroc20=float(E.bedroc(y, s, 20.0)),
                         ef1=float(E.enrichment_factor(y, s, 0.01)),
                         ef5=float(E.enrichment_factor(y, s, 0.05))))
    return rows


# ------------------------------------------------------------ (b) FDA control recovery
def fda_recovery(df, Ptest, mask, Y, split, label=""):
    fda = pd.read_csv(mdata.FDA)
    key = {k: i for i, k in enumerate(df["inchikey"].tolist())}
    rows = []
    for _, r in fda.iterrows():
        i = key.get(r["inchikey"])
        t = r["gene"]
        if i is None or t not in TASKS:
            continue
        ti = TASKS.index(t)
        p = Ptest[i, ti]
        pool = Ptest[np.where((split == "test") & np.isfinite(Ptest[:, ti]))[0], ti]
        rows.append(dict(model=label, gene=t, drug=r["drug_names"], inchikey=r["inchikey"],
                         label_on_target=r.get("label_on_target"),
                         measured_pchembl=r.get("measured_pchembl"),
                         predicted_prob=float(p) if np.isfinite(p) else np.nan,
                         rank_percentile=float((pool <= p).mean() * 100) if np.isfinite(p) else np.nan,
                         n_test_pool=int(len(pool))))
    return rows


# --------------------------------------------------------------- (c) marine holdout
def marine_holdout(df, Ptest, label=""):
    mh = pd.read_csv(mdata.MARINE_HOLDOUT)
    key = {k: i for i, k in enumerate(df["inchikey"].tolist())}
    rows = []
    for _, r in mh.iterrows():
        i = key.get(r["inchikey"])
        if i is None:
            continue
        for ti, t in enumerate(TASKS):
            lab = r.get(f"label_{t}")
            if pd.isna(lab):
                continue
            rows.append(dict(model=label, compound=r["cmnpd_name"], cmnpd_id=r["cmnpd_compound_id"],
                             inchikey=r["inchikey"], task=t, label=int(lab),
                             pchembl=r.get(f"pchembl_{t}"),
                             predicted_prob=float(Ptest[i, ti]) if np.isfinite(Ptest[i, ti]) else np.nan))
    return rows
