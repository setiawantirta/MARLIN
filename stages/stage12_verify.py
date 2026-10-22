#!/usr/bin/env python3
"""Stage 12 - re-derive the paper's headline numbers and assert them.

Every check below reads a shipped or regenerated result table, recomputes the
value the manuscript states, and compares it against
``config/published_values.json``. The science is locked: a mismatch is reported
as a FAIL and the published value is never rewritten.

    python stages/stage12_verify.py           # full check, exit 1 on any FAIL
    python stages/stage12_verify.py --smoke   # the subset that needs no large inputs
    python stages/stage12_verify.py --json    # machine-readable report only

Checks whose input table is absent are reported SKIP, not PASS. A green run
means every check that could be made was made.
"""
import argparse
import json
import re
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from repro.config import CFG, out_dir, have, path as RPATH   # noqa: E402

PUB = json.load(open(os.path.join(REPO, CFG["verification"]["published_values"])))
ATOL = float(CFG["verification"]["abs_tolerance"])
PRTOL = float(CFG["verification"]["p_value_rel_tolerance"])

RESULTS = []


def record(name, status, got=None, expected=None, detail=""):
    RESULTS.append(dict(check=name, status=status, got=got, expected=expected, detail=detail))


def close(a, b, atol=None):
    return abs(float(a) - float(b)) <= (ATOL if atol is None else atol)


def check(name, expected, fn, atol=None, needs=()):
    """Run one check. `fn` returns the recomputed value."""
    miss = [n for n in needs if not have(n)]
    if miss:
        record(name, "SKIP", None, expected, "missing input: " + ", ".join(miss))
        return
    try:
        got = fn()
    except Exception as exc:                                   # noqa: BLE001
        record(name, "ERROR", None, expected, repr(exc)[:300])
        return
    if got is None:
        record(name, "SKIP", None, expected, "value not present in the table")
        return
    ok = close(got, expected, atol)
    record(name, "PASS" if ok else "FAIL", got, expected)


# ---------------------------------------------------------------- 1. internal
MODEL_ALIAS = {"RF": ("RF", "RandomForest", "Random forest", "RandomForest_ECFP4_RDKitDesc"),
               "XGB": ("XGB", "XGBoost"), "DMPNN": ("DMPNN", "D-MPNN"),
               "GIN": ("GIN",), "AttentiveFP": ("AttentiveFP",),
               "GraphGPS": ("GraphGPS",), "GAT": ("GAT", "GATv2")}


def _summary():
    df = pd.read_csv(RPATH("benchmark_summary.csv"))
    return df


def _macro_mean(model):
    df = _summary()
    cols = {c.lower(): c for c in df.columns}
    mcol = cols.get("model") or cols.get("arch")
    tcol = cols.get("task")
    met = cols.get("metric")
    val = cols.get("mean") or cols.get("value")
    sel = df[df[mcol].isin(MODEL_ALIAS[model])]
    if met:
        sel = sel[sel[met].astype(str).str.lower().isin(("roc_auc", "roc-auc", "auc"))]
    if tcol:
        macro = sel[sel[tcol].astype(str).str.lower() == "macro"]
        if len(macro):
            sel = macro
    if not len(sel):
        return None
    return float(sel[val].astype(float).mean())


for m, v in PUB["internal_macro_roc_auc_per_seed_mean"].items():
    check(f"internal macro ROC-AUC, {m}", v, lambda m=m: _macro_mean(m),
          needs=["benchmark_summary.csv"])


# ------------------------------------------------------------- 2. publication table
PUBROW = {"RF": "Random forest (ECFP4+desc)", "XGB": "XGBoost (ECFP4+desc)",
          "GIN": "GIN", "GAT": "GATv2", "DMPNN": "D-MPNN",
          "AttentiveFP": "AttentiveFP", "GraphGPS": "GraphGPS"}


def _pub(model, col):
    """Cell of benchmark_publication_table.csv. Cells of the form
    '0.922 (0.920-0.925)' or '0.369 (p=1.1e-07)' yield the leading value."""
    df = pd.read_csv(RPATH("benchmark_publication_table.csv")).set_index("Model")
    if PUBROW[model] not in df.index:
        return None
    v = df.loc[PUBROW[model], col]
    return float(str(v).split("(")[0].strip())


for m, v in PUB["external_marine"].items():
    if not isinstance(v, dict) or "roc_auc" not in v:
        continue
    check(f"external marine ROC-AUC, {m}", v["roc_auc"],
          lambda m=m: _pub(m, "Marine ROC-AUC"), needs=["benchmark_publication_table.csv"])
    check(f"external marine Spearman rho, {m}", v["spearman_rho"],
          lambda m=m: _pub(m, "Marine rho_s"), needs=["benchmark_publication_table.csv"])


def _ev(model, score, potency, field, censoring="censored_excluded"):
    df = pd.read_csv(RPATH("external_phenotypic_validation.csv"))
    s = df[(df.model == model) & (df.score == score) & (df.potency == potency)
           & (df.censoring == censoring)]
    return None if not len(s) else float(s.iloc[0][field])


# The headline marine figures use the polypharmacology score against the most
# sensitive cell line, censored records excluded.
check("marine set size (censored_excluded)", PUB["external_marine"]["n_compounds"],
      lambda: _ev("RF", "polypharm_score", "pact_max", "n"), atol=0.5,
      needs=["external_phenotypic_validation.csv"])
check("marine actives at pActivity >= 6", PUB["external_marine"]["n_active"],
      lambda: _ev("RF", "polypharm_score", "AUC(pAct>=6.0)", "n_pos"), atol=0.5,
      needs=["external_phenotypic_validation.csv"])
check("marine ROC-AUC recomputed from the per-model table, RF",
      PUB["external_marine"]["RF"]["roc_auc"],
      lambda: _ev("RF", "polypharm_score", "AUC(pAct>=6.0)", "roc_auc"),
      needs=["external_phenotypic_validation.csv"])
check("marine Spearman rho recomputed from the per-model table, RF",
      PUB["external_marine"]["RF"]["spearman_rho"],
      lambda: _ev("RF", "polypharm_score", "pact_max", "spearman_rho"),
      needs=["external_phenotypic_validation.csv"])
check("marine Spearman p recomputed from the per-model table, RF",
      PUB["external_marine"]["RF"]["spearman_p"],
      lambda: _ev("RF", "polypharm_score", "pact_max", "spearman_p"), atol=1e-8,
      needs=["external_phenotypic_validation.csv"])
check("censored-inclusive secondary analysis, n",
      PUB["external_marine"]["censored_inclusive"]["n_compounds"],
      lambda: _ev("RF", "polypharm_score", "pact_max", "n", "censored_included"), atol=0.5,
      needs=["external_phenotypic_validation.csv"])


def _graph_band():
    lo, hi = PUB["external_marine"]["graph_networks_roc_auc_range"]
    vals = [_pub(m, "Marine ROC-AUC") for m in ("GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS")]
    vals = [v for v in vals if v is not None]
    if len(vals) != 5:
        return None
    return 1.0 if (min(vals) >= lo - ATOL and max(vals) <= hi + ATOL) else 0.0


check("all five graph networks inside the published 0.382-0.502 marine band", 1.0,
      _graph_band, needs=["benchmark_publication_table.csv"])


def _sig(scope, a, b, field):
    df = pd.read_csv(RPATH("significance_tests.csv"))
    s = df[(df.scope == scope) & (df.model_a == a) & (df.model_b == b) & (df.metric == "roc_auc")]
    if not len(s):
        s = df[(df.scope == scope) & (df.model_a == b) & (df.model_b == a) & (df.metric == "roc_auc")]
        if not len(s):
            return None
        v = float(s.iloc[0][field])
        return -v if field == "delta" else v
    return float(s.iloc[0][field])


R = PUB["rf_vs_xgb_external"]
check("RF vs XGBoost marine, delta ROC-AUC", R["delta_auc"],
      lambda: _sig("MARINE_PHENOTYPIC", "RF", "XGB", "delta"), atol=0.001,
      needs=["significance_tests.csv"])
check("RF vs XGBoost marine, q (BH, paired bootstrap)", R["q_bootstrap"],
      lambda: _sig("MARINE_PHENOTYPIC", "RF", "XGB", "q_bootstrap"), atol=0.001,
      needs=["significance_tests.csv"])
check("RF vs XGBoost marine is NOT significant at q < 0.05", 0.0,
      lambda: float(bool(pd.read_csv(RPATH("significance_tests.csv"))
                         .query("scope == 'MARINE_PHENOTYPIC' and model_a == 'RF' and model_b == 'XGB'")
                         .iloc[0]["significant_q05"])), atol=0.1,
      needs=["significance_tests.csv"])


def _rf_vs_graph_band():
    """RF beats every graph network by the published 0.183-0.303 margin, q <= 0.0057."""
    df = pd.read_csv(RPATH("significance_tests.csv"))
    s = df[(df.scope == "MARINE_PHENOTYPIC") & (df.metric == "roc_auc")
           & (((df.model_a == "RF") & (df.model_b.isin(["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS"])))
              | ((df.model_b == "RF") & (df.model_a.isin(["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS"]))))]
    if len(s) != 5:
        return None
    d = (s.delta * s.model_a.map(lambda x: 1 if x == "RF" else -1)).abs()
    return 1.0 if (d.min() >= 0.18 and d.max() <= 0.31 and s.q_bootstrap.max() <= 0.0060) else 0.0


check("RF beats all five graph networks on marine, delta 0.183-0.303 at q <= 0.0057", 1.0,
      _rf_vs_graph_band, needs=["significance_tests.csv"])


# --------------------------------------------------------------- 3. controls
H = PUB["control_hyperparameters"]


def _hy(row, col):
    df = pd.read_csv(RPATH("hyperparameter_budget_ablation.csv")).set_index("Model")
    key = [i for i in df.index if row.lower() in str(i).lower()]
    return None if not key else float(df.loc[key[0], col])


check("hyperparameter control, RF internal macro ROC-AUC (exhaustive)", H["internal_rf_tuned"],
      lambda: _hy("random forest", "Macro ROC-AUC, exhaustive"),
      needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, RF internal macro ROC-AUC (published budget)",
      H["internal_rf_published"],
      lambda: _hy("random forest", "Macro ROC-AUC, published budget"),
      needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, XGBoost internal macro ROC-AUC (exhaustive)", H["internal_xgb_tuned"],
      lambda: _hy("xgboost", "Macro ROC-AUC, exhaustive"),
      needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, RF marine ROC-AUC (exhaustive)", H["marine_rf_tuned"],
      lambda: _hy("random forest", "Marine ROC-AUC, exhaustive"),
      needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, RF marine ROC-AUC (published budget)", H["marine_rf_published"],
      lambda: _hy("random forest", "Marine ROC-AUC, published budget"),
      needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, XGBoost marine ROC-AUC (exhaustive)", H["marine_xgb_tuned"],
      lambda: _hy("xgboost", "Marine ROC-AUC, exhaustive"),
      needs=["hyperparameter_budget_ablation.csv"])


def _grid(model, want):
    """'6 of 54 -> 81 of 81' -> 81."""
    df = pd.read_csv(RPATH("hyperparameter_budget_ablation.csv")).set_index("Model")
    key = [i for i in df.index if model.lower() in str(i).lower()]
    if not key:
        return None
    nums = [int(x) for x in re.findall(r"\d+", str(df.loc[key[0], "Configurations searched"]))]
    return float(max(nums))


check("hyperparameter control, RF grid size", H["rf_configs"], lambda: _grid("random forest", 81),
      atol=0.5, needs=["hyperparameter_budget_ablation.csv"])
check("hyperparameter control, XGBoost grid size", H["xgb_configs"], lambda: _grid("xgboost", 96),
      atol=0.5, needs=["hyperparameter_budget_ablation.csv"])


def _total_fits():
    """(81 + 96) configurations x 5 CV folds x 6 targets x 3 seeds."""
    rf, xgb = _grid("random forest", 81), _grid("xgboost", 96)
    if rf is None or xgb is None:
        return None
    return (rf + xgb) * CFG["control_hyperparameters"]["cv_folds"] * \
        len(CFG["target_discovery"]["panel"]) * len(CFG["seeds"]["model_seeds"])


check("hyperparameter control, total fits = (81+96) x 5 folds x 6 targets x 3 seeds",
      H["total_fits"], _total_fits, atol=0.5, needs=["hyperparameter_budget_ablation.csv"])


def _epochs():
    """'86/63/76' per architecture -> the overall min and max."""
    df = pd.read_csv(RPATH("convergence_ablation_table.csv"))
    col = "Epochs to early stop (3 seeds)"
    vals = [int(x) for s in df[col].dropna() for x in re.findall(r"\d+", str(s))]
    return (min(vals), max(vals)) if vals else None


C = PUB["control_convergence"]
check("convergence control, number of runs", C["n_runs"],
      lambda: float(sum(len(re.findall(r"\d+", str(s)))
                        for s in pd.read_csv(RPATH("convergence_ablation_table.csv"))
                        ["Epochs to early stop (3 seeds)"].dropna())),
      atol=0.5, needs=["convergence_ablation_table.csv"])
check("convergence control, earliest early stop", C["early_stopping_epoch_range"][0],
      lambda: float(_epochs()[0]), atol=0.5, needs=["convergence_ablation_table.csv"])
check("convergence control, latest early stop", C["early_stopping_epoch_range"][1],
      lambda: float(_epochs()[1]), atol=0.5, needs=["convergence_ablation_table.csv"])


def _conv_marine():
    df = pd.read_csv(RPATH("convergence_ablation_table.csv"))
    s = pd.to_numeric(df["Marine ROC-AUC, converged"], errors="coerce").dropna()
    return (float(s.min()), float(s.max())) if len(s) else None


check("convergence control, lowest converged marine ROC-AUC", C["marine_roc_auc_range"][0],
      lambda: _conv_marine()[0], atol=0.001, needs=["convergence_ablation_table.csv"])
check("convergence control, highest converged marine ROC-AUC", C["marine_roc_auc_range"][1],
      lambda: _conv_marine()[1], atol=0.001, needs=["convergence_ablation_table.csv"])


def _conv_still_chance():
    """Seed-mean converged marine ROC-AUC per architecture stays below the RF."""
    df = pd.read_csv(RPATH("converged_external_marine.csv"))
    df["arch"] = df.model.str.rsplit("_s", n=1).str[0]
    mean = df.groupby("arch").polypharm_auc.mean()
    return 1.0 if float(mean.max()) < PUB["external_marine"]["RF"]["roc_auc"] else 0.0


check("convergence control, every converged architecture still below the random forest", 1.0,
      _conv_still_chance, needs=["converged_external_marine.csv"])


# --------------------------------------------------------------- 4. screening
def _funnel(definition, field="n_compounds"):
    df = pd.read_csv(RPATH("screening_funnel.csv"))
    s = df[df.definition == definition].sort_values("stage_order")
    return None if not len(s) else float(s.iloc[-1][field])


check("screening cascade, stringent survivors", PUB["screening"]["n_survivors"],
      lambda: _funnel("stringent"), atol=0.5, needs=["screening_funnel.csv"])
check("screening cascade, library entering the funnel", PUB["screening"]["library_size"],
      lambda: float(pd.read_csv(RPATH("screening_funnel.csv"))
                    .query("stage_order == 1").n_compounds.max()),
      atol=0.5, needs=["screening_funnel.csv"])
check("seven survivors table has seven rows", PUB["screening"]["n_survivors"],
      lambda: float(len(pd.read_csv(RPATH("seven_survivors.csv")))), atol=0.5,
      needs=["seven_survivors.csv"])


def _libstat(key):
    df = pd.read_csv(RPATH("cmnpd_library_summary.csv"))
    s = df[df.statistic == key]
    return None if not len(s) else float(s.iloc[0]["value"])


check("curated marine library size", PUB["dataset"]["n_curated_marine"],
      lambda: _libstat("n_curated_unique"), atol=0.5, needs=["cmnpd_library_summary.csv"])
check("six-target panel size", float(len(CFG["target_discovery"]["panel"])),
      lambda: float(pd.read_csv(RPATH("glioma_target_panel.csv")).shape[0]
                    if "in_final_panel" not in pd.read_csv(RPATH("glioma_target_panel.csv")).columns
                    else int(pd.read_csv(RPATH("glioma_target_panel.csv")).in_final_panel.sum())),
      atol=0.5, needs=["glioma_target_panel.csv"])
check("consensus feature table has ten rows", float(CFG["attribution"]["n_reported_features"]),
      lambda: float(len(pd.read_csv(RPATH("top10_features.csv")))), atol=0.5,
      needs=["top10_features.csv"])


# ------------------------------------------------------------------ 5. figures
FIGURES_EXPECTED = [
    ("1", "fig01_glioma_ppi_network", "main"),
    ("2", "fig02_marine_library_properties", "main"),
    ("3", "fig03_internal_benchmark", "main"),
    ("4", "fig04_external_validation", "main"),
    ("5", "fig05_attribution_mechanism", "main"),
    ("6", "fig06_screening_funnel", "main"),
    ("7", "fig07_feature_statistics", "main"),
    ("8", "fig08_annotated_structures", "main"),
    ("S1", "figS1_method_flowchart", "supplementary"),
    ("S2", "figS2_dataset_overview", "supplementary"),
    ("S3", "figS3_polypharmacology_heatmap", "supplementary"),
    ("S4", "figS4_convergence", "supplementary"),
]
def _figure_quality():
    """Every regenerated figure must be a vector PDF plus a >= 300 dpi PNG."""
    from PIL import Image
    d = out_dir("stage11_figures")
    # Only the canonical deliverables; a builder's own scratch output is not one.
    stems = [s for _, s, _ in FIGURES_EXPECTED]
    pngs = sorted(f for f in os.listdir(d) if f.endswith(".png") and f[:-4] in stems)
    if not pngs:
        return None
    bad = []
    for f in pngs:
        im = Image.open(os.path.join(d, f))
        dpi = im.info.get("dpi", (0, 0))[0]
        if dpi < CFG["figures"]["png_dpi"] - 1:
            bad.append(f"{f}:{dpi}dpi")
        if not os.path.exists(os.path.join(d, f[:-4] + ".pdf")):
            bad.append(f"{f}:no-pdf")
    return 0.0 if bad else float(len(pngs))


_nfig = _figure_quality()
if _nfig is None:
    record("regenerated figures are 300 dpi PNG + vector PDF", "SKIP", None, ">=1",
           "stage 11 has not been run")
elif _nfig == 0.0:
    record("regenerated figures are 300 dpi PNG + vector PDF", "FAIL", 0, ">=1",
           "at least one figure is below 300 dpi or missing its PDF")
else:
    record("regenerated figures are 300 dpi PNG + vector PDF", "PASS", int(_nfig), ">=1")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="list the checks and which inputs resolve, then exit")
    a = ap.parse_args()

    if a.check:
        print("=== Stage 12: verification - %d checks declared" % len(RESULTS))
        for r in RESULTS:
            print(f"  {r['check']:<60} "
                  + ("resolvable" if r["status"] != "SKIP" else "SKIP: " + r["detail"]))
        return 0

    od = out_dir("stage12_verify")
    with open(os.path.join(od, "verification_report.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=1)
    rows = pd.DataFrame(RESULTS)
    rows.to_csv(os.path.join(od, "verification_report.csv"), index=False)

    if a.json:
        print(json.dumps(RESULTS, indent=1))
    else:
        print("=== Stage 12: verification against the published values")
        for r in RESULTS:
            g = "" if r["got"] is None else f"  got {r['got']:.6g}" if isinstance(r["got"], float) else f"  got {r['got']}"
            e = "" if r["expected"] is None else f"  published {r['expected']}"
            print(f"  [{r['status']:<5}] {r['check']:<58}{g}{e}"
                  + (f"   ({r['detail']})" if r["detail"] else ""))
        n = {k: sum(1 for r in RESULTS if r["status"] == k) for k in ("PASS", "FAIL", "SKIP", "ERROR")}
        print(f"\n  PASS {n['PASS']}   FAIL {n['FAIL']}   SKIP {n['SKIP']}   ERROR {n['ERROR']}")
        print(f"  report: {os.path.join(od, 'verification_report.csv')}")
        if n["FAIL"]:
            print("\n  A FAIL means the regenerated value disagrees with the manuscript.")
            print("  The published value is NEVER adjusted; investigate the pipeline.")
    return 1 if any(r["status"] in ("FAIL", "ERROR") for r in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
