#!/usr/bin/env python3
"""Stage 11 - regenerate every figure in the paper at publication quality.

Eight main figures (Fig. 1-8) and four supplementary figures (Fig. S1-S4), each
emitted as a vector PDF plus a 300 dpi PNG. Figure numbering follows the
UPLOADED manuscript: the convergence figure is supplementary Fig. S4, not a
main figure.

Every builder is run in its own subprocess so that one missing input cannot
take the rest of the stage down, and so that matplotlib state cannot leak
between figures.

    python stages/stage11_figures.py              # all twelve
    python stages/stage11_figures.py --only 3 5   # a subset
    python stages/stage11_figures.py --check      # resolve inputs, render nothing
    python stages/stage11_figures.py --smoke      # two cheap figures only
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from repro.config import CFG, out_dir, have, MANIFEST   # noqa: E402
from repro.stagelib import _bar, _run_streaming

FIGURES = [
    ("1",  "fig01_glioma_ppi_network",          ["string_net.json", "modules.json", "centrality.csv"]),
    # curation_raw.pkl was listed here but fig02 never reads it: the vendor BBB column it
    # supplied is already carried in cmnpd_curated.parquet. Declaring it blocked the figure
    # on an input that is not required.
    ("2",  "fig02_marine_library_properties",   ["cmnpd_curated.parquet", "cmnpd_glioma_cellline_activity.csv"]),
    ("3",  "fig03_internal_benchmark",          ["ens_preds.npz", "benchmark_summary.csv",
                                                 "chembl_multitask_matrix.parquet"]),
    ("4",  "fig04_external_validation",         ["ens_preds.npz", "decoy_roc_inputs.npz", "enrichment_results.csv",
                                                 "fda_control_recovery.csv", "external_phenotypic_validation.csv",
                                                 "marine_holdout_predictions.csv", "cmnpd_predictions.parquet"]),
    ("5",  "fig05_attribution_mechanism",       ["shap_test_EGFR.npz", "shap_marine_EGFR.npz",
                                                 "gnn_vs_rf_attribution.csv", "gnn_rf_atom_attrib.csv",
                                                 "gnnexplainer_atom_attrib.csv"]),
    ("6",  "fig06_screening_funnel",            ["cmnpd_predictions.parquet", "cmnpd_ecfp4.npz",
                                                 "cmnpd_curated.parquet", "models_bundle.tar.gz",
                                                 "chembl_multitask_matrix.parquet"]),
    ("7",  "fig07_feature_statistics",          ["top10_features.csv", "bit_collision_prevalence.csv"]),
    ("8",  "fig08_annotated_structures",        ["fig_exemplars.json", "cmnpd_raw.npz", "chembl_raw.npz",
                                                 "featnames.npy", "top10_features.csv"]),
    ("S1", "figS1_method_flowchart",            []),
    ("S2", "figS2_dataset_overview",            ["chembl_multitask_matrix.parquet", "Y_v2.parquet", "P_v2.parquet",
                                                 "EV_v2.parquet", "chemspace.parquet",
                                                 "applicability_domain.csv", "ad_loo_train.npy"]),
    ("S3", "figS3_polypharmacology_heatmap",    ["cmnpd_predictions.parquet", "cmnpd_curated.parquet",
                                                 "cmnpd_ecfp4.npz", "models_bundle.tar.gz",
                                                 "cmnpd_glioma_cellline_activity.csv"]),
    ("S4", "figS4_convergence",                 []),   # consumes stage 09 outputs, see note below
]

# Figures whose inputs are produced by an earlier stage of THIS module rather
# than being shippable artifacts. They are skipped with an explicit reason when
# that stage has not been run.
STAGE_OUTPUT_INPUTS = {
    "S4": ("stage09_control_convergence",
           ["metrics", "marine194_converged_preds.npz"],
           "Fig. S4 plots the per-epoch validation-loss trajectories of the 15 convergence runs. "
           "Those trajectories exist only inside stage 09's metrics/ directory; no summary table "
           "can substitute for them. Run stage 09 first."),
}
SMOKE_ONLY = {"7", "S1"}


def _missing(names):
    return [n for n in names if not have(n)]


def _s4_ready():
    d = out_dir("stage09_control_convergence")
    need = STAGE_OUTPUT_INPUTS["S4"][1]
    return [n for n in need if not os.path.exists(os.path.join(d, n))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="figure numbers, e.g. 3 5 S2")
    ap.add_argument("--check", action="store_true", help="resolve inputs and exit")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="redraw figures whose output files already exist")
    ap.add_argument("--no-progress", action="store_true",
                    help="disable the tqdm progress bar")
    ap.add_argument("--quiet-steps", action="store_true",
                    help="capture figure output instead of streaming it live")
    args = ap.parse_args()

    sel = FIGURES
    if args.smoke:
        sel = [f for f in FIGURES if f[0] in SMOKE_ONLY]
    if args.only:
        want = {str(x).upper() for x in args.only}
        sel = [f for f in FIGURES if f[0].upper() in want]

    out = out_dir("stage11_figures")
    # A few builders call fig.savefig() with a bare filename of their own before
    # the canonical save_figure() call. Run them in a scratch subdirectory so
    # those side-outputs cannot land among the deliverables.
    scratch = os.path.join(out, "_intermediate")
    os.makedirs(scratch, exist_ok=True)
    report, rc = [], 0
    bar = _bar(sel, len(sel), "stage 11 figures", not args.no_progress)
    for num, stem, inputs in bar:
        # Resume on outputs: a figure whose PNG and PDF are both already in the output
        # directory is not redrawn. Re-running the stage after a dropped session then
        # costs only the figures that are actually missing. --force redraws everything.
        if not args.check and not args.force:
            have_png = os.path.exists(os.path.join(out, stem + ".png"))
            have_pdf = os.path.exists(os.path.join(out, stem + ".pdf"))
            if have_png and have_pdf:
                print(f"Fig {num:>2}  {stem:<36} skipped: already rendered  "
                      f"[--force to redraw]", flush=True)
                report.append(dict(figure=num, stem=stem, status="skipped_already_rendered"))
                continue
        miss = _missing(inputs)
        if num in STAGE_OUTPUT_INPUTS:
            miss += ["stage09:" + m for m in _s4_ready()]
        if args.check:
            print(f"Fig {num:>2}  {stem:<36} {'OK' if not miss else 'MISSING ' + ', '.join(miss)}")
            report.append(dict(figure=num, stem=stem, status="ok" if not miss else "missing_inputs",
                               missing=miss))
            continue
        if miss:
            reason = (STAGE_OUTPUT_INPUTS[num][2] if num in STAGE_OUTPUT_INPUTS
                      else "missing inputs: " + ", ".join(miss))
            print(f"Fig {num:>2}  SKIPPED - {reason}", flush=True)
            report.append(dict(figure=num, stem=stem, status="skipped", reason=reason, missing=miss))
            continue
        t0 = time.time()
        # run in the output directory so a builder's own relative savefig lands there
        cmd = [sys.executable, os.path.join(REPO, "figures", stem + ".py")]
        env = dict(os.environ, PYTHONPATH=REPO, PYTHONUNBUFFERED="1")
        if args.quiet_steps:
            p = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, env=env)
            code, tail = p.returncode, (p.stderr or p.stdout)
        else:
            code, tail = _run_streaming(cmd, cwd=scratch, env=env)
        dt = time.time() - t0
        ok = code == 0 and os.path.exists(os.path.join(out, stem + ".png"))
        print(f"Fig {num:>2}  {stem:<36} {'ok' if ok else 'FAILED'}  {dt:6.1f}s", flush=True)
        if not ok:
            rc = 1
            print(tail[-2500:], file=sys.stderr, flush=True)
        report.append(dict(figure=num, stem=stem, status="ok" if ok else "failed",
                           seconds=round(dt, 1),
                           stderr_tail=("" if ok else tail[-2000:])))
    with open(os.path.join(out, "stage11_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    if args.check:
        nmiss = sum(1 for r in report if r.get("status") == "missing_inputs")
        if nmiss:
            print(f"  {nmiss} of {len(report)} figure(s) cannot be drawn: inputs missing.",
                  file=sys.stderr)
            return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
