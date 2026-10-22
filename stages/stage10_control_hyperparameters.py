#!/usr/bin/env python3
"""Stage 10 - hyperparameter-budget control experiment.

The published baselines were chosen by a 6-point random search per target and
seed (36 fits in total). This control replaces that with an exhaustive grid -
81 random-forest and 96 XGBoost configurations under StratifiedKFold(5), tuned
independently for all three seeds, 15,930 fits - to establish that the
fingerprint ensembles' advantage is not a search-budget artefact.

Published result: internal macro ROC-AUC 0.922 -> 0.923 (RF) and 0.920 -> 0.922
(XGBoost); marine ROC-AUC 0.685 -> 0.682 and 0.640 -> 0.649; tuned-versus-
published random-forest dAUC -0.0027, p = 0.40.

The legacy driver carries per-configuration checkpoint/resume: killing it and
restarting resumes from the last completed configuration. That behaviour is
preserved here deliberately - do not delete outputs/stage10_.../ckpt/ between
restarts.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from repro.config import CFG, out_dir, have      # noqa: E402

LEGACY = os.path.join(HERE, "legacy", "stage10")
INPUTS = ["chembl_multitask_matrix.parquet", "cmnpd_curated.parquet",
          "cmnpd_glioma_cellline_activity.csv", "decoy_set.parquet",
          "applicability_domain.csv", "marine_holdout.csv", "fda_positive_controls.csv"]
WALL = ("15,930 fits. ~30 h on 8 CPU cores; ~5 h with the cuML screening arm on one GPU. "
        "Checkpointed per configuration - safe to interrupt and resume.")


def main():
    smoke = "--smoke" in sys.argv
    check = "--check" in sys.argv
    c = CFG["control_hyperparameters"]
    print("=== Stage 10: hyperparameter-budget control "
          f"({c['rf_grid_size']} RF + {c['xgb_grid_size']} XGB configurations, "
          f"{c['cv']}({c['cv_folds']}), {c['total_fits']} fits)")
    print("    hardware: CPU reference path, GPU optional.  wall clock: " + WALL)
    missing = [i for i in INPUTS if not have(i)]
    if check:
        for i in INPUTS:
            print(f"  {i:<42} {'OK' if have(i) else 'MISSING'}")
        print("  driver:", os.path.join(LEGACY, "run_tuning.sh"))
        return 0
    if missing:
        print("  NOT RUN: missing " + ", ".join(missing), file=sys.stderr)
        return 2

    od = out_dir("stage10_control_hyperparameters")
    script = "smoke_test_baselines.py" if smoke else None
    if script:
        p = subprocess.run([sys.executable, os.path.join(LEGACY, script)],
                           cwd=od, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=REPO))
        print(p.stdout[-2000:])
        if p.returncode:
            print(p.stderr[-2000:], file=sys.stderr)
        return p.returncode
    p = subprocess.run(["bash", os.path.join(LEGACY, "run_tuning.sh")], cwd=od,
                       env=dict(os.environ, PYTHONPATH=REPO, TUNE_OUT=od))
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
