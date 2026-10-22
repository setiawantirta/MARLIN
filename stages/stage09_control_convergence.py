#!/usr/bin/env python3
"""Stage 9 - convergence control experiment.

The published graph networks were trained under a 30-epoch ceiling with early
stopping patience 6. This control retrains all five architectures x three seeds
with the CEILING raised to 1,000 and the patience to 25, at reference capacity,
and re-measures marine transfer, to establish that the out-of-distribution
failure is not an under-training artefact.

"Ceiling 1,000" is not "train for 1,000 epochs". Early stopping still decides
when each run ends, and it fired in every one of the fifteen runs between epochs
34 and 86. Forcing a literal 1,000 epochs would train past the validation minimum
- the AttentiveFP, GraphGPS and D-MPNN curves rise after theirs - and would
reproduce no published number.

Published result: early stopping fired in all 15 runs between epochs 34 and 86,
and marine ROC-AUC stayed in 0.394-0.538 - still chance.

IMPORTANT. This stage writes its per-epoch trajectories into
``outputs/stage09_control_convergence/metrics/``. Figure S4 plots exactly those
trajectories and nothing else can substitute for them, so the directory must be
preserved (the original run lost it to a scratch sweep, which is why the
shipped package cannot regenerate Fig. S4 until this stage is run).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from repro.config import CFG, out_dir, have      # noqa: E402

LEGACY = os.path.join(HERE, "legacy", "stage09")
INPUTS = ["chembl_multitask_matrix.parquet", "cmnpd_curated.parquet",
          "cmnpd_glioma_cellline_activity.csv", "decoy_set.parquet",
          "applicability_domain.csv", "marine_holdout.csv", "fda_positive_controls.csv"]
WALL = ("15 runs. ~40 h on 8 CPU cores at 3 concurrent workers; "
        "~4 h on one NVIDIA A10G or better. This is the single most expensive stage.")


def main():
    smoke = "--smoke" in sys.argv
    check = "--check" in sys.argv
    print("=== Stage 9: convergence control (max %d epochs, patience %d)"
          % (CFG["control_convergence"]["epochs_max"], CFG["control_convergence"]["patience"]))
    print("    hardware: CPU reference path, GPU optional.  wall clock: " + WALL)
    missing = [i for i in INPUTS if not have(i)]
    if check:
        for i in INPUTS:
            print(f"  {i:<42} {'OK' if have(i) else 'MISSING'}")
        print("  driver:", os.path.join(LEGACY, "run_convergence.sh"))
        return 0
    # The smoke path builds synthetic batches and reads none of the study inputs, so
    # it must not be gated on them: a reviewer's first question is "does this machine
    # train these five architectures at all, and how fast", and that is answerable
    # before a single data file has been downloaded.
    if missing and not smoke:
        print("  NOT RUN: missing " + ", ".join(missing), file=sys.stderr)
        return 2

    od = out_dir("stage09_control_convergence")
    os.makedirs(os.path.join(od, "metrics"), exist_ok=True)
    if smoke:
        if missing:
            print("  (smoke path uses synthetic batches; the %d missing study inputs "
                  "are not read)" % len(missing))
        p = subprocess.run([sys.executable, os.path.join(LEGACY, "smoke_test.py")],
                           cwd=od, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=REPO))
        print(p.stdout[-2000:])
        if p.returncode:
            print(p.stderr[-2000:], file=sys.stderr)
        return p.returncode

    env = dict(os.environ, PYTHONPATH=REPO, CONV_OUT=od,
               EPOCHS=str(CFG["control_convergence"]["epochs_max"]),
               PATIENCE=str(CFG["control_convergence"]["patience"]))
    p = subprocess.run(["bash", os.path.join(LEGACY, "run_convergence.sh")],
                       cwd=od, env=env)
    # Archive the trajectories Fig. S4 needs. The original run lost these.
    print("  metrics/ retained at", os.path.join(od, "metrics"))
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
