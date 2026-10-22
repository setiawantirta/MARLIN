#!/usr/bin/env python3
"""Stage 3 - ChEMBL label matrix and the scaffold-disjoint split.

Two-pass ChEMBL activity retrieval, filtering, censored-bound recovery,\nstandardisation, the six-task label matrix and the Bemis-Murcko split.

Block III of the published method flowchart (Fig. S1).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
from repro.config import CFG, out_dir, have, path as RPATH, MANIFEST   # noqa: E402
from repro.stagelib import Stage                                        # noqa: E402

from repro.stagelib import Step

STAGE = Stage(
    number=3, slug="label_matrix", title="ChEMBL label matrix and the scaffold-disjoint split",
    description="""Two-pass ChEMBL activity retrieval, filtering, censored-bound recovery,\nstandardisation, the six-task label matrix and the Bemis-Murcko split.""",
    wall_clock="~3 h (dominated by ChEMBL retrieval)", hardware="CPU, network",
    steps=[
    Step("fetch_chembl_activities", script="stages/legacy/stage03/fetch_chembl_activities.py",
         inputs=[],
         outputs=[],
         minutes=90, requires_network=True),
    # ORDER IS LOAD-BEARING. fetch_molecules reads work/agg_per_compound_target.csv.gz,
    # which filter_activities writes (filter_activities.py line 84 -> fetch_molecules.py
    # line 28). Declaring fetch_molecules first made a clean run fail on its first step
    # after retrieval. Do not swap these back.
    Step("filter_activities", script="stages/legacy/stage03/filter_activities.py",
         inputs=[],
         outputs=[],
         minutes=3, requires_network=False),
    Step("fetch_molecules", script="stages/legacy/stage03/fetch_molecules.py",
         inputs=[],
         outputs=[],
         minutes=20, requires_network=True),
    Step("fetch_censored", script="stages/legacy/stage03/fetch_censored.py",
         inputs=[],
         outputs=[],
         minutes=25, requires_network=True),
    Step("recover_censored", script="stages/legacy/stage03/recover_censored.py",
         inputs=[],
         outputs=[],
         minutes=3, requires_network=False),
    Step("standardize", script="stages/legacy/stage03/standardize.py",
         inputs=[],
         outputs=[],
         minutes=20, requires_network=False),
    Step("build_matrix_v2", script="stages/legacy/stage03/build_matrix_v2.py",
         inputs=[],
         outputs=['chembl_multitask_matrix.parquet'],
         minutes=5, requires_network=False),
    Step("project_chemspace", script="stages/legacy/stage03/project.py",
         inputs=[],
         outputs=['chemspace.parquet'],
         minutes=10, requires_network=False),
    Step("applicability_domain", script="stages/legacy/stage03/ad_umap.py",
         inputs=[],
         outputs=['applicability_domain.csv', 'ad_loo_train.npy'],
         minutes=25, requires_network=False),
    Step("finalize", script="stages/legacy/stage03/finalize_v2.py",
         inputs=['CMNPD_1.0_calc_prop.tsv'],
         outputs=['Y_v2.parquet', 'P_v2.parquet', 'EV_v2.parquet'],
         minutes=10, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
