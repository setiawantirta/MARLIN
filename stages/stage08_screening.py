#!/usr/bin/env python3
"""Stage 8 - Marine screening cascade.

The 26,754 curated marine compounds are scored on all six targets by the\nselected random forest, then filtered: >= 2 targets active, inside the 0.30\nECFP4 applicability domain, blood-brain barrier not excluded, PAINS-free.

Block VIII of the published method flowchart (Fig. S1).
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
    number=8, slug="screening", title="Marine screening cascade",
    description="""The 26,754 curated marine compounds are scored on all six targets by the\nselected random forest, then filtered: >= 2 targets active, inside the 0.30\nECFP4 applicability domain, blood-brain barrier not excluded, PAINS-free.""",
    wall_clock="~25 min", hardware="CPU",
    steps=[
    Step("polypharmacology_shortlist", script="stages/legacy/stage08/01_polypharmacology_shortlist.py",
         inputs=['cmnpd_predictions.parquet', 'cmnpd_curated.parquet', 'cmnpd_ecfp4.npz', 'models_bundle.tar.gz'],
         outputs=['polypharmacology_combinations.csv', 'screening_funnel.csv', 'seven_survivors.csv', 'cmnpd_top_candidates.csv'],
         minutes=25, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
