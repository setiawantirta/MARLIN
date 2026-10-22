#!/usr/bin/env python3
"""Stage 4 - Control-set assembly: approved drugs and property-matched decoys.

Positive controls are ChEMBL max_phase 4 drug-target pairs forced into the\ntest partition; negative controls are ZINC20 decoys matched on six DUD-E\nproperties with ECFP4 Tanimoto < 0.35 to every active.

Block IV of the published method flowchart (Fig. S1).
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
    number=4, slug="controls", title="Control-set assembly: approved drugs and property-matched decoys",
    description="""Positive controls are ChEMBL max_phase 4 drug-target pairs forced into the\ntest partition; negative controls are ZINC20 decoys matched on six DUD-E\nproperties with ECFP4 Tanimoto < 0.35 to every active.""",
    wall_clock="~70 min", hardware="CPU, network",
    steps=[
    Step("fetch_zinc_pool", script="stages/legacy/stage04/fetch_zinc_pool.py",
         inputs=[],
         outputs=[],
         minutes=40, requires_network=True),
    Step("build_decoys", script="stages/legacy/stage04/decoys.py",
         inputs=['chembl_multitask_matrix.parquet'],
         outputs=['decoy_set.parquet'],
         minutes=25, requires_network=False),
    Step("putative_inactives", script="stages/legacy/stage04/putative_decoys.py",
         inputs=['decoy_set.parquet'],
         outputs=[],
         minutes=5, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
