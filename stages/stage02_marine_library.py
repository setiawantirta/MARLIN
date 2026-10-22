#!/usr/bin/env python3
"""Stage 2 - Marine library curation (CMNPD 1.0).

Standardisation, element/size QC, InChIKey de-duplication and property\nannotation of the CMNPD marine natural-product library.

Block II of the published method flowchart (Fig. S1).
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
    number=2, slug="marine_library", title="Marine library curation (CMNPD 1.0)",
    description="""Standardisation, element/size QC, InChIKey de-duplication and property\nannotation of the CMNPD marine natural-product library.""",
    wall_clock="~45 min on 8 cores", hardware="CPU",
    steps=[
    # Only the property table is a hard input. The 2D SDF is opened solely to recover
    # molblocks for SMILES that RDKit fails to parse, so declaring it here made
    # --check report the stage unrunnable over a file it may never read - and under a
    # name the script does not use: the code opens CMNPD_1.0_2d.sdf, while this list
    # said CMNPD_1.0.sdf, so the check and the run disagreed about which file mattered.
    Step("run_curation", script="stages/legacy/stage02/run_curation.py",
         inputs=['CMNPD_1.0_calc_prop.tsv'],
         outputs=['cmnpd_curated.parquet', 'cmnpd_curation_attrition.csv'],
         minutes=45, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
