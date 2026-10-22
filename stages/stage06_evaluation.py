#!/usr/bin/env python3
"""Stage 6 - Evaluation, external validation and model selection.

Internal scaffold-disjoint metrics with DeLong and paired-bootstrap tests,\nthen the three external arms (ZINC decoy enrichment, FDA control recovery,\nmarine phenotypic transfer). The model is selected on the marine arm.

Block VI of the published method flowchart (Fig. S1).
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
    number=6, slug="evaluation", title="Evaluation, external validation and model selection",
    description="""Internal scaffold-disjoint metrics with DeLong and paired-bootstrap tests,\nthen the three external arms (ZINC decoy enrichment, FDA control recovery,\nmarine phenotypic transfer). The model is selected on the marine arm.""",
    wall_clock="~90 min", hardware="CPU",
    steps=[
    Step("internal_metrics", script="src/evaluate.py",
         inputs=[],
         outputs=['benchmark_metrics.csv', 'benchmark_summary.csv', 'significance_tests.csv', 'calibration.csv'],
         minutes=40, requires_network=False),
    Step("ablation_metrics", script="src/eval_ablation.py",
         inputs=[],
         outputs=['ablation_metrics.csv', 'ablation_summary.csv'],
         minutes=10, requires_network=False),
    Step("score_marine_library", script="src/predict_all_cmnpd.py",
         inputs=['cmnpd_curated.parquet'],
         outputs=['cmnpd_predictions.parquet'],
         minutes=25, requires_network=False),
    Step("external_validation", script="src/run_external.py",
         inputs=['fda_positive_controls.csv', 'decoy_set.parquet', 'cmnpd_glioma_cellline_activity.csv'],
         outputs=['enrichment_results.csv', 'fda_control_recovery.csv', 'external_phenotypic_validation.csv', 'benchmark_publication_table.csv'],
         minutes=20, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
