#!/usr/bin/env python3
"""Stage 7 - Consensus feature attribution.

Exact path-dependent TreeSHAP, permutation importance and mean decrease in\nimpurity are computed independently and combined into a consensus rank; the\ntop ten features are then annotated structurally and compared against the\ngraph networks' integrated-gradients attribution.

Block VII of the published method flowchart (Fig. S1).
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
    number=7, slug="attribution", title="Consensus feature attribution",
    description="""Exact path-dependent TreeSHAP, permutation importance and mean decrease in\nimpurity are computed independently and combined into a consensus rank; the\ntop ten features are then annotated structurally and compared against the\ngraph networks' integrated-gradients attribution.""",
    wall_clock="~4 h", hardware="CPU",
    steps=[
    Step("prepare_bundle", script="stages/legacy/stage07/prepare_bundle.py",
         # The legacy scripts of this stage open bundle/glioma_models/... relative
         # to the working directory; nothing recreated that directory in a clean
         # checkout, so each of them failed with a bare FileNotFoundError.
         inputs=['models_bundle.tar.gz'],
         outputs=[],
         minutes=2, requires_network=False),
    Step("build_feature_matrices", script="stages/legacy/stage07/build_feats.py",
         # This step BUILDS the feature cache from the bioactivity matrix and the
         # curated marine library; it does not consume chembl_raw.npz. Those two
         # names were listed here as inputs, which made --check report the step as
         # blocked on files it is upstream of, and left Fig. 5 and Fig. 8
         # unreachable even when every real input was present.
         inputs=['chembl_multitask_matrix.parquet', 'cmnpd_curated.parquet'],
         outputs=[],
         minutes=15, requires_network=False),
    Step("refit_per_seed", script="stages/legacy/stage07/refit_seeds.py",
         inputs=['models_bundle.tar.gz'],
         outputs=[],
         minutes=60, requires_network=False),
    Step("rf_attribution", script="stages/legacy/stage07/interp_rf.py",
         inputs=[],
         outputs=['feature_importance_full.csv']
                 # interp_rf.py writes a marine cache beside every test cache (see its
                 # line 140); the marine ones were missing from this manifest, so --check
                 # under-reported what the step delivers.
                 + [f'shap_{part}_{t}.npz' for part in ('test', 'marine')
                    for t in ('EGFR', 'PDGFRA', 'PIK3CA', 'KDR', 'CDK6', 'IDH1')],
         minutes=120, requires_network=False),
    Step("gnn_attribution", script="stages/legacy/stage07/interp_gnn.py",
         inputs=['models_bundle.tar.gz'],
         outputs=['gnn_vs_rf_attribution.csv', 'gnn_rf_atom_attrib.csv', 'gnnexplainer_atom_attrib.csv'],
         minutes=70, requires_network=False),
    Step("consensus_rank", script="stages/legacy/stage07/interp_consensus.py",
         inputs=[],
         outputs=['top10_features.csv', 'ecfp4_bit_dictionary.csv', 'bit_collision_prevalence.csv'],
         minutes=15, requires_network=False),
    Step("structural_annotation", script="stages/legacy/stage07/stage_c.py",
         inputs=['cmnpd_curated.parquet'],
         outputs=['fig_exemplars.json'],
         minutes=25, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
