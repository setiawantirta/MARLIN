#!/usr/bin/env python3
"""Stage 5 - Training of all seven model families.

Graph caching and flat feature build, then five graph architectures x three\nseeds and the two fingerprint ensembles. Shared 2-layer 128-unit trunk with\nsix task-specific heads and a masked BCE that excludes unmeasured cells.

Block V of the published method flowchart (Fig. S1).
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
    number=5, slug="training", title="Training of all seven model families",
    description="""Graph caching and flat feature build, then five graph architectures x three\nseeds and the two fingerprint ensembles. Shared 2-layer 128-unit trunk with\nsix task-specific heads and a masked BCE that excludes unmeasured cells.""",
    wall_clock="~26 h on 8 CPU cores (3 concurrent workers); ~3 h on one modern GPU", hardware="CPU (GPU optional)",
    steps=[
    Step("build_graph_cache", script="src/build_cache.py",
         inputs=['chembl_multitask_matrix.parquet', 'cmnpd_curated.parquet', 'decoy_set.parquet'],
         outputs=[],
         minutes=180, requires_network=False),
    Step("flatten_arrays", script="src/build_flat.py",
         inputs=[],
         outputs=[],
         minutes=20, requires_network=False),
    Step("fingerprint_features", script="src/fp_features.py",
         inputs=[],
         # src/fp_features.py writes these three fingerprint-plus-descriptor caches.
         # The manifest previously named chembl_raw.npz / cmnpd_raw.npz / featnames.npy,
         # which this step does not produce: those are written by stage 07's
         # rf_attribution (stages/legacy/stage07/interp_rf.py).
         outputs=['chembl_fp.npz', 'cmnpd_fp.npz', 'decoy_fp.npz'],
         minutes=60, requires_network=False),
    Step("pretrain_graph_networks", script="src/run_pretrain.py",
         inputs=[],
         outputs=[],
         minutes=1200, requires_network=False),
    Step("fit_baselines", script="src/run_baselines.py",
         inputs=[],
         outputs=[],
         minutes=120, requires_network=False),
    Step("ablation_finetune", script="src/run_ablation.py",
         inputs=[],
         outputs=[],
         minutes=180, requires_network=False),
    Step("persist_final_rf", script="src/fit_final_rf.py",
         inputs=[],
         outputs=['models_bundle.tar.gz', 'model_config.json'],
         minutes=20, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
