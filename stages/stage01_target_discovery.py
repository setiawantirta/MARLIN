#!/usr/bin/env python3
"""Stage 1 - Target discovery from the protein-interaction network.

Open Targets + CIViC seeds -> STRING subnetwork -> greedy-modularity modules ->\ncentrality -> pathway enrichment -> ChEMBL tractability -> the six-target panel.

Block I of the published method flowchart (Fig. S1).
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
    number=1, slug="target_discovery", title="Target discovery from the protein-interaction network",
    description="""Open Targets + CIViC seeds -> STRING subnetwork -> greedy-modularity modules ->\ncentrality -> pathway enrichment -> ChEMBL tractability -> the six-target panel.""",
    wall_clock="~25 min (dominated by API calls)", hardware="CPU, network",
    steps=[
    Step("01_fetch_opentargets_civic", script="stages/legacy/stage01/01_fetch_opentargets_civic.py",
         inputs=[],
         outputs=['step1_raw.json', 'civic_genes.json'],
         minutes=3, requires_network=True),
    Step("02_build_seed_gene_table", script="stages/legacy/stage01/02_build_seed_gene_table.py",
         inputs=['step1_raw.json', 'civic_genes.json'],
         outputs=['glioma_seed_genes.csv'],
         minutes=1, requires_network=False),
    Step("03_fetch_string_network", script="stages/legacy/stage01/03_fetch_string_network.py",
         inputs=[],
         outputs=['string_net.json'],
         minutes=2, requires_network=True),
    Step("04_detect_modules", script="stages/legacy/stage01/04_detect_modules.py",
         inputs=['string_net.json'],
         outputs=['modules.json'],
         minutes=1, requires_network=False),
    Step("05_build_centrality_table", script="stages/legacy/stage01/05_build_centrality_table.py",
         inputs=['string_net.json', 'modules.json'],
         outputs=['centrality.csv'],
         minutes=1, requires_network=False),
    Step("06_pathway_enrichment", script="stages/legacy/stage01/06_pathway_enrichment.py",
         inputs=['reactome_raw.json', 'go_raw.json', 'centrality.csv'],
         outputs=['glioma_pathway_enrichment.csv'],
         minutes=4, requires_network=True),
    Step("07_fetch_chembl_hub_counts", script="stages/legacy/stage01/07_fetch_chembl_hub_counts.py",
         inputs=['centrality.csv'],
         outputs=['chembl_counts.json'],
         minutes=10, requires_network=True),
    Step("08_select_target_panel", script="stages/legacy/stage01/08_select_target_panel.py",
         inputs=['chembl_counts.json', 'centrality.csv'],
         outputs=['glioma_target_panel.csv'],
         minutes=1, requires_network=False),
    Step("09_export_centrality_table", script="stages/legacy/stage01/09_export_centrality_table.py",
         inputs=['centrality.csv'],
         outputs=['glioma_network_centrality.csv'],
         minutes=1, requires_network=False),
    ],
)

if __name__ == "__main__":
    sys.exit(STAGE.main())
