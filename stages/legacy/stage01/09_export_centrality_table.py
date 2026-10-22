import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from repro.config import path as RPATH, CFG, out_dir
import pandas as pd

cen = pd.read_csv(RPATH("centrality.csv"))

cen_out = cen[["gene_symbol","protein_name","module","degree","betweenness","closeness","eigenvector",
               "composite_hub_score","composite_hub_rank","ot_max_score","civic_evidence_items","sources"]]
cen_out.to_csv("glioma_network_centrality.csv", index=False)