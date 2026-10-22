"""Figure 1 - glioma protein-interaction network, module-block layout.

Drives ``_fig1_ppi_redraw.build / module_block_layout / draw``, which is the
source that produced the published Figure 1.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from repro.config import path as P, out_dir
from repro.figstyle import save_figure, uppercase_existing_panel_letters
import _fig1_ppi_redraw as F

OUT = out_dir("stage11_figures")
G, Gc, mod, cen, iso = F.build(P("string_net.json"), P("modules.json"), P("centrality.csv"))
pos, ring, members, unass = F.module_block_layout(Gc, mod)
import tempfile
with tempfile.TemporaryDirectory() as _tmp:          # draw() saves internally; discard that copy
    fig, _pos, _labels = F.draw(Gc, pos, mod, cen, out=_os.path.join(_tmp, "raw"), members=members)
uppercase_existing_panel_letters(fig)
pdf, png = save_figure(fig, "fig01_glioma_ppi_network", outdir=OUT)
print("FIGURE_OK", "fig01_glioma_ppi_network", pdf, png,
      "nodes=%d edges=%d isolated=%d" % (Gc.number_of_nodes(), Gc.number_of_edges(), len(iso)), flush=True)
