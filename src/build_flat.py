"""Convert the pickled graph lists into flat .npy arrays that every training process
memory-maps, so concurrent runs share one physical copy of the featurisation."""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata


def flatten(pkl, outdir):
    if os.path.exists(os.path.join(outdir, "node_ptr.npy")):
        print("exists", outdir, flush=True); return
    with open(pkl, "rb") as f:
        graphs, ok = pickle.load(f)
    os.makedirs(outdir, exist_ok=True)
    nn = [0 if g is None else g["x"].shape[0] for g in graphs]
    ne = [0 if g is None else g["edge_index"].shape[1] for g in graphs]
    node_ptr = np.concatenate([[0], np.cumsum(nn)]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum(ne)]).astype(np.int64)
    X = np.zeros((node_ptr[-1], graphs[next(i for i,g in enumerate(graphs) if g is not None)]["x"].shape[1]), dtype=np.float32)
    PE = np.zeros((node_ptr[-1], 16), dtype=np.float32)
    EI = np.zeros((2, edge_ptr[-1]), dtype=np.int32)
    EA = np.zeros((edge_ptr[-1], 13), dtype=np.float32)
    RV = np.zeros(edge_ptr[-1], dtype=np.int32)
    for i, g in enumerate(graphs):
        if g is None:
            continue
        a, b = node_ptr[i], node_ptr[i + 1]
        c, d = edge_ptr[i], edge_ptr[i + 1]
        X[a:b] = g["x"]; PE[a:b] = g["pe"]
        EI[:, c:d] = g["edge_index"]; EA[c:d] = g["edge_attr"]; RV[c:d] = g["rev"]
    for n, arr in [("x", X), ("pe", PE), ("edge_index", EI), ("edge_attr", EA),
                   ("rev", RV), ("node_ptr", node_ptr), ("edge_ptr", edge_ptr),
                   ("ok", ok.astype(np.bool_))]:
        np.save(os.path.join(outdir, n + ".npy"), arr)
    print("wrote", outdir, X.shape, EA.shape, flush=True)
    del graphs, X, PE, EI, EA, RV


if __name__ == "__main__":
    for tag in sys.argv[1:] or ["chembl", "cmnpd", "decoy"]:
        flatten(os.path.join(mdata.CACHE, f"{tag}_graphs.pkl"),
                os.path.join(mdata.CACHE, f"flat_{tag}"))
    print("FLAT DONE", flush=True)
