"""Forward-pass a trained checkpoint over a cached external set (cmnpd / decoy / chembl)."""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata, models


def predict(arch, ckpt, tag, rows=None, batch=512, threads=2):
    torch.set_num_threads(threads)
    store = mdata.FlatStore(tag)
    rows = np.arange(store.n) if rows is None else np.asarray(rows)
    rows = rows[store.ok[rows]]
    model = models.build(arch)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    P, I = [], []
    dl = mdata.flat_loader(store, rows, batch=batch)
    with torch.no_grad():
        for k, b in enumerate(dl):
            P.append(torch.sigmoid(model(b)).numpy())
            I.append(b.idx.numpy())
            if (k + 1) % 20 == 0:
                print(f"[pred:{tag}] {(k+1)*batch}/{len(rows)}", flush=True)
    return np.concatenate(P), np.concatenate(I)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    P, I = predict(a.arch, a.ckpt, a.set, threads=a.threads)
    np.savez_compressed(a.out, P=P, I=I)
    print("wrote", a.out, P.shape, flush=True)
