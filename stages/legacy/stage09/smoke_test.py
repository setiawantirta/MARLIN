"""Verify the environment and MEASURE per-epoch time before spending GPU hours.

Run this first on Kaggle/Colab/GCP. It checks that every architecture builds and
runs a forward pass at both capacities, reports the GPU it found, and times a
handful of real training batches so you can extrapolate the full experiment
instead of guessing at it.

    python smoke_test.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import models                                    # noqa: E402
from feat import RWSE_K                          # noqa: E402
from models import ATOM_DIM, BOND_DIM            # noqa: E402
from train_gnn import FULL_CAPACITY, pick_device  # noqa: E402
from torch_geometric.data import Batch, Data     # noqa: E402

ARCHS = ["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS"]


def synthetic_batch(n_graphs=256, atoms=25):
    """A batch shaped like the real one: ~25 heavy atoms per molecule, chain topology."""
    gs = []
    for _ in range(n_graphs):
        ei = torch.stack([torch.arange(atoms - 1), torch.arange(1, atoms)])
        ei = torch.cat([ei, ei.flip(0)], 1)
        d = Data(x=torch.randn(atoms, ATOM_DIM), edge_index=ei,
                 edge_attr=torch.randn(ei.size(1), BOND_DIM))
        d.pe = torch.randn(atoms, RWSE_K)
        d.rev = torch.arange(ei.size(1))
        d.y = torch.randint(0, 2, (1, 6)).float()
        d.mask = torch.ones(1, 6)
        gs.append(d)
    return Batch.from_data_list(gs)


def main():
    dev = pick_device("auto")
    print("torch", torch.__version__, "| device:", dev)
    if dev.type == "cuda":
        print("gpu   :", torch.cuda.get_device_name(0),
              "| %.1f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
    else:
        print("WARNING: no GPU visible. On Kaggle enable Settings -> Accelerator -> GPU;")
        print("         on Colab, Runtime -> Change runtime type -> T4 GPU.")

    b = synthetic_batch().to(dev)
    pw = torch.ones(6, device=dev)
    print("\n%-12s %14s %14s %12s" % ("arch", "paper params", "full params", "s/epoch*"))

    n_batches = 89   # 22,683 training compounds / batch 256
    for arch in ARCHS:
        row = [arch]
        for cap in ("paper", "full"):
            kw = FULL_CAPACITY[arch] if cap == "full" else {}
            m = models.build(arch, **kw).to(dev)
            row.append(format(sum(p.numel() for p in m.parameters()), ","))
            if cap == "full":
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
                m.train()
                for _ in range(2):                      # warm-up, not timed
                    opt.zero_grad()
                    models.masked_bce(m(b), b.y, b.mask, pw).backward()
                    opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                reps = 5
                for _ in range(reps):
                    opt.zero_grad()
                    models.masked_bce(m(b), b.y, b.mask, pw).backward()
                    opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                per_batch = (time.time() - t0) / reps
                row.append("%.0f" % (per_batch * n_batches))
        print("%-12s %14s %14s %12s" % tuple(row))

    print("\n* estimated seconds per epoch at FULL capacity, %d batches of 256." % n_batches)
    print("  Multiply by the epochs you expect early stopping to reach (60-150 is typical")
    print("  with patience 25) and by 15 runs (5 architectures x 3 seeds) to size the job.")
    print("  Add ~20 min once for the graph cache build.")


if __name__ == "__main__":
    main()
