# Lifting the 30-epoch bound: what to run, where, and what it can and cannot prove

## Why this experiment is worth doing

The manuscript's central claim is that the five graph architectures fail to transfer to marine
chemical space while a descriptor-augmented fingerprint ensemble does. Two sentences in the paper
currently protect that claim with a caveat, and both are compute artefacts rather than science:

> Graph training was capped at 30 epochs and several runs were still improving at epochs 26–29, so the
> graph figures are lower bounds. Architectures were also not capacity-matched: AttentiveFP and
> GraphGPS were narrowed relative to the other three to fit the memory budget.

A reviewer reading that has an easy objection available: *your graph models were undertrained and two
of them were also undersized, so of course they lost.* As long as that objection stands, the title
claim is weaker than the evidence could support. Running to early-stopping convergence at reference
capacity is the single highest-value addition you can make, and it is cheap — an estimated 5–10 GPU-hours.

Note what the 30-epoch cap actually implies, because it is stronger than it looks. Early stopping with
patience 6 was already active in the published run. A run that consumed all 30 epochs therefore
improved its validation loss at some point within its final 6 epochs — otherwise stopping would have
fired. The runs were genuinely still descending when the budget expired; that inference follows from the
design and does not depend on the training logs.

## Be prepared for the result to go against you

This is a real experiment, which means it can refute your paper. Three outcomes, and you should decide
now how you will report each:

**1. Internal scores rise, marine transfer stays at chance (most likely).** The applicability-domain
analysis is the reason to expect this: only 8.23% of the marine library sits inside a 0.30 ECFP4
Tanimoto domain of the training pool, and marine median max-similarity to the training set is 0.207
against 0.823 training self-similarity. Longer training fits ChEMBL kinase-inhibitor chemistry better;
it creates no information about marine scaffolds. Converged graph models may well beat the random
forest on the internal test partition and still sit at ρ ≈ 0 externally. **This is the strong version
of your paper** — the failure is then a property of the representation under covariate shift, not of
an interrupted optimiser, and the reviewer's objection is dead.

**2. Marine transfer improves substantially.** If a converged D-MPNN reaches, say, marine ρ ≈ 0.30, the
central claim does not survive and the paper must be rewritten around a compute-budget finding instead
of an architectural one. Unwelcome, but far better discovered by you than by a referee.

**3. Marine transfer degrades further.** Entirely possible, and mechanistically coherent: more epochs
sharpen a representation that is specialised to the training distribution. This would let you make the
sharper claim that graph pre-training on synthetic inhibitor chemistry actively harms extrapolation.

Whichever occurs, **report the 30-epoch and converged results side by side as a compute-budget
ablation.** That framing is more valuable than either number alone, and it makes the new run additive
rather than a replacement.

## Platform

| | Free GPU | Session limit | Verdict |
|---|---|---|---|
| **Kaggle Notebooks** | P100 16 GB, or T4×2; ~30 h/week | 9 h per session, 20 GB working disk, persistent Datasets | **Recommended.** The 9-hour ceiling is comfortable because each run is 20–60 min and `run_convergence.sh` skips completed runs, so a stopped session resumes cleanly. Persistent Datasets are the cleanest way to hold the 16 MB of parquets and the rebuilt cache. |
| Colab free | T4, no quota guarantee | Preemptible; disconnects without warning | Workable but the preemptions cost you time. Colab Pro (~USD 10/month) removes most of the pain. |
| GCP | none (paid) | none | Use if you want one uninterrupted run. A `g2-standard-4` (L4) spot instance is roughly USD 0.25–0.30/h, so the whole experiment lands near USD 3. Needs VM setup, which is the only real cost. |

A P100 or T4 is ample. These are small models — 0.4 M to 1.2 M parameters at full capacity — and the
molecular graphs are small, so GPU utilisation will be partial and the speedup over CPU will be roughly
5–10×, not 50×. Do not pay for an A100; it will not help.

## What to prepare

**1. Three data files (16.3 MB total).** From this project's artifacts:

| File | Size |
|---|---|
| `chembl_multitask_matrix.parquet` | 4.5 MB |
| `decoy_set.parquet` | 3.4 MB |
| `cmnpd_curated.parquet` | 8.4 MB |

Put them in `data/`. Paths are read from environment variables with `data/` defaults, so nothing needs
editing; override with `GLIOMA_MATRIX`, `GLIOMA_DECOYS`, `GLIOMA_CMNPD` if you prefer another layout.

**2. This bundle.** `src/` is the pipeline with three patches already applied (see below), plus
`run_convergence.sh` and `smoke_test.py`.

**3. Environment.** Kaggle and Colab both ship PyTorch with CUDA; you need PyTorch Geometric and RDKit
on top:

```bash
pip install torch-geometric rdkit-pypi
```

`torch-scatter` and `torch-sparse` are *not* required — the code uses only PyG operators that ship with
the base package.

**4. Do not upload the existing checkpoints.** `models_bundle.tar.gz` holds all 35 best-validation
checkpoints and `train_gnn.py` accepts `--init` to warm-start from them, but for this experiment train
from scratch. Continuing from a 30-epoch checkpoint gives a different optimisation trajectory than a
clean long run — different effective learning-rate schedule, different plateau history — and a referee
would be right to ask which one the reported number came from.

## What was patched in this bundle

Three changes to `src/`, and nothing else:

1. **GPU support in `train_gnn.py`.** The published script was CPU-only: it called
   `torch.set_num_threads` and never moved anything to a device. Added `--device` (default `auto`),
   with the model, each batch, and the per-task positive-weight vector moved to the device, and
   predictions brought back to CPU for scoring.
2. **`--capacity {paper,full}`.** `paper` reproduces the published widths exactly. `full` restores
   reference-implementation scale:

   | Architecture | Paper | Full | Params paper → full |
   |---|---|---|---|
   | GIN | hidden 128, 3 layers | hidden 256, 5 layers | 207 k → 1,203 k (5.8×) |
   | GATv2 | hidden 128, 3 layers, 4 heads | hidden 256, 5 layers, 8 heads | 162 k → 890 k (5.5×) |
   | D-MPNN | hidden 128, depth 3 | hidden 300, depth 3 (Chemprop default) | 96 k → 432 k (4.5×) |
   | AttentiveFP | hidden 96, 2 layers, 1 timestep | hidden 200, 2 layers, 2 timesteps | 250 k → 1,056 k (4.2×) |
   | GraphGPS | hidden 64, 2 layers, 4 heads | hidden 128, 5 layers, 8 heads | 131 k → 1,054 k (8.1×) |

   All ten configurations were verified here by construction **and a real forward pass** on a synthetic
   batch, so none of them is a guess that will fail an hour into your session. GraphGPS was the most
   starved in the published run — 2 layers is very shallow for a graph transformer — which makes it the
   architecture most likely to move.
3. **Portable data paths and longer defaults.** Absolute paths into this project's artifact store
   replaced by `data/` defaults, and `--epochs`/`--patience` defaults raised from 30/6 to 300/25 with
   `--lr_patience` exposed (the plateau scheduler was hard-coded at 4, which is too impatient over a
   300-epoch horizon).

Everything else is untouched: the scaffold-disjoint split, the three seeds, batch 256, lr 1e-3, weight
decay 1e-5, gradient clipping at 5.0, the masked multi-task BCE and its per-task positive weights.

## Sizing the job

Measured on the 8-core CPU of the machine that prepared this bundle, at **full** capacity, 89 batches
of 256 per epoch:

| Architecture | s/epoch (this CPU) |
|---|---|
| D-MPNN | 43 |
| AttentiveFP | 54 |
| GIN | 75 |
| GATv2 | 87 |
| GraphGPS | 93 |

Run `smoke_test.py` first on your GPU; it prints the same table measured there. Then:

```
total ≈ 15 runs × (epochs to early stop) × (s/epoch) + ~20 min cache build
```

With early stopping at patience 25 typically firing between 60 and 150 epochs, and a 5–10× GPU
speedup over the numbers above, expect **5–10 GPU-hours** for all 15 runs. That is one to two Kaggle
sessions, or a third of a week's free quota.

## Running it

```bash
pip install torch-geometric rdkit-pypi
python smoke_test.py                 # verify GPU + measure s/epoch
bash run_convergence.sh              # 15 runs; skips any already finished
tar czf converged_results.tar.gz metrics models runs_converged
```

**Archive `metrics/` before the session dies.** Each run writes
`metrics/<name>_run.json` containing `epochs_run`, `best_val_loss`, the full per-epoch `history`, the
parameter count, wall time and the resolved hyperparameters, plus
`metrics/<name>_preds.npz` with validation and test predictions. The published run wrote all of this
too, but it lived in a scratch directory that was swept, which is why the paper can assert that runs
were still improving at epochs 26–29 without being able to show the curves. Do not repeat that: the
per-epoch histories are what let you plot validation loss against epoch and *demonstrate* convergence
rather than assert it.

## After the run

Bring `converged_results.tar.gz` back and the remaining work is mechanical, since the evaluation and
external-validation code is unchanged:

1. `src/evaluate.py` → internal metrics on the scaffold-disjoint test partition.
2. `src/run_external.py` → the marine validation: Spearman ρ and ROC-AUC of the polypharmacology score
   against measured glioma cell-line pActivity on the 194 CMNPD compounds, plus the ZINC20 decoy
   enrichment.
3. New figure panel: validation loss against epoch for all 15 runs, with the 30-epoch line marked. This
   is the panel that retires the reviewer's objection, and it is the one the published run cannot draw.
4. New table: internal and external metrics at 30 epochs against convergence, per architecture.
5. Manuscript edits — the two caveat sentences quoted at the top, the Methods epoch ceiling, the
   Conclusions phrase "Under a defined 30-epoch compute budget", and the abstract if the external result
   moves.

I can do all five once you bring the tarball back.
