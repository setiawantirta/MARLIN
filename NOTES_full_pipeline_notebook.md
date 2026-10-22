# Notes on `stages/full_pipeline_colab.ipynb`

What the notebook is, what was executed while building it, what it could not execute,
and the nine defects in the module that building it exposed.

---

## 1. What it is

A single notebook that runs **stages 01 to 12** of this module end to end on a free Colab
or Kaggle session and finishes on the 44-assertion verification, written for a referee
reproducing the study without contacting the authors.

It reimplements nothing. Every stage is invoked through the module's own
`./run_stage.sh NN`; the stage-01 retrieval reuses `repro/fetch_sources.py` and
`repro/host_shim.py`, which were already verified against the published seed set. 83
cells: 43 code, 40 markdown.

---

## 2. The epoch policy, stated plainly

The notebook was asked to run the graph training "at the full 1,000-epoch ceiling with no
epoch reduction". It does, and section 1 of the notebook spells out why that is not the
same as training for 1,000 epochs.

* **Stage 09, the convergence control: ceiling 1,000, early-stopping patience 25,
  capacity `full`.** Read from `config/config.yaml` and passed to the trainer by
  `stages/legacy/stage09/run_convergence.sh`. Early stopping is **not** removed. In the
  published run it fired in all fifteen runs between epochs 34 and 86, so the ceiling was
  never approached.
* **Stage 05, the main benchmark: ceiling 30, patience 6 - unchanged.** Those two numbers
  *are* the published benchmark (internal macro ROC-AUC 0.922 for the random forest down
  to 0.867 for GATv2). Raising them would change every headline number in the paper.
* **Why not a literal 1,000 epochs.** The validation-loss curves of AttentiveFP, GraphGPS
  and D-MPNN rise after their minima. Training past early stopping does not refine the
  models, it overfits them; it would reproduce no published number and would cost roughly
  15-20x the compute for a worse model. Relative to the 30-epoch main benchmark, a 1,000
  ceiling *is* "no epoch reduction" - it is 33x the main one.

Before this change, one setting had three values in three places: `config.yaml` said 200
epochs / patience 20, `run_convergence.sh` hardcoded `--epochs 300 --patience 25` and
ignored the `EPOCHS`/`PATIENCE` the stage exported, and `stage09_colab.ipynb` and the
manuscript both described a 1,000 ceiling. All three now read 1,000 / 25. Patience 25 is
what the driver already used, so no published number moves; the ceiling is non-binding at
any value above 86.

---

## 3. Per-stage wall clock, and where a GPU actually helps

Wall clock is the module's own declaration (`Stage.wall_clock`, README section 2),
measured on its reference machine - 8 CPU cores, 8 GB RAM, no GPU. The GPU column was
**re-derived from the code**, not copied from the README.

| stage | CPU | GPU | GPU helps? | why |
|---|---|---|---|---|
| 01 target discovery | ~25 min | ~25 min | no | Open Targets / CIViC / STRING / ChEMBL latency |
| 02 marine library | ~45 min | ~45 min | no | RDKit standardisation on CPU |
| 03 label matrix | ~3 h | ~3 h | no | ChEMBL API latency |
| 04 controls | ~70 min | ~70 min | no | ~40 min is the ZINC20 download |
| 05 training | ~26 h | ~26 h | **no** | see below |
| 06 evaluation | ~90 min | ~90 min | no | numpy/sklearn metrics and bootstrap |
| 07 attribution | ~4 h | ~4 h | **no** | exact TreeSHAP on a scikit-learn forest; no CUDA path exists |
| 08 screening | ~25 min | ~25 min | no | forest inference plus RDKit filters |
| 09 convergence control | ~40 h | **~4 h** | **yes** | torch trainer takes `--device auto` |
| 10 hyperparameter control | ~30 h | **~5 h** | **partly** | XGBoost arm only; the RF arm is sklearn CPU by design |
| 11 figures | ~8 min | ~8 min | no | matplotlib |
| 12 verify | ~2 s | ~2 s | no | reads CSVs |

Summing the table: **~107 h (4.5 days)** on CPU throughout, falling to **~46 h (1.9
days)** once stages 09 and 10 move to a GPU. Stage 05's 26 h is then the single largest
item and the reason the GPU total is not smaller.

**Stage 05 gets nothing from a GPU, contrary to the README's "~3 h" claim.** Checked
against the source: `src/train_gnn.py` declares no `--device` argument and contains no
`.to(device)` call anywhere, and `src/baselines.py` constructs
`XGBClassifier(..., tree_method="hist")` with no `device=`. Both run on CPU on a GPU
machine. Making stage 05 CUDA-capable would mean editing the code that produced the
published numbers, so it was not done; the README has been corrected instead and the
notebook tells reviewers not to spend GPU quota there.

### Measured on this machine (8 cores, 8 GB, macOS arm64, no GPU)

Measurements, not projections:

| what | measured |
|---|---|
| `./smoke_test.sh` | ~60 s, `44 PASS 0 FAIL 0 SKIP` |
| `./run_all.sh --check` | 1.9 s, exit 0 |
| `./run_stage.sh 09 --smoke` | 21 s |
| `./run_stage.sh 11 --only 1 7 S1` | 9.7 s (3.6 / 4.1 / 2.0 s per figure) |
| `./run_stage.sh 12` | ~2 s, 44 PASS |
| one live Open Targets query, 25 rows | 0.6 s |
| stage 01 full retrieval path, live APIs | ~22 min (9.3 min of it steps 07-09) |

The stage-09 smoke path reports per-architecture training speed at **full** capacity on
this machine: GIN 33 s/epoch, GATv2 50, D-MPNN 18, AttentiveFP 35, GraphGPS 44 - 180 s per
epoch across the five. At 60 epochs x 3 seeds that projects to roughly **9 h** of serial
CPU, and ~13 h at the 86-epoch upper end. Treat that as a **lower bound**, not a
replacement for the module's ~40 h declaration: the probe uses synthetic chain-topology
batches, and it excludes the validation pass, data loading and the one-off ~20 min graph
cache build. The notebook prints the same measurement on the reviewer's own machine so
they can size the job rather than trust either number.

---

## 4. Multi-session design

The full pipeline cannot run in one session and the notebook does not pretend otherwise.
Kaggle hard-stops at **12 hours**; Colab recycles runtimes without warning.

* **Persistent storage is mounted first, before anything else.** Drive on Colab,
  `/kaggle/working` on Kaggle, `~/glioma_repro` locally. The code checkout goes on the
  fast session disk - the stage scripts do many small reads and Drive's filesystem is slow
  for that - and every output, cache and checkpoint goes to the persistent root.
* **Environment variables are set through `os.environ`, never `!export`.** Each `!` line
  in a notebook is its own subshell, so `!export VAR=x` on a line of its own is lost the
  moment it ends. `os.environ` is inherited by every `!` line and subprocess.
* **Resume operates at three levels.** (a) A `.done_stageNN` marker in the persistent
  output directory, written only when a *whole* stage exits 0 - a `--only` group, a
  `--smoke` run or a `--check` never marks a stage complete. (b) The module's own
  `metrics/<tag>_run.json` convention, now honoured by `src/sched.py`, so the fifteen
  pretraining runs of stage 05 and the fifteen convergence runs of stage 09 each skip what
  is already finished. (c) Stage 10's per-configuration checkpoints under `GLIOMA_CKPT`.
* **A session ledger** (`logs/session_ledger.jsonl`) records stage, arguments, exit
  status, minutes, platform, GPU and whether the run completed the stage.
* **An explicit twelve-session order** is given in notebook section 8, with the
  accelerator marked per session - only two of the twelve want a GPU.

---

## 5. Smoke path

Four cells, about ten minutes, exercising retrieval, training and figure rendering before
any long stage:

1. `./smoke_test.sh` - imports, config, all twelve `--check`, two figure renders, the full
   44-assertion verification.
2. Reachability of all six services, then one real Open Targets query through
   `repro/fetch_sources.py`.
3. `./run_stage.sh 09 --smoke` - builds all five architectures at full capacity on
   synthetic batches and times real training steps.
4. `./run_stage.sh 11 --only 1 7 S1` and `./run_stage.sh 12`.

Stage 10 has its own one-target rehearsal in its section: `ONLY_TASK=CDK6` restricts every
fit to a single target, which is the ~10-minute end-to-end test for that stage.

---

## 6. Defects found in the module while building this

Every one was found by **running** the module, and every one is fixed in this package.
The first two were already known; the rest were not.

1. **Stage 03 step order.** `fetch_molecules` was declared before `filter_activities`, but
   `fetch_molecules.py` line 28 reads `work/agg_per_compound_target.csv.gz`, which
   `filter_activities.py` line 84 writes. A clean run died on its first post-retrieval
   step. Order swapped, with a comment saying why.
2. **No scratch directories.** Nothing created `work/`, `raw/` or `handoff/`, so a clean
   run failed on the first write into them. `repro/stagelib.py` now creates all three
   under the stage output directory before any step runs.
3. **Stage 09 could not run at all - wrong trainer.** `stages/legacy/stage09/src/train_gnn.py`
   was a copy of the *main-benchmark* trainer: no `--capacity`, no `--device`, no
   `--lr_patience`, defaults 30/6. `run_convergence.sh` passes all three of those flags,
   so argparse would have exited 2 on every one of the fifteen runs, and
   `smoke_test.py`'s `from train_gnn import FULL_CAPACITY, pick_device` was an ImportError.
   Replaced with the trainer from the `gpu_convergence` bundle, which is the code that
   actually ran the published control. The module's `repro.config`-based `mdata.py` is
   kept, so no absolute path from the authors' machine is introduced.
4. **Stage 09 and 10 drivers resolved their scripts against the caller's cwd.**
   `run_convergence.sh` ran `python src/train_gnn.py` and `run_tuning.sh` ran
   `python tune_baselines.py`, while the stage entry points set `cwd` to the stage *output*
   directory, where no such file exists. Demonstrated: running `run_convergence.sh` from an
   empty directory fails instantly with `can't open file '.../src/build_cache.py'`. Both now
   resolve from `$(dirname "${BASH_SOURCE[0]}")`, cwd unchanged.
5. **Stage 05's job drivers had the same bug**, in Python: `run_pretrain.py`,
   `run_ablation.py`, `run_baselines.py` and `run_external.py` invoked
   `[PY, "src/train_gnn.py", ...]` and friends relative to the caller's cwd. Six
   invocations across four files, now resolved from `__file__`.
6. **No resume in stage 05.** `src/sched.py` had no skip-if-done, so a dropped session
   restarted all fifteen pretraining runs from zero - fatal for a ~20 h step under a 12 h
   session limit. It now honours the same `metrics/<tag>_run.json` marker that
   `run_convergence.sh` already used. Verified: a job with an existing marker prints
   `SKIP` and is not run; a job without one runs normally.
7. **`$EPOCHS` and `$PATIENCE` were exported and ignored.**
   `stage09_control_convergence.py` exported both from `config.yaml`;
   `run_convergence.sh` hardcoded 300 and 25. See section 2.
8. **The stage-09 smoke path was gated on inputs it does not read.** It builds synthetic
   batches, but the stage refused to start without all seven study data products, because
   `mdata.py` resolved every input path **at import time** and `train_gnn` imports `mdata`.
   Path resolution is now lazy (PEP 562 module `__getattr__`, memoised; identical
   behaviour and identical `MissingInput` when the data is present), and `--smoke` is no
   longer blocked by missing inputs. The rehearsal now runs before any download.
9. **`--smoke` silently ran the full stage for stages 01-08.** No `Step` declares
   `smoke_args`, and `repro.config.smoke()` is defined but called from nowhere in the
   package, so `./run_stage.sh 05 --smoke` would have started a 26-hour run. `stagelib`
   now skips a step that has no subset path and says so, instead of starting hours of work
   unannounced.
10. **No way to put outputs on persistent storage.** `out_dir()` read only
    `paths.output_dir` from `config.yaml`, so on Colab or Kaggle every output - including
    the stage-09 per-epoch trajectories Fig. S4 needs and stage 10's checkpoints - landed
    on the ephemeral session disk. `out_dir()` now honours `$MARINE_REPRO_OUTPUT`.
    Verified: `MARINE_REPRO_OUTPUT=/tmp/x ./run_stage.sh 12` writes its report to
    `/tmp/x/stage12_verify/`.
11. **Stage 09 rebuilt the ~20 min graph cache every session.** Its guard tested
    `cache/flat_chembl/node_ptr.npy` relative to cwd while `build_cache.py` writes to
    `mdata.CACHE`; with the cache on persistent storage those are different directories.
    The guard now asks `repro.config` for the real location.

Two further defects were fixed in the notebook rather than the module:

* The GPU probe died on machines with **no `nvidia-smi` binary at all** -
  `subprocess.run` raises `FileNotFoundError` rather than returning non-zero - which left
  `HAS_GPU` undefined and took every subsequent cell down. Found by executing the
  notebook on this CPU-only machine; now caught.
* Bundling the optional `06_pathway_enrichment` step with steps 07-09 in one `--only`
  group would have let a Reactome outage block the panel selection, because `stagelib`
  refuses an entire group when any selected step has a missing input. They are now
  separate cells.

### Noted but not fixed (reported rather than touched)

* `stages/legacy/stage10/smoke_test_baselines.py` does `os.chdir(_HERE)` at import, so its
  outputs land inside the package rather than the stage output directory. Harmless for a
  rehearsal; left alone.
* `README.md` section 9 records a genuine discrepancy between the manuscript and a
  specification handed to the module's author for the RF-vs-XGBoost marine comparison.
  Neither value was altered, here or previously.

---

## 7. What was executed while building this, and what was not

All execution on 8 CPU cores, 8 GB RAM, macOS arm64, **no GPU**, Python 3.12.14.

**Executed, zero exit status.** `bash -n` on all five shell scripts; `ast.parse` on all
121 Python files; `./smoke_test.sh` (44 PASS); `./run_all.sh --check` (exit 0); the module
runner for stages 01-12 in `--check`; `./run_stage.sh 09 --smoke`; `./run_stage.sh 11`
and `--only 1 7 S1`; `./run_stage.sh 12` (44 PASS, exit 0); the stages whose inputs are
absent here (02, 06, 07, 08, 10) run for real and exiting 2 with the expected
missing-input report; a live Open Targets query; the `MARINE_REPRO_OUTPUT` override; and
the `sched.py` resume guard as a unit test.

**The stage-01 retrieval path was run for real against the live APIs**, not merely
`--check`ed, and reproduced the published target panel:

| | rebuilt live, this machine | published |
|---|---|---|
| seed genes | 186 | 186 |
| STRING nodes / edges / isolated | 186 / **1,564** / 10 | 186 / **1,421** / 10 |
| modules | **5** (75/66/18/8/7, 12 unassigned) | **4** (64/56/43/8, 15 unassigned) |
| seed genes mapped to a ChEMBL target | 135 of 186 | 135 of 186 |
| **target panel** | EGFR, PDGFRA, PIK3CA, KDR, CDK6, IDH1 | **identical** |

The edge count and module partition differ because STRING has gained evidence since the
study ran; the panel does not depend on the partition. Reactome refused this network with
its documented Cloudflare error, and the notebook's optional-enrichment cell skipped
cleanly and let steps 07-09 proceed - which is exactly the failure mode the 6b/6c split
was introduced to handle, confirmed in practice rather than in principle.

**Notebook validation.** Valid JSON, nbformat 4; every code cell compiles. The magic
stripper used for that check is bracket-depth aware and leaves `!=` alone - a naive
`"!"` replacement breaks `!=` and has produced a false failure in this project before, so
the validator asserts both that it handles the real cells and that the naive version
still fails on the control string. Scanned clean for absolute paths from this machine,
credentials and personal data.

**Could not be executed here, with the reason.**

| cell | why not |
|---|---|
| stage 01 enrichment (step 06) | Reactome refused this network (Cloudflare 1010); the optional cell skipped as designed and steps 07-09 still ran |
| stage 03 full run | ~3 h of live ChEMBL traffic; `--check` run instead, exit 0 |
| stage 04 full run | ~70 min, ~40 min of it a ZINC20 download; `--check` run instead, exit 0 |
| stage 05 groups A/B/C | ~26 h of CPU training; `--check` and `--list` run instead, exit 0 |
| stage 09 full run | needs a GPU (~40 h here); `--check` and the synthetic `--smoke` both run, exit 0 |
| stage 10 full grid | 15,930 fits; the command line was exercised and exits 2 on missing inputs as designed |

The stage-09 and stage-10 *command lines* were therefore verified without the data, and
the stage-09 trainer's flag surface was verified directly against
`stages/legacy/stage09/src/train_gnn.py` after the fix: `--capacity`, `--device`,
`--lr_patience`, `--epochs`, `--patience` all exist.

---

## 8. What a reviewer must download

1. **CMNPD 1.0 vendor files** - `CMNPD_1.0_calc_prop.tsv` and `CMNPD_1.0.sdf`, from
   <https://www.cmnpd.org> after registering. Not redistributable, no public API, no
   automated path. Needed by stage 02 and by stage 03's final step only.
2. **The study data products** - 21 files, ~173 MB, from the deposit alongside the paper.
   A convenience, not a dependency: stages 01-08 regenerate every one of them.
3. Everything else - 61 result tables, 9.2 MB - ships inside the tarball, which is why
   `./run_stage.sh 12` answers `44 PASS` the moment the package is unpacked.
