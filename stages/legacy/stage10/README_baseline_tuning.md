# Were the fingerprint baselines maximally tuned? No. What this bundle fixes.

## The honest answer to the question

They were not. `src/baselines.py` in the published run sampled **six configurations at
random** from the grid and scored each **once on the validation split**:

```
ap.add_argument("--n_search", type=int, default=6)
...
cands = sample_grid(RF_GRID if a.model == "RF" else XGB_GRID, a.n_search, rng)
```

The grids contain 54 (random forest) and 96 (XGBoost) configurations, so roughly
**11% and 6% of the search space** was explored, on a single split, with no
cross-validation. Seeds 1 and 2 did not repeat the search at all — they reused the
hyperparameters chosen on seed 0. Nothing was refit on train plus validation.

This bundle replaces that with an exhaustive grid and stratified 5-fold
cross-validation on the training rows only. The validation and test splits are never
consulted during selection.

| | Published run | This bundle |
|---|---|---|
| Search | 6 random configurations | every configuration in the grid |
| RF space | 54 combinations, 6 tried | 81 combinations, all tried |
| XGB space | 96 combinations, 6 tried | 96 combinations, all tried |
| Selection score | one validation split | mean ROC-AUC over StratifiedKFold(5) on train |
| Seeds | tuned on seed 0, reused | each seed tuned independently |
| Refit | best fold model kept | winner refit on all train rows |

Held identical so the comparison stays clean: the scaffold-disjoint split column, the
ECFP4(2048, r=2) + 30-descriptor matrix, the per-target binary formulation,
`class_weight="balanced_subsample"` for the forest and `scale_pos_weight` for
XGBoost, and the three seeds.

One deliberate asymmetry, stated because it is a choice and not an oversight:
XGBoost's `n_estimators` is not a grid axis. It is fixed at 2000 with
`early_stopping_rounds=50` on a 10% holdout carved out of **each fold's training
part**, so the outer fold remains untouched for scoring and the tree count is chosen
per fit. This is the conventional protocol for boosted trees and it removes an axis
that would otherwise triple the grid.

## Using the GPU for the forest: a screening arm, not a substitution

XGBoost already runs on the GPU through its own `device="cuda"` path, which
`run_tuning.sh` enables when it sees a device. cuML does not provide XGBoost and there
is nothing to add there.

The forest is a different matter. cuML has a GPU random forest, but it is **not a
faster scikit-learn forest**, and three differences decide how it can be used here:

1. `class_weight='balanced_subsample'` — the setting the published protocol uses — is
   rejected by cuML with a `ValueError`; only `None`, a dict and `'balanced'` are
   accepted. `'balanced'` reweights once on the full label vector rather than per
   bootstrap sample.
2. cuML's documentation states that its tree-node split algorithm differs from
   scikit-learn's, and it quantises each feature into `n_bins` candidate thresholds
   instead of considering every observed value.
3. The interpretability chapter loads joblib-pickled scikit-learn forests into
   `shap.TreeExplainer` and reads `feature_importances_`. A cuML forest cannot stand in
   there, so replacing the reported forest would change the model behind the ten
   features, the screening funnel and the seven prioritised compounds.

Because of (3) especially, cuML is used as a **ranking device over a much larger search
space**, never as the reported model:

```
screen_rf_cuml.py   320 configurations on GPU, class_weight='balanced'
        |           StratifiedKFold(5) on the same train rows
        v
    top k = 24  (+ a max_depth=None variant of each, an arm cuML cannot represent)
        |
        v
tune_baselines.py   refit in scikit-learn, class_weight='balanced_subsample',
  --candidates_from the same folds; the winner is chosen on the scikit-learn score
```

Every number that reaches the manuscript therefore comes from a scikit-learn forest
with the published class weighting, while the space searched grows from 81
configurations to 320 or 960.

Turn it on with `RF_MODE=screen` (see the notebook's step 2); `RF_MODE=grid` keeps the
built-in 81-configuration CPU grid and needs no GPU at all.

**The proxy is imperfect and the bundle measures it rather than assuming it — with a
control, not a correlation.** Alongside the shortlist, `--n_control` configurations
(12 by default) are drawn uniformly at random from the whole screened grid and refit in
scikit-learn on the same folds. Two numbers then appear in the log and in
`tuned_RF_s<seed>.json` under `screen_diagnostics`:

- `screening_gain` = the best shortlisted candidate's scikit-learn CV minus the best
  control's. **This is the decision-relevant number.** At or below zero, the screening
  bought nothing over drawing configurations at random from the same grid, and the
  result should be reported as an unscreened search or rerun with `RF_MODE=grid`.
- `spearman_spanning` = rank agreement between screen and scikit-learn scores over the
  shortlist *and* the controls together.

The first version of this diagnostic correlated the two rankings over the shortlist
alone, and a plumbing test returned rho = -0.83 — which says nothing about ranking
quality. The top candidates of a 96-configuration grid are near-tied, so restricting
the range to that slice leaves fold noise as the only variance and the coefficient
becomes meaningless. That is why the controls exist and why the gain, not the
correlation, decides whether to trust a screened result. A negative rank agreement
together with a positive gain is the normal signature of near-tied candidates, not a
failure.

This is not a theoretical caveat. In the plumbing test, the configuration the
confirmation stage finally selected was a `max_depth=None` variant — precisely the arm
the screen cannot represent, which is why that arm is added back on CPU.

### Class weighting in the screen is probed, not assumed

cuML's support for class weighting is version-dependent, and the two failure modes are
not the same thing. Some builds accept `class_weight` and reject only the value
`'balanced_subsample'` (a `ValueError`); others do not recognise the argument at all (a
`TypeError`), and on those a screen that passes `class_weight='balanced'` would crash on
its first fit rather than run.

The screen therefore probes with two-tree fits before doing any work and picks the best
available route, printing which one it took:

1. `class_weight='balanced'` if the build accepts it;
2. otherwise balanced `sample_weight` computed at fit time, using
   `n / (n_classes · count(class))` — verified here to reproduce
   `sklearn.utils.class_weight.compute_sample_weight('balanced', y)` exactly, and to give
   an identical screen score to route 1 on the scikit-learn backend, so the fallback is
   equivalent rather than merely similar;
3. otherwise unweighted, with a warning, and the choice recorded in
   `screen_RF_s<seed>_topk.json` under `weight_strategy`.

Override with `--weight_mode {auto,class_weight,sample_weight,none}`. The confirmation
stage is unaffected in every case: it refits in scikit-learn with
`class_weight='balanced_subsample'`, the published setting.

### Does the random forest ever run on the GPU?

No, and that is the design, not a misconfiguration. In `RF_MODE=screen` the GPU ranks
candidates and a **scikit-learn CPU forest** produces every reported number. In
`RF_MODE=grid` no GPU is involved on the forest side at all. Only XGBoost trains on the
GPU. So a message confirming that cuML refuses the published class weighting is the
screen reporting its own constraint — it says nothing about whether the reported forest
is accelerated, because by construction it never is.

One honest limitation of this bundle: the cuML code path itself has **not** been
executed here, because this machine has no GPU. The orchestration around it — the
grids, the fold construction, the shortlist handoff, the checkpointing and the
provenance recording — was tested end to end with `--backend sklearn`, which runs the
identical logic on CPU. The first real cuML call will be in your session, which is why
`screen_rf_cuml.py` refuses to start unless cuML imports and the driver reports a GPU,
and why it verifies in-session that `balanced_subsample` really is rejected rather than
trusting this README.

## What happens if the session dies

The grid runs for hours, so this is not a hypothetical. The run keeps two kinds of
checkpoint and resumes from them automatically.

**Per configuration.** Every configuration's cross-validation score is appended to
`<ckpt>/tuned_<MODEL>_s<seed>__<target>__cv.jsonl` and forced to disk with `fsync`
before the next configuration starts. A power cut costs at most the one configuration
that was mid-flight. On restart the run reads the log, skips every configuration it
already has, and keeps the best-so-far from those records.

**Per target.** When a target's winner has been refit and the evaluation sets scored,
its prediction columns are written to `<ckpt>/..._<target>__done.npz` through a
temporary file and an atomic rename, with the chosen hyperparameters in a sidecar
JSON. A restart reloads that target and does not refit it — so a crash during target
five never costs the four before it.

`run_tuning.sh` also skips any (model, seed) whose final JSON exists, and reports how
many targets are already checkpointed when it resumes.

Resumption is exact, not approximate. Each configuration's fold seed is derived from
`(seed, config_index)` rather than from a running generator, so a resumed run
evaluates the same folds in the same way as an uninterrupted one. This is tested: a
run was interrupted after 10 of 27 configurations, restarted, and all 27 CV scores,
the chosen hyperparameters and the CMNPD prediction column came out identical to the
uninterrupted run.

**The one thing you must do yourself.** A local directory dies with the virtual
machine, and then the checkpoints are worthless. Point them at storage that outlives
the session:

```bash
# Colab, after mounting Drive
export GLIOMA_CKPT=/content/drive/MyDrive/glioma_tuning/checkpoints
export GLIOMA_CACHE=/content/drive/MyDrive/glioma_tuning/cache
```

Step 0 of the notebook does this for you and prints what is already checkpointed.
`run_tuning.sh` prints a warning if the checkpoint directory does not look persistent.
Putting `GLIOMA_CACHE` there too means a restart does not repeat the fingerprint
featurisation, which is about ten minutes for the training matrix and the two
evaluation libraries together.

On Kaggle, `/kaggle/working` survives a **committed** session, so the defaults are
fine if you use *Save version*; an interactive session killed before a save loses it.
For the full grid across three seeds, Colab with Drive is the safer choice.

To start over deliberately rather than resume, pass `--fresh`.

## Run order

```bash
python smoke_test_baselines.py          # measures one fit, projects the whole run
GRID=full bash run_tuning.sh             # or GRID=reduced if the projection is too long
tar czf tuned_baselines.tar.gz metrics logs
```

`smoke_test_baselines.py` reports, from a real timed fit on this machine: the feature
matrix shape, whether XGBoost sees a GPU, the per-target training sizes and positive
counts, the minimum positives per fold, and a projected wall time for each of the four
grid/model combinations. **Read that projection before starting.** A run that dies at
hour eight leaves nothing behind.

Platform notes. The random forest is CPU-bound and does not benefit from a GPU, so its
cost is set by the core count — Kaggle's 4 vCPU beats Colab's 2 on the free tier.
XGBoost does benefit: pass a GPU session and `run_tuning.sh` will detect it and set
`device=cuda` for XGBoost while leaving the forest on CPU. The two models write
separate output files, so running one per session is a valid way to stay inside a time
limit. `run_tuning.sh` skips any run whose output JSON already exists, so a resumed
session continues where it stopped.

## If the runner prints its header and then stops

Fixed in this version. `run_tuning.sh` used to begin with `set -euo pipefail`, and the
line that counts existing checkpoints was `ls "$GLIOMA_CKPT"/..._done.npz | wc -l`. On a
first run nothing matches that glob, `ls` exits non-zero, `pipefail` propagates the
failure to the assignment, and `set -e` terminates the script — before any message from
inside the loop. The symptom was exactly: the `== grid=... ==` header, then silence, and
nothing written to `metrics/`.

The count is now a glob test that cannot fail, `set -e` is gone, and each run's exit
status is checked and reported so a failure names itself instead of ending the script.
A syntax check (`bash -n`) cannot catch this class of bug; only running the script can,
which is now part of the pre-release check.

### Setting the switches when you run with `!bash` rather than `%%bash`

Each `!` line in a notebook is its own subshell, so `!export RF_MODE=screen` on a line
by itself is forgotten the instant that line finishes. Three things that work:

```python
# 1. prefix on the same line
!RF_MODE=screen SCREEN_GRID=medium TOPK=24 bash run_tuning.sh

# 2. set them in the kernel environment, which `!` inherits
import os
os.environ['RF_MODE'] = 'screen'
os.environ['SCREEN_GRID'] = 'medium'
!bash run_tuning.sh

# 3. the cell magic, with %%bash alone on the first line
%%bash
export RF_MODE=screen
bash run_tuning.sh
```

The same applies to `GLIOMA_CKPT` and `GLIOMA_CACHE`: step 0 of the notebook sets them
through `os.environ`, which route 1 and 2 both inherit. If the runner warns that the
checkpoint directory is not persistent, those variables did not reach it.

Other switches worth knowing:

- `ONLY_TASK=CDK6` restricts everything to one target. Useful for a 10-minute end-to-end
  rehearsal on a new platform before committing hours to the real run.
- If a run fails, the message names its log and reminds you that whatever it had already
  checkpointed is kept, so rerunning resumes rather than restarting.

## Reading the smoke-test projection

On a Kaggle T4 session with 4 visible cores the measured projection is about 2.6 h for
the random-forest grid and 1.2 h for XGBoost, so **one seed is roughly 3.8 h and three
seeds roughly 11.4 h — more than Kaggle's 9 h limit.** Two consequences:

1. Persistent checkpoints are not optional for a three-seed run. Set `GLIOMA_CKPT` and
   `GLIOMA_CACHE` before starting, then continue in a second session.
2. `SEEDS='0'` alone fits comfortably and already answers the question the experiment
   asks, since the published run also tuned only seed 0.

Note also what the projection says about the GPU: XGBoost drops from about 10.7 s to
3.5 s per fit on a T4 — a real gain but a factor of three, not of thirty, because
5,604 rows by 2,078 features is a small problem for a GPU. The same ceiling applies to
a cuML forest, which is why the screening arm is justified by search-space size rather
than by raw speed.

## What this can do to the manuscript, before you run it

The result is not cosmetic and you should decide now how each outcome gets reported.

**Outcome 1 — the forest improves, the ranking is unchanged.** The most likely case.
The forest already beats every graph architecture while under-tuned, and a fuller
search can only raise its selection score. The central claim strengthens, the numbers
in Table 3 and the external columns move, and the manuscript needs the affected values
updated plus a Methods sentence describing the new protocol.

**Outcome 2 — XGBoost overtakes the forest internally.** Possible: the two are 0.002
apart in macro ROC-AUC in the published run (0.922 against 0.920), which is well inside
the seed spread, and XGBoost has the larger unexplored space. This one has a cascade.
The declared model-selection criterion for this study is **external marine performance,
not internal**, and on that criterion the forest led by a wide margin (Spearman 0.369
against 0.235; marine ROC-AUC 0.685 against 0.640). So an internal reversal alone does
not change the selected model — but it does change Table 3, the selection narrative in
Section 3.5, and it obliges the manuscript to state that the internal ranking of the two
ensembles is not stable under tuning.

**Outcome 3 — XGBoost overtakes the forest externally as well.** Then the selected model
changes, and with it the interpretability chapter (Figures 7 and 8, the ten features,
Table 6), the screening funnel, and the seven prioritised compounds. This is the
expensive branch. It is also the one a referee would most want tested, which is why the
experiment is worth running rather than asserting.

**Outcome 4 — neither improves materially.** Also informative: it says the baselines were
already at their ceiling and the six-candidate search was adequate, which retires a
reviewer objection at the cost of one sentence in Methods.

Whichever occurs, the published under-tuned numbers stay in the manuscript as a
hyperparameter-budget ablation, exactly as the 30-epoch graph results now stand beside
the converged ones. Nothing is overwritten and nothing is quietly dropped.

## What comes back, and what happens next

`metrics/` will hold, per model and seed:

- `tuned_<MODEL>_s<seed>.json` — the chosen configuration per target with its CV score,
  the number of configurations searched, and the training sizes
- `tuned_<MODEL>_s<seed>_cvtable.csv` — every configuration against every target with
  mean and standard deviation of the fold scores; this is the artefact that proves the
  search was exhaustive
- `tuned_<MODEL>_s<seed>_preds.npz` — validation and test probabilities in the same
  layout as the graph runs, so the existing evaluation code reads them unchanged
- `tuned_<MODEL>_s<seed>_{cmnpd,decoy,fda}.npz` — probabilities over the curated marine
  library, the ZINC20 decoy set and the FDA positive controls

Those external predictions are the reason this bundle re-scores the three evaluation
sets itself: without them the marine Spearman correlation, the decoy enrichment, the
FDA recovery and the screening funnel cannot be recomputed for the tuned models, and
the manuscript update would stop at the internal table.

Bring `tuned_baselines.tar.gz` back and the remaining work is mechanical: recompute the
internal benchmark, the external marine validation, the decoy enrichment and FDA
recovery, the paired tests against the graph architectures, a hyperparameter-budget
ablation table alongside the convergence one, and the edits those numbers force in the
abstract, Sections 2.11, 3.4, 3.5 and 3.13, and Table 3.
