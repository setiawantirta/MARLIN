# Reproducibility module

### *AI-guided triage of marine natural products against a six-target glioma panel: seven prioritized candidates and the limitations of graph neural networks in out-of-distribution chemical space*

This is the package referenced by the paper's **Availability of data and
materials** statement. It reproduces the study end to end: target discovery
from the glioma protein-interaction network, curation of the CMNPD marine
library, the six-task ChEMBL label matrix and its scaffold-disjoint split, the
control sets, training of all seven model families, internal and external
evaluation, consensus feature attribution, the marine screening cascade, the
two control experiments, every figure in the paper, and a verification stage
that asserts the published numbers.

Nothing here requires access to the authors; a stranger with this tarball, the
deposited data and a conda installation can run it.

---

## 1. Quick start

```bash
conda env create -f env/environment.yml
conda activate glioma-gnn
pip install -r env/requirements.txt          # torch, torch-geometric, shap, xgboost

./smoke_test.sh                              # ~1 min: does everything import and resolve?
./run_all.sh --check                         # which inputs are present, stage by stage
./run_stage.sh 11                            # regenerate every figure (~8 min)
./run_stage.sh 12                            # verify the headline numbers (~2 s)
```

`./run_stage.sh 12` is the one-command answer to "did this reproduce?" - it
prints a PASS/FAIL line per published value and exits non-zero on any failure.
It runs straight out of the box, because the result tables it checks ship
inside the package.

---

## 1b. Where to start

- `GUIDE_recomputing_figures.md` - the practical guide: which figure regenerates now,
  what unlocks the three that do not, what runs on a laptop versus what wants a GPU,
  how to re-render one figure during a review round, and how to put significance stars
  on a model-comparison panel.
- `stages/full_pipeline_colab.ipynb` - **the whole study, stages 01 to 12, on Colab or
  Kaggle**, designed for a referee who wants to reproduce it independently. Mounts
  persistent storage first, splits the run across sessions, says per stage whether a GPU
  helps, carries a ten-minute rehearsal, and ends on the 44-assertion verification.
  `NOTES_full_pipeline_notebook.md` records what was executed when it was built, the
  defects it fixed, and the per-stage timings it claims.
- `stages/stage09_colab.ipynb` - ready-to-run Colab notebook for the one stage that
  genuinely needs a GPU.
- `stages/retrieval_notebook.ipynb` - re-derives the glioma PPI network, target panel,
  ChEMBL label matrix and control sets from the public APIs. Replaces the two stage-01
  steps that depended on a hosted connector; the marine library is out of scope.

---

## 2. Hardware and wall clock

The **CPU-only path is the reference**. A GPU shortens only the two control
experiments (stages 9 and 10); it changes no result. Reference machine: 8 CPU
cores, 8 GB RAM, macOS arm64, no GPU, Python 3.12.

**Correction, found by reading the code rather than the table below.** Stage 5
does **not** use a GPU. `src/train_gnn.py`, the main-benchmark trainer, declares
no `--device` argument and contains no `.to(device)` call, and `src/baselines.py`
builds `XGBClassifier(tree_method="hist")` with no `device=`. Both run on CPU on
a GPU machine, so the "~3 h" GPU figure in the stage-5 row is not reachable with
the shipped code and is struck through. Only stages 9 and 10 reward an
accelerator; stage 10 only in its XGBoost arm, because the reported random forest
is scikit-learn on CPU by design.

| stage | what it does | CPU wall clock | GPU | peak RAM |
|---|---|---|---|---|
| 01 | target discovery from the PPI network | ~25 min (API-bound) | n/a | < 1 GB |
| 02 | marine library curation | ~45 min | n/a | ~3 GB |
| 03 | ChEMBL label matrix + scaffold split | ~3 h (API-bound) | n/a | ~4 GB |
| 04 | control sets (approved drugs, decoys) | ~70 min | n/a | ~3 GB |
| 05 | training, 7 model families x 3 seeds | **~26 h** | ~~~3 h~~ n/a | ~6 GB |
| 06 | evaluation, external validation, selection | ~90 min | n/a | ~4 GB |
| 07 | consensus feature attribution | ~4 h | n/a | ~6 GB |
| 08 | marine screening cascade | ~25 min | n/a | ~3 GB |
| 09 | convergence control (15 runs to convergence) | **~40 h** | ~4 h | ~6 GB |
| 10 | hyperparameter-budget control (15,930 fits) | **~30 h** | ~5 h | ~5 GB |
| 11 | regenerate all 12 figures | ~8 min | n/a | ~2 GB |
| 12 | verify the published numbers | ~2 s | n/a | < 1 GB |

Full `./run_all.sh` on the reference machine is roughly **four and a half days**
of wall clock. Stages 9 and 10 are independent controls and can be skipped
without affecting any main-text result; stage 10 is checkpointed per
configuration and is safe to interrupt and resume.

---

## 3. What ships inside and what must be downloaded

| | files | size |
|---|---|---|
| ships inside (`data/shipped/`) | 61 result tables | 9.2 MB |
| download: study data products | 21 files | 173 MB |
| download: CMNPD vendor release (not redistributable) | 2 files | see cmnpd.org |

`config/inputs_manifest.tsv` lists every input with its SHA-256, origin and the
stage that regenerates it. `data/README.md` explains how to populate the
directory. The shipped tables are enough to run stages 11 and 12; the
downloads are needed for stages 1-10.

---

## 4. Layout

```
config/config.yaml           every path, threshold and seed - no constant is buried in code
config/published_values.json the locked headline numbers that stage 12 asserts against
config/inputs_manifest.tsv   every input: size, sha256, origin, producing stage
env/environment.yml          conda environment, pinned to the versions used
env/requirements.txt         pip layer (torch, torch-geometric, shap, xgboost)
repro/config.py              the only path resolver; raises an actionable error, never a bare IOError
repro/figstyle.py            figure style; UPPERCASE BOLD SERIF panel letters; PDF + 300 dpi PNG
repro/compact.py             removes in-panel titles without moving any plotted value
repro/artist_digest.py       before/after digest proving the compaction changed nothing
repro/stagelib.py            the uniform stage driver (--check / --smoke / --list / --only)
src/                         the study's core modules (featurisation, models, training, evaluation)
stages/stageNN_*.py          the twelve numbered entry points
stages/legacy/stageNN/       the per-step scripts each stage drives
figures/                     one builder per figure
data/shipped/                the result tables that ship inside
outputs/                     everything a run produces
run_stage.sh  run_all.sh  smoke_test.sh
```

The stage numbering follows the published method flowchart (Fig. S1): stages
01-08 are its blocks **I-VIII**, in order. Stages 09-10 are the control
experiments, 11 regenerates the figures and 12 verifies.

Each stage runs alone:

```bash
./run_stage.sh 07 --list      # what its steps are
./run_stage.sh 07 --check     # whether its inputs resolve
./run_stage.sh 07 --smoke     # small-subset run, where one exists
./run_stage.sh 07             # the real thing
```

---

## 5. Figures

Stage 11 regenerates **eight main figures and four supplementary figures**, each
as a vector PDF plus a 300 dpi PNG. The numbering is that of the uploaded
manuscript: the convergence figure is supplementary **Fig. S4**, not a main
figure.

| figure | file stem | content |
|---|---|---|
| Fig. 1 | `fig01_glioma_ppi_network` | glioma protein-interaction network, four modules |
| Fig. 2 | `fig02_marine_library_properties` | chemical space and properties of the curated library |
| Fig. 3 | `fig03_internal_benchmark` | internal scaffold-disjoint benchmark, seven models |
| Fig. 4 | `fig04_external_validation` | decoy enrichment, FDA recovery, marine transfer, selection |
| Fig. 5 | `fig05_attribution_mechanism` | fingerprint versus graph-network attribution |
| Fig. 6 | `fig06_screening_funnel` | screening cascade and chemical-space position |
| Fig. 7 | `fig07_feature_statistics` | statistics of the ten consensus features |
| Fig. 8 | `fig08_annotated_structures` | annotated structures, panels A-J |
| Fig. S1 | `figS1_method_flowchart` | full method flowchart (pure matplotlib, no graphviz) |
| Fig. S2 | `figS2_dataset_overview` | label matrix and chemical-space overview |
| Fig. S3 | `figS3_polypharmacology_heatmap` | polypharmacology heatmap of the shortlist |
| Fig. S4 | `figS4_convergence` | convergence control (needs stage 09 - see below) |

Two conventions are enforced on **every** figure, main and supplementary:

* panel letters are **uppercase, bold, serif** - `repro.figstyle.panel_letter`
  ignores any `case="lower"` passed by a call site, so the supplementary
  figures come out in the main-text convention even though the author's own
  supplementary files left them lowercase;
* output is always a vector PDF next to a 300 dpi PNG.

Main-text figures additionally carry **no in-panel titles**; the title text
lives in the caption. `repro/compact.py` removes them from the finished figure
and closes the band they occupied without resizing any data axes, and
`outputs/stage11_figures/digests/` holds the before/after artist digests that
prove no plotted value moved. Supplementary figures keep their in-panel titles,
matching the manuscript.

This is the direct fix for the resolution problem in the Word manuscript, whose
embedded main-text figures are 110-153 dpi at 7-inch width (Fig. 1 is 287 dpi)
against the journal's 300 dpi requirement.

**Fig. S4 is the one figure stage 11 cannot produce from shipped inputs.** It
plots the per-epoch validation-loss trajectories of the 15 convergence runs,
which exist only inside `outputs/stage09_control_convergence/metrics/`. No
summary table substitutes for them. Run stage 9 first; stage 11 then picks the
figure up automatically. Until then it is reported `SKIPPED` with that reason.

---

## 6. Verification

```bash
./run_stage.sh 12
```

Stage 12 re-derives **44 published values** from the shipped result tables and
asserts each against `config/published_values.json`:

* the seven internal macro ROC-AUC values (per-seed-mean convention);
* the external marine arm - set size 194, 51 actives at pActivity >= 6, random
  forest rho 0.369 (p = 1.1e-07) and ROC-AUC 0.685, XGBoost 0.235 and 0.640,
  all five graph networks inside the 0.382-0.502 chance band;
* the random forest's significant advantage over every graph network
  (delta 0.183-0.303, q <= 0.0057) and its **non**-significant advantage over
  XGBoost (delta 0.045, q = 0.085);
* the hyperparameter-budget control (81 and 96 configurations, 15,930 fits,
  0.922 -> 0.923 internal and 0.685 -> 0.682 marine for the random forest);
* the convergence control (15 runs, early stopping between epochs 34 and 86,
  converged marine ROC-AUC 0.394-0.538, every architecture still below the
  random forest);
* the screening cascade (26,754 -> 7) and the dataset sizes;
* that every regenerated figure is a 300 dpi PNG with a vector PDF beside it.

A check whose input table is absent is reported `SKIP`, never `PASS`. The
science is locked: **a FAIL means the regenerated value disagrees with the
manuscript, and the published value is never adjusted.**

Reference run of this package: **44 PASS, 0 FAIL, 0 SKIP.**

---

## 7. Configuration

`config/config.yaml` is the single source of truth. Thresholds that change a
result and therefore change what stage 12 reports include:

```yaml
seeds:         model_seeds [0, 1, 2]
target_discovery:  string_combined_score_min 0.70, hub_chembl_min_compounds 300
label_matrix:  pchembl_active_min 6.0, pchembl_inactive_max 5.0, confidence_score_min 8
controls:      max_tanimoto_to_any_active 0.35
features:      fp_radius 2, fp_bits 2048, 30 RDKit descriptors
training:      epochs 30, patience 6, batch 256, lr 1e-3        # the MAIN benchmark
control_convergence: epochs_max 1000, patience 25, capacity full  # stage 9 only
evaluation:    selection_criterion external_marine_roc_auc
screening:     min_targets_active 2, applicability_domain_tanimoto_min 0.30
```

The selection criterion is the load-bearing one: **the model is chosen on
external marine ROC-AUC, not on internal test performance.** That is what makes
the random forest, not a graph network, the selected model.

**`control_convergence.epochs_max: 1000` is a ceiling, not a training length.**
Stage 9 raises the main benchmark's 30-epoch ceiling to 1,000 and its patience
from 6 to 25, then lets early stopping decide: in the published run it fired in
all fifteen runs between epochs 34 and 86, so no run came near the ceiling.
Training a literal 1,000 epochs would carry three of the five architectures past
their validation minimum and reproduce nothing. The two values were previously
recorded here as 200 and 20 while `run_convergence.sh` hardcoded 300 and 25 and
ignored them; the driver now reads them, and they are set to the values the
manuscript describes.

---

## 8. Paper-to-stage map

Every figure, table and headline number in the paper, with the stage that
produces it and the file it lands in. Output paths are relative to
`outputs/`; files marked *shipped* are already in `data/shipped/`.

### Main figures

| paper | stage | output |
|---|---|---|
| Fig. 1 glioma PPI network | 01 -> 11 | `stage11_figures/fig01_glioma_ppi_network.{pdf,png}` |
| Fig. 2 marine library properties | 02 -> 11 | `stage11_figures/fig02_marine_library_properties.{pdf,png}` |
| Fig. 3 internal benchmark | 06 -> 11 | `stage11_figures/fig03_internal_benchmark.{pdf,png}` |
| Fig. 4 external validation | 06 -> 11 | `stage11_figures/fig04_external_validation.{pdf,png}` |
| Fig. 5 attribution mechanism | 07 -> 11 | `stage11_figures/fig05_attribution_mechanism.{pdf,png}` |
| Fig. 6 screening funnel | 08 -> 11 | `stage11_figures/fig06_screening_funnel.{pdf,png}` |
| Fig. 7 feature statistics | 07 -> 11 | `stage11_figures/fig07_feature_statistics.{pdf,png}` |
| Fig. 8 annotated structures (A-J) | 07 -> 11 | `stage11_figures/fig08_annotated_structures.{pdf,png}` |

### Supplementary figures

| paper | stage | output |
|---|---|---|
| Fig. S1 method flowchart | 11 | `stage11_figures/figS1_method_flowchart.{pdf,png}` |
| Fig. S2 dataset overview | 03 -> 11 | `stage11_figures/figS2_dataset_overview.{pdf,png}` |
| Fig. S3 polypharmacology heatmap | 08 -> 11 | `stage11_figures/figS3_polypharmacology_heatmap.{pdf,png}` |
| Fig. S4 convergence control | **09** -> 11 | `stage11_figures/figS4_convergence.{pdf,png}` |

### Main tables

| paper | stage | source file |
|---|---|---|
| Table 1 six-protein multi-task panel | 01 | `glioma_target_panel.csv` *(shipped)* |
| Table 2 label accounting per task | 03 | `dataset_summary.csv` *(shipped)* |
| Table 3 internal benchmark | 06 | `benchmark_publication_table.csv`, `benchmark_summary.csv` *(shipped)* |
| Table 4 external validation, 194 + 217 | 06 | `external_phenotypic_validation.csv`, `enrichment_results.csv`, `fda_control_recovery.csv` *(shipped)* |
| Table 5 twelve highest-ranked candidates | 08 | `cmnpd_top_candidates.csv` *(shipped)* |
| Table 6 ten consensus features | 07 | `top10_features.csv`, `ecfp4_bit_dictionary.csv` *(shipped)* |
| Table 7 provenance of the seven survivors | 08 | `seven_survivors.csv`, `seven_compound_provenance.csv` *(shipped)* |
| Table 8 convergence and capacity control | 09 | `convergence_ablation_table.csv`, `converged_per_task.csv` *(shipped)* |
| Table 9 hyperparameter-budget control | 10 | `hyperparameter_budget_ablation.csv`, `tuned_per_task.csv` *(shipped)* |

### Headline numbers

Each row is asserted by stage 12; the check name is what `./run_stage.sh 12`
prints.

| number | value | stage | source file |
|---|---|---|---|
| internal macro ROC-AUC, per-seed mean | RF 0.922 / XGB 0.920 / D-MPNN 0.893 / GIN 0.882 / AttentiveFP 0.877 / GraphGPS 0.869 / GATv2 0.867 | 06 | `benchmark_summary.csv` |
| internal macro ROC-AUC, seed-averaged prediction | 0.923 / 0.921 / 0.902 / 0.894 / 0.888 / 0.881 / 0.874 | 06 | `benchmark_metrics.csv` |
| external marine set | 194 compounds, 51 active at pActivity >= 6 | 06 | `external_phenotypic_validation.csv` |
| random forest, marine | rho 0.369 (p = 1.1e-07), ROC-AUC 0.685 | 06 | `benchmark_publication_table.csv` |
| XGBoost, marine | rho 0.235 (p = 9.6e-04), ROC-AUC 0.640 | 06 | `benchmark_publication_table.csv` |
| all five graph networks, marine | ROC-AUC 0.382-0.502 (chance) | 06 | `benchmark_publication_table.csv` |
| RF over each graph network, marine | delta 0.183-0.303, q <= 0.0057 | 06 | `significance_tests.csv` |
| RF over XGBoost, marine | delta 0.045, q = 0.085 - **not** significant | 06 | `significance_tests.csv` |
| censored-inclusive replicate | 217 compounds, RF ROC-AUC 0.673 | 06 | `external_phenotypic_validation.csv` |
| applicability domain | 8.2 % of the marine library inside Tanimoto 0.30 | 03 | `applicability_domain.csv` |
| screening cascade | 26,754 -> 7 survivors, six staurosporine-class indolocarbazoles | 08 | `screening_funnel.csv`, `seven_survivors.csv` |
| convergence control | 15 runs, early stop epochs 34-86, marine ROC-AUC 0.394-0.538 | 09 | `convergence_ablation_table.csv`, `converged_external_marine.csv` |
| hyperparameter control | 81 + 96 configurations, 15,930 fits; internal 0.922 -> 0.923 and 0.920 -> 0.922; marine 0.685 -> 0.682 and 0.640 -> 0.649 | 10 | `hyperparameter_budget_ablation.csv` |

---

## 9. Discrepancy on record

One number differs between sources, and this module does not silently pick a
winner.

For the **random forest versus XGBoost comparison on the external marine set**,
the uploaded manuscript states *"its advantage over the gradient-boosted
ensemble was not significant (delta 0.045, q = 0.085)"*, and
`significance_tests.csv` carries delta 0.045249, 95 % CI -0.005348 to 0.097962,
q (BH, paired bootstrap) 0.0845 - the manuscript and the result table agree.

A specification handed to the author of this module quoted a different triple
for the same comparison: delta +0.0339, 95 % CI -0.010 to +0.076, p = 0.119.
That triple appears nowhere in the manuscript text and does not match the
result table. **Neither value has been altered.** Stage 12 verifies against the
manuscript, because the uploaded Word document is the authoritative source of
record; the other triple is recorded in `config/published_values.json` under
`rf_vs_xgb_external._task_brief_variant` so that the disagreement stays visible.

The qualitative conclusion is identical either way: the difference between the
two fingerprint ensembles is not statistically significant, and the manuscript
says so.

---

## 10. Known limitations

* **Fig. S4 needs stage 9.** See section 5.
* **Live APIs drift.** Stages 1, 3 and 4 query services that change. The cached
  responses from the original run are in the data deposit and are preferred
  when present; delete them to force a live re-fetch, and expect small
  differences downstream.
* **CMNPD is not redistributable.** Stages 2 and 3 need two vendor files you
  must download yourself (`data/README.md`).
* **PyYAML is optional.** `repro/config.py` carries a fallback parser for
  `config.yaml`; the reference run used it.
* **No GPU is required anywhere.** The CUDA path differs only in the torch
  wheel.
