# Recomputing the figures: what to run, and where

A practical guide for the review rounds. The short answer to "do I need Colab?" is
**no, for almost everything**. Only one figure genuinely needs a GPU session; the
other two that are currently missing run on your own laptop overnight.

> **If you want the whole study rather than the figures**, use
> `stages/full_pipeline_colab.ipynb` instead of this guide. It runs stages 01-12 end to
> end on Colab or Kaggle, mounts persistent storage first, splits the ~4.5-day run across
> twelve sessions, marks per stage whether a GPU helps, and finishes on the 44-assertion
> verification. `NOTES_full_pipeline_notebook.md` has the per-stage timings and the
> defects that building it exposed - including that **stage 5 does not use a GPU**
> despite the README's old claim, so the only stages worth an accelerator are 9 and 10.

---

## 1. Where each figure stands today

| figure | regenerates now? | what it needs | where to run it |
|---|---|---|---|
| Fig. 1 PPI network | **yes** | shipped data | laptop, 4 s |
| Fig. 2 marine library | **yes** | shipped data | laptop, 2 s |
| Fig. 3 internal benchmark | **yes** | shipped data | laptop, 13 s |
| Fig. 4 external validation | **yes** | shipped data | laptop, 11 s |
| Fig. 5 attribution mechanism | no | stage 07 SHAP caches | **laptop, ~4 h** |
| Fig. 6 screening funnel | **yes** | shipped data | laptop, 145 s |
| Fig. 7 feature statistics | **yes** | shipped data | laptop, 4 s |
| Fig. 8 annotated structures | no | stage 03 raw caches | **laptop, ~3 h** |
| Fig. S1 flowchart | **yes** | shipped data | laptop, 2 s |
| Fig. S2 dataset overview | **yes** | shipped data | laptop, 3 s |
| Fig. S3 polypharmacology | **yes** | shipped data | laptop, 198 s |
| Fig. S4 convergence | no | stage 09 per-epoch metrics | **Colab/GPU, ~4 h** |

Nine of twelve already regenerate from what ships in the package plus the study's
stored data products. The three that do not are missing *intermediate* files that
the original runs wrote to scratch and never kept.

**Only Fig. S4 wants a GPU.** Stage 09 retrains fifteen graph networks to
convergence: about 40 hours on eight CPU cores against roughly 4 hours on one
GPU. Stages 03 and 07 are not GPU work at all — stage 03 is network-bound on the
ChEMBL API and stage 07 is exact TreeSHAP on a random forest, which has no GPU path.

---

## 2. One-time setup

```bash
tar xzf marine_antiglioma_reproducibility.tar.gz
cd marine_antiglioma_reproducibility

conda env create -f env/environment.yml     # or: pip install -r env/requirements.txt
conda activate marine-antiglioma

export MARINE_REPRO_DATA=/path/to/your/data_directory
```

`MARINE_REPRO_DATA` is where the large inputs live. The 61 result tables under
`data/shipped/` come with the package; the 21 larger study data products go in the
data directory. `config/inputs_manifest.tsv` lists every input with its SHA-256,
its origin and the stage that regenerates it, and `data/README.md` explains how to
populate the directory.

Check before committing to a long run — this resolves inputs and runs nothing:

```bash
./run_stage.sh 11 --check
```

Each figure reports `OK` or names exactly which files are missing.

---

## 3. The three commands you will actually use during review

```bash
./run_stage.sh 11                 # regenerate every figure, ~8 min
./run_stage.sh 11 --only 3 4      # just figures 3 and 4, ~25 s
./run_stage.sh 12                 # re-assert the published numbers, ~2 s
```

Stage 11 writes both formats for every figure into
`outputs/stage11_figures/`: a vector PDF to submit and a 300 dpi PNG at the
figure's designed width.

Stage 12 is the one to run after *any* change. It re-derives 44 headline numbers
from the shipped outputs and prints `PASS`/`FAIL` per item, so a figure edit that
silently moved a value is caught immediately rather than at proof stage.

---

## 4. Unlocking Fig. 5 and Fig. 8 — on your laptop

Both need raw caches that stage 03 and stage 07 produce.

**Fig. 8** needs `cmnpd_raw.npz` and `chembl_raw.npz` from stage 03. Stage 03 also
needs the two CMNPD vendor files, which are not redistributable and are not in the
package — but **you already have them**, in your `cmnpd_screening` folder:
`CMNPD_1.0_calc_prop.tsv` and `CMNPD_1.0.sdf`. Copy them into your data directory
and run:

```bash
./run_stage.sh 03          # ~3 h, API-bound, no GPU
./run_stage.sh 11 --only 8
```

**Fig. 5** needs the per-target SHAP caches `shap_test_*.npz` and
`shap_marine_*.npz` from stage 07:

```bash
./run_stage.sh 07          # ~4 h CPU, ~6 GB peak RAM
./run_stage.sh 11 --only 5
```

Run stage 07 overnight. It is the single heaviest CPU step you will do locally, and
8 GB of RAM is enough but not generous — close other applications.

---

## 5. Fig. S4 — the one Colab job

```python
# Colab cell 1: mount Drive FIRST, so a recycled runtime does not lose the run
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['MARINE_REPRO_DATA'] = '/content/drive/MyDrive/glioma/data'
os.environ['GLIOMA_CKPT']       = '/content/drive/MyDrive/glioma/checkpoints'
```

```python
# Colab cell 2
!tar xzf marine_antiglioma_reproducibility.tar.gz
%cd marine_antiglioma_reproducibility
!pip install -q -r env/requirements.txt
```

```python
# Colab cell 3 — about 4 h on a T4
!./run_stage.sh 09
!./run_stage.sh 11 --only S4
```

Two things that cost real time if you skip them:

- **Mount Drive and point the checkpoint variable at it before starting.** Colab
  recycles runtimes. A four-hour run on local VM disk is lost entirely when that
  happens; on Drive it resumes.
- **If you drive the shell with `!bash` rather than `%%bash`,** remember each `!`
  line is its own subshell, so `!export VAR=x` on its own line is forgotten
  immediately. Set variables through `os.environ` as above, or prefix them on the
  same line as the command.

`stages/legacy/stage09/convergence_notebook.ipynb` is the notebook from the
original run and carries the same Drive-mounting pattern.

---

## 6. Adding significance stars to a figure

This is what `repro/siglabels.py` is for, and it is designed for exactly the
review-round case: annotate an existing figure without touching the code that drew it.

```python
from repro.siglabels import load, stars, vs_reference, star_legend

sig = load()                                  # significance_tests.csv
stars(0.0042)                                 # -> '**'
vs_reference(sig, 'MACRO')                    # every model vs the random forest
star_legend()                                 # the caption sentence
```

Default thresholds are **`*` q < 0.05** and **`**` q < 0.005**. Pass
`thresholds=(0.05, 0.01, 0.001)` for the more common three-tier scheme.

Two helpers place the stars:

- `annotate_heatmap(ax, rows, cols, sig)` appends a star to the number already in
  each heatmap cell. Appending rather than adding a second text object is
  deliberate: those cells are set in 6 pt and a separate artist would collide.
- `star_points(ax, labels, star_map, values)` puts a star just past each marker in
  a one-model-per-row panel.

Both are already wired in: Fig. 3 panel C carries per-cell stars and Fig. 4 panel E
carries per-model stars. Re-running stage 11 reproduces them.

### Read the corrected q, not the raw p

`significance_tests.csv` holds both, and the number of comparisons differs by scope:
the six per-target scopes and `MACRO` each carry all 21 pairs of the seven model
families, while `MARINE_PHENOTYPIC` carries 6 — the random forest against each of the
others, because that scope exists to document the model-selection decision. Both
counts warrant correction: 21 uncorrected tests at 0.05 expect about one false
positive per panel and 6 expect about one in three, and multiplicity is the first
thing a referee raises about a starred benchmark figure. The helpers default to
`q_bootstrap`, the
Benjamini-Hochberg corrected value, and `star_legend()` writes the thresholds, the
test and the correction into one sentence so the caption cannot drift from what was
actually drawn.

### What the stars currently say

On the internal benchmark (Fig. 3C) every graph network differs from the random
forest at `*` on the macro column, and XGBoost does not differ anywhere. On the
marine set (Fig. 4E) GIN, D-MPNN and AttentiveFP reach `**`, GATv2 and GraphGPS
reach `*`, and XGBoost again does not separate. IDH1 is the one target where
nothing is significant.

Note what is **not** starred and must not be: the random forest is the reference, so
it carries no star against itself, and the forest-versus-XGBoost comparison is not
significant on the marine set. The manuscript states that explicitly; the figure
should not appear to contradict it.

---

## 7. Changing a figure during review

1. Edit the script in `figures/` — one file per figure, named for the figure.
2. `./run_stage.sh 11 --only <n>` to re-render just that one.
3. `./run_stage.sh 12` to confirm no reported number moved.
4. Replace the PDF in your submission package.

The figure scripts read their inputs through `repro/config.py` and their style
through `repro/figstyle.py`, so a change to the panel-letter convention or the
output size is made once in `figstyle.py` rather than in twelve places.

---

## 8. Caption corrections found while adding the stars

Two numbers in the uploaded captions do not match the stored results. Both were
checked against the code that wrote `significance_tests.csv` and against the file
itself; neither affects any conclusion, but both are wrong as printed.

**Fig. 3** — "a 2,000-resample stratified bootstrap". The resample count is 500 for
the macro column and 1,000 for the per-target columns; 2,000 is `paired_bootstrap`'s
function default and is overridden at every call site. The p-value floors in the file
confirm it: 1/501 = 0.001996 for macro rows, 1/1001 = 0.000999 for per-target rows.
And it is not stratified - resampling draws from the pooled rows and skips any draw
that leaves one class empty. It is paired, since the same resampled rows score both
models.

**Fig. 4, panel E** — "All five graph networks fall within the interval 0.457-0.502".
The range is 0.382-0.502. The caption's own delta range, 0.183-0.303, proves it: with
the random forest at 0.685, a delta of 0.303 requires a floor of 0.382, not 0.457.
0.457 is AttentiveFP, the second-lowest; D-MPNN at 0.382 was left out. The delta range
and "q <= 0.0057" are both correct.

## 9. Annotating Fig. 5

Fig. 5 is a different annotation problem from Figs. 3 and 4 and should not get bare
stars. All eight of its displayed contrasts reach ** after BH correction across the
eight, because the comparisons are over molecules rather than seven models: n runs
from 126 to 6,445. Seven of the eight sit at 1,164-6,445; the GNNExplainer row of
panel F is a 126-molecule subsample, which is why its q is 4.6e-10 rather than the
e-56 and smaller of its neighbours. A star alone would
print the same symbol on Cohen's d = 1.55 in panel B and d = -0.17 in panel D, which
inverts the message: panel D's difference is negligible and significant only because
n = 6,445 against 1,164.

`repro.siglabels.attribution_tests()` therefore returns the effect size with the q,
and the figure labels each panel `d = X.XX, **`. Panel E is deliberately unlabelled:
it shows six cosine similarities with no comparison group, so there is no test to
report and a star there would mean inventing a null.

The statistics come from `gnn_vs_rf_attribution.csv`, which is already stored, so the
numbers for the caption are available now. Drawing them onto the figure needs stage 07
first, because Fig. 5 cannot render without the SHAP caches.

---

## 10. Re-running the data retrieval

`stages/retrieval_notebook.ipynb` re-derives everything the study pulled from public
databases: the glioma network and its target panel (stage 01), the ChEMBL label matrix
(stage 03) and the control sets (stage 04). The marine library is out of scope - CMNPD
has no public API and the curated library ships as stored data.

Two defects in the shipped stage 01 are worked around there, both found by running it:

**The connector dependency.** `01_fetch_opentargets_civic.py` and
`03_fetch_string_network.py` reach Open Targets, CIViC and STRING through a hosted
connector (`host.mcp`) that exists only in the environment the study ran in, so neither
executes on a reader's machine. `repro/fetch_sources.py` replaces those calls with the
services' public APIs, and `repro/host_shim.py` feeds the results to the study's own
scripts so the seed-selection logic is not reimplemented. Verified result: 186 seed
genes with 184 shared with the published set, a 186-node network with a 176-node giant
component and 10 isolated seeds - all matching - and the identical six-target panel.

**A missing input.** `07_fetch_chembl_hub_counts.py` reads
`handoff/chembl_targets.json`, which no script in the module writes; it was produced
outside the shipped pipeline. `fetch_chembl_targets()` builds it from the ChEMBL target
API, keeping only reviewed human single-protein targets. It maps 135 of the 186 seed
genes, which is enough for every hub that clears the compound-count threshold.

Two further notes. Stage-01 scripts write some outputs to the working directory and
some to `handoff/`, so look in both. And any subprocess running a stage script needs the
repo root on `PYTHONPATH` - that is what `run_stage.sh` sets, and step 08 fails without
it.

**Reactome is not reachable from every network.** reactome.org sits behind Cloudflare
and answers some automated clients with error 1010; it does so from the environment this
module was prepared in. Do not disguise the client to get around it. The pathway
enrichment is the only thing affected and it is optional:
`08_select_target_panel.py` reads only `chembl_counts.json` and `centrality.csv`, so the
panel is selected without it. GO annotations, which the same enrichment step also needs,
come through UniProt and QuickGO and are reliable.
