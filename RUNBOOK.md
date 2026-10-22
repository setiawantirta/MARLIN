# Runbook

How to run this module, what each stage takes, and where the data has to sit. This is
the operating document; `README.md` describes what the module *is*.

---

## 1. Invoke scripts with `bash`, never `./`

```python
subprocess.run(["bash", "smoke_test.sh"], cwd=str(REPO),
               env=dict(os.environ, PYTHONPATH=str(REPO)), check=True)
```

A tarball unpacked onto Google Drive does not keep the POSIX executable bit, and
`chmod +x` does not stick on a Drive mount — so `["./smoke_test.sh"]` raises
`PermissionError` there. `bash script.sh` ignores the bit entirely and behaves the
same on Colab, on Kaggle and on a local checkout. Every notebook cell in this module
now uses that form; so should anything you write.

The stage number may be given either way: `bash run_stage.sh 3` and
`bash run_stage.sh 03` are the same call.

---

## 2. Where the data goes

Inputs are resolved **by basename, recursively, under a list of roots**. You do not
have to reproduce any particular folder layout — put a file anywhere under a root and
it is found.

Roots are searched in this order:

1. `$MARINE_REPRO_DATA`, if set
2. every entry of `paths.data_roots` in `config/config.yaml`
3. `<repo>/data/shipped`
4. the output directory (so a later stage finds what an earlier one wrote)

A root that does not exist is skipped silently, and so is one that exists but cannot
be read, so listing several spellings of the same folder costs nothing.

`config/config.yaml` already lists the current project folder in both spellings:

```
/content/drive/MyDrive/Projek/tirta/BMC-cmnpd_screening/glioma_final          # Colab
~/Library/CloudStorage/GoogleDrive-.../BMC-cmnpd_screening/glioma_final       # macOS client
```

### The two CMNPD vendor files

Stage 02 needs `CMNPD_1.0_calc_prop.tsv`. Put it anywhere under a root — for example
`glioma_final/CMNPD_Data/`. The 2D SDF (`CMNPD_1.0_2d.sdf`, or `CMNPD_1.0.sdf`; either
name is accepted) is **optional**: it is opened only to recover molblocks for SMILES
that RDKit cannot parse, and skipped entirely when none fails.

> Earlier versions asked for a *directory* called `cmnpd_screening` and then joined
> `data semua senyawa` onto it. The resolver returns files, never directories, so that
> lookup could not succeed on any machine but the original one and failed with a
> misleading "intermediate written by an earlier stage". Both files are now resolved by
> their own basenames.

---

## 3. The command surface

Every stage answers the same flags.

| Flag | Effect |
|---|---|
| *(none)* | run the stage |
| `--check` | resolve inputs, run nothing |
| `--list` | print the step list |
| `--only A B` | run only these steps (or, for stage 11, these figure numbers) |
| `--smoke` | small-subset rehearsal |
| `--force` | re-run steps whose declared outputs already exist |
| `--no-progress` | turn off the progress bar |
| `--quiet-steps` | buffer step output instead of streaming it |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | everything asked for succeeded |
| `1` | `--check`: some declared input is absent here. A report, not a crash. |
| `2` | a real run refused to start: inputs missing, or the stage cannot run here |

`--check` used to return `0` unconditionally, so a wrapper that reported only the exit
code called a stage ready when every step had just printed `MISSING`.

---

## 4. Resume: nothing is recomputed

Two independent mechanisms, both on by default.

**Per step, by declared output.** Every step declares what it writes. A step whose
declared outputs all resolve is skipped:

```
  run_curation                           skipped: all 2 output(s) already present  [--force to re-run]
```

Stage 11 applies the same rule per figure, testing for the PNG and the PDF. This is
what makes a dropped Colab session cheap: re-run the same cell and only the unfinished
work executes. Use `--force` to override.

**Inside the two long controls.** Stage 10 checkpoints each hyperparameter
configuration to JSONL as it finishes, keyed by an md5 fingerprint of the candidate
list, so a changed shortlist starts fresh rather than resuming against the wrong
configurations. Point `GLIOMA_CKPT` at persistent storage.

For this to survive a session, **the output directory must be on persistent storage**:

```python
os.environ["MARINE_REPRO_OUTPUT"] = str(PERSIST / "outputs")
```

On Colab, `/content` is wiped when the runtime ends. Point it at Drive.

---

## 5. Progress

`tqdm` advances once per completed step, and step output is **streamed live** rather
than buffered — the two together are what make a long stage legible: the bar says how
far through the stage you are, the stream says the running step is alive. The longest
step in the pipeline, the curation of 31,561 CMNPD records, carries its own bar over
compounds with an ETA.

A stage with one long step will sit at `0/1` for hours. That is expected; watch the
stream, not the bar. `tqdm` is optional — without it the loops fall back to periodic
prints.

---

## 6. The stages

Wall clocks are from the original 8-core CPU run. Stages marked **network** call
public APIs and need outbound access.

| # | Stage | Hardware | Wall clock | Steps |
|---|---|---|---|---|
| 01 | target discovery | CPU, **network** | ~25 min, API-bound | 9 |
| 02 | marine library curation | CPU | ~45 min | 1 |
| 03 | ChEMBL label matrix | CPU, **network** | ~3 h, retrieval-bound | 10 |
| 04 | control sets | CPU, **network** | ~70 min | 3 |
| 05 | training, seven families | CPU, GPU optional | ~26 h CPU / ~3 h GPU | 7 |
| 06 | evaluation and selection | CPU | ~90 min | 4 |
| 07 | consensus attribution | CPU | ~4 h | 7 |
| 08 | marine screening cascade | CPU | ~25 min | 1 |
| 09 | convergence control | GPU preferred | ~40 h CPU | — |
| 10 | hyperparameter-budget control | GPU preferred | ~5 h GPU / ~30 h CPU | — |
| 11 | regenerate every figure | CPU | ~8 min | 12 figures |
| 12 | verify published numbers | CPU | ~2 s | 44 checks |

Stages 09–12 carry no `wall_clock` metadata in code, so the two figures above are the
ones `README.md` documents for the reference run. Re-measured on an 8-core CPU while
writing this: stage 11 took 5.4 min for all twelve figures with `--force`, stage 12
took 0.5 s for the 44 checks.

Stage 05 is too long for one Colab session. Run it in the three groups the notebook
defines, in order; group B is resumable.

### Knobs that are not command-line flags

Stages 09 and 10 read their settings from `config/config.yaml`
(`control_convergence.epochs_max`, `control_convergence.patience`) and pass them down
as `EPOCHS` and `PATIENCE`. Stage 10 accepts `ONLY_TASK=CDK6` in the environment to
restrict every fit to one target, which turns a 30-hour stage into a ten-minute
rehearsal.

### Environment variables

| Variable | Purpose |
|---|---|
| `MARINE_REPRO_DATA` | first input root, searched before the config list |
| `MARINE_REPRO_OUTPUT` | output directory — **put this on persistent storage** |
| `MARINE_REPRO_CACHE` | cache directory |
| `MARINE_REPRO_CONFIG` | alternative config file |
| `GLIOMA_CKPT` | stage-10 checkpoint directory |

---

## 7. Figure conventions

Every figure is generated by `stages/stage11_figures.py`, which calls the scripts in
`figures/`. The shared rules live in `repro/figstyle.py` and `repro/siglabels.py` and
are applied centrally, so they hold for any figure added later.

- **Panel letters** uppercase bold serif, via `uppercase_existing_panel_letters()`.
- **No titles inside panels.** That information belongs in the caption; the journal
  sets figure titles separately.
- **Axis labels** are capitalised by `capitalise_axis_labels(fig)`, called inside
  `save_figure()`. The rule: capitalise the first letter *unless* it is immediately
  followed by a capital, which identifies a scientific symbol whose lowercase prefix
  carries meaning — so `pActivity`, `pChEMBL` and `cLogP` are left alone.
- **Significance asterisks** on three tiers — `*` q < 0.05, `**` q < 0.01,
  `***` q < 0.001 — Benjamini–Hochberg corrected within each panel's family of
  comparisons. `DEFAULT_THRESHOLDS = (0.05, 0.01, 0.001)` in `repro/siglabels.py`;
  `stars()` counts cleared thresholds, so changing the tuple changes the scale and
  `star_legend()` regenerates the caption sentence to match.
  Where every contrast saturates at `***`, the brackets also carry Cohen's *d*: on a
  panel whose largest q is 10⁻⁹, asterisks alone cannot distinguish a negligible
  effect from a large one.
- **Annotation text is black; reference rules are coloured.** A median or cutoff rule
  is drawn red or grey-dashed to read as annotation rather than as another data
  series, with its label centred directly above it and set in black.
- **Every marker carries its own value** where the panel is a ranked list.
- **Output**: a vector PDF (submit this) plus a 300 dpi PNG at the figure's designed
  width. `save_figure()` writes both.

Re-render one figure with `bash run_stage.sh 11 --only 7`, and force a redraw of an
existing one with `--force`.

---

## 8. If something fails

| Symptom | Cause |
|---|---|
| `PermissionError` on `./script.sh` | Drive does not carry the exec bit — use `bash script.sh` |
| `MissingInput: ... found as a DIRECTORY` | the stage asked for a folder; it should name the file it reads |
| `MissingInput` on a vendor file | put it anywhere under a configured root; see §2 |
| `--check` exits 1 | inputs absent on this machine. On a checkout with no data roots, measured: stage 01 and stage 11 only |
| a stage "does nothing" | its outputs are already present — that is the resume rule. `--force` to re-run |
| outputs vanish between sessions | `MARINE_REPRO_OUTPUT` points at ephemeral disk |

`bash smoke_test.sh` exercises imports, config parsing, every stage's `--check`, a
figure render and the 44 verification checks in about two minutes. Run it first.
