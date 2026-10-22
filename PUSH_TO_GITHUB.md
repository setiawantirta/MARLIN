# Pushing this to github.com/setiawantirta/MARLIN

This directory is already a git repository: one commit on `main`, with `origin` set to
`https://github.com/setiawantirta/MARLIN.git`. It was prepared but **not pushed** — GitHub
was unreachable from the environment that built it. Three commands finish the job:

```bash
cd marine_antiglioma_reproducibility
git log --oneline          # expect one commit on main
git push -u origin main
```

If git asks for a password, paste a personal access token with **Contents: write** on this
repository, not your account password.

## What is and is not in the commit

Tracked: all 211 source, config and shipped-data files — the twelve stage drivers, the
legacy step scripts they call, `repro/` (config resolver, stage driver, figure style,
significance labels), the twelve figure scripts, `data/shipped/` (61 derived tables), the
Colab notebook, `smoke_test.sh`, `run_all.sh`, `run_stage.sh`, `README.md` and `RUNBOOK.md`.

Excluded by `.gitignore`, deliberately:

- `outputs/` — every stage writes here. Keeping it out means a reader can clone, run, and
  compare against the released tables without a conflict on every byte of their own run.
- `checkpoints/`, `cache/`, `logs/` — run-time state.
- `data/CMNPD_1.0*` and `data/*.sdf` — the vendor library is not redistributable. Obtain it
  from https://www.cmnpd.org under that database's terms and place it anywhere under a
  configured data root; see `RUNBOOK.md` §2.

## First thing to run after cloning

```bash
bash smoke_test.sh
```

About two minutes: imports, config parsing, every stage's `--check`, one figure render and
the 44 verification checks. On a fresh clone with no data roots configured, stage 01 and
stage 11 will report "inputs not yet present" — that is expected and not a failure.
