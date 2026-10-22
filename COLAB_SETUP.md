# Running the stages from a Colab or Kaggle notebook

## The two errors you will hit, and why

**`PermissionError: [Errno 13] Permission denied: './smoke_test.sh'`**

Google Drive does not carry the POSIX executable bit, and a tarball extracted onto
Drive does not restore it, so `./script.sh` cannot be executed no matter what the
archive contained. Do not `chmod +x` - on Drive it will not stick. Invoke the
interpreter explicitly instead:

    subprocess.run(["bash", "run_stage.sh", "03"], cwd=str(REPO), ...)

`repro.notebook_run.run()` always does this, so using it avoids the problem entirely.

**`./run_stage.sh 02 --check` exits 0 but `./run_stage.sh 02` exits 2**

This is not a path error and not a contradiction. `--check` is a REPORT, not a gate:
it prints `OK` or `MISSING: ...` for each step and then always exits 0. A real run
gates on the same information and exits 2 when anything is missing, printing
`NOT RUN <step>: missing <files>` **to stderr** - which a notebook wrapper that only
shows the exit code will swallow. That is why you saw a bare `2`.

Stage 02 needs two files that cannot be redistributed with this module:

    CMNPD_1.0_calc_prop.tsv
    CMNPD_1.0.sdf

Branch on the return value of `check()`, which gives you the missing map, rather than
on the exit code.

## Pointing the stages at your CMNPD files without copying them

Input resolution walks a list of roots and searches each one RECURSIVELY by
basename, so the files may sit in any subdirectory of any root. The roots are:

  1. `$MARINE_REPRO_DATA`
  2. every entry of `paths.data_roots` in `config/config.yaml`
  3. `<repo>/data/shipped`
  4. the current output directory

Your CMNPD export already lives on the same Drive, in a different folder, so add it
as a second root rather than copying gigabytes:

```yaml
# config/config.yaml
paths:
  data_roots:
    - /content/drive/MyDrive/Projek/tirta/BMC-cmnpd_screening/glioma_final/data
    - /content/drive/MyDrive/Projek/tirta/cmnpd_screening        # <- the CMNPD export
```

Then `check(2)` should report READY.

## Progress while a stage runs

```python
import sys; sys.path.insert(0, str(REPO))
from repro.notebook_run import run, check, steps

steps(3)        # the ten step names, in order
check(2)        # what stage 2 needs; prints the search roots when something is missing
run(3)          # runs it, with a tqdm bar that advances once per completed step
run(11, "--only", "3", "4")   # just figures 3 and 4
```

`run()` streams stdout and stderr as they arrive, so a long step shows its own
progress rather than going silent; the bar's postfix names the step that just
finished and how long it took. It refuses to start a stage whose inputs are missing
and tells you which ones - pass `gate=False` to override.

Stages 11 and 12 have their own `main()` and do not report per-step timing, so their
bar completes in one jump on a clean exit; that is expected, not a stall.
