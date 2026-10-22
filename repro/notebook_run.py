"""Run a stage from a notebook with live per-step progress.

Why this exists rather than a bare subprocess call
--------------------------------------------------
Three things go wrong when a stage is launched with
`subprocess.run(["./run_stage.sh", "03"], capture_output=True)`:

  1. `PermissionError: './smoke_test.sh'`. Google Drive does not carry the POSIX
     executable bit, and neither does a tarball extracted onto it, so `./script.sh`
     cannot be executed. Invoking the interpreter explicitly - `bash script.sh` -
     sidesteps the bit entirely. `run()` below always does that.
  2. The diagnosis disappears. When a stage cannot run it prints
     `NOT RUN <step>: missing <files>` to STDERR and exits 2. A wrapper that shows
     only the exit code reports a bare `2` and the reason is lost.
  3. No progress. A three-hour stage prints nothing until a step finishes, so there
     is no way to tell a slow step from a hung one.

`run()` fixes all three: it invokes through `bash`, streams both streams as they
arrive, and advances a tqdm bar once per completed step, having first asked the
stage how many steps it has.

    from repro.notebook_run import run, check
    check(2)        # what stage 2 needs, and whether it is present
    run(2)          # run it, with a progress bar

Note on exit codes, because they are easy to misread:
    0  the stage ran, or --check/--list completed
    2  the stage did NOT run: a required input is missing
`--check` ALWAYS exits 0, even when it has just printed MISSING for every step. It
is a report, not a gate. Use `check()` below, which returns the missing map so a
notebook can branch on it instead of on the exit code.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Stages 01-10 route through Stage.main and print "  <step>   ok   12.3s" per step.
# Stages 11 and 12 have their own main() and print a different line - "Fig  3
# fig03_internal_benchmark   ok   13.1s" and "  [PASS ] <assertion> ..." - so both
# shapes are matched. A bar left at 0 on a successful run would misreport, so the
# caller completes it on a clean exit when no step line was ever seen.
_STEP_DONE = re.compile(r"^\s{2}(\S.*?)\s{2,}(ok|FAILED)\s+([0-9.]+)s\s*$")
_FIG_DONE = re.compile(r"^(?:Fig|Tab)\s+\S+\s+(\S+)\s+(ok|FAILED)\s+([0-9.]+)s\s*$")
_ASSERT = re.compile(r"^\s{2}\[(PASS|FAIL|SKIP|ERROR)\s*\]")
_NOT_RUN = re.compile(r"NOT RUN\s+(\S+):\s*missing\s*(.*)")


def _repo() -> Path:
    return Path(__file__).resolve().parent.parent


def _bar(total, desc):
    """A tqdm bar when tqdm is importable, otherwise a quiet stand-in."""
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit="step", leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    except ImportError:
        class _Null:
            n = 0
            def update(self, k=1): self.n += k
            def set_postfix_str(self, s): print(f"    ... {s}", flush=True)
            def close(self): pass
            def write(self, s): print(s, flush=True)
        return _Null()


def steps(stage: int, repo: Path | None = None):
    """The step names of a stage, in order, from the stage's own --list."""
    repo = repo or _repo()
    script = repo / "stages" / f"stage{stage:02d}_{_slug(stage, repo)}.py"
    p = subprocess.run([sys.executable, str(script), "--list"],
                       cwd=str(repo), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(repo)))
    out = []
    for line in p.stdout.splitlines():
        m = re.match(r"\s{2}- (\S+)", line)
        if m:
            out.append(m.group(1))
    return out


def _slug(stage: int, repo: Path) -> str:
    hits = sorted(repo.glob(f"stages/stage{stage:02d}_*.py"))
    if not hits:
        raise FileNotFoundError(f"no stage script for stage {stage} under {repo/'stages'}")
    return hits[0].stem.split("_", 1)[1]


def check(stage: int, repo: Path | None = None, verbose: bool = True):
    """Resolve a stage's inputs. Returns {step: [missing, ...]} - empty when ready.

    Branch on the RETURN VALUE, not on the exit code: --check always exits 0.
    """
    repo = repo or _repo()
    script = repo / f"stages/stage{stage:02d}_{_slug(stage, repo)}.py"
    p = subprocess.run([sys.executable, str(script), "--check"], cwd=str(repo),
                       capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(repo)))
    missing = {}
    for line in (p.stdout + p.stderr).splitlines():
        m = re.match(r"\s{2}(\S+)\s+MISSING:\s*(.+)", line)
        if m:
            missing[m.group(1)] = [x.strip() for x in m.group(2).split(",")]
        if verbose:
            print(line, flush=True)
    if verbose:
        if missing:
            want = sorted({f for v in missing.values() for f in v})
            print(f"\n  NOT READY. {len(want)} input(s) unresolved: {', '.join(want)}")
            print("  Searched, in order:")
            try:
                from repro.config import _roots
                for r in _roots():
                    print(f"    {r}{'' if Path(r).is_dir() else '   (does not exist)'}")
            except Exception:
                print("    set MARINE_REPRO_DATA, or add a path to paths.data_roots in config/config.yaml")
            print("  Resolution searches each root RECURSIVELY by basename, so a file may sit in a "
                  "subdirectory of any root.")
        else:
            print("\n  READY: every declared input resolves.")
    return missing


def run(stage: int, *args, repo: Path | None = None, gate: bool = True):
    """Run a stage with a per-step progress bar. Returns the exit code.

    gate=True refuses to launch when `check()` reports a missing input, and names
    what is missing - rather than letting the stage exit 2 with the reason on stderr
    where a notebook will not show it. Pass gate=False to launch anyway.
    """
    repo = repo or _repo()
    extra = [a for a in args]

    if gate and not extra:
        miss = check(stage, repo=repo, verbose=False)
        if miss:
            want = sorted({f for v in miss.values() for f in v})
            print(f"[stage {stage:02d}] NOT STARTED - {len(want)} input(s) missing: {', '.join(want)}")
            print("  Steps blocked: " + ", ".join(sorted(miss)))
            print("  Run check(%d) for the search paths, or pass gate=False to try anyway." % stage)
            return 2

    # How many units the bar should expect. With --only the selection itself is the
    # count: stages 11 and 12 take figure numbers rather than the step names that
    # steps() returns, so filtering the step list against them would give the wrong
    # total (and a bar that reads 2/1).
    if "--only" in extra:
        i = extra.index("--only")
        total = max(1, len([a for a in extra[i + 1:] if not a.startswith("-")]))
    elif extra:
        total = 1
    else:
        total = max(1, len(steps(stage, repo)))

    cmd = ["bash", "run_stage.sh", f"{stage:02d}"] + extra   # bash, never ./ - see the docstring
    t0 = time.time()
    bar = _bar(total, f"stage {stage:02d}")
    done, failed, missing = 0, [], {}

    proc = subprocess.Popen(cmd, cwd=str(repo), text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONPATH=str(repo)))
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            m = _STEP_DONE.match(line) or _FIG_DONE.match(line)
            if m:
                name, status, secs = m.group(1).strip(), m.group(2), float(m.group(3))
                done += 1
                bar.update(1)
                bar.set_postfix_str(f"{name} {status} {secs:.0f}s")
                if status == "FAILED":
                    failed.append(name)
                continue
            mm = _NOT_RUN.search(line)
            if mm:
                missing[mm.group(1)] = [x.strip() for x in mm.group(2).split(",")]
            # Everything else - including the stderr the shell merged in - is shown.
            if line.strip():
                print(line, flush=True)
    finally:
        rc = proc.wait()
        # Stages with a custom main() emit no step lines. Leaving the bar at zero on a
        # clean exit would read as "nothing happened", so complete it instead.
        if rc == 0 and done == 0:
            bar.update(total)
            bar.set_postfix_str("done (stage reports no per-step timing)")
            done = total
        bar.close()

    mins = (time.time() - t0) / 60
    print(f"[stage {stage:02d}] exit {rc} after {mins:.1f} min, {done}/{total} steps completed")
    if failed:
        print("  FAILED steps: " + ", ".join(failed))
    if missing:
        want = sorted({f for v in missing.values() for f in v})
        print("  missing inputs: " + ", ".join(want))
    return rc
