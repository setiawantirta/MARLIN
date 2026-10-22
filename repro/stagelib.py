"""Uniform stage driver.

Every numbered stage is a list of steps. A step is either a python script
(``script=``) or a callable (``fn=``). The driver gives every stage the same
command-line surface:

    python stages/stageNN_<name>.py            # run it
    python stages/stageNN_<name>.py --check    # resolve inputs, run nothing
    python stages/stageNN_<name>.py --smoke    # small-subset run
    python stages/stageNN_<name>.py --list     # print the steps

and the same contract: resolve every declared input first, refuse to start if
one is missing, write a JSON report into ``outputs/<stage>/``, and return a
non-zero exit status on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from collections import deque

from repro.config import out_dir, have, MANIFEST, repo_root


def _bar(iterable, total, desc, enabled):
    """A tqdm bar over the steps, degrading to the plain iterable if tqdm is absent.

    The bar advances once per completed step, so on a stage whose single step runs
    for hours it will sit at 0/1 - which is why step output is streamed rather than
    captured. The two together are what make a long stage legible: the bar says how
    far through the stage you are, the stream says the step is alive.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")


def _run_streaming(cmd, cwd, env, keep=40):
    """Run a step and echo its output live, keeping the last `keep` lines for the report.

    capture_output=True held everything until the step exited, so a forty-five minute
    curation printed nothing at all and was indistinguishable from a hang. Reading the
    merged stream line by line prints the step's own progress as it happens and still
    leaves a tail to put in the JSON report when it fails.
    """
    tail = deque(maxlen=keep)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        print("      " + line, flush=True)
    p.wait()
    return p.returncode, "\n".join(tail)[-1500:]


class Step:
    def __init__(self, name, script=None, fn=None, args=(), inputs=(), outputs=(),
                 minutes=None, smoke_args=(), note="", requires_network=False,
                 executable_here=True):
        self.name, self.script, self.fn = name, script, fn
        self.args, self.smoke_args = list(args), list(smoke_args)
        self.inputs, self.outputs = list(inputs), list(outputs)
        self.minutes, self.note = minutes, note
        self.requires_network = requires_network
        self.executable_here = executable_here


class Stage:
    def __init__(self, number, slug, title, steps, description="",
                 wall_clock="", hardware="CPU", not_executable_reason=None):
        self.number, self.slug, self.title = number, slug, title
        self.steps, self.description = steps, description
        self.wall_clock, self.hardware = wall_clock, hardware
        self.not_executable_reason = not_executable_reason

    # ------------------------------------------------------------------ cli
    def main(self, argv=None):
        ap = argparse.ArgumentParser(description=f"Stage {self.number} - {self.title}")
        ap.add_argument("--check", action="store_true", help="resolve inputs and exit")
        ap.add_argument("--smoke", action="store_true", help="small-subset run")
        ap.add_argument("--list", action="store_true", help="print the step list")
        ap.add_argument("--only", nargs="*", default=None, help="run only these step names")
        ap.add_argument("--force", action="store_true",
                        help="re-run steps whose declared outputs already exist")
        ap.add_argument("--no-progress", action="store_true",
                        help="disable the tqdm progress bar")
        ap.add_argument("--quiet-steps", action="store_true",
                        help="capture step output instead of streaming it live")
        a = ap.parse_args(argv)

        print(f"=== Stage {self.number}: {self.title}")
        if self.description:
            print(self.description.strip())
        print(f"    hardware: {self.hardware}   reference wall clock: {self.wall_clock or 'n/a'}")

        if a.list:
            for s in self.steps:
                print(f"  - {s.name:<38} {'(' + str(s.minutes) + ' min)' if s.minutes else ''} {s.note}")
            return 0

        steps = self.steps if not a.only else [s for s in self.steps if s.name in set(a.only)]

        missing = {}
        for s in steps:
            m = [i for i in s.inputs if not have(i)]
            if m:
                missing[s.name] = m
        if a.check:
            for s in steps:
                m = missing.get(s.name, [])
                print(f"  {s.name:<38} {'OK' if not m else 'MISSING: ' + ', '.join(m)}")
            if self.not_executable_reason:
                print(f"  NOTE: {self.not_executable_reason}")
            if missing:
                n = sum(len(v) for v in missing.values())
                print(f"  {len(missing)} of {len(steps)} step(s) cannot run: {n} input(s) missing.",
                      file=sys.stderr)
                print("  (see config/inputs_manifest.tsv and data/README.md)", file=sys.stderr)
            # --check used to return 0 unconditionally, so a wrapper that reported only
            # the exit code called a stage ready when every step had just printed
            # MISSING. It now returns 1 when something is missing, which is distinct
            # from the 2 a real run returns, so a caller can tell "inputs absent" from
            # "the run refused". 0 still means every declared input resolved.
            return 1 if missing else 0

        if self.not_executable_reason and not a.smoke:
            print(f"  NOT RUN: {self.not_executable_reason}")
            return 0
        if missing:
            for k, v in missing.items():
                print(f"  NOT RUN {k}: missing {', '.join(v)}", file=sys.stderr)
            print("  (see config/inputs_manifest.tsv and data/README.md)", file=sys.stderr)
            return 2

        od = out_dir(f"stage{self.number:02d}_{self.slug}")
        # The legacy step scripts write into work/, raw/ and handoff/ relative to their
        # working directory and none of them creates those directories. A clean run
        # therefore died on the first write. Create them here, once, before any step.
        for sub in ("work", "raw", "handoff"):
            os.makedirs(os.path.join(od, sub), exist_ok=True)
        report, rc = [], 0
        bar = _bar(steps, len(steps), f"stage {self.number:02d} {self.slug}",
                   not a.no_progress)
        for s in bar:
            # Resume on outputs. Every step already declares what it writes, so a step
            # whose declared outputs all resolve under some data root has nothing left
            # to do and is skipped. This is what makes a dropped Colab session cheap to
            # recover: re-run the same cell and only the unfinished steps execute. A
            # step that declares no outputs is never skipped, because there is nothing
            # to test. --force runs everything regardless.
            if s.outputs and not a.force and not a.smoke:
                present = [o for o in s.outputs if have(o)]
                if len(present) == len(s.outputs):
                    print(f"  {s.name:<38} skipped: all {len(present)} output(s) already "
                          f"present  [--force to re-run]")
                    report.append(dict(step=s.name, status="skipped_outputs_present",
                                       seconds=0.0, stderr_tail=""))
                    continue
            # --smoke must never silently mean "run the full stage". Only steps that
            # declare a subset path (smoke_args) or a callable have one; the rest are
            # skipped with a reason rather than starting hours of work unannounced.
            if a.smoke and s.fn is None and not s.smoke_args:
                print(f"  {s.name:<38} skipped (no subset path; --smoke would run the "
                      f"full {s.minutes or '?'} min step)")
                report.append(dict(step=s.name, status="skipped_no_smoke_path", seconds=0.0,
                                   stderr_tail=""))
                continue
            t0 = time.time()
            if s.fn is not None:
                try:
                    s.fn(smoke=a.smoke)
                    ok, tail = True, ""
                except Exception as exc:                    # noqa: BLE001
                    ok, tail = False, repr(exc)[-1500:]
            else:
                cmd = [sys.executable, os.path.join(repo_root(), s.script)] + \
                      (s.smoke_args if a.smoke else s.args)
                env = dict(os.environ, PYTHONPATH=str(repo_root()),
                           # unbuffered, so a step's progress prints reach the stream as
                           # they happen rather than in 8 kB blocks at the end
                           PYTHONUNBUFFERED="1")
                if a.quiet_steps:
                    p = subprocess.run(cmd, cwd=od, capture_output=True, text=True, env=env)
                    ok, tail = p.returncode == 0, (p.stderr or p.stdout)[-1500:]
                else:
                    code, tail = _run_streaming(cmd, cwd=od, env=env)
                    ok = code == 0
            dt = time.time() - t0
            print(f"  {s.name:<38} {'ok' if ok else 'FAILED'}  {dt:7.1f}s", flush=True)
            if not ok:
                rc = 1
                print(tail, file=sys.stderr, flush=True)
            report.append(dict(step=s.name, status="ok" if ok else "failed",
                               seconds=round(dt, 1), stderr_tail="" if ok else tail))
        with open(os.path.join(od, f"stage{self.number:02d}_report.json"), "w") as fh:
            json.dump(dict(stage=self.number, slug=self.slug, smoke=a.smoke, steps=report), fh, indent=1)
        return rc
