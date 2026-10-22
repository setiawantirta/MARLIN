"""Run a job list with a fixed number of concurrent worker processes."""
import subprocess, sys, time, os, json
def run(jobs, conc=3, logdir="logs"):
    os.makedirs(logdir, exist_ok=True)
    # Resume. train_gnn.py and baselines.py each write metrics/<tag>_run.json only
    # after a run completes, and run_convergence.sh already treats that file as the
    # "done" marker. Honour the same convention here: without it a dropped Colab or
    # Kaggle session restarted all fifteen pretraining runs from zero, which makes
    # the ~20 h step impossible to finish inside a 12 h session limit.
    _skip = {t for t, _ in jobs if os.path.exists(os.path.join("metrics", t + "_run.json"))}
    for t in sorted(_skip):
        print(f"SKIP  {t} (metrics/{t}_run.json exists)", flush=True)
    jobs = [(t, c) for t, c in jobs if t not in _skip]

    pend, live, done = list(jobs), [], []
    while pend or live:
        while pend and len(live) < conc:
            tag, cmd = pend.pop(0)
            f = open(os.path.join(logdir, tag + ".log"), "w")
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            live.append((tag, p, f, time.time()))
            print(f"START {tag} ({len(pend)} pending)", flush=True)
        time.sleep(5)
        for it in list(live):
            tag, p, f, t0 = it
            if p.poll() is not None:
                f.close(); live.remove(it); done.append((tag, p.returncode))
                print(f"END   {tag} rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    return done
