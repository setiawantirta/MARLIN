import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.dirname(os.path.abspath(__file__))
from sched import run
PY = sys.executable
for model in ["XGB", "RF"]:
    run([(f"{model}_s0_baseline", [PY, os.path.join(_SRC, "baselines.py"), "--model", model, "--seed", "0",
                                   "--jobs", "2", "--predict_cmnpd", "--predict_decoy"])], conc=1)
    jobs = [(f"{model}_s{s}_baseline",
             [PY, os.path.join(_SRC, "baselines.py"), "--model", model, "--seed", str(s), "--jobs", "2",
              "--params", f"metrics/{model}_s0_baseline_run.json",
              "--predict_cmnpd", "--predict_decoy"])
            for s in [1, 2]]
    run(jobs, conc=1)
print("BASELINES ALL DONE", flush=True)
