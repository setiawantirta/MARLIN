import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Child scripts were addressed as "src/<name>.py" relative to the CALLER's
# working directory. The stage runner sets cwd to the stage output directory,
# where no src/ exists, so every job died instantly. Resolve them from this
# file instead; cwd stays the output directory so metrics/, models/ and logs/
# still land there.
_SRC = os.path.dirname(os.path.abspath(__file__))
from sched import run
PY = sys.executable
jobs = []
for arch in ["GIN", "GAT", "DMPNN", "AttentiveFP", "GraphGPS"]:
    for seed in [0, 1, 2]:
        tag = f"{arch}_s{seed}_pretrain"
        jobs.append((tag, [PY, os.path.join(_SRC, "train_gnn.py"), "--arch", arch, "--seed", str(seed),
                           "--stage", "pretrain", "--tag", tag]))
# cheapest first so short jobs clear the queue early
order = {"GIN": 0, "GAT": 1, "DMPNN": 2, "AttentiveFP": 3, "GraphGPS": 4}
jobs.sort(key=lambda j: order[j[0].split("_")[0]])
run(jobs, conc=3)
print("PRETRAIN ALL DONE", flush=True)
