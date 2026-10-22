"""Fine-tuning ablation (3 arms) and the putative-inactive ablation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.dirname(os.path.abspath(__file__))
from sched import run
PY = sys.executable
mode = sys.argv[1]                      # finetune | putative
archs = sys.argv[2].split(",")
jobs = []
if mode == "finetune":
    for arch in archs:
        for seed in [0, 1, 2]:
            init = f"models/{arch}_s{seed}_pretrain.pt"
            for stage, ft in [("ft_np", "np_subset"), ("ft_npw", "np_weight")]:
                tag = f"{arch}_s{seed}_{stage}"
                jobs.append((tag, [PY, os.path.join(_SRC, "train_gnn.py"), "--arch", arch, "--seed", str(seed),
                                   "--stage", stage, "--ft_mode", ft, "--init", init,
                                   "--epochs", "20", "--patience", "5", "--lr", "3e-4",
                                   "--tag", tag]))
else:
    for arch in archs:
        for seed in [0, 1, 2]:
            tag = f"{arch}_s{seed}_noputative"
            jobs.append((tag, [PY, os.path.join(_SRC, "train_gnn.py"), "--arch", arch, "--seed", str(seed),
                               "--stage", "noputative", "--no_putative", "--tag", tag]))
run(jobs, conc=int(os.environ.get("ABL_CONC", "1")))
print(f"ABLATION {mode} DONE", flush=True)
