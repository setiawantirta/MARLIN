"""Stage the trained-model bundle into this stage's working directory.

The attribution scripts of this stage (`refit_seeds.py`, `interp_rf.py`,
`interp_consensus.py`, `stage_c.py`) were written to run inside the original
project workspace, where the models had already been unpacked to
``bundle/glioma_models/``. They therefore open paths such as
``bundle/glioma_models/model_config.json`` relative to the current directory.

Nothing in the packaged module recreated that directory, so every one of those
scripts failed with a bare FileNotFoundError the first time the stage was run
from a clean checkout -- which is what blocked Figure 5. This step closes the gap
without editing the legacy scripts: it extracts the bundle, addressed by its
logical name through the resolver, into the working directory the stage runner
already uses as cwd.

Re-running is cheap and idempotent: an existing, complete bundle is left alone.
"""
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from repro.config import path as RPATH

SENTINEL = os.path.join("bundle", "glioma_models", "model_config.json")


def main():
    if os.path.exists(SENTINEL):
        with open(SENTINEL) as fh:
            cfg = json.load(fh)
        print("BUNDLE PRESENT", SENTINEL, "| tasks:",
              len(cfg.get("decision_thresholds", {})), flush=True)
        return

    src = RPATH("models_bundle.tar.gz")
    os.makedirs("bundle", exist_ok=True)
    with tarfile.open(src) as tf:
        tf.extractall("bundle")
    if not os.path.exists(SENTINEL):
        raise SystemExit(
            f"{src} did not contain glioma_models/model_config.json; the bundle "
            "layout is not what this stage expects")
    with open(SENTINEL) as fh:
        cfg = json.load(fh)
    n_ckpt = sum(1 for _, _, fs in os.walk("bundle") for f in fs if f.endswith(".pt"))
    print("BUNDLE STAGED from", src, "| checkpoints:", n_ckpt,
          "| tasks:", len(cfg.get("decision_thresholds", {})), flush=True)


if __name__ == "__main__":
    main()
