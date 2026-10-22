#!/usr/bin/env bash
# Convergence re-run of the five graph architectures, at reference capacity and
# with early stopping allowed to decide when training ends.
#
# Everything except epochs, patience and capacity is identical to the published run:
# same scaffold-disjoint split, same three seeds, same batch size, learning rate,
# weight decay, gradient clipping and loss. Only the compute bound is lifted.
#
# Usage:  bash run_convergence.sh [OUTDIR]
set -euo pipefail
OUT="${1:-runs_converged}"
mkdir -p "$OUT"

# The stage entry point runs this script with cwd set to the stage OUTPUT directory,
# so "src/..." relative to cwd resolves to nothing and every run died on the first
# line. Resolve the trainer from this script's own directory instead; cwd stays the
# output directory so metrics/, models/ and the cache still land there.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${PYTHON:-python}"

# Epoch ceiling and early-stopping patience. These are the ONLY knobs this control
# changes relative to the main benchmark, and they are what the stage exports from
# config/config.yaml (control_convergence.epochs_max / .patience). The published
# control ran with a ceiling of 1,000 and patience 25; early stopping fired in all
# fifteen runs between epochs 34 and 86, so the ceiling never bound. Everything else
# - split, seeds, batch size, learning rate, weight decay, gradient clipping, loss -
# is identical to the published run.
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-25}"
LR_PATIENCE="${LR_PATIENCE:-10}"

# One-off: build the graph cache from the three parquet files in data/.
# The guard tested "cache/..." relative to cwd while build_cache.py writes to
# mdata.CACHE. With the cache on persistent storage - which is the only way a
# multi-session Colab/Kaggle run survives - those are different directories, and
# the ~20 min cache build was repeated every session. Ask for the real location.
CACHE_DIR="$("$PY" -c 'import os; from repro.config import out_dir; print(os.environ.get("MARINE_REPRO_CACHE") or os.path.join(out_dir(), "cache"))')"
echo "== graph cache: $CACHE_DIR =="
if [ ! -f "$CACHE_DIR/flat_chembl/node_ptr.npy" ]; then
  echo "== building graph cache (once, ~10-20 min) =="
  "$PY" "$SRC/build_cache.py"
  "$PY" "$SRC/build_flat.py"
fi

ARCHS="GIN GAT DMPNN AttentiveFP GraphGPS"
SEEDS="0 1 2"

for arch in $ARCHS; do
  for seed in $SEEDS; do
    name="${arch}_s${seed}_converged"
    if [ -f "metrics/${name}_run.json" ]; then
      echo "== skip $name (already done) =="
      continue
    fi
    echo "== $name  (ceiling $EPOCHS epochs, early-stopping patience $PATIENCE) =="
    "$PY" "$SRC/train_gnn.py" \
      --arch "$arch" --seed "$seed" --stage pretrain \
      --capacity full \
      --epochs "$EPOCHS" --patience "$PATIENCE" --lr_patience "$LR_PATIENCE" \
      --batch 256 --lr 1e-3 --wd 1e-5 \
      --device auto --tag "$name" \
      2>&1 | tee "$OUT/${name}.log"
  done
done

echo "== all runs finished =="
echo "Archive BOTH directories before the session ends -- metrics/ holds the"
echo "per-epoch histories that the published run lost:"
echo "  tar czf converged_results.tar.gz metrics models $OUT"
