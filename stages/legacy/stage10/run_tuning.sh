#!/usr/bin/env bash
# Exhaustive grid search with 5-fold stratified cross-validation for both
# fingerprint baselines, three seeds each.
#
# RUN smoke_test_baselines.py FIRST. It prints a measured projection of the wall
# time on the machine you are actually on; if the projection for --grid full does
# not fit your session, pass GRID=reduced below rather than letting a session die
# halfway.
#
# Seeds: the published run tuned on seed 0 and reused those hyperparameters for
# seeds 1 and 2. Here every seed is tuned independently, which is stricter and
# also three times the cost. If you are short of time, run seed 0 only and say so
# -- a single tuned seed still answers the question this experiment asks.
# NOT `set -e`: a non-matching glob makes `ls` return non-zero, and with pipefail
# that aborted the whole script at the checkpoint-counting line before printing
# anything. Failures are reported explicitly below instead.
set -uo pipefail

# The stage entry point runs this script with cwd set to the stage OUTPUT directory,
# so "tune_baselines.py" relative to cwd resolves to nothing. Resolve the drivers
# from this script's own directory; cwd stays the output directory so metrics/,
# logs/ and the checkpoints still land there.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

GRID="${GRID:-full}"
FOLDS="${FOLDS:-5}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-auto}"
JOBS="${JOBS:--1}"

# Where checkpoints and the fingerprint cache live. THE DEFAULT IS NOT CRASH-SAFE:
# a local directory dies with the virtual machine. On Colab set
#   export GLIOMA_CKPT=/content/drive/MyDrive/glioma_tuning/checkpoints
#   export GLIOMA_CACHE=/content/drive/MyDrive/glioma_tuning/cache
# after mounting Drive; on Kaggle leave them under /kaggle/working and use
# "Save version" so the output is kept when the session ends.
export GLIOMA_CKPT="${GLIOMA_CKPT:-checkpoints}"
export GLIOMA_CACHE="${GLIOMA_CACHE:-cache}"
mkdir -p "$GLIOMA_CKPT" "$GLIOMA_CACHE"
case "$GLIOMA_CKPT" in
  /content/drive/*|/kaggle/working*|/gdrive/*) : ;;
  *) echo "WARNING: GLIOMA_CKPT=$GLIOMA_CKPT is not on persistent storage."
     echo "         If this session dies, the checkpoints die with it, and the full"
     echo "         three-seed run does not fit in one session. Stop now and set:"
     echo "           Colab   (after mounting Drive):"
     echo "             export GLIOMA_CKPT=/content/drive/MyDrive/glioma_tuning/checkpoints"
     echo "             export GLIOMA_CACHE=/content/drive/MyDrive/glioma_tuning/cache"
     echo "           Kaggle  (then use Save version):"
     echo "             export GLIOMA_CKPT=/kaggle/working/glioma_tuning/checkpoints"
     echo "             export GLIOMA_CACHE=/kaggle/working/glioma_tuning/cache"
     echo "         Continuing in 15 s. Interrupt to fix it."
     sleep 15 ;;
esac

if [ "$DEVICE" = "auto" ]; then
  # xgboost >= 2.1 silently falls back to CPU rather than failing, so a trial
  # train with device=cuda cannot be used as a probe. Ask the driver instead.
  if nvidia-smi -L 2>/dev/null | grep -q GPU; then DEVICE=cuda; else DEVICE=cpu; fi
fi
echo "== grid=$GRID folds=$FOLDS seeds='$SEEDS' xgb_device=$DEVICE jobs=$JOBS =="

mkdir -p metrics logs

for s in $SEEDS; do
  for model in RF XGB; do
    name="tuned_${model}_s${s}"
    if [ -f "metrics/${name}.json" ]; then
      echo "-- $name complete, skipping"
      continue
    fi
    # RF_MODE=screen runs the cuML pass first and hands its shortlist to the CPU
    # confirmation. RF_MODE=grid (default) uses the built-in 81-configuration grid.
    cand_arg=""
    if [ "$model" = "RF" ] && [ "${RF_MODE:-grid}" = "screen" ]; then
      shortlist="metrics/screen_RF_s${s}_topk.json"
      if [ ! -f "$shortlist" ]; then
        echo "-- GPU screen for seed $s (grid=${SCREEN_GRID:-medium}, topk=${TOPK:-24})"
        "$PY" "$HERE/screen_rf_cuml.py" --seed "$s" --folds "$FOLDS" \
          --grid "${SCREEN_GRID:-medium}" --topk "${TOPK:-24}" \
          --backend "${SCREEN_BACKEND:-cuml}" --ckpt_dir "$GLIOMA_CKPT" \
          2>&1 | tee -a "logs/screen_RF_s${s}.log"
      else
        echo "-- GPU screen for seed $s already done, reusing $shortlist"
      fi
      cand_arg="--candidates_from $shortlist"
    fi
    # Counted with a glob test rather than `ls`, which fails when nothing matches.
    n_done=0
    for f in "$GLIOMA_CKPT"/${name}__*__done.npz; do
      [ -e "$f" ] && n_done=$((n_done + 1))
    done
    if [ "$n_done" != "0" ]; then
      echo "-- $name resuming: $n_done of 6 targets already checkpointed"
    else
      echo "-- $name"
    fi
    dev=cpu
    [ "$model" = "XGB" ] && dev="$DEVICE"
    only_arg=""
    [ -n "${ONLY_TASK:-}" ] && only_arg="--only_task $ONLY_TASK"
    "$PY" "$HERE/tune_baselines.py" $only_arg \
      --model "$model" --seed "$s" --folds "$FOLDS" --grid "$GRID" \
      --jobs "$JOBS" --device "$dev" --ckpt_dir "$GLIOMA_CKPT" $cand_arg \
      2>&1 | tee -a "logs/${name}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
      echo "!! $name exited with status $rc. The tail of logs/${name}.log says why."
      echo "   Whatever it had already checkpointed is kept, so rerunning resumes."
      echo "   Continuing with the next run."
      continue
    fi
  done
done

echo "== all runs finished =="
echo "Archive metrics/ BEFORE the session ends -- it holds the CV tables and the"
echo "external-set predictions, and a lost session cannot be reconstructed:"
echo "  tar czf tuned_baselines.tar.gz metrics logs"
