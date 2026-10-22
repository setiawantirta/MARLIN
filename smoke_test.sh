#!/usr/bin/env bash
# Exercise every stage on a small subset, in minutes, before committing to a
# full run. Each stage is asked to resolve its inputs (--check) and, where a
# subset path exists, to run it (--smoke).
#
#   ./smoke_test.sh
#
# Exit status 0 means: every stage entry point imports, parses its arguments,
# resolves the inputs it can see, and the stages with a subset path produced
# their outputs. It does NOT mean the full pipeline will reproduce the paper -
# that is stage 12's job.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"

shopt -s nullglob
STAGES=("$REPO"/stages/stage[0-9][0-9]_*.py)
shopt -u nullglob
if [ "${#STAGES[@]}" -eq 0 ]; then echo "no stages found" >&2; exit 2; fi

echo "=== 1. environment"
"$PY" - <<'PYEOF'
import importlib.util, sys
need = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "rdkit"]
opt = ["torch", "torch_geometric", "xgboost", "networkx", "shap", "PIL", "yaml"]
miss = [m for m in need if importlib.util.find_spec(m) is None]
print("  python", sys.version.split()[0])
for m in need + opt:
    spec = importlib.util.find_spec(m)
    print(f"  {m:<18} {'present' if spec else ('MISSING (required)' if m in need else 'absent (optional)')}")
if miss:
    raise SystemExit("required packages missing: " + ", ".join(miss))
PYEOF

echo
echo "=== 2. configuration and input resolution"
"$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(os.pathsep)[0])
from repro.config import CFG, MANIFEST, have
assert CFG["seeds"]["model_seeds"] == [0, 1, 2], CFG["seeds"]
present = sum(1 for n in MANIFEST if have(n))
print(f"  config.yaml parsed: {len(CFG)} top-level sections")
print(f"  manifest: {len(MANIFEST)} declared inputs, {present} resolvable here")
PYEOF

echo
echo "=== 3. every stage entry point answers --check"
RC=0
for f in "${STAGES[@]}"; do
    b="$(basename "$f" .py)"
    # `; crc=$?` would not survive `set -e`: a non-zero --check aborts the script
    # before the assignment runs. `|| crc=$?` keeps the failure from being fatal and
    # still records the code.
    crc=0
    "$PY" "$f" --check >/tmp/smoke_$$.log 2>&1 || crc=$?
    # --check now distinguishes "every input resolved" (0) from "some declared input is
    # absent on this machine" (1). The second is not a failure: a stage whose inputs are
    # fetched from public services, or produced by a stage not yet run, has nothing to
    # resolve on a bare checkout, and reporting that as a smoke-test failure made a
    # healthy install look broken. Measured on a checkout with no data roots, stage 01
    # (Open Targets, CIViC, STRING, ChEMBL) and stage 11 (figure inputs from stages 03,
    # 05 and 07) report missing inputs; every other stage resolves from data/shipped.
    # A crash in the checker itself is still a failure.
    case "$crc" in
        0) echo "  $b  check ok";;
        1) echo "  $b  check ok (inputs not yet present on this machine)";;
        *) echo "  $b  CHECK FAILED (exit $crc)"
           sed 's/^/      /' /tmp/smoke_$$.log | tail -8; RC=1;;
    esac
done
rm -f /tmp/smoke_$$.log

echo
echo "=== 4. subset runs"
for n in 11 12; do
    shopt -s nullglob
    M=("$REPO"/stages/stage"$n"_*.py)
    shopt -u nullglob
    [ "${#M[@]}" -eq 1 ] || continue
    b="$(basename "${M[0]}" .py)"
    # --force on the figure stage: without it the resume-on-outputs rule skips any
    # figure already on disk, so a second smoke run would report success without
    # rendering anything. The point of this section is to prove rendering works.
    EXTRA=()
    [ "$n" = "11" ] && EXTRA=(--force)
    if "$PY" "${M[0]}" --smoke ${EXTRA[@]+"${EXTRA[@]}"}; then
        echo "  $b  smoke ok"
    else
        echo "  $b  SMOKE FAILED"; RC=1
    fi
done

echo
if [ "$RC" -eq 0 ]; then echo "smoke_test: PASS"; else echo "smoke_test: FAIL" >&2; fi
exit "$RC"
