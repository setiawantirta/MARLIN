#!/usr/bin/env bash
# Run every stage in order, then verify.
#
#   ./run_all.sh                 # the whole study (days of CPU - read the README first)
#   ./run_all.sh --check         # resolve every stage's inputs and stop
#   ./run_all.sh --from 11       # start at stage 11
#   ./run_all.sh --only 11 12    # just these
#
# Stages whose inputs are absent are reported and skipped; the run continues so
# that one missing download does not hide the rest of the report.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PASSTHRU=()
FROM=1
ONLY=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --from)  FROM="$((10#${2#0}))"; shift 2;;
        --only)  shift; while [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do ONLY+=("$((10#${1#0}))"); shift; done;;
        *)       PASSTHRU+=("$1"); shift;;
    esac
done

shopt -s nullglob
STAGES=("$REPO"/stages/stage[0-9][0-9]_*.py)
shopt -u nullglob
if [ "${#STAGES[@]}" -eq 0 ]; then echo "no stages found under $REPO/stages" >&2; exit 2; fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
FAILED=()
for f in "${STAGES[@]}"; do
    b="$(basename "$f" .py)"
    n="$((10#${b:5:2}))"
    [ "$n" -ge "$FROM" ] || continue
    if [ "${#ONLY[@]}" -gt 0 ]; then
        hit=0
        for o in "${ONLY[@]}"; do [ "$o" -eq "$n" ] && hit=1; done
        [ "$hit" -eq 1 ] || continue
    fi
    echo
    echo "############################################################ stage $n"
    # A failing stage must not abort the whole report.
    if ! "${PYTHON:-python}" "$f" ${PASSTHRU[@]+"${PASSTHRU[@]}"}; then
        FAILED+=("$n")
    fi
done

echo
# In --check mode a non-zero exit means "declared inputs are absent", not "failed",
# so the summary says so: reporting a check run as a failure sent readers hunting
# for a crash that had not happened.
IS_CHECK=0
for a in ${PASSTHRU[@]+"${PASSTHRU[@]}"}; do [ "$a" = "--check" ] && IS_CHECK=1; done

if [ "${#FAILED[@]}" -eq 0 ]; then
    if [ "$IS_CHECK" -eq 1 ]; then
        echo "run_all --check: every declared input of every stage resolved on this machine"
    else
        echo "run_all: every stage attempted completed without error"
    fi
else
    if [ "$IS_CHECK" -eq 1 ]; then
        echo "run_all --check: stages with inputs still missing: ${FAILED[*]}" >&2
        echo "  (this is a report, not a failure - populate the inputs and re-check)" >&2
    else
        echo "run_all: stages with a non-zero exit: ${FAILED[*]}" >&2
    fi
    exit 1
fi
