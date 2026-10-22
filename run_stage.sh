#!/usr/bin/env bash
# Run one stage by number.
#
#   ./run_stage.sh 11                 # regenerate every figure
#   ./run_stage.sh 06 --check         # resolve inputs, run nothing
#   ./run_stage.sh 05 --smoke         # small-subset run
#
# Extra arguments are passed through to the stage.
set -euo pipefail

usage() {
    echo "usage: $0 <stage-number 01..12> [--check|--smoke|...]" >&2
    echo "stages:" >&2
    for f in "$(dirname "$0")"/stages/stage[0-9][0-9]_*.py; do
        [ -e "$f" ] || continue
        b="$(basename "$f" .py)"
        echo "  ${b:5:2}  ${b:8}" >&2
    done
}

if [ "$#" -lt 1 ]; then usage; exit 2; fi
case "$1" in -h|--help|help) usage; exit 0;; esac

REPO="$(cd "$(dirname "$0")" && pwd)"
N="$(printf '%02d' "$((10#${1#0}))" 2>/dev/null || echo "$1")"
shift

# A glob that matches nothing must not be passed through as a literal path:
# this is why nullglob is set before the match rather than relying on the glob
# expanding to something.
shopt -s nullglob
MATCHES=("$REPO"/stages/stage"$N"_*.py)
shopt -u nullglob

if [ "${#MATCHES[@]}" -eq 0 ]; then
    echo "no stage numbered '$N'" >&2
    usage
    exit 2
fi
if [ "${#MATCHES[@]}" -gt 1 ]; then
    echo "ambiguous stage number '$N': ${MATCHES[*]}" >&2
    exit 2
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" "${MATCHES[0]}" "$@"
