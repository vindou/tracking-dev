#!/usr/bin/env bash
# One benchmark run for tools/cpu_bench_table.py:   cpu_bench_run.sh BUILD_DIR OUT_DIR NAME SCENES HINTS STEP [options of trk_hint_eval]
# (run from tracker/). STEP: every STEP-th scene. A build without --step / --cycles (the reference build): STEP = 0.
set -e
B=$1; O=$2; N=$3; SC=$4; HI=$5; STEP=$6; shift 6
if [ "$STEP" = 0 ]; then EXTRA_H=""; EXTRA_S=""; else EXTRA_H="--step $STEP --cycles $O/${N}_hints_cycles.csv"; EXTRA_S="--step $STEP --cycles $O/${N}_search_cycles.csv"; fi
$B/trk_hint_eval --scenes $SC --hints $HI --baseline 0 --max-cycles 1 --out $O/${N}_hints.csv $EXTRA_H "$@" > /dev/null
# SEARCH=0: skip the search round (options that only concern the hint trials)
[ "${SEARCH:-1}" = 0 ] || $B/trk_hint_eval --scenes $SC --config config/bench_noslice.json --max-cycles 1 --out $O/${N}_search.csv $EXTRA_S "$@" > /dev/null
