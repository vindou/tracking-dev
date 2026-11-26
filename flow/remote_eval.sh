#!/bin/bash
# Proposals on the GPU machine, tracker evaluation here.   flow/remote_eval.sh NAME REMOTE_MODEL [SOLVER_STEPS] [EVAL_SET] [WEIGHTS] [SOLVER] ["EXTRA ARGS of infer_rnode"]
# NAME: local folder flow/data/eval_NAME (hints and results); REMOTE_MODEL: path under ~/tracking-flow on linbox01;
# EVAL_SET: scenes folder under flow/data (default eval). SKIP_REMOTE=1: the proposals are already there (made by the
# queue on linbox01); only copy them and run the tracker.
set -e
cd "$(dirname "$0")/.."
NAME=$1; MODEL=$2; STEPS=${3:-4}; SET=${4:-eval}; WEIGHTS=${5:-model}; SOLVER=${6:-rk4}; EXTRA=${7:-}
H=chang940@linbox01
try() { for i in 1 2 3 4 5 6; do "$@" && return 0; sleep 8; done; return 1; }
[ -n "$SKIP_REMOTE" ] || try ssh -o BatchMode=yes -o ConnectTimeout=20 $H "cd ~/tracking-flow && mkdir -p flow/data/eval_$NAME && ln -sf ../$SET/scenes.bin flow/data/eval_$NAME/scenes.bin && ln -sf ../$SET/meta.npz flow/data/eval_$NAME/meta.npz && FLOW_GPU_GB=${FLOW_GPU_GB:-7.0} .venv/bin/python -m flow.infer_rnode --model $MODEL --eval flow/data/eval_$NAME --solver-steps $STEPS --weights $WEIGHTS --solver $SOLVER $EXTRA --device cuda 2>&1 | grep -v -i 'triton\|warn'"
mkdir -p flow/data/eval_$NAME
for f in scenes.bin meta.npz baseline.csv; do ln -sf ../$SET/$f flow/data/eval_$NAME/$f; done
try scp -q $H:tracking-flow/flow/data/eval_$NAME/hints.bin $H:tracking-flow/flow/data/eval_$NAME/flow.npz flow/data/eval_$NAME/
.venv/bin/python -m flow.evaluate --method rnode --eval flow/data/eval_$NAME --skip-infer | tail -n 8
