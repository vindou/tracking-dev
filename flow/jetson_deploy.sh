#!/bin/bash
# The graphs of the inference process (python -m flow.export --deploy --out flow/runs/export_NAME ...) on the Jetson:
#   flow/jetson_deploy.sh start NAME...   copy to ~/flow_probe/NAME, build the float16 engines and time them (detached)
#   flow/jetson_deploy.sh fetch NAME...   print deploy.tsv (graph, median ms of three runs) once deploy.done is there
# trk_flow --engines ~/flow_probe/NAME then runs them. See tracker/MACHINE_CHANGES.md.
set -e
cd "$(dirname "$0")/.."
cmd=$1; shift
for n in "$@"; do
  if [ "$cmd" = start ]; then
    ssh jetson "mkdir -p ~/flow_probe/$n"
    rsync -a flow/runs/export_$n/ jetson:flow_probe/$n/
  else
    ssh jetson "cd ~/flow_probe/$n && ls deploy.done > /dev/null && cat deploy.tsv"
  fi
done
[ "$cmd" = start ] || exit 0
ssh -f jetson "setsid nohup bash -s $* > ~/flow_probe/deploy.out 2>&1 < /dev/null <<'REMOTE' &
TRT=/usr/src/tensorrt/bin/trtexec
for n in \"\$@\"; do
  cd ~/flow_probe/\$n; rm -f deploy.done deploy.tsv
  for g in stage1e stage2e stage2e1 refine refine1 score score1; do
    [ -f \$g.fp16.engine ] || \$TRT --onnx=\$g.onnx --fp16 --saveEngine=\$g.fp16.engine --skipInference > build_\$g.log 2>&1 || echo \"BUILD FAILED \$g\" >> deploy.tsv
  done
  for g in stage1e stage2e stage2e1 refine refine1 score score1; do
    m=\"\"
    for r in 1 2 3; do m=\"\$m \$(\$TRT --loadEngine=\$g.fp16.engine --duration=3 --useSpinWait 2>&1 | grep '\\] GPU Compute Time' | sed 's/.*median = \\([0-9.]*\\) ms.*/\\1/')\"; done
    echo \"\$g \$(echo \$m | tr ' ' '\\n' | sort -n | sed -n 2p)\" >> deploy.tsv
  done
  touch deploy.done
done
REMOTE"
echo started
