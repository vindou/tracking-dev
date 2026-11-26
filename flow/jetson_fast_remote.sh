#!/bin/bash
# Runs on the Jetson in ~/flow_probe/frozen (started by flow/jetson_fast.sh). Times one field evaluation with every
# arrangement of flow.export.FastNet (parts/evalx_*.onnx) next to the model as it is (parts/eval_b128.onnx), and the
# whole stages given as arguments ("graph:input-graph", e.g. stage2f:stage2). Results: fast.tsv, fast.log.
cd ~/flow_probe/frozen
TRT=/usr/src/tensorrt/bin/trtexec
med() { grep '\] GPU Compute Time' "$1" | sed 's/.*median = \([0-9.]*\) ms.*/\1/'; }
build() {   # onnx, engine, flags...
  local onnx=$1 eng=$2; shift 2
  if [ ! $eng -nt $onnx ]; then
    local t0=$(date +%s)
    $TRT --onnx=$onnx --saveEngine=$eng --skipInference "$@" > ${eng%.engine}.build.log 2>&1 || { echo "BUILD FAILED $eng" >> fast.log; tail -4 ${eng%.engine}.build.log >> fast.log; return 1; }
    printf "build\t%s\t%s\t%s\n" $eng $(( $(date +%s) - t0 )) $(stat -c %s $eng) >> fast.tsv
  fi
}
rm -f fast.done
echo "start $(date +%H:%M) $*" >> fast.log

if [ -z "$1" ]; then      # ---- one evaluation, every arrangement ----
  for g in eval_b128 $(cd parts; ls evalx_*.onnx | sed 's/.onnx//'); do
    for tag in fp32 fp16; do
      flag=""; [ $tag = fp16 ] && flag="--fp16"
      build parts/$g.onnx parts/$g.$tag.engine $flag || continue
    done
  done
  for rep in 1 2 3; do
    for g in eval_b128 $(cd parts; ls evalx_*.onnx | sed 's/.onnx//'); do
      for tag in fp32 fp16; do
        [ -f parts/$g.$tag.engine ] || continue
        out=""; [ $rep = 1 ] && out="--exportOutput=trt_${g}_$tag.json"
        ( cd parts; $TRT --loadEngine=$g.$tag.engine --loadInputs=$(cat $g.inputs) $out --duration=3 --useSpinWait > run_${g}_$tag.log 2>&1 \
          && printf "evalx\t%s\t%s\t%s\t%s\n" $g $tag $rep $(med run_${g}_$tag.log) >> ../fast.tsv || echo "RUN FAILED $g $tag" >> ../fast.log )
      done
    done
  done
else                      # ---- whole stages ----
  for spec in "$@"; do
    g=${spec%%:*}; src=${spec##*:}
    for tag in fp32 fp16; do
      flag=""; [ $tag = fp16 ] && flag="--fp16"
      build $g.onnx $g.$tag.engine $flag
    done
  done
  for rep in 1 2 3; do
    for spec in "$@"; do
      g=${spec%%:*}; src=${spec##*:}
      for tag in fp32 fp16; do
        [ -f $g.$tag.engine ] || continue
        for s in s3 s700 s1500; do
          out=""; [ $rep = 1 ] && out="--exportOutput=$s/trt_${g}_$tag.json"
          $TRT --loadEngine=$g.$tag.engine --loadInputs=pts_in:$s/in_${src}_pts.bin,cam_in:$s/in_${src}_cam.bin,q_in:$s/in_${src}_q.bin $out --duration=3 --useSpinWait > $s/run_${g}_$tag.log 2>&1 \
            && printf "stage\t%s\t%s\t%s\t%s\t%s\n" $g $tag $s $rep $(med $s/run_${g}_$tag.log) >> fast.tsv || echo "RUN FAILED $g $tag $s" >> fast.log
        done
      done
    done
  done
fi
echo "done $(date +%H:%M)" >> fast.log
touch fast.done
