#!/bin/bash
# Runs on the Jetson in ~/flow_probe/frozen (started by flow/jetson_tune.sh). Results: tune.tsv, tune.log.
cd ~/flow_probe/frozen
TRT=/usr/src/tensorrt/bin/trtexec
med() { grep '\] GPU Compute Time' "$1" | sed 's/.*median = \([0-9.]*\) ms.*/\1/'; }
build() {   # onnx, engine, flags...: builds unless the engine is newer; logs seconds and size
  local onnx=$1 eng=$2; shift 2
  if [ ! $eng -nt $onnx ]; then
    local t0=$(date +%s)
    $TRT --onnx=$onnx --saveEngine=$eng --skipInference "$@" > ${eng%.engine}.build.log 2>&1 || { echo "BUILD FAILED $eng" >> tune.log; tail -4 ${eng%.engine}.build.log >> tune.log; return 1; }
    printf "build\t%s\t%s\t%s\n" $eng $(( $(date +%s) - t0 )) $(stat -c %s $eng) >> tune.tsv
  fi
}
rm -f tune.tsv tune.log tune.done

# ---- experiment 1: the time split ----
cd parts
for rep in 1 2; do
  for g in eval_b32 eval_b64 eval_b128 eval_b128_n256 net_b128 feat_b128 look_b128; do
    for tag in fp32 fp16; do
      flag=""; [ $tag = fp16 ] && flag="--fp16"
      ( cd .. ; build parts/$g.onnx parts/$g.$tag.engine $flag ) || continue
      $TRT --loadEngine=$g.$tag.engine --loadInputs=$(cat $g.inputs) --duration=3 --useSpinWait > run_${g}_$tag.log 2>&1 \
        && printf "split\t%s\t%s\t%s\t%s\n" $g $tag $rep $(med run_${g}_$tag.log) >> ../tune.tsv || echo "RUN FAILED $g $tag" >> ../tune.log
    done
  done
done
for tag in fp32 fp16; do      # per-layer profile of one evaluation
  flag=""; [ $tag = fp16 ] && flag="--fp16"
  ( cd .. ; build parts/eval_b128.onnx parts/prof.$tag.engine $flag --profilingVerbosity=detailed ) || continue
  $TRT --loadEngine=prof.$tag.engine --loadInputs=$(cat eval_b128.inputs) --duration=3 --useSpinWait --dumpProfile --separateProfileRun \
    --profilingVerbosity=detailed --exportProfile=profile_$tag.json --exportLayerInfo=layers_$tag.json > run_prof_$tag.log 2>&1 || echo "PROFILE FAILED $tag" >> ../tune.log
done
cd ..
echo "experiment 1 done $(date +%H:%M)" >> tune.log

# ---- experiment 2: build and run settings on the pose stage ----
cp -u stage2.fp32.engine stage2.fp32L3.engine; cp -u stage2.fp16.engine stage2.fp16L3.engine
ENG="fp32L3 fp16L3"
for lvl in 4 5; do
  build stage2.onnx stage2.fp32L$lvl.engine --builderOptimizationLevel=$lvl && ENG="$ENG fp32L$lvl"
  build stage2.onnx stage2.fp16L$lvl.engine --fp16 --builderOptimizationLevel=$lvl && ENG="$ENG fp16L$lvl"
done
build stage2mixed.onnx stage2.mixedL3.engine --stronglyTyped && ENG="$ENG mixedL3"
echo "builds done $(date +%H:%M): $ENG" >> tune.log
for rep in 1 2 3; do
  for e in $ENG; do
    for cg in off on; do
      for s in s3 s700 s1500; do
        flag=""; [ $cg = on ] && flag="--useCudaGraph"
        out=""; [ $rep = 1 ] && [ $cg = off ] && out="--exportOutput=$s/trt_stage2_$e.json"
        $TRT --loadEngine=stage2.$e.engine --loadInputs=pts_in:$s/in_stage2_pts.bin,cam_in:$s/in_stage2_cam.bin,q_in:$s/in_stage2_q.bin $out $flag --duration=3 --useSpinWait > $s/run_stage2_${e}_$cg.log 2>&1 \
          && printf "tune\t%s\t%s\t%s\t%s\t%s\t%s\n" $e $cg $s $rep $(med $s/run_stage2_${e}_$cg.log) "$(grep -o 'device memory size: [0-9.]* MiB' $s/run_stage2_${e}_$cg.log | head -1 | cut -d' ' -f4)" >> tune.tsv \
          || echo "RUN FAILED $e $cg $s" >> tune.log
      done
    done
  done
done
echo "experiment 2 done $(date +%H:%M)" >> tune.log
touch tune.done
