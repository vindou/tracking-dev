#!/bin/bash
# Times the exported graphs of the frozen model (python -m flow.export) with TensorRT on the Jetson, on the inputs
# of real scenes, in float32 and float16, and copies the outputs back for `python -m flow.export --check fp32,fp16`.
#   flow/jetson_time.sh            copy, build the engines, run, fetch
# Everything on the Jetson is in ~/flow_probe/frozen (see tracker/MACHINE_CHANGES.md).
set -e
cd "$(dirname "$0")/.."
OUT=flow/runs/export
ssh jetson 'mkdir -p ~/flow_probe/frozen'
rsync -a --exclude 'trt_*' $OUT/ jetson:flow_probe/frozen/
ssh jetson 'bash -s' <<'REMOTE'
cd ~/flow_probe/frozen
TRT=/usr/src/tensorrt/bin/trtexec
inputs() {   # graph, scene directory
  case $1 in
    stage1|stage2) echo "pts_in:$2/in_$1_pts.bin,cam_in:$2/in_$1_cam.bin,q_in:$2/in_$1_q.bin" ;;
    refine) echo "pts_in:$2/in_$1_pts.bin,q_in:$2/in_$1_q.bin,p_in:$2/in_$1_p.bin" ;;
    score) echo "pts_in:$2/in_$1_pts.bin,cam_in:$2/in_$1_cam.bin,q_in:$2/in_$1_q.bin,p_in:$2/in_$1_p.bin" ;;
  esac
}
for tag in fp32 fp16; do
  for g in stage1 stage2 refine score; do
    flag=""; [ $tag = fp16 ] && flag="--fp16"
    [ $g.$tag.engine -nt $g.onnx ] || $TRT --onnx=$g.onnx $flag --saveEngine=$g.$tag.engine --skipInference > build_${g}_$tag.log 2>&1 || { echo "BUILD FAILED $g $tag"; tail -5 build_${g}_$tag.log; continue; }
    for s in s*/; do
      s=${s%/}
      $TRT --loadEngine=$g.$tag.engine --loadInputs=$(inputs $g $s) --exportOutput=$s/trt_${g}_$tag.json --duration=4 --useSpinWait > $s/run_${g}_$tag.log 2>&1 || { echo "RUN FAILED $g $tag $s"; tail -5 $s/run_${g}_$tag.log; continue; }
      echo "$tag $g $s $(grep '\] GPU Compute Time' $s/run_${g}_$tag.log | sed 's/.*min = \([0-9.]*\) ms.*median = \([0-9.]*\) ms.*percentile(99%) = \([0-9.]*\) ms/min \1 median \2 p99 \3 ms/')"
    done
  done
done
REMOTE
rsync -a --include 's*/' --include 'trt_*.json' --include 'run_*.log' --exclude '*' jetson:flow_probe/frozen/ $OUT/
