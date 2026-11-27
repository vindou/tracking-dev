#!/usr/bin/env bash
# Tracker CPU benchmark on the Jetson, in one command (Mac):   tracker/tools/bench_cpu_jetson.sh [STEP=8]
# Copies tracker/ to ~/tracker_bench on the Jetson (the demo copy ~/tracker and its builds are not touched), builds
# three variants in separate folders, runs the unit tests of the tuned build and the hint tests under the thread
# sanitizer, and runs the benchmark
# (tools/cpu_bench_run.sh: the hint cycle and one search round without time slicing) on every STEP-th scene of the
# evaluation set with the deployed hints, for each switchable path:
#   flags:   generic (-O3 as before), tuned (-mcpu=cortex-a78ae), lto (tuned + link-time optimization)
#   search:  kdtree (the port), bounded and table_exact (same correspondences), table (may differ slightly)
#   hints:   1 thread, 4 threads
# Needs on the Jetson: ~/flow_probe/frozen/trt_eval/scenes.bin and cpp_hints.bin (flow/live_test.sh puts them there),
# ~/models, ~/capture. Prints the table of tools/cpu_bench_table.py; the files are copied to flow/runs/cpuopt/jetson/.
# Nothing else should run on the Jetson meanwhile. About 25 minutes with STEP = 8.
set -euo pipefail
cd "$(dirname "$0")/../.."
STEP=${1:-8}
rsync -rlp --delete --exclude 'build*' --exclude 'logs' --exclude 'bench' tracker/ jetson:tracker_bench/
ssh jetson "STEP=$STEP bash -s" <<'REMOTE'
set -e
cd ~/tracker_bench
find . -path './build*' -prune -o -type f -exec touch {} +
build() { cmake -S . -B $1 -DCMAKE_BUILD_TYPE=Release -DCAPTURE_DIR=$HOME/capture "${@:2}" > /dev/null && make -C $1 -j5 trk_hint_eval trk_tests 2>&1 | grep -E 'error|warning:' | grep -v 'Clock skew' || true; }
build build-generic -DTRK_TUNE=OFF
build build-tuned -DTRK_TUNE=ON
build build-lto -DTRK_TUNE=ON -DTRK_LTO=ON
echo "compiler: $(c++ --version | head -1)"
for b in generic tuned lto; do echo "flags $b: $(grep -m1 'CXX_FLAGS = ' build-$b/CMakeFiles/trk_core.dir/flags.make)"; done
build-tuned/trk_tests > tests_tuned.log 2>&1 && tail -1 tests_tuned.log || { echo "TESTS FAILED"; grep -E 'FAIL' tests_tuned.log | head; }
# thread sanitizer: the tests of the hint pool (the sanitizer build of the tests does not start on the Mac)
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Release -DCAPTURE_DIR=$HOME/capture -DTRK_TSAN=ON > /dev/null && make -C build-tsan -j5 trk_tests 2>&1 | grep -E 'error' || true
build-tsan/trk_tests hint > tests_tsan.log 2>&1; echo "thread sanitizer, tests with 'hint' in the name: $(tail -1 tests_tsan.log); reports: $(grep -c 'WARNING: ThreadSanitizer' tests_tsan.log)"
D=bench; mkdir -p $D
S=~/flow_probe/frozen/trt_eval
run() { name=$1; bin=$2; shift 2; tools/cpu_bench_run.sh $bin $D $name $S/scenes.bin $S/cpp_hints.bin $STEP "$@"; }
run generic build-generic
run tuned build-tuned
run lto build-lto
run bounded build-tuned --corr bounded
run table_exact build-tuned --corr table_exact
run table build-tuned --corr table
SEARCH=0 run threads4 build-tuned --hint-threads 4
SEARCH=0 run table_exact_threads4 build-tuned --corr table_exact --hint-threads 4
SEARCH=0 run table_threads4 build-tuned --corr table --hint-threads 4
python3 tools/cpu_bench_table.py $D generic tuned lto bounded table_exact table threads4 table_exact_threads4 table_threads4
REMOTE
mkdir -p flow/runs/cpuopt/jetson
rsync -a jetson:tracker_bench/bench/ flow/runs/cpuopt/jetson/
