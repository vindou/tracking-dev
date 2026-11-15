# Shared helpers for the revised single-camera experiments (camera B, USB 3.2). Sourced by expB*.sh.
cd ~/rs_investigation
SER=$(sed -n "s/^B=//p" cams.conf)
export LD_LIBRARY_PATH=$HOME/.local/lib
cap(){ local n=$1; shift; src/run_capture.sh $n --cam B:$SER --global-time 1 "$@"; sleep 2; }
# cap with background CPU/memory load: capl NTHREADS RUN ARGS... (load runs 4 s longer than the capture)
capl(){ local nt=$1 n=$2; shift 2; local secs; secs=$(echo "$@" | sed -n "s/.*--seconds \([0-9.]*\).*/\1/p")
  mkdir -p logs/$n; src/build/cpuload $nt $(( ${secs%.*} + 8 )) > logs/$n/cpuload.txt & local P=$!; sleep 2
  cap $n "$@"; wait $P; }
