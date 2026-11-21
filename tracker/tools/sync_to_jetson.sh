#!/usr/bin/env bash
# Developer helper (Mac): copy tracker/ and models/ to the Jetson (~/tracker, ~/models) and build.
#   sync_to_jetson.sh [normal] [tsan] [asan]      (default: normal)
# File times are set to the Jetson's clock (its wall clock differs from the Mac's) so that make sees no clock skew.
set -euo pipefail
cd "$(dirname "$0")/../.."
# --delete keeps the Jetson copy equal to the Mac copy, except: builds, logs, and files made on the Jetson in config/
# (background maps, calibration files: config/*.bin, config/calib*.json)
rsync -rlp --delete --exclude 'build*' --exclude 'logs' --exclude 'config/*.bin' --exclude 'config/calib*.json' tracker/ jetson:tracker/
rsync -rlp models/ jetson:models/
VARIANTS=${*:-normal}
ssh jetson "cd ~/tracker && find . -path './build*' -prune -o -type f -exec touch {} + &&
  for v in $VARIANTS; do
    case \$v in normal) d=build; o='' ;; tsan) d=build-tsan; o='-DTRK_TSAN=ON' ;; asan) d=build-asan; o='-DTRK_ASAN=ON' ;; esac
    cmake -S . -B \$d -DCMAKE_BUILD_TYPE=Release \$o > /dev/null && make -C \$d -j6 2>&1 | grep -E 'error|warning:|Built target' | grep -v 'Clock skew' || true
  done"
