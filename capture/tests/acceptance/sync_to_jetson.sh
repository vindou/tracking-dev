#!/usr/bin/env bash
# Developer helper (runs on the Mac): copy capture/ to ~/capture on the Jetson and build there.
# File times are set to the Jetson's clock (its wall clock differs from the Mac's) so that make does not see clock skew.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -rlp --delete --exclude 'build*' --exclude capture.conf ./ jetson:capture/
ssh jetson 'cd ~/capture && find . -path ./build -prune -o -path ./build-tsan -prune -o -type f -exec touch {} + &&
  cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/.local -DCMAKE_BUILD_TYPE=RelWithDebInfo > /dev/null && make -C build -j6 2>&1 | grep -E "warning|error|Built target" || true'
