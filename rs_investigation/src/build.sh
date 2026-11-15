#!/usr/bin/env bash
set -e
cd ~/rs_investigation/src && mkdir -p build
g++ -O2 -g -std=c++17 -Wall -I$HOME/.local/include rs_probe.cpp -o build/rs_probe -L$HOME/.local/lib -Wl,-rpath,$HOME/.local/lib -lrealsense2 -lpthread
g++ -O2 -std=c++17 -Wall cpuload.cpp -o build/cpuload -lpthread
echo built
