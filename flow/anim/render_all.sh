#!/bin/sh
# All eight scenes at 1080p / 30 fps into media/videos/<file>/1080p30/.
cd "$(dirname "$0")"
for x in "s1_pipeline PipelineOverview" "s2_state_flow StateAndFlow" "s3_inputs NetworkInputs" "s4_network Network" \
         "s5_ode SolvingODE" "s6_two_stage TwoStageInference" "s7_verifier VerifierScene" "s8_results Results"; do
    set -- $x
    ../../.venv/bin/manim -qh --disable_caching "$1.py" "$2" || exit 1
done
