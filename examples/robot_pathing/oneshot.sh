#!/bin/bash

python do_simulation.py \
    --simulation-dir simulation \
    --n-x-steps 10 \
    --n-y-steps 5 \
    --noise-type "gaussian" \
    --n-observations 10 \
    --n-pgm-samples 0 \
    --n-iters 200 \
    --initial-state "[0, 0]" \
    --seed 42 \
    --verbose \
    | tee log_0_simulation_baseline.log

python do_simulation.py \
    --simulation-dir simulation \
    --n-x-steps 10 \
    --n-y-steps 5 \
    --noise-type "gaussian" \
    --n-observations 10 \
    --n-pgm-samples 200\
    --n-iters 200 \
    --initial-state "[0, 0]" \
    --seed 42 \
    --verbose \
    | tee log_0_simulation_pgm.log
