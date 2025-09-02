#!/bin/bash

set -eo pipefail
export OMP_NUM_THREADS=8

# python -u do_sample_params.py \
#     --n-samples 400 \
#     --db-dir hifi_db/ \
#     | tee log_0_do_sample_params.log

# python -u do_snapshots_hifi.py \
#     --overwrite \
#     --prune \
#     --db-dir hifi_db/ \
#     | tee log_1_do_snapshots_hifi.log

# python -u do_roms_sensors.py \
#     --sensor-type displacement \
#     --n-sensors 4 \
#     --db-dir hifi_db/ \
#     --roms-dir roms/ \
#     | tee log_2_do_roms_sensors.log

python -u do_roms.py \
    --truncation-rank 5 \
    --db-dir hifi_db/ \
    --roms-dir roms/ \
    | tee log_3_do_roms.log

python -u do_sample_params.py \
    --n-samples 2000 \
    --db-dir lofi_db/ \
    | tee log_4_do_sample_params.log

python -u do_snapshots_roms.py \
    --overwrite \
    --prune \
    --db-dir lofi_db/ \
    --roms-dir roms/ \
    | tee log_5_do_snapshots_roms.log

python -u do_inverse_problem.py \
    --dmg-loc-model xgboost \
    --dmg-lvl-model xgboost \
    --noise-type "gaussian" \
    --db-dir lofi_db/ \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    | tee log_6_do_inverse_problem.log

python -u do_inverse_problem_cpd.py \
    --noise-type "gaussian" \
    --db-dir hifi_db/ \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    | tee log_7_do_inverse_problem_cpd.log

python -u do_policy.py \
    --algorithm "tabular_q_learning" \
    --policy "tabular_q_learning_stochastic.pkl" \
    --n-iters 1000000 \
    --n-episode-steps 100 \
    --noise-type "" \
    --update-type "stochastic" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --n-evals 20 \
    --verbose \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    --train-stochastic \
    | tee log_8_do_policy.log

python -u do_policy_eval.py \
    --n-evals 20 \
    --n-episode-steps 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "gaussian" \
    --update-type "stochastic" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    | tee log_9_do_policy_eval_baseline.log

python -u do_policy_eval.py \
    --n-evals 20 \
    --n-episode-steps 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "gaussian" \
    --update-type "stochastic" \
    --n-observations 1 \
    --n-pgm-samples 10 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    | tee log_9_do_policy_eval_pgm.log

python -u do_simulation.py \
    --n-iters 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "gaussian" \
    --update-type "stochastic" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --verbose \
    --initial-state "[0, 0]" \
    | tee log_10_do_simulation_baseline.log

python -u do_simulation.py \
    --n-iters 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "gaussian" \
    --update-type "stochastic" \
    --n-observations 1 \
    --n-pgm-samples 10 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --verbose \
    --initial-state "[0, 0]" \
    | tee log_10_do_simulation_pgm.log

# python -u do_simulation.py \
#     --n-iters 11 \
#     --policy "tabular_q_learning_stochastic.pkl" \
#     --noise-type "gaussian" \
#     --update-type "stochastic" \
#     --n-observations 1 \
#     --n-pgm-samples 10 \
#     --policy-update-rate 10 \
#     --roms-dir roms/ \
#     --assimilation-dir assimilation/ \
#     --policy-dir policy/ \
#     --simulation-dir simulation/ \
#     --seed 42 \
#     --verbose \
#     --initial-state "[0, 0]" \
#     | tee log_draw_actions.log

python -u do_evolution.py \
    --n-iters 30 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --n-pgm-samples 50 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --evolution-dir evolution/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    --verbose \
    | tee log_11_do_evolution.log

