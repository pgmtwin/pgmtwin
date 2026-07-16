#!/bin/bash

set -eo pipefail
export OMP_NUM_THREADS=8

n_hifis=400

n_lofis=2000
sensor_type="displacement"
n_sensors=4

noise="gaussian"
n_episode_steps=100
update="stochastic"
n_pgm_samples=2

time python -u do_sample_params.py \
    --n-samples "${n_hifis}" \
    --db-dir hifi_db/ \
    --seed 42 \
    2>&1 | tee log_0_do_sample_params.log

time python -u do_snapshots_hifi.py \
    --overwrite \
    --prune \
    --db-dir hifi_db/ \
    2>&1 | tee log_1_do_snapshots_hifi.log

time python -u do_roms_sensors.py \
    --sensor-type "${sensor_type}" \
    --n-sensors "${n_sensors}" \
    --db-dir hifi_db/ \
    --roms-dir roms/ \
    2>&1 | tee log_2_do_roms_sensors.log

set +e
set +o pipefail
time python -u do_roms.py \
    --truncation-rank 5 \
    --db-dir hifi_db/ \
    --roms-dir roms/ \
    2>&1 | tee log_3_do_roms.log
set -eo pipefail

time python -u do_sample_params.py \
    --n-samples "${n_lofis}" \
    --db-dir lofi_db/ \
    --seed 42531 \
    2>&1 | tee log_4_do_sample_params.log

time python -u do_snapshots_roms.py \
    --overwrite \
    --prune \
    --db-dir lofi_db/ \
    --roms-dir roms/ \
    2>&1 | tee log_5_do_snapshots_roms.log

time python -u do_inverse_problem.py \
    --dmg-loc-model xgboost \
    --dmg-lvl-model xgboost \
    --noise-type "${noise}" \
    --db-dir lofi_db/ \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --seed 42 \
    2>&1 | tee log_6_do_inverse_problem.log

time python -u do_inverse_problem_cpd.py \
    --noise-type "${noise}" \
    --db-dir hifi_db/ \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --seed 42 \
    2>&1 | tee log_7_do_inverse_problem_cpd.log

time python -u do_policy.py \
    --algorithm "tabular_q_learning" \
    --policy "tabular_q_learning_stochastic.pkl" \
    --n-iters 1000000 \
    --n-episode-steps "${n_episode_steps}" \
    --noise-type "" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --n-evals 20 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    --train-stochastic \
    2>&1 | tee log_8_do_policy.log
    # --verbose \

time python -u do_policy_eval.py \
    --n-evals 20 \
    --n-episode-steps "${n_episode_steps}" \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    2>&1 | tee log_9_do_policy_eval_baseline.log

time python -u do_policy_eval.py \
    --n-evals 20 \
    --n-episode-steps "${n_episode_steps}" \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples "${n_pgm_samples}" \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --seed 42 \
    2>&1 | tee log_9_do_policy_eval_pgm.log

time python -u do_simulation.py \
    --n-iters 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    2>&1 | tee log_10_do_simulation_baseline.log
    # --verbose \

time python -u do_simulation.py \
    --n-iters 100 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples "${n_pgm_samples}" \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    2>&1 | tee log_10_do_simulation_pgm.log
    # --verbose \

time python -u do_simulation.py \
    --n-iters 10000 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples 0 \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    2>&1 | tee log_11_do_long_simulation_baseline.log
    # --verbose \

time python -u do_simulation.py \
    --n-iters 10000 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --noise-type "${noise}" \
    --update-type "${update}" \
    --n-observations 1 \
    --n-pgm-samples "${n_pgm_samples}" \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --simulation-dir simulation/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    2>&1 | tee log_11_do_long_simulation_pgm.log
    # --verbose \

time python -u do_evolution.py \
    --n-iters 30 \
    --policy "tabular_q_learning_stochastic.pkl" \
    --n-pgm-samples "${n_pgm_samples}" \
    --roms-dir roms/ \
    --assimilation-dir assimilation/ \
    --policy-dir policy/ \
    --evolution-dir evolution/ \
    --seed 42 \
    --initial-state "[0, 0]" \
    --verbose \
    2>&1 | tee log_11_do_evolution.log

