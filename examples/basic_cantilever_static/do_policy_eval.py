import os
import argparse
import time

import cloudpickle
import numpy as np

import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from common import (
    BasicCantileverEnv,
    BasicCantileverInverseSolver,
    BasicCantileverROM,
    BasicCantileverSetup,
)
from pgmtwin.core.tabular_policy import (
    get_landing_rewards,
)

if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    policy_algos_sb3 = [
        "PPO",
        "DQN",
        "A2C",
        "ARS",
        "QRDQN",
        "TRPO",
    ]
    policy_algos_tabular = [
        "tabular_val_iter",
        "tabular_q_learning",
    ]
    policy_algos = policy_algos_tabular + policy_algos_sb3

    noise_types = ["", "gaussian", "snr"]
    state_update_types = ["deterministic", "stochastic", "damage_model"]

    parser = argparse.ArgumentParser(description="Compute the maintenance policy")

    parser.add_argument("--roms-dir", type=str, required=True, help="roms directory")
    parser.add_argument(
        "--assimilation-dir", type=str, required=True, help="inverse problem directory"
    )
    parser.add_argument(
        "--policy-dir", type=str, required=True, help="policy directory"
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="policy filename under policy-dir",
    )
    parser.add_argument(
        "--noise-type",
        choices=noise_types,
        default=noise_types[0],
        help="noise model for environment",
    )
    parser.add_argument(
        "--update-type",
        choices=state_update_types,
        default=state_update_types[0],
        help="state update model for environment",
    )
    parser.add_argument(
        "--n-observations", type=int, default=1, help="number of observation samples"
    )
    parser.add_argument(
        "--n-pgm-samples",
        type=int,
        default=0,
        help="number of pgm samples - 0 to disable",
    )

    parser.add_argument(
        "--n-episode-steps",
        type=int,
        default=50,
        help="maximum number of steps per episode",
    )
    parser.add_argument(
        "--eval-stochastic", action="store_true", help="evaluate a stochastic policy"
    )
    parser.add_argument(
        "--n-evals", type=int, default=50, help="number of evaluation episodes"
    )

    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    assimilation_dir = args.assimilation_dir
    policy_dir = args.policy_dir

    policy_fname = args.policy
    noise_type = args.noise_type
    state_update_type = args.update_type
    n_obs_assimilation = args.n_observations
    n_pgm_samples_assimilation = args.n_pgm_samples

    n_episode_steps = args.n_episode_steps
    eval_deterministic = not args.eval_stochastic
    n_eval_episodes = args.n_evals

    verbose = args.verbose
    seed = args.seed
    # endregion

    os.makedirs(policy_dir, exist_ok=True)

    if verbose:
        print(f"landing rewards from actions")
    landing_rewards = get_landing_rewards(
        setup.state_domain,
        setup.actions,
        reward=setup.reward,
        verbose=verbose,
    )

    # region env setup
    # load the instances to be used for the assets and the policy
    with open(os.path.join(roms_dir, "rom_sensors.pkl"), "rb") as fin:
        rom: BasicCantileverROM = cloudpickle.load(fin)

    with open(os.path.join(assimilation_dir, "inv_solver.pkl"), "rb") as fin:
        inv_solver: BasicCantileverInverseSolver = cloudpickle.load(fin)
    inv_solver_confusion_matrix = np.load(
        os.path.join(assimilation_dir, "conf_matrix.npy")
    )

    env = gym.make(
        "basic_cantilever",
        rom=rom,
        inv_solver=inv_solver,
        inv_solver_confusion_matrix=inv_solver_confusion_matrix,
        noise_type=noise_type,
        state_update_type=state_update_type,
        n_obs_assimilation=n_obs_assimilation,
        pgm_n_samples_assimilation=n_pgm_samples_assimilation,
        max_episode_steps=n_episode_steps,
    )
    check_env(env)

    env_raw: BasicCantileverEnv = env.unwrapped
    # endregion

    policy_fname = os.path.join(policy_dir, policy_fname)
    with open(policy_fname, "rb") as fin:
        policy = cloudpickle.load(fin)

    print(f"policy loaded from {policy_fname}")

    if n_eval_episodes:
        eval_env = Monitor(env)

        print(f"evaluating for {n_eval_episodes} episodes")
        elapsed = -time.perf_counter()
        mean_reward, std_reward = evaluate_policy(
            policy,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=eval_deterministic,
        )
        elapsed += time.perf_counter()
        print(f"evaluation took {elapsed:.3f}")
        print(f"mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
