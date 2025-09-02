import os
import argparse
import shutil
import time

import cloudpickle
import numpy as np
import matplotlib as mpl

mpl.use("agg")
from matplotlib import pyplot as plt

import gymnasium as gym

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from common import (
    BasicCantileverEnv,
    BasicCantileverInverseSolver,
    BasicCantileverROM,
    BasicCantileverSetup,
)
from pgmtwin.core.tabular_policy import (
    TabularPolicy,
    get_landing_rewards,
    plot_landing_rewards,
    plot_tabular_policy,
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
        "--algorithm",
        choices=policy_algos,
        default=policy_algos[0],
        help="policy protocol",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="output filename under policy-dir, defaults to the chosen <algorithm>.pkl",
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
        "--n-iters", type=int, default=1000, help="maximum number of iterations"
    )
    parser.add_argument(
        "--tabular-map",
        action="store_true",
        help="set the tabular policy to use the MAP value from observations",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3, help="tolerance for value convergence"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="weight decay of future values"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="threshold for epsilon-greedy exploration policy",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
    parser.add_argument(
        "--n-episode-steps",
        type=int,
        default=100,
        help="maximum number of steps per episode",
    )
    parser.add_argument(
        "--train-stochastic", action="store_true", help="train a stochastic policy"
    )
    parser.add_argument(
        "--eval-stochastic", action="store_true", help="evaluate a stochastic policy"
    )
    parser.add_argument(
        "--n-evals", type=int, default=20, help="number of evaluation episodes"
    )

    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    assimilation_dir = args.assimilation_dir
    policy_dir = args.policy_dir

    algorithm = args.algorithm
    policy_fname = args.policy
    if not policy_fname:
        policy_fname = algorithm + ".pkl"
    noise_type = args.noise_type
    state_update_type = args.update_type
    n_obs_assimilation = args.n_observations
    n_pgm_samples_assimilation = args.n_pgm_samples

    n_iters = args.n_iters
    tabular_map = args.tabular_map
    tol = args.tol
    gamma = args.gamma
    epsilon = args.epsilon
    alpha = args.alpha
    n_episode_steps = args.n_episode_steps
    train_deterministic = not args.train_stochastic
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

    plot_landing_rewards(
        setup.state_domain,
        setup.actions,
        landing_rewards,
    )
    plt.savefig(os.path.join(policy_dir, "landing_rewards.pdf"))
    plt.savefig(os.path.join(policy_dir, "landing_rewards.png"))

    plt.close()

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

    elapsed_train = -time.perf_counter()
    policy = None
    if algorithm in policy_algos_sb3:
        if algorithm == "PPO":
            from stable_baselines3 import PPO

            model = PPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="cpu",
            )
        elif algorithm == "DQN":
            from stable_baselines3 import DQN

            model = DQN(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="auto",
            )
        elif algorithm == "A2C":
            from stable_baselines3 import A2C

            model = A2C(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="cpu",
            )
        elif algorithm == "ARS":
            from sb3_contrib import ARS

            model = ARS(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="auto",
            )
        elif algorithm == "QRDQN":
            from sb3_contrib import QRDQN

            model = QRDQN(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="auto",
            )
        elif algorithm == "TRPO":
            from sb3_contrib import TRPO

            model = TRPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                device="cpu",
            )

        eval_env = gym.make(
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
        eval_env = Monitor(eval_env)

        logs_dir = os.path.join(policy_dir, f"logs_{os.path.basename(policy_fname)}")
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=max(1, min(n_iters // 10, 2000)),
            n_eval_episodes=n_eval_episodes,
            log_path=logs_dir,
            best_model_save_path=logs_dir,
            deterministic=train_deterministic,
            render=False,
        )

        model.learn(
            total_timesteps=n_iters,
            callback=eval_callback,
        )

        model = type(model).load(os.path.join(logs_dir, "best_model.zip"))

        policy = model.policy

        shutil.rmtree(logs_dir)
    elif algorithm in policy_algos_tabular:
        policy = TabularPolicy(
            setup.state_domain,
            setup.actions,
            reward=setup.reward,
            use_map_state=tabular_map,
            rng=np.random.default_rng(seed),
        )

        if algorithm == "tabular_val_iter":
            policy.fit_value_iteration(
                n_iters=n_iters,
                tol=tol,
                gamma=gamma,
                deterministic=train_deterministic,
                verbose=verbose,
            )
        elif algorithm == "tabular_q_learning":
            policy.fit_q_learning(
                env_raw._physical_asset.state_update_component,
                n_episodes=n_iters // n_episode_steps,
                n_iters=n_episode_steps,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
                deterministic=train_deterministic,
                verbose=verbose,
            )

        plot_tabular_policy(policy)

        plt.savefig(os.path.join(policy_dir, "policy.pdf"))
        plt.savefig(os.path.join(policy_dir, "policy.png"))

        plt.close()

        if n_eval_episodes:
            eval_env = Monitor(env)

            print(f"evaluating for {n_eval_episodes} episodes")
            elapsed_eval = -time.perf_counter()
            mean_reward, std_reward = evaluate_policy(
                policy,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=eval_deterministic,
            )
            elapsed_eval += time.perf_counter()
            print(f"evaluation took {elapsed_eval:.3f}")
            print(f"mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

    elapsed_train += time.perf_counter()

    print(f"training took {elapsed_train:.3f}")

    policy_fname = os.path.join(policy_dir, policy_fname)
    with open(policy_fname, "wb") as fout:
        cloudpickle.dump(policy, fout)

    print(f"policy saved to {policy_fname}")
