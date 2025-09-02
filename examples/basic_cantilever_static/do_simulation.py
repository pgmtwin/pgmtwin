import ast
import glob
import os
import argparse
import pickle
import cloudpickle

import numpy as np
import matplotlib as mpl

from pgmtwin.core.action import plot_action_transitions

mpl.use("agg")

import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

from pgmtwin.core.tabular_policy import TabularPolicy, plot_tabular_policy
from pgmtwin.core.utils import (
    CMAP_BLUES,
    get_number_from_text,
    plot_discrete_domain_confusion_matrix,
)
from pgmtwin.toolkits.shm.prediction import Predictor

from common import (
    BasicCantileverInverseSolver,
    BasicCantileverROM,
    BasicCantileverSetup,
)


def plot_simulation_history(
    history_df: pd.DataFrame,
    output_dir: str,
    title: str = None,
):
    setup = BasicCantileverSetup()

    n_plots = 3
    fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=(10, n_plots * 2.5))

    frames = history_df.index

    n_iters = len(frames)
    n_states = len(setup.state_domain)

    # dstate_distro
    dstate_distro = history_df[
        [c for c in history_df.columns if c.startswith("digital_state_distro")]
    ].values
    # pdmg_lvl = history_df[f"physical_state[{setup.state_domain.damage_level_id}]"]
    dstate_true_idx = history_df["digital_state_true_idx"].values
    dstate_assim_idx = history_df["digital_state_map_idx"].values

    curr_axes = -1

    # region assimilation
    curr_axes += 1
    axs[curr_axes].imshow(
        dstate_distro.T,
        aspect="auto",
        origin="lower",
        cmap=CMAP_BLUES,
        extent=[0, n_iters, 0, n_states],
        zorder=-10,
    )

    axs[curr_axes].plot(
        frames + 0.5,
        dstate_true_idx + 0.5,
        linestyle="--",
        color="black",
        label="perfect assimilation",
        zorder=10,
    )

    axs[curr_axes].scatter(
        frames + 0.5,
        dstate_assim_idx + 0.5,
        color="tab:orange",
        alpha=0.7,
        label="assimilation",
        zorder=10,
    )

    with np.printoptions(precision=1):
        yticks = []
        yticklabels = []
        for i in range(n_states):
            yticks.append(i + 0.5)
            yticklabels.append(str(setup.state_domain.index2values(i)))
    axs[curr_axes].set_yticks(yticks)
    axs[curr_axes].set_yticklabels(yticklabels)

    axs[curr_axes].legend(bbox_to_anchor=(1, 1), loc="upper left")

    axs[curr_axes].title.set_text(f"assimilation")
    axs[curr_axes].set_ylabel("digital state distribution")
    # endregion

    # region applied action
    curr_axes += 1
    axs[curr_axes].scatter(
        frames + 0.5,
        history_df["action_true_idx"].values,
        color="black",
        label="perfect choice",
        zorder=11,
        alpha=0.7,
    )
    axs[curr_axes].scatter(
        frames + 0.5,
        history_df["action_idx"].values,
        color="tab:orange",
        label="actual action",
        zorder=11,
        alpha=0.7,
    )

    xticks = axs[curr_axes].get_xticks()[:-1]
    axs[curr_axes].set_xticks(xticks + 0.5, [int(x) for x in xticks])
    axs[curr_axes].set_xlim(0, len(frames))

    axs[curr_axes].set_yticks(list(range(len(setup.actions))))
    axs[curr_axes].set_yticklabels([a.name for a in setup.actions])

    axs[curr_axes].set_ylabel("actions")
    axs[curr_axes].legend(bbox_to_anchor=(1, 1), loc="upper left")
    # endregion

    # region reward
    curr_axes += 1
    axs[curr_axes].scatter(
        frames + 0.5,
        history_df["reward_cumulative"].values,
        color="tab:orange",
        label="",
        zorder=11,
        alpha=0.7,
    )

    axs[curr_axes].set_ylabel("cumulative reward")
    # endregion

    # region true damage
    # curr_axes += 1
    # axs[curr_axes].plot(
    #     frames + 0.5,
    #     pdmg_lvl,
    #     linestyle="--",
    #     color="black",
    #     label="physical damage",
    #     zorder=10,
    # )

    # axs[curr_axes].set_ylabel("damage level")
    # endregion

    axs[-1].set_xlabel("time")

    fig.align_ylabels()
    if title:
        axs[0].set_title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "history.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "history.pdf"), bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    noise_types = ["", "gaussian", "snr"]
    state_update_types = ["deterministic", "stochastic", "damage_model"]

    parser = argparse.ArgumentParser(description="Perform a simulation")

    parser.add_argument("--roms-dir", type=str, required=True, help="roms directory")
    parser.add_argument(
        "--assimilation-dir", type=str, required=True, help="assimilation directory"
    )
    parser.add_argument(
        "--policy-dir", type=str, required=True, help="policy directory"
    )
    parser.add_argument(
        "--simulation-dir", type=str, required=True, help="simulation output directory"
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
        "--initial-state",
        type=str,
        default="",
        help="[dmg-loc, dmg-lvl] pair",
    )

    parser.add_argument("--n-iters", type=int, default=100, help="number of iterations")
    parser.add_argument(
        "--tabular-map",
        action="store_true",
        help="set the tabular policy to use the MAP value from observations",
    )
    parser.add_argument(
        "--policy-update-rate",
        type=int,
        default=0,
        help="number of iterations after which the policy is updated",
    )

    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument(
        "--skip-logging", action="store_true", help="do not save logs to disk"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    assimilation_dir = args.assimilation_dir
    planner_dir = args.policy_dir
    simulation_dir = args.simulation_dir

    policy_fname = args.policy
    noise_type = args.noise_type
    state_update_type = args.update_type
    n_obs_assimilation = args.n_observations
    n_pgm_samples_assimilation = args.n_pgm_samples
    initial_state = args.initial_state

    policy_update_rate = args.policy_update_rate
    # policy_update_cfg_fname = args.policy_update_cfg

    # TODO: fix these with argparse
    n_iters = args.n_iters
    tabular_map = args.tabular_map
    policy_fname = os.path.join(planner_dir, policy_fname)

    verbose = args.verbose
    skip_logging = args.skip_logging
    seed = args.seed
    # endregion

    if not skip_logging:
        os.makedirs(simulation_dir, exist_ok=True)

        prev_run_dirs = list(glob.glob(os.path.join(simulation_dir, "[0-9]*/")))
        if prev_run_dirs:
            simulation_idx = 1 + max(map(get_number_from_text, prev_run_dirs))
        else:
            simulation_idx = 0
        simulation_run_dir = os.path.join(simulation_dir, f"{simulation_idx}")
        os.makedirs(simulation_run_dir)

    # load the instances to be used for the assets and the policy
    with open(os.path.join(roms_dir, "rom_sensors.pkl"), "rb") as fin:
        rom: BasicCantileverROM = pickle.load(fin)

    with open(os.path.join(assimilation_dir, "inv_solver.pkl"), "rb") as fin:
        inv_solver: BasicCantileverInverseSolver = pickle.load(fin)
    inv_solver_confusion_matrix = np.load(
        os.path.join(assimilation_dir, "conf_matrix.npy")
    )

    with open(policy_fname, "rb") as fin:
        policy = cloudpickle.load(fin)

    if isinstance(policy, TabularPolicy):
        policy.use_map_state = tabular_map

    env = gym.make(
        "basic_cantilever",
        rom=rom,
        inv_solver=inv_solver,
        inv_solver_confusion_matrix=inv_solver_confusion_matrix,
        noise_type=noise_type,
        state_update_type=state_update_type,
        n_obs_assimilation=n_obs_assimilation,
        pgm_n_samples_assimilation=n_pgm_samples_assimilation,
        pgm_policy_matrix=(
            None if not isinstance(policy, TabularPolicy) else policy.policy
        ),
        max_episode_steps=n_iters,
    )

    predictor = Predictor(
        env,
        policy,
    )

    rng = np.random.default_rng(seed)

    n_states = len(setup.state_domain)
    n_actions = len(setup.action_domain)

    if not initial_state:
        initial_state = setup.state_domain.sample_values(1, rng=rng)
    else:
        initial_state = np.array(ast.literal_eval(initial_state))

    reset_options = {
        "physical_state": initial_state,
        "digital_state_distro": np.ones(n_states) / n_states,
    }

    if not skip_logging:
        if policy_update_rate:
            for a, action in enumerate(policy.actions):
                plot_action_transitions(action)
                plt.title(action.name)

                plt.savefig(
                    os.path.join(
                        simulation_run_dir, f"action_{action.name}_original.pdf"
                    )
                )
                plt.savefig(
                    os.path.join(
                        simulation_run_dir, f"action_{action.name}_original.png"
                    )
                )

                plt.close()

            plot_tabular_policy(policy)

            plt.savefig(os.path.join(simulation_run_dir, "policy_original.pdf"))
            plt.savefig(os.path.join(simulation_run_dir, "policy_original.png"))

            plt.close()

    history_df = predictor.simulate(
        n_iters=n_iters,
        n_iters_fit_actions_policy=policy_update_rate,
        action_deterministic=True,
        reset_options=reset_options,
        rng=rng,
        verbose=verbose,
    )

    if not skip_logging:
        history_df.to_csv(os.path.join(simulation_run_dir, "history.csv"))

    print(f"cumulative reward {history_df['reward_cumulative'].values[-1]}")

    if not skip_logging:
        if policy_update_rate:
            for a, action in enumerate(policy.actions):
                plot_action_transitions(action)
                plt.title(action.name)

                plt.savefig(
                    os.path.join(simulation_run_dir, f"action_{action.name}_final.pdf")
                )
                plt.savefig(
                    os.path.join(simulation_run_dir, f"action_{action.name}_final.png")
                )

                plt.close()

            plot_tabular_policy(policy)

            plt.savefig(os.path.join(simulation_run_dir, "policy_final.pdf"))
            plt.savefig(os.path.join(simulation_run_dir, "policy_final.png"))

            plt.close()

        plot_simulation_history(
            history_df,
            simulation_run_dir,
            title=f"""policy: {os.path.basename(policy_fname)}
update: {state_update_type} noise: {noise_type} 
# obs: {n_obs_assimilation} # pgm_samples: {n_pgm_samples_assimilation}""",
        )

    # region assimilation_accuracy
    assimilation_confusion_matrix = np.zeros((n_states, n_states))
    for i, j in history_df[["digital_state_true_idx", "digital_state_map_idx"]].values:
        assimilation_confusion_matrix[i, j] += 1

    assimilation_accuracy = np.sum(np.diag(assimilation_confusion_matrix)) / len(
        history_df
    )
    print(f"assimilation accuracy {assimilation_accuracy}")

    if not skip_logging:
        plot_discrete_domain_confusion_matrix(
            setup.state_domain, assimilation_confusion_matrix
        )

        plt.savefig(
            os.path.join(simulation_run_dir, "assimilation_confusion_matrix.pdf")
        )
        plt.savefig(
            os.path.join(simulation_run_dir, "assimilation_confusion_matrix.png")
        )

        plt.close()
    # endregion

    # region action_accuracy
    action_confusion_matrix = np.zeros((n_actions, n_actions))
    for i, j in zip(
        history_df["action_true_idx"],
        history_df["action_idx"],
    ):
        action_confusion_matrix[i, j] += 1

    action_accuracy = np.sum(np.diag(action_confusion_matrix)) / len(history_df)
    print(f"action accuracy {action_accuracy}")

    if not skip_logging:
        plot_discrete_domain_confusion_matrix(
            setup.action_domain,
            action_confusion_matrix,
            ticklabels=[a.name for a in setup.actions],
        )

        plt.savefig(os.path.join(simulation_run_dir, "action_confusion_matrix.png"))
        plt.savefig(os.path.join(simulation_run_dir, "action_confusion_matrix.pdf"))

        plt.close()
    # endregion
