import ast
import glob
import os
import argparse
import cloudpickle

import numpy as np

import matplotlib as mpl

from pgmtwin.core.tabular_policy import TabularPolicy

mpl.use("agg")
import matplotlib.pyplot as plt

import pandas as pd

from pgmtwin.core.utils import CMAP_BLUES, get_number_from_text
from pgmtwin.toolkits.shm.pgm_helper import PGMHelper

from common import (
    BasicCantileverSetup,
)


def plot_evolution_history(
    history_df: pd.DataFrame,
    title: str = None,
):
    setup = BasicCantileverSetup()

    n_plots = 2
    fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=(10, n_plots * 2.5))

    frames = history_df.index

    n_iters = len(frames)
    n_states = len(setup.state_domain)
    n_actions = len(setup.actions)

    # dstate_distro
    dstate_distro = history_df[
        [c for c in history_df.columns if c.startswith("digital_state_distro")]
    ].values
    # action_distro
    action_distro = history_df[
        [c for c in history_df.columns if c.startswith("action_distro")]
    ].values

    curr_axes = -1

    # region assimilation
    curr_axes += 1
    im = axs[curr_axes].imshow(
        dstate_distro.T,
        aspect="auto",
        origin="lower",
        cmap=CMAP_BLUES,
        extent=[0, n_iters, 0, n_states],
        zorder=-10,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im)

    xticks = axs[curr_axes].get_xticks()[:-1]
    axs[curr_axes].set_xticks(xticks + 0.5, [int(x) for x in xticks])
    axs[curr_axes].set_xlim(0, len(frames))

    with np.printoptions(precision=1):
        yticks = []
        yticklabels = []
        for i in range(n_states):
            yticks.append(i + 0.5)
            yticklabels.append(str(setup.state_domain.index2values(i)))
    axs[curr_axes].set_yticks(yticks)
    axs[curr_axes].set_yticklabels(yticklabels)

    axs[curr_axes].title.set_text(f"assimilation")
    axs[curr_axes].set_ylabel("digital state distribution")
    # endregion

    # region action
    curr_axes += 1
    im = axs[curr_axes].imshow(
        action_distro.T,
        aspect="auto",
        origin="lower",
        cmap=CMAP_BLUES,
        extent=[0, n_iters, 0, n_actions],
        zorder=-10,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im)

    with np.printoptions(precision=1):
        yticks = []
        yticklabels = []
        for i in range(n_actions):
            yticks.append(i + 0.5)
            yticklabels.append(setup.actions[i].name)
    axs[curr_axes].set_yticks(yticks)
    axs[curr_axes].set_yticklabels(yticklabels)

    # axs[curr_axes].legend(bbox_to_anchor=(1, 1), loc="upper left")

    # axs[curr_axes].title.set_text(f"actions")
    axs[curr_axes].set_ylabel("action distribution")
    # endregion

    axs[-1].set_xlabel("time")

    fig.align_ylabels()
    if title:
        axs[0].set_title(title)

    plt.tight_layout()


if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    parser = argparse.ArgumentParser(description="Perform a simulation")

    parser.add_argument("--roms-dir", type=str, required=True, help="roms directory")
    parser.add_argument(
        "--assimilation-dir", type=str, required=True, help="assimilation directory"
    )
    parser.add_argument(
        "--policy-dir", type=str, required=True, help="policy directory"
    )
    parser.add_argument(
        "--evolution-dir", type=str, required=True, help="evolution output directory"
    )

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="policy filename under policy-dir",
    )
    parser.add_argument(
        "--n-pgm-samples",
        type=int,
        default=100,
        help="number of sampled pgm evolutions",
    )
    parser.add_argument(
        "--initial-state",
        type=str,
        default=None,
        help="[dmg-loc, dmg-lvl] pair",
    )

    parser.add_argument("--n-iters", type=int, default=100, help="number of iterations")

    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    assimilation_dir = args.assimilation_dir
    planner_dir = args.policy_dir
    evolution_dir = args.evolution_dir

    policy_fname = args.policy
    n_pgm_samples_assimilation = args.n_pgm_samples
    initial_state = args.initial_state

    n_iters = args.n_iters
    policy_fname = os.path.join(planner_dir, policy_fname)

    verbose = args.verbose
    seed = args.seed
    # endregion

    assert (
        n_pgm_samples_assimilation > 0
    ), f"number of pgm evolutions must be > 1, got {n_pgm_samples_assimilation}"

    os.makedirs(evolution_dir, exist_ok=True)

    prev_run_dirs = list(glob.glob(os.path.join(evolution_dir, "[0-9]*/")))
    if prev_run_dirs:
        evolution_idx = 1 + max(map(get_number_from_text, prev_run_dirs))
    else:
        evolution_idx = 0
    evolution_run_dir = os.path.join(evolution_dir, f"{evolution_idx}")
    os.makedirs(evolution_run_dir)

    # load the instances to be used for the assets and the policy
    inv_solver_confusion_matrix = np.load(
        os.path.join(assimilation_dir, "conf_matrix.npy")
    )

    with open(policy_fname, "rb") as fin:
        policy = cloudpickle.load(fin)

    if not isinstance(policy, TabularPolicy):
        raise NotImplementedError(
            f"policy must be of type TabularPolicy, got {type(policy)}"
        )

    pgm_helper = PGMHelper(
        setup.state_domain,
        setup.actions,
        inv_problem_confusion_matrix=inv_solver_confusion_matrix,
        policy=policy.policy,
    )

    dbn_evolve = pgm_helper.init_dbn_evolution()

    rng = np.random.default_rng(seed)

    n_states = len(setup.state_domain)
    dstate_distro = np.full(n_states, 1 / n_states)
    if initial_state is not None:
        initial_state = np.array(ast.literal_eval(initial_state))
        dstate_distro[:] = 0.0
        dstate_distro[setup.state_domain.values2index(initial_state)] = 1.0

    (dstate_distro_evolution, action_distro_evolution) = pgm_helper.evolve(
        dbn_evolve,
        dstate_distro=dstate_distro,
        n_samples=n_pgm_samples_assimilation,
        n_steps=n_iters,
        rng=rng,
    )

    episode_dfdict = {
        "digital_state_distro": dstate_distro_evolution,
        "action_distro": action_distro_evolution,
    }

    # region history
    # expand digital_state_distro columns
    dstate_distro_cols = [
        f"digital_state_distro[{setup.state_domain.index2values(i)}]"
        for i in range(len(setup.state_domain))
    ]
    dstate_distro_vals = episode_dfdict.pop("digital_state_distro")
    action_distro_cols = [f"action_distro[{a}]" for a in setup.actions]
    action_distro_vals = episode_dfdict.pop("action_distro")

    episode_df = pd.DataFrame(episode_dfdict)

    episode_df[dstate_distro_cols] = np.array(dstate_distro_vals)
    episode_df[action_distro_cols] = np.array(action_distro_vals)
    # endregion

    episode_df.to_csv(os.path.join(evolution_run_dir, "evolution.csv"))

    plot_evolution_history(
        episode_df, title=f"policy: {os.path.basename(policy_fname)}"
    )

    plt.savefig(os.path.join(evolution_run_dir, "history.png"), bbox_inches="tight")
    plt.savefig(os.path.join(evolution_run_dir, "history.pdf"), bbox_inches="tight")

    plt.close()
