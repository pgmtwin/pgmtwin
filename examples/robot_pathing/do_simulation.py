import ast
import glob
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib import animation, patches

import gymnasium as gym

from pgmtwin.core.utils import (
    CMAP_BLUES,
    get_number_from_text,
    plot_discrete_domain_confusion_matrix,
)

from examples.robot_pathing.common import (
    RobotPathingPredictor,
    RobotPathingPolicy,
    RobotPathingSetup,
)

from examples.robot_pathing.pgm_env import RobotPathingPGMEnv


def plot_simulation_history(
    history_df: pd.DataFrame,
    output_dir: str,
    title: str = None,
    fps: int = 10,
    show_timeprogress: bool = True,
    show_colorbar: bool = True,
    dpi: float | None = None,
):
    setup = RobotPathingSetup()

    frames = history_df.index

    # dstate_distro
    dstate_distro = history_df[
        [c for c in history_df.columns if c.startswith("digital_state_distro")]
    ].values
    pstate = history_df[
        [c for c in history_df.columns if c.startswith("physical_state")]
    ].values

    goal_idxs = setup.state_domain.index2multi_index(
        history_df["goal_state_idx"].values.astype(int)
    )

    fig, ax = plt.subplots()

    # draw full first frame and keep the artist objects
    field = ax.imshow(
        dstate_distro[0].reshape(setup.n_world_xsteps, setup.n_world_ysteps).T,
        vmin=0,
        vmax=1,
        cmap=CMAP_BLUES,
        origin="lower",
    )

    im_ratio = setup.n_world_ysteps / setup.n_world_xsteps
    if show_colorbar:
        fig.colorbar(field, fraction=0.096 * im_ratio, pad=0.04)

    if title:
        ax.set_title(title)

    pstate_post = patches.Rectangle(
        pstate[0] - 0.5,
        width=1,
        height=1,
        linewidth=1,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(pstate_post)

    goal_post = patches.Rectangle(
        goal_idxs[0] - 0.5,
        width=1,
        height=1,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(goal_post)

    for beacon_coord in setup.beacons_coords:
        beacon = patches.Rectangle(
            beacon_coord - 0.5,
            width=1,
            height=1,
            linewidth=1,
            edgecolor="tab:orange",
            facecolor="none",
        )
        ax.add_patch(beacon)

    # region time_progress
    main_pos = ax.get_position()

    # progress bar
    bar_height = 0.02  # as fraction of figure
    bar_bottom = main_pos.y0 - 0.1
    bar_ax = fig.add_axes([main_pos.x0, bar_bottom, main_pos.width, bar_height])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis("off")

    if show_timeprogress:
        bg_rect = patches.Rectangle(
            (0, 0), 1, 1, facecolor="lightgray", edgecolor="black"
        )
        bar_ax.add_patch(bg_rect)
        progress_rect = patches.Rectangle((0, 0), 0, 1, facecolor="tab:blue")
        bar_ax.add_patch(progress_rect)
    # endregion

    def update(frame):
        # for each frame, update the data stored on each artist
        field.set_data(
            dstate_distro[frame].reshape(setup.n_world_xsteps, setup.n_world_ysteps).T
        )
        pstate_post.set_xy(pstate[frame] - 0.5)
        goal_post.set_xy(goal_idxs[frame] - 0.5)

        if show_timeprogress:
            progress_rect.set_width(frame / len(frames))

        return (field,)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(frames), interval=1000 / fps
    )

    ani.save(os.path.join(output_dir, "history.gif"), dpi=dpi)


if __name__ == "__main__":
    # region argparse
    noise_types = ["", "gaussian", "snr"]
    state_update_types = ["deterministic", "stochastic"]

    parser = argparse.ArgumentParser(description="Perform a simulation")

    parser.add_argument(
        "--simulation-dir", type=str, required=True, help="simulation output directory"
    )

    parser.add_argument(
        "--n-x-steps", type=int, default=5, help="number of x cells in map"
    )
    parser.add_argument(
        "--n-y-steps", type=int, default=5, help="number of y cells in map"
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
        help="[x, y] pair",
    )

    parser.add_argument("--n-iters", type=int, default=100, help="number of iterations")

    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--seed", type=int, default=None, help="random seed")

    args = parser.parse_args()
    print(args)

    simulation_dir = args.simulation_dir

    n_x_steps = args.n_x_steps
    n_y_steps = args.n_y_steps
    noise_type = args.noise_type
    state_update_type = args.update_type
    n_obs_assimilation = args.n_observations
    n_pgm_samples = args.n_pgm_samples
    initial_state = args.initial_state

    n_iters = args.n_iters

    verbose = args.verbose
    seed = args.seed
    # endregion

    os.makedirs(simulation_dir, exist_ok=True)

    prev_run_dirs = list(glob.glob(os.path.join(simulation_dir, "[0-9]*/")))
    if prev_run_dirs:
        simulation_idx = 1 + max(map(get_number_from_text, prev_run_dirs))
    else:
        simulation_idx = 0
    simulation_run_dir = os.path.join(simulation_dir, f"{simulation_idx}")
    os.makedirs(simulation_run_dir)

    setup = RobotPathingSetup(n_world_xsteps=n_x_steps, n_world_ysteps=n_y_steps)

    inv_solver_conf_matrix = np.load("inverse_solver_conf_matrix.npy")

    env: RobotPathingPGMEnv = gym.make(
        "robot_pathing_pgm",
        noise_type=noise_type,
        state_update_type=state_update_type,
        n_obs_assimilation=n_obs_assimilation,
        inv_solver_confusion_matrix=inv_solver_conf_matrix,
        pgm_n_samples_assimilation=n_pgm_samples,
        pgm_n_workers=1,
        max_episode_steps=n_iters,
    )

    rng = np.random.default_rng(seed)

    policy = RobotPathingPolicy(
        setup.state_domain,
        actions=setup.actions,
        goals=setup.goals_coords,
        goal_reached_tol=0.5,
        rng=rng,
    )

    predictor = RobotPathingPredictor(
        env,
        policy,
    )

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

    history_df = predictor.simulate(
        n_iters=n_iters,
        reset_options=reset_options,
        rng=rng,
        verbose=verbose,
    )

    history_df.to_csv(os.path.join(simulation_run_dir, "history.csv"))

    # region assimilation_accuracy
    assimilation_conf_matrix = np.zeros((n_states, n_states))
    for i, j in zip(
        history_df["digital_state_true_idx"],
        history_df["digital_state_map_idx"],
    ):
        assimilation_conf_matrix[i, j] += 1

    plot_discrete_domain_confusion_matrix(setup.state_domain, assimilation_conf_matrix)

    accuracy = np.sum(np.diag(assimilation_conf_matrix)) / len(history_df)

    plt.title(
        f"assimilation accuracy {accuracy:.3f}"
        + "\n"
        + f"noise: {noise_type} # obs: {n_obs_assimilation} # pgm samples {n_pgm_samples}"
    )

    plt.savefig(os.path.join(simulation_run_dir, "assimilation_confusion_matrix.png"))
    plt.savefig(os.path.join(simulation_run_dir, "assimilation_confusion_matrix.pdf"))

    plt.close()
    # endregion

    # region action_accuracy
    action_conf_matrix = np.zeros((n_actions, n_actions))
    for i, j in zip(
        history_df["action_true_idx"],
        history_df["action_idx"],
    ):
        action_conf_matrix[i, j] += 1

    plot_discrete_domain_confusion_matrix(
        setup.action_domain,
        action_conf_matrix,
        ticklabels=[a.name for a in setup.actions],
    )

    accuracy = np.sum(np.diag(action_conf_matrix)) / len(history_df)

    plt.title(
        f"action accuracy {accuracy:.3f}"
        + "\n"
        + f"noise: {noise_type} # obs: {n_obs_assimilation} # pgm samples {n_pgm_samples}"
    )

    plt.savefig(os.path.join(simulation_run_dir, "action_confusion_matrix.png"))
    plt.savefig(os.path.join(simulation_run_dir, "action_confusion_matrix.pdf"))

    plt.close()
    # endregion

    plot_simulation_history(
        history_df,
        simulation_run_dir,
        title=f"noise: {noise_type} # obs: {n_obs_assimilation} # pgm samples {n_pgm_samples}",
        show_timeprogress=True,
        show_colorbar=True,
        dpi=300,
    )
