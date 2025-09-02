from typing import Optional
import numpy as np
import matplotlib as mpl

from pgmtwin.core.domain import DiscreteDomain

mpl.use("agg")
from matplotlib import pyplot as plt

import gymnasium as gym

from pgmtwin.core.utils import plot_discrete_domain_confusion_matrix
from examples.robot_pathing.common import (
    RobotPathingSetup,
    RobotPathingPolicy,
    RobotPathingEnv,
)
from examples.robot_pathing.pgm_env import RobotPathingPGMEnv


def eval_env(
    env: RobotPathingEnv,
    n_simulation_iters: int,
    physical_state: np.ndarray = np.array([0, 0]),
    seed: int = None,
):
    setup = RobotPathingSetup()

    policy = RobotPathingPolicy(
        setup.state_domain,
        actions=setup.actions,
        goals=setup.goals_coords,
    )

    obs, info = env.reset(seed=seed, options={"physical_state": physical_state})

    goals = [policy.get_current_goal()]
    pstates = [info["physical_state"]]
    dstates = [setup.state_domain.index2values(np.argmax(obs))]

    action, _ = policy.predict(obs, deterministic=True)

    actions = [setup.actions[0].name]

    for _ in range(n_simulation_iters):
        obs, _, terminated, truncated, info = env.step(action)

        goals.append(policy.get_current_goal())
        pstates.append(info["physical_state"])
        dstates.append(setup.state_domain.index2values(np.argmax(obs)))

        action, _ = policy.predict(obs, deterministic=True)

        actions.append(setup.actions[action].name)

        if terminated or truncated:
            break

    return (
        pstates,
        dstates,
        goals,
        actions,
    )


def plot_state_conf_matrix(
    state_domain: DiscreteDomain, pstates: np.ndarray, dstates: np.ndarray
):
    n_states = len(state_domain)

    assimilation_conf_matrix = np.zeros((n_states, n_states))
    for i, j in zip(
        state_domain.values2index(pstates),
        state_domain.values2index(dstates),
    ):
        assimilation_conf_matrix[i, j] += 1

    plot_discrete_domain_confusion_matrix(state_domain, assimilation_conf_matrix)

    accuracy = np.sum(np.diag(assimilation_conf_matrix)) / len(dstates)
    plt.title(f"assimilation accuracy {accuracy:.3f}")

    plt.close()


def run_baseline_env(
    n_simulation_iters: int = 20,
    n_obs_assimilation: int = 5,
    seed: int = 42,
):
    env = gym.make(
        "robot_pathing",
        noise_type="gaussian",
        state_update_type="deterministic",
        n_obs_assimilation=n_obs_assimilation,
        max_episode_steps=n_simulation_iters,
    )

    pstates, dstates, goals, actions = eval_env(env, n_simulation_iters, seed=seed)

    assert len(pstates) == n_simulation_iters + 1
    assert len(dstates) == n_simulation_iters + 1
    assert len(goals) == n_simulation_iters + 1
    assert len(actions) == n_simulation_iters + 1

    setup = RobotPathingSetup()

    plot_state_conf_matrix(setup.state_domain, pstates, dstates)


def run_pgm_env(
    n_simulation_iters: int = 20,
    n_obs_assimilation: int = 5,
    n_pgm_samples: int = 5,
    n_workers: Optional[int] = 1,
    seed: int = 42,
):
    pgm_env = gym.make(
        "robot_pathing_pgm",
        noise_type="gaussian",
        state_update_type="deterministic",
        n_obs_assimilation=n_obs_assimilation,
        pgm_n_samples_assimilation=n_pgm_samples,
        pgm_n_workers=n_workers,
        max_episode_steps=n_simulation_iters,
    )

    pstates, dstates, goals, actions = eval_env(pgm_env, n_simulation_iters, seed=seed)

    assert len(pstates) == n_simulation_iters + 1
    assert len(dstates) == n_simulation_iters + 1
    assert len(goals) == n_simulation_iters + 1
    assert len(actions) == n_simulation_iters + 1

    setup = RobotPathingSetup()

    plot_state_conf_matrix(setup.state_domain, pstates, dstates)


# region baseline_env
def test_baseline_env_obs1():
    run_baseline_env(n_obs_assimilation=1)


def test_baseline_env_obs5():
    run_baseline_env(n_obs_assimilation=5)


# endregion


# region pgm_env_singlejob
def test_pgm_env_singlejob_obs1_pgm1():
    run_pgm_env(n_workers=1, n_obs_assimilation=1, n_pgm_samples=1)


def test_pgm_env_singlejob_obs5_pgm1():
    run_pgm_env(n_workers=1, n_obs_assimilation=5, n_pgm_samples=1)


def test_pgm_env_singlejob_obs5_pgm5():
    run_pgm_env(n_workers=1, n_obs_assimilation=5, n_pgm_samples=5)


# endregion


# region pgm_env_multiojob_none
def test_pgm_env_multijob_obs1_pgm1():
    run_pgm_env(n_workers=None, n_obs_assimilation=1, n_pgm_samples=1)


def test_pgm_env_multijob_obs5_pgm1():
    run_pgm_env(n_workers=None, n_obs_assimilation=5, n_pgm_samples=1)


def test_pgm_env_multijob_obs5_pgm5():
    run_pgm_env(n_workers=None, n_obs_assimilation=5, n_pgm_samples=5)


# endregion
