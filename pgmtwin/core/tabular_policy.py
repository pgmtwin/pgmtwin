"""
Tabular policy utilities and classes
"""

from typing import Callable, List, Optional, Tuple, Union

from matplotlib.colors import Colormap
import numpy as np
import matplotlib.pyplot as plt

from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.action import BaseAction
from pgmtwin.core.asset_components.state_update import StateUpdateComponent
from pgmtwin.core.utils import CMAP_BLUES, get_epsilon_greedy_policy, softmax

from stable_baselines3.common.type_aliases import PolicyPredictor


def get_landing_rewards(
    state_domain: DiscreteDomain,
    actions: List[BaseAction],
    reward: Callable[[BaseAction], np.ndarray],
    verbose: bool = False,
) -> np.ndarray:
    """
    Computes the rewards for each combination of (action, ending state)

    Args:
        state_domain (DiscreteDomain): description of the digital state domain
        actions (List[BaseAction]): list of actions
        reward (Callable[[BaseAction], np.ndarray]): reward function that takes an action and a state
        verbose (bool, optional): whether to print debug messages. Defaults to False.

    Returns:
        np.ndarray: a 2d array of rewards, with shape (n_actions, n_states)
    """
    landing_rewards = []
    for state_idx in range(len(state_domain)):
        state = state_domain.index2values(state_idx)
        rewards = [reward(a, state) for a in actions]
        rewards = np.array(rewards)

        if verbose:
            print(f"state {state} rewards {rewards}")

        landing_rewards.append(rewards)
    landing_rewards = np.array(landing_rewards).T

    return landing_rewards


def plot_landing_rewards(
    state_domain: DiscreteDomain,
    actions: List[BaseAction],
    landing_rewards: np.ndarray,
    cmap: Union[str, Colormap] = CMAP_BLUES,
):
    """
    Helper method to plot the landing rewards with action and value labels

    Args:
        state_domain (DiscreteDomain): the digital state domain
        actions (List[BaseAction]): the list of actions
        landing_rewards (np.ndarray): a 2d array of rewards, with shape (n_actions, n_states)
        cmap (Union[str, Colormap], optional): the colormap to use. Defaults to CMAP_BLUES.
    """
    im = plt.imshow(landing_rewards, cmap=cmap)

    plt.xlabel("state")
    plt.ylabel("action")

    ticks = list(range(len(state_domain)))
    ticklabels = list(str(s_values) for s_values in state_domain.index2values(ticks))
    plt.xticks(ticks, labels=ticklabels, rotation=90)

    ticks = list(range(len(actions)))
    ticklabels = [action.name for action in actions]
    plt.yticks(ticks, labels=ticklabels)

    plt.colorbar(im)
    plt.tight_layout()


class TabularPolicy(PolicyPredictor):
    """
    A tabular policy that predicts actions based on a discrete state domain and a set of actions.
    Compatible with stable-baselines3 and gym environments.
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        reward: Callable[[BaseAction, np.ndarray], float],
        use_map_state: bool = False,
        rng: np.random.Generator = None,
    ):
        """
        Initializes the TabularPolicy with a state domain, actions, and a reward function.

        Args:
            state_domain (DiscreteDomain): the discrete state domain
            actions (List[BaseAction]): the list of actions available in the policy
            reward (Callable[[BaseAction, np.ndarray], float]): the reward function that evaluates the action and state
            use_map_state (bool, optional): whether the policy whould use the maximum a posteriori state to act. Defaults to False.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.state_domain = state_domain
        self.actions = actions
        self.reward = reward

        n_states = len(self.state_domain)
        n_actions = len(self.actions)

        self.policy = np.empty((n_actions, n_states))

        self.use_map_state = use_map_state

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).

        Args:
            observation (Union[np.ndarray, dict[str, np.ndarray]]): the observation which describes the environment state
            state (Optional[tuple[np.ndarray, ...]], optional): optional hidden state. Defaults to None.
            episode_start (Optional[np.ndarray], optional): additional information to denote the start of a new episode. Defaults to None.
            deterministic (bool, optional): whether the policy selects the action deterministically or via sampling. Defaults to False.

        Returns:
            tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]: the index of the selected action, and the new hidden state (if applicable)
        """
        squeeze = observation.ndim < 2
        observation = np.atleast_2d(observation)
        if self.use_map_state:
            ret = np.array([self.policy[:, i] for i in np.argmax(observation, axis=1)])
        else:
            ret = observation @ self.policy.T

        if deterministic:
            ret = np.argmax(ret, axis=1)
        else:
            ret = np.array([self.rng.choice(len(self.actions), p=dist) for dist in ret])

        return ret if not squeeze else np.squeeze(ret, axis=0), None

    def get_action_rewards_for_state(
        self,
        state_idx: int,
        state_values: np.ndarray,
        landing_rewards: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Computes the action values for a given state based on the landing rewards and state values.

        Args:
            state_idx (int): index of the starting state
            state_values (np.ndarray): array of current values, n_states elements
            landing_rewards (np.ndarray): array of shape (n_actions, n_states) with the landing rewards for each combination
            gamma (float): weighting factor for future rewards

        Returns:
            np.ndarray: an array of action values
        """
        n_actions = len(self.actions)

        action_values = np.empty(n_actions)
        for a, action in enumerate(self.actions):
            action_values[a] = action.get_transition_probabilities()[:, state_idx] @ (
                landing_rewards[a] + gamma * state_values
            )

        return action_values

    def fit_value_iteration(
        self,
        n_iters: int,
        tol: float = 1e-3,
        gamma: float = 0.9,
        deterministic: bool = False,
        verbose: bool = False,
    ):
        """
        Fit the policy using value iteration.

        Args:
            n_iters (int): number of iterations to run for training
            tol (float, optional): tolerance for convergence, checcked on the maximum change in state values. Defaults to 1e-3.
            gamma (float, optional): weighting factor. Defaults to 0.9.
            deterministic (bool, optional): whether the resulting policy has a single action outcome, or a full probability distribution. Defaults to False.
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the policy matrix and the Q-values for each state-action pair
        """
        n_states = len(self.state_domain)
        n_actions = len(self.actions)

        landing_rewards = get_landing_rewards(
            self.state_domain, self.actions, self.reward, verbose=False
        )
        state_values = np.ones(n_states)
        # TODO: handle terminal states?

        if verbose:
            for action, action_rewards in zip(self.actions, landing_rewards):
                print(f"{action}")
                print(f"    {action_rewards}")

            print(f"initial state_values {state_values}")

        for i in range(n_iters):
            delta = 0

            if verbose:
                print(f"iter {i}")
                print(f"    state_values {state_values}")

            for s in range(n_states):
                v = state_values[s]
                if verbose:
                    print(f"    state {s}")

                action_values = self.get_action_rewards_for_state(
                    s, state_values, landing_rewards=landing_rewards, gamma=gamma
                )

                if verbose:
                    print(f"        action_values {action_values}")

                state_values[s] = max(action_values)
                delta = max(delta, abs(v - state_values[s]))

            if delta < tol:
                break

        print(f"exiting at iter {i}")

        self.policy = np.zeros((n_actions, n_states))
        q_values = np.empty_like(self.policy.T)
        for s in range(n_states):
            q_values[s] = self.get_action_rewards_for_state(
                s, state_values, landing_rewards=landing_rewards, gamma=gamma
            )
            action_values = q_values[s]

            if deterministic:
                self.policy[np.argmax(action_values), s] = 1
            else:
                self.policy[:, s] = softmax(action_values)

        return self.policy, q_values

    def fit_q_learning(
        self,
        state_update: StateUpdateComponent,
        n_episodes: int,
        n_iters: int,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        deterministic: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the policy using Q-learning.

        Args:
            state_update (StateUpdateComponent): the state update component to use for state transitions
            n_episodes (int): number of episodes to run for training
            n_iters (int): number of iterations per episode
            alpha (float, optional): weighting factor for rewards+values. Defaults to 0.1.
            epsilon (float, optional): the threshold for the construction of the epsilon-greedy policy. Defaults to 0.1.
            gamma (float, optional): weighting factor for current values. Defaults to 0.9.
            deterministic (bool, optional): whether the resulting policy has a single action outcome, or a full probability distribution. Defaults to False.
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: policy matrix and Q-values
        """
        n_states = len(self.state_domain)
        n_actions = len(self.actions)

        q_values = np.ones((n_states, n_actions))
        # TODO: handle terminal states?

        for e in range(n_episodes):
            physical_state = self.state_domain.sample_values(1, rng=self.rng)
            s = self.state_domain.values2index(physical_state)

            if verbose:
                print(f"episode {e}")
                print(f"    q_values")
                print(q_values)
                print(f"    initial state {s} {physical_state}")

            for _ in range(n_iters):
                eps_greedy_policy = get_epsilon_greedy_policy(
                    q_values[s].reshape(-1, 1), epsilon=epsilon
                )
                a = self.rng.choice(n_actions, p=eps_greedy_policy.flatten())

                physical_state = state_update.step(
                    physical_state, self.actions[a], self.rng
                )
                ss = self.state_domain.values2index(physical_state)
                reward = self.reward(self.actions[a], physical_state)

                q_values[s, a] += alpha * (
                    reward + gamma * np.max(q_values[ss]) - q_values[s, a]
                )
                s = ss

        self.policy = np.zeros((n_actions, n_states))
        for s in range(n_states):
            action_values = q_values[s]

            if deterministic:
                self.policy[np.argmax(action_values), s] = 1
            else:
                self.policy[:, s] = softmax(action_values)

        return self.policy, q_values.T


def plot_tabular_policy(
    policy: TabularPolicy,
    cmap: Union[str, Colormap] = CMAP_BLUES,
):
    """
    Helper method to represent the tabular policy with action and value labels

    Args:
        policy (TabularPolicy): the tabular policy to plot
        cmap (Union[str, Colormap], optional): the colormap to use. Defaults to CMAP_BLUES.
    """
    im = plt.imshow(policy.policy, cmap=cmap, vmin=0, vmax=1)

    plt.xlabel("state")
    plt.ylabel("action")

    ticks = list(range(len(policy.state_domain)))
    ticklabels = list(
        str(s_values) for s_values in policy.state_domain.index2values(ticks)
    )
    plt.xticks(ticks, labels=ticklabels, rotation=90)

    ticks = list(range(len(policy.actions)))
    ticklabels = [action.name for action in policy.actions]
    plt.yticks(ticks, labels=ticklabels)

    plt.colorbar(im)
    plt.tight_layout()
