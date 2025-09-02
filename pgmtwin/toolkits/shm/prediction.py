"""
Module to track the evolution of a digital twin built with the Structural Health Monitoring entities
"""

import copy
from typing import Optional

import numpy as np
import pandas as pd

from stable_baselines3.common.policies import BasePolicy

from pgmtwin.core.asset_components.state_update import ActionUpdateComponent
from pgmtwin.core.tabular_policy import TabularPolicy
from pgmtwin.core.utils import get_transition_matrix_from_history

from pgmtwin.toolkits.shm.action import MaintenanceAction
from pgmtwin.toolkits.shm.env import DigitalTwinEnv


class Predictor:
    """
    Class to simulate the evolution of a Structural Health Monitoring digital twin environment using a policy.
    """

    def __init__(
        self,
        env: DigitalTwinEnv,
        policy: BasePolicy,
    ):
        """
        Initializes the Predictor with a digital twin environment and a policy.

        Args:
            env (BaseDigitalTwinEnv): a digital twin environment to simulate the Structural Health Monitoring
            policy (BasePolicy): a stable_baselines3-compatible policy to use for action selection
        """
        self.env: DigitalTwinEnv = env.unwrapped
        assert isinstance(self.env, DigitalTwinEnv)

        self.policy = policy

    def simulate(
        self,
        n_iters: int,
        n_iters_fit_actions_policy: int = 0,
        action_deterministic: bool = True,
        reset_options: Optional[dict] = None,
        rng: np.random.Generator = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Simulates the evolution of the digital twin environment for a specified number of iterations.
        Supports refitting the actions and policy at specified intervals.

        Args:
            n_iters (int): the number of iterations to simulate
            n_iters_fit_actions_policy (int, optional): intervals after which the actions and policy are re-fit. Defaults to 0.
            action_deterministic (bool, optional): whether the actions act deterministic or the outcomes are sampled. Defaults to True.
            reset_options (Optional[dict], optional): dictionary of options for the environment reset. Defaults to None.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Raises:
            ValueError: if n_iters_fit_actions_policy is specified and the policy is not a TabularPolicy

        Returns:
            pd.DataFrame: the DataFrame containing the history of the simulation, including actions, states, rewards, and digital state distributions
        """
        if n_iters_fit_actions_policy and not isinstance(self.policy, TabularPolicy):
            raise ValueError(
                f"actions and policy can be refit only for TabularPolicy; found {type(self.policy)}"
            )

        if rng is None:
            rng = np.random.default_rng()

        # region history
        cols = [  # physical state
            "physical_state",
            "digital_state_true",
            "digital_state_true_idx",
            # digital state
            "digital_state_distro",
            "digital_state_map",
            "digital_state_map_idx",
            # policy
            "action_idx",
            "action_true_idx",
            # reward at the start of the frame
            "reward",
            "reward_cumulative",
        ]
        episode_dfdict = {c: [] for c in cols}

        def history_append():
            # physical state
            episode_dfdict[f"physical_state"].append(info["physical_state"])

            pstate_dstate_idx = self.env._state_domain.values2index(
                info["physical_state"]
            )
            pstate_dstate = self.env._state_domain.index2values(pstate_dstate_idx)
            episode_dfdict[f"digital_state_true"].append(pstate_dstate)
            episode_dfdict[f"digital_state_true_idx"].append(pstate_dstate_idx)

            # digital state
            episode_dfdict[f"digital_state_distro"].append(obs)
            dstate_idx = np.argmax(obs)
            dstate = self.env._state_domain.index2values(dstate_idx)
            episode_dfdict[f"digital_state_map"].append(dstate)
            episode_dfdict[f"digital_state_map_idx"].append(dstate_idx)

            # policy
            episode_dfdict[f"action_idx"].append(action)

            pstate2dstate_obs = np.zeros_like(obs)
            pstate2dstate_obs[pstate_dstate_idx] = 1
            pstate2dstate_action, _ = self.policy.predict(
                pstate2dstate_obs, deterministic=action_deterministic
            )
            episode_dfdict[f"action_true_idx"].append(pstate2dstate_action)

            # reward at the start of the frame
            episode_dfdict[f"reward"].append(reward)
            episode_dfdict[f"reward_cumulative"].append(reward_cumulative)

            if verbose:
                print(f'frame {len(episode_dfdict[f"physical_state"])-1}')

                print(f'    physical_state {episode_dfdict["physical_state"][-1]}')
                print(
                    f'    physical_state to digital {episode_dfdict["digital_state_true"][-1]}'
                )
                print(
                    f'    digital_state map {episode_dfdict["digital_state_map"][-1]}'
                )
                print(
                    f'    digital_state_distro action {episode_dfdict["action_idx"][-1]}'
                )
                print(
                    f'    physical_state to digital action {episode_dfdict["action_true_idx"][-1]}'
                )

        # endregion

        if verbose:
            print(f"env reset")

        # frame 0
        reward = 0
        reward_cumulative = 0

        obs, info = self.env.reset(seed=int(rng.integers(2**32)), options=reset_options)

        action, _ = self.policy.predict(obs, deterministic=action_deterministic)

        history_append()

        for i in range(n_iters):
            # physical state update and assimilation
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward_cumulative += reward

            # policy invocation
            action, _ = self.policy.predict(obs, deterministic=action_deterministic)

            history_append()

            if terminated or truncated:
                break

            if n_iters_fit_actions_policy and i and i % n_iters_fit_actions_policy == 0:
                if verbose:
                    print(f"    refit policy's actions")

                hist_actions = episode_dfdict[f"action_idx"][:-1]
                hist_transitions = np.stack(
                    [
                        episode_dfdict[f"digital_state_map_idx"][:-1],
                        episode_dfdict[f"digital_state_map_idx"][1:],
                    ],
                    axis=0,
                )

                # fit a new set of action from the history collected so far
                actions_new = copy.deepcopy(self.env._actions)
                for a, action_new in enumerate(actions_new):
                    if verbose:
                        print(f"    processing transitions for {action_new}")

                    if isinstance(action_new, MaintenanceAction):
                        if verbose:
                            print(
                                f"        previous prior {action_new.transition_prior}"
                            )

                        mask = hist_actions == a
                        if np.any(mask):
                            trx = get_transition_matrix_from_history(
                                n_states=len(self.env._state_domain),
                                state_idx_from=hist_transitions[mask, 0],
                                state_idx_to=hist_transitions[mask, 1],
                            )
                            action_new.update(trx)

                            if verbose:
                                print(
                                    f"        updated prior {action_new.transition_prior}"
                                )
                        else:
                            if verbose:
                                print(f"        skipped unobserved action {action_new}")
                    else:
                        if verbose:
                            print(f"        no prior to fit")

                if verbose:
                    print(f"refit policy")

                # override the actions in the policy
                policy: TabularPolicy = self.policy
                policy.actions = actions_new
                policy.rng = rng

                # refit the policy on the new actions
                # TODO: pick parameters from somewhere
                policy.fit_q_learning(
                    state_update=ActionUpdateComponent(action_deterministic),
                    n_episodes=100,
                    n_iters=100,
                    deterministic=action_deterministic,
                    verbose=verbose,
                )

                if self.env.pgm_helper:
                    # update the pgmhelper with the new actions
                    self.env.init_pgm_objects()

        # region history
        # expand physical_state columns
        # TODO: this probably needs a specific state_domain if metadata are needed
        pstate_cols = [
            f"physical_state[{d}]" for d in self.env._state_domain.var2values
        ]
        pstate_vals = episode_dfdict.pop("physical_state")

        # expand digital_state_distro columns
        dstate_distro_cols = [
            f"digital_state_distro[{self.env._state_domain.index2values(i)}]"
            for i in range(len(self.env._state_domain))
        ]
        dstate_distro_vals = episode_dfdict.pop("digital_state_distro")

        episode_df = pd.DataFrame(episode_dfdict)

        episode_df[pstate_cols] = np.array(pstate_vals)
        episode_df[dstate_distro_cols] = np.array(dstate_distro_vals)
        # endregion

        return episode_df
