"""
Module for the implementation of a base Env compatible with gym
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from pgmtwin.core.action import BaseAction
from pgmtwin.core.digital_asset import BaseDigitalAsset
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.physical_asset import BasePhysicalAsset


class BaseDigitalTwinEnv(gym.Env):
    """
    Base class for an Env compatible with gym and stable-baselines, for policy training and simulation
    It encapsulates the Physical and Digital assets, exposing the digital state distribution as an observation in a Box space
    This base implementation does not employ a Probabilistic Graphical Model, as the Assets only communicate through the inverse problem
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        physical_asset: BasePhysicalAsset,
        digital_asset: BaseDigitalAsset,
        reward: Callable[[BaseAction, np.ndarray], float],
        n_samples_assimilation: int = 1,
    ):
        """
        Initializes the BaseDigitalTwinEnv with the given parameters.

        Args:
            state_domain (DiscreteDomain): description of the digital state domain
            actions (List[BaseAction]): list of actions that can be applied to the physical asset
            physical_asset (BasePhysicalAsset): physical asset that is being modeled
            digital_asset (BaseDigitalAsset): digital asset that holds the state distribution and performs assimilation
            reward (Callable[[BaseAction, np.ndarray], float]): reward function
            n_samples_assimilation (int, optional): number of samples for sensor reading assimilation . Defaults to 1.
        """
        self._state_domain = state_domain
        self._actions = actions
        self._physical_asset = physical_asset
        self._digital_asset = digital_asset
        self._reward = reward
        self._n_samples_assimilation = n_samples_assimilation

        n_states = len(self._state_domain)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_states,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(self._actions))

    def _get_obs(self) -> np.ndarray:
        return self._digital_asset.state_distro.copy()

    def _get_qois(self) -> np.ndarray:
        return np.empty(0)

    def _get_info(self) -> Dict[str, np.ndarray]:
        return {
            "physical_state": self._physical_asset.state.copy(),
            "qois": self._get_qois(),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Resets the internal state, with the option of setting the physical state and digital state distribution.
        The options dictionary keys are "physical_state" and "digital_state_distro" respectively.

        Args:
            seed (Optional[int], optional): a seed for the environment's pseudo random generator. Defaults to None.
            options (Optional[dict], optional): a dictionary of options. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: an observation of the new state, and a dictionary of extra informations
        """
        super().reset(seed=seed)

        self._physical_asset.rng = self.np_random
        self._digital_asset.rng = self.np_random

        # randomize physical asset's state
        if options is not None and "physical_state" in options:
            self._physical_asset.set_state(options["physical_state"])
        else:
            self._physical_asset.set_state(
                self._state_domain.sample_values(1, rng=self.np_random)
            )

        # fixed starting state for digital asset
        if options is not None and "digital_state_distro" in options:
            self._digital_asset.state_distro = options["digital_state_distro"]
        else:
            self._digital_asset.reset_state_distribution()

        # observe
        sensor_readings = self._step_sensor_readings(None)

        # assimilate
        self._step_assimilate(None, sensor_readings)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _terminated(self) -> bool:
        """
        Whether the simulation must be terminated and the environment reset.
        Override for specialization

        Returns:
            bool: whether the simulation must be terminated and the environment reset
        """
        return False

    def _step_update(self, action: int):
        """
        Applies the action identified by the index to the physical asset
        Override for specialization

        Args:
            action (int): index of the action to be applied
        """
        self._physical_asset.update(self._actions[action])

    def _step_sensor_readings(self, action: Optional[int]) -> np.ndarray:
        """
        Interrogate the physical asset, with optional knowledge of the last applied action
        Override for specialization

        Args:
            action (int, optional): index of the last applied action

        Returns:
            np.ndarray: a (n_samples_assimilation, sensor_reading_shape) array
        """
        sensor_readings = self._physical_asset.get_observations(
            self._n_samples_assimilation
        )

        return sensor_readings

    def _step_assimilate(self, action: Optional[int], sensor_readings: np.ndarray):
        """
        Assimilates the sensor readings in the digital asset, with optional knowledge of the last applied action
        Override for specialization

        Args:
            action (int, optional): index of the last applied action, or None if it is performed during the environment reset
            sensor_readings (np.ndarray): a (n_samples_assimilation, sensor_reading_shape) array
        """
        self._digital_asset.assimilate(sensor_readings)

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]:
        """
        Applies the given action to the PhysicalAsset and assimilates a new set of observations of the updated physical state
        through the digital asset

        Args:
            action (int): the action to be applied

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]: observation, reward, terminated, truncated, info dictionary
        """
        # apply action
        self._step_update(action)

        # observe
        sensor_readings = self._step_sensor_readings(action)

        # assimilate
        self._step_assimilate(action, sensor_readings)

        terminated = self._terminated()
        reward = self._reward(self._actions[action], self._physical_asset.state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
