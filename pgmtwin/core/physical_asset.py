"""
Abstract and base implementation for a PhysicalAsset, which holds a state updated by Actions and generates observations
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pgmtwin.core.action import BaseAction
from pgmtwin.core.asset_components.observer import ObserverComponent
from pgmtwin.core.asset_components.observation_noise import (
    BaseObservationNoiseComponent,
)
from pgmtwin.core.asset_components.state_update import StateUpdateComponent


class BasePhysicalAsset(ABC):
    """
    Abstract class for a PhysicalAsset, which holds a state updated by Actions and generates observations
    """

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        rng: np.random.Generator = None,
    ):
        """
        Initializes the PhysicalAsset with an initial state and a random number generator

        Args:
            initial_state (np.ndarray, optional): the initial state of the PhysicalAsset
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
        """
        self.state = initial_state

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def set_state(self, state: np.ndarray):
        """
        Sets the current state of the PhysicalAsset

        Args:
            state (np.ndarray): the new state to set
        """
        self.state = state.copy()

    @abstractmethod
    def update(self, action: BaseAction) -> np.ndarray:
        """
        Updates the state of the PhysicalAsset by applying the given Action

        Args:
            action (BaseAction): the action to apply to the current state

        Returns:
            np.ndarray: the updated state of the PhysicalAsset
        """
        pass

    @abstractmethod
    def get_observations(self, n_observations: int = 1) -> np.ndarray:
        """
        Retrieves a set of observations from the current state of the PhysicalAsset

        Args:
            n_observations (int, optional): number of observation records to retrieve. Defaults to 1.

        Returns:
            np.ndarray: the observation generated from the current state
        """
        pass


class ComposablePhysicalAsset(BasePhysicalAsset):
    """
    Component-based implementation of a PhysicalAsset, which enables the
    customization of state updates, observations, and noise
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        state_update_component: StateUpdateComponent,
        sensor_component: ObserverComponent,
        noise_component: BaseObservationNoiseComponent = None,
        rng: np.random.Generator = None,
    ):
        """
        Initializes the ComposablePhysicalAsset with an initial state and components for state updates, observations, and noise

        Args:
            initial_state (np.ndarray): the initial state of the PhysicalAsset
            state_update_component (StateUpdateComponent): state update component to apply actions
            sensor_component (ObserverComponent): component to generate observations from the state
            noise_component (BaseObservationNoiseComponent, optional): component to add noise to observations
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Raises:
            ValueError: if state_update_component is None
            ValueError: if sensor_component is None
        """
        super().__init__(initial_state=initial_state, rng=rng)

        self.state_update_component = state_update_component
        if self.state_update_component is None:
            raise ValueError(f"state_update_component is required")

        self.sensor_component = sensor_component
        if self.state_update_component is None:
            raise ValueError(f"sensor_component is required")

        self.noise_component = noise_component

    def update(self, action: BaseAction) -> np.ndarray:
        self.state = self.state_update_component.step(self.state, action, self.rng)
        return self.state

    def get_observations(self, n_observations: int = 1) -> np.ndarray:
        ret = np.array(
            [
                self.sensor_component.get_observation(self.state, self.rng)
                for _ in range(n_observations)
            ]
        )

        if self.noise_component is not None:
            ret = self.noise_component.apply_noise(self.state, ret, self.rng)

        return ret
