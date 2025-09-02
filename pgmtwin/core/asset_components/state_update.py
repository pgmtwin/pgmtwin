"""
Abstract and base implementations of components for the state update of a PhysicalAsset, due to an Action
"""

from abc import ABC, abstractmethod

import numpy as np

from pgmtwin.core.action import BaseAction


class StateUpdateComponent(ABC):
    """
    Abstract class for a state update component based on actions
    """

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        action: BaseAction,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Applies the action to the given state record

        Args:
            state (np.ndarray): the current asset state value record
            action (BaseAction): the action to apply
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: the updated state value record after applying the action
        """
        pass


class ActionUpdateComponent(StateUpdateComponent):
    """
    Updates the state by applying the given Action
    """

    def __init__(self, deterministic: bool):
        """
        Specifies if the Actions are applied in a deterministic or stochastic manner
        If deterministic, the output state is given as a value record
        If stochastic, the output is the MAP of the post-action state distribution

        Args:
            deterministic (bool): whether the actions are applied in a deterministic manner
        """
        self.deterministic = deterministic

    def step(
        self,
        state: np.ndarray,
        action: BaseAction,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if self.deterministic:
            ret = action.apply(
                state, input_deterministic=True, output_deterministic=True
            )
        else:
            if rng is None:
                rng = np.random.default_rng()

            distro = action.apply(
                state, input_deterministic=True, output_deterministic=False
            )
            state_idx = rng.choice(len(action.state_domain), p=distro)
            ret = action.state_domain.index2values(state_idx)

        return ret


class IsolatedRngStateUpdateComponent(StateUpdateComponent):
    """
    Wrap a StateUpdateComponent with a per-instance rng, so that the updates can be isolated by other pseudo-random processes
    """

    def __init__(
        self, state_update: StateUpdateComponent, rng: np.random.Generator = None
    ):
        """
        Sets the base update component and the pseudo-random generator

        Args:
            state_update (StateUpdateComponent): the state update component to use for the step
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
        """
        super().__init__(self)

        self.state_update = state_update
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def step(
        self,
        state: np.ndarray,
        action: BaseAction,
        rng: np.random.Generator = None,
    ):
        return self.state_update(state, action, self.rng)
