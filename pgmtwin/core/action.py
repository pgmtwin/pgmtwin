"""
Base implementations of of Actions for the state update of a PhysicalAsset
"""

from abc import ABC
from typing import Optional, Union

import numpy as np
from matplotlib.colors import Colormap
from matplotlib import pyplot as plt

from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.utils import CMAP_BLUES


class BaseAction(ABC):
    """
    Models a transition between elements of a discrete state domain.
    This base class doesn't implement any actual transition table
    and is to be used by an external update model.
    """

    def __init__(
        self,
        name: str,
        state_domain: DiscreteDomain,
    ):
        """
        Sets the action's name and discrete domain

        Args:
            name (str): the name of the action
            state_domain (DiscreteDomain): the discrete domain of the action's state
        """
        self.name = name
        self.state_domain = state_domain

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def get_transition_probabilities(self) -> np.ndarray:
        """
        Concretizes the transition probabilities
        Returns a 2d (n_states, n_states) matrix whose columns sum to 1

        Returns:
            np.ndarray: a 2d (n_states, n_states) matrix whose columns sum to 1
        """
        return np.empty(0)

    def apply(
        self,
        state: np.ndarray,
        input_deterministic: bool,
        output_deterministic: bool,
    ) -> np.ndarray:
        """
        Applies the transition from the given state
        Handles all cases of deterministic/stochastic inputs/outputs

        Args:
            state (np.ndarray): the input state, given as value record
            input_deterministic (bool): whether the given state is a value record, or a distribution
            output_deterministic (bool): whether the output is to be given as a value record, or a distribution

        Returns:
            np.ndarray: the output state, given as value record or distribution
        """
        raise NotImplementedError(
            f"BaseAction {self.name} does not implement a discretized state transition"
        )


class ActionCPD(BaseAction):
    """
    Implements a static transition table for the state update
    """

    def __init__(
        self,
        name: str,
        state_domain: DiscreteDomain,
        transition_probabilities: Optional[np.ndarray] = None,
    ):
        super().__init__(name=name, state_domain=state_domain)

        if transition_probabilities is None:
            transition_probabilities = np.eye(len(self.state_domain))

        target_shape = (len(self.state_domain), len(self.state_domain))
        if transition_probabilities.shape != target_shape:
            raise ValueError(
                f"invalid cpd with shape {transition_probabilities.shape}, expected {target_shape}"
            )

        self.transition_probabilities = transition_probabilities

    def get_transition_probabilities(self):
        return self.transition_probabilities

    def apply(
        self,
        state: np.ndarray,
        input_deterministic: bool,
        output_deterministic: bool,
    ) -> np.ndarray:
        """
        Retrieves the transition outcome

        Args:
            state (np.ndarray): input state
            input_deterministic (bool): whether to treat the input state as values (True) or as a probability distribution (False)
            output_deterministic (bool): whether to return the outcome as values (True) or as a probability distribution (False)

        Raises:
            ValueError: if the input state is not a valid element of the domain, or if it is not a valid distribution for the domain
            ValueError: if the input state size does not match the domain size

        Returns:
            np.ndarray: the output state, given as value record or distribution
        """
        if input_deterministic:
            if state.size != len(self.state_domain.shape):
                raise ValueError(
                    f"input state is not a valid element of the domain; got {state.size} instead of {len(self.state_domain.shape)}"
                )

            idx = self.state_domain.values2index(state)
            y = self.transition_probabilities[:, idx].squeeze()
        else:
            if state.size != len(self.state_domain):
                raise ValueError(
                    "input state is not a valid distribution for the domain"
                )

            y = self.transition_probabilities @ state

        if output_deterministic:
            idx = np.argmax(y)
            y = self.state_domain.index2values(idx)

        return y


def plot_action_transitions(
    action: ActionCPD, colorbar: bool = True, cmap: Union[str, Colormap] = CMAP_BLUES
):
    """
    Helper method to represent the state transition matrix with the value labels

    Args:
        action (ActionCPD): the action to be represented
        colorbar (bool, optional): whether to draw a colorbar. Defaults to True.
        cmap (Union[str, Colormap], optional): colormap for the transitions. Defaults to CMAP_BLUES.
    """
    plt.imshow(action.get_transition_probabilities(), cmap=cmap, vmin=0, vmax=1)

    plt.xlabel("start state")
    plt.ylabel("end state")

    ticks = list(range(len(action.state_domain)))
    ticklabels = list(
        str(s_values) for s_values in action.state_domain.index2values(ticks)
    )
    plt.xticks(ticks, labels=ticklabels, rotation=90)
    plt.yticks(ticks, labels=ticklabels)

    if colorbar:
        plt.colorbar()
    plt.tight_layout()
