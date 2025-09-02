"""
Base abstract class for a DigitalAsset, holding a state distribution
Encapsulates the assimilation of observations through an inverse problem
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.utils import normalize_cpd


class BaseDigitalAsset(ABC):
    """
    Holds a state distribution and updates based on observations
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        state_distro: Optional[np.ndarray] = None,
        rng: np.random.Generator = None,
    ):
        """
        Initializes the DigitalAsset's internal objects.

        Args:
            state_domain (DiscreteDomain): the discrete domain of the state
            state_distro (Optional[np.ndarray], optional): the initial probability distribution of the state.
            If None, it is initialized with a uniform distribution. Defaults to None.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
        """
        self.state_domain = state_domain
        if state_distro is None:
            self.state_distro = self.reset_state_distribution()
        else:
            self.state_distro: np.ndarray = state_distro.copy()

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def reset_state_distribution(self) -> np.ndarray:
        """
        Reset the state distribution to be uniform

        Returns:
            np.ndarray: the updated state distribution
        """
        n_states = len(self.state_domain)
        self.state_distro = np.full(n_states, 1 / n_states)

        return self.state_distro

    @abstractmethod
    def get_assimilation(self, observations: np.ndarray) -> np.ndarray:
        """
        Solves the inverse problem for multiple observation records, returns the corresponding digital states

        Args:
            observations (np.ndarray): array of observation records

        Returns:
            np.ndarray: array of inverse problem solutions, one for each input observation record
        """
        pass

    @abstractmethod
    def get_assimilation_distribution(self, observations: np.ndarray) -> np.ndarray:
        """
        Retrieves the digital state distribution from the inverse problem solutions of a set of observations

        Args:
            observations (np.ndarray): array of observation records

        Returns:
            np.ndarray: the state distribution
        """
        pass

    def assimilate(self, observations: np.ndarray) -> np.ndarray:
        """
        Sets the internal digital state to the distribution computed from the given observations

        Args:
            observations (np.ndarray): array of observations

        Returns:
            np.ndarray: the updated state distribution
        """
        self.state_distro = self.get_assimilation_distribution(observations)

        return self.state_distro


class InverseProblemConfusionDigitalAsset(BaseDigitalAsset):
    """
    Holds a state distribution and provides uncertainty quantification of the digital state via the inverse problem's confusion matrix
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        inv_solver: Callable[[np.ndarray], np.ndarray],
        inv_solver_confusion_matrix: np.ndarray,
        state_distro: Optional[np.ndarray] = None,
    ):
        """
        Initializes the DigitalAsset with the given parameters.

        Args:
            state_domain (DiscreteDomain): the domain of the digital asset
            inv_solver (Callable[[np.ndarray], np.ndarray]): a callable that solves the inverse problem
            inv_solver_confusion_matrix (np.ndarray): the confusion matrix for the inverse solver
            state_distro (Optional[np.ndarray], optional): the initial probability distribution of the state.
        """
        super().__init__(state_domain=state_domain, state_distro=state_distro)

        self.inv_solver = inv_solver
        self.inv_solver_cpd = normalize_cpd(inv_solver_confusion_matrix.copy())

    def get_assimilation_cpd(self) -> np.ndarray:
        """
        Retrieve the assimilation CPD, which is the inverse solver's confidence matrix

        Returns:
            np.ndarray: the inverse solver's confidence matrix
        """
        return self.inv_solver_cpd

    def get_assimilation(self, observations: np.ndarray) -> np.ndarray:
        return self.inv_solver(observations)

    def get_assimilation_distribution(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations)
        observations = np.atleast_2d(observations)

        ret = np.zeros(len(self.state_domain))
        for o in observations:
            state_val = self.get_assimilation(o)
            state_idx = self.state_domain.values2index(state_val)

            ret[state_idx] += 1

        ret /= np.sum(ret)

        return ret

    def assimilate(self, observations: np.ndarray) -> np.ndarray:
        ret = self.get_assimilation_cpd() @ self.get_assimilation_distribution(
            observations
        )
        self.state_distro[:] = ret / np.sum(ret)

        return self.state_distro
