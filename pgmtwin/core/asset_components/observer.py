"""
Abstract and base implementations of components for the generation of observations of a PhysicalAsset

"""

from abc import ABC, abstractmethod

import numpy as np


class ObserverComponent(ABC):
    """
    Abstract class to generate observations from a state
    """

    @abstractmethod
    def get_observation(
        self,
        state: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """
        Generates a single observation from a given state

        Args:
            state (np.ndarray): the current state value record
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: the generated observation value record
        """
        pass


class DatabaseObserverComponent(ObserverComponent):
    """
    Base class to retrieve offline observations of states, based on the runtime state
    The inverse of the distance betweeen runtime state and db keys will be used as probability of selection

    """

    def __init__(self, state_db: np.ndarray, observation_db: np.ndarray):
        """
        Loads a database of states and observations

        Args:
            state_db (np.ndarray): the database of states to sample from
            observation_db (np.ndarray): the database of observations to sample from
        """
        assert state_db.ndim >= 2
        assert observation_db >= 2

        self.state_db = state_db
        self.observation_db = observation_db

    def get_observation(
        self,
        state: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        distances = np.linalg.norm(self.state_db - state, axis=-1)
        relevance = 1 / (distances + 1e-5)
        idx = rng.choice(len(self.state_db), size=1, p=relevance)

        return self.observation_db[idx]
