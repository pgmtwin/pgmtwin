"""
Base implementation of classes to handle domains
"""

from __future__ import annotations
import copy
from typing import Dict, Sequence, Tuple

import numpy as np


class DiscreteDomain:
    """
    A multi-dimensional discrete domain
    Enables conversions from/to value records, index records, and multi indices of each dimension
    """

    def __init__(self, var2values: Dict[str, np.ndarray]):
        """
        Initializes the DiscreteDomain with a mapping of variable names to their possible values

        Args:
            var2values (Dict[str, np.ndarray]): a dictionary mapping variable names to their possible values.
        """
        self.labels = list(var2values.keys())
        self.var2values: Dict[str, np.ndarray] = copy.deepcopy(var2values)
        self.shape: Tuple[int] = tuple(
            len(choices) for choices in self.var2values.values()
        )
        self.value_shape: Tuple[int] = (len(self.var2values),)

    def __str__(self) -> str:
        return f"DiscreteDomain{list(self.var2values.keys())}"

    def __repr__(self) -> str:
        ret = f"{self} len {len(self)} value shape {self.value_shape}" + "\n"
        ret += "\n".join([f"    {k} {v}" for k, v in self.var2values.items()])
        return ret

    def __len__(self) -> int:
        return np.prod(self.shape)

    def __getitem__(self, key) -> np.ndarray:
        return self.var2values[key]

    def values2values(self, values: np.ndarray) -> np.ndarray:
        """
        Converts an array of values to the closest values in the domain

        Args:
            values (np.ndarray): a 1d or 2d array of value records

        Returns:
            np.ndarray: an array of value records, with the same shape as the input
        """
        values = np.asarray(values)
        ret_squeeze = values.ndim < 2
        if ret_squeeze:
            values = values.reshape(1, -1)

        ret = np.empty((len(values), len(self.labels)))
        for i, v in enumerate(values):
            for j, choices in enumerate(self.var2values.values()):
                idx = np.argmin(np.abs(choices - v[j]))
                ret[i, j] = choices[idx]

        return ret.squeeze(0) if ret_squeeze else ret

    def values2multi_index(self, values: np.ndarray) -> np.ndarray:
        """
        Converts an array of values to the corresponding multi indices in the domain

        Args:
            values (np.ndarray): a 1d or 2d array of value records

        Returns:
            np.ndarray: an array of multi indices, with the same shape as the input
        """
        values = np.asarray(values)
        ret_squeeze = values.ndim < 2
        if ret_squeeze:
            values = values.reshape(1, -1)

        ret = np.empty((len(values), len(self.labels)), dtype=int)
        for i, v in enumerate(values):
            for j, choices in enumerate(self.var2values.values()):
                idx = np.argmin(np.abs(choices - v[j]))
                ret[i, j] = idx

        return ret.squeeze(0) if ret_squeeze else ret

    def multi_index2values(self, m_indices: np.ndarray) -> np.ndarray:
        """
        Converts an array of multi indices to the corresponding values in the domain

        Args:
            m_indices (np.ndarray): a 1d or 2d array of multi indices

        Returns:
            np.ndarray: an array of value records, with the same shape as the input
        """
        m_indices = np.asarray(m_indices)
        ret_squeeze = m_indices.ndim < 2
        if ret_squeeze:
            m_indices = m_indices.reshape(1, -1)

        ret = np.empty((len(m_indices), len(self.labels)))
        for i, v in enumerate(m_indices):
            for j, choices in enumerate(self.var2values.values()):
                ret[i, j] = choices[v[j]]

        return ret.squeeze(0) if ret_squeeze else ret

    def index2multi_index(self, indices: np.ndarray) -> np.ndarray:
        """
        Converts an array of indices to the corresponding multi indices in the domain

        Args:
            indices (np.ndarray): a scalar index or 1d array of indices

        Returns:
            np.ndarray: a 1d or 2d array of multi indices
        """
        indices = np.asarray(indices)
        ret_squeeze = indices.ndim < 1
        if ret_squeeze:
            indices = np.atleast_1d(indices)

        ret = np.unravel_index(indices, self.shape)
        ret = np.array(list(zip(*ret)))

        return ret.squeeze(0) if ret_squeeze else ret

    def multi_index2index(self, m_indices: np.ndarray) -> np.ndarray:
        """
        Converts an array of multi indices to the corresponding indices in the domain

        Args:
            m_indices (np.ndarray): a 1d or 2d array of multi indices

        Returns:
            np.ndarray: a scalar index or 1d array of indices
        """
        m_indices = np.asarray(m_indices)
        ret_squeeze = m_indices.ndim < 2
        if ret_squeeze:
            m_indices = m_indices.reshape(1, -1)

        ret = np.array([np.ravel_multi_index(tuple(v), self.shape) for v in m_indices])

        return ret.squeeze(0) if ret_squeeze else ret

    def values2index(self, values: np.ndarray) -> np.ndarray:
        """
        Converts an array of values to the corresponding indices in the domain

        Args:
            values (np.ndarray): a 1d or 2d array of value records

        Returns:
            np.ndarray: a scalar index or 1d array of indices
        """
        values = np.asarray(values)
        ret_squeeze = values.ndim < 2
        if ret_squeeze:
            values = values.reshape(1, -1)

        m_indices = self.values2multi_index(values)

        ret = self.multi_index2index(m_indices)

        return ret.squeeze(0) if ret_squeeze else ret

    def index2values(self, indices: np.ndarray) -> np.ndarray:
        """
        Converts an array of indices to the corresponding values in the domain

        Args:
            indices (np.ndarray): a scalar index or 1d array of indices

        Returns:
            np.ndarray: a 1d or 2d array of value records
        """
        indices = np.asarray(indices)
        ret_squeeze = indices.ndim < 1
        if ret_squeeze:
            indices = np.atleast_1d(indices)

        m_indices = self.index2multi_index(indices)
        ret = self.multi_index2values(m_indices)

        return ret.squeeze(0) if ret_squeeze else ret

    def sample_indices(
        self, n_samples: int, p: np.ndarray = None, rng: np.random.Generator = None
    ) -> np.ndarray:
        """
        Generates samples of index records

        Args:
            n_samples (int): number of samples to generate
            p (np.ndarray, optional): probability distribution to sample with, or None for uniform sampling. Defaults to None.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: a scalar index, or array of index records with shape (n_samples,)
        """
        if rng is None:
            rng = np.random.default_rng()

        return rng.choice(len(self), size=n_samples, p=p).squeeze()

    def sample_values(
        self, n_samples: int, p: np.ndarray = None, rng: np.random.Generator = None
    ) -> np.ndarray:
        """
        Generate samples of values records

        Args:
            n_samples (int): number of samples to generate
            p (np.ndarray, optional): probability distribution to sample with, or None for uniform sampling. Defaults to None.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: an array of value records, with shape (n_samples, len(self.labels))
        """
        return self.index2values(self.sample_indices(n_samples, p, rng))

    @staticmethod
    def merge_domains(domains: Sequence[DiscreteDomain]) -> DiscreteDomain:
        """
        Generate a new DiscreteDomain instance, concatenating the dimensions of the given domain

        Args:
            domains (Sequence[DiscreteDomain]): a sequence of DiscreteDomain instances to merge

        Returns:
            DiscreteDomain: a new DiscreteDomain instance with the merged dimensions
        """
        d = {}
        for domain in domains:
            d = d | domain.var2values
        ret = DiscreteDomain(d)

        return ret
