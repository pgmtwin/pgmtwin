"""
Definitions for the metadata of a variable in a Probabilistic Graphical Model
Such metadata are used during construction of the underlying DynamicBayesianNetwork
"""

from __future__ import annotations
from typing import Union

import numpy as np

from pgmtwin.core.domain import DiscreteDomain


class Variable:
    """
    Metadata for a variable in a Probabilistic Graphical Model
    """

    def __init__(
        self,
        name: str,
        distribution: bool = False,
        discrete: bool = True,
        domain: Union[np.ndarray, DiscreteDomain] = None,
    ):
        """
        Sets the metadata for the variable
        The domain can either be an array of shape (n_elems, 2) so that domain[i] = [lb, ub] for a continuous variable, or a DiscreteDomain

        Args:
            name (str): the name of the variable
            distribution (bool, optional): whether the variable's evidence comes as a single record, or as a probability distribution. Defaults to False.
            discrete (bool, optional): whether the variables has discrete state values or not. Defaults to True.
            domain (Union[np.ndarray, DiscreteDomain], optional): the domain for the states. Discrete variables use a DiscreteDomain instance, while continuous variables take an array of shape (n_features, 2) with lower and upper bounds of each feature. Defaults to None.
        """
        self.name = name
        self.distribution = distribution

        self.discrete = discrete
        self.domain = None
        if domain is not None:
            self.domain = domain

            if isinstance(self.domain, np.ndarray):
                assert (
                    self.domain.shape[-1] == 2
                ), "the domain of a continuous variable must be an array of shape (n_elems, 2) so that domain[i] = [lb, ub]"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other):
        if isinstance(other, Variable):
            return self.name < other.name
        return NotImplemented
