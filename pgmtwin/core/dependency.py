"""
Definitions for the metadata of a variable dependency in a Probabilistic Graphical Model
Such metadata are used during construction of the underlying DynamicBayesianNetwork
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np

from pgmtwin.core.utils import get_combined_cpd_pair

from .variable import Variable


class Dependency:
    """
    Specifies the inputs, output and, optionally, the cpd of relationships between variables in a Probabilistic Graphical Model
    """

    def __init__(
        self,
        name: str,
        inputs: Sequence[Tuple[Variable, int]],
        output: Tuple[Variable, int],
        cpd: Optional[np.ndarray] = None,
    ):
        """
        Sets the features of the dependency between variables, to be used as metadata for the construction of a DynamicBayesianNetwork

        Args:
            name (str): the name of the dependency
            inputs (Sequence[Tuple[Variable, int]]): sequence of variables and their observation frame
            output (Tuple[Variable, int]): the output variable and its observation frame
            cpd (Optional[np.ndarray], optional): conditional probability distribution, given as an array of shape (cardinality of output, prod(cardinalities of inputs)). Defaults to None.
        """
        self.name = name

        assert isinstance(output, tuple)
        var_out, f_out = output
        assert isinstance(var_out, Variable)
        assert isinstance(f_out, int)

        for i in inputs:
            assert isinstance(i, tuple)
            var_in, f_in = i
            assert isinstance(var_in, Variable)
            assert isinstance(f_in, int)

            if var_out is var_in:
                assert f_out > f_in

        self.inputs = inputs
        self.output = output

        self.cpd = cpd

    def get_timestep_copy(self, d: int) -> Dependency:
        """
        Generates a copy of this dependency, where all frame values are incremented by d

        Args:
            d (int): the frame increment

        Returns:
            Dependency: a new Dependency instance with incremented frame values
        """
        o_v, o_f = self.output
        ret = Dependency(
            self.name,
            inputs=[(v, f + d) for v, f in self.inputs],
            output=(o_v, o_f + d),
            cpd=self.cpd,
        )

        return ret

    def merge(self, other: Dependency, cpd_smoothing: float = 0) -> Dependency:
        """
        Creates a new dependenccy that combines the current one and other
        The inputs are concatenated and the cpds are combined

        Args:
            other (Dependency): the other dependency to merge with this one
            cpd_smoothing (bool, optional): smoothing when normalizing combined cpds. Defaults to 0.

        Returns:
            Dependency: a new Dependency instance that combines the current one and other
        """
        assert (
            self.output == other.output
        ), f"cannot merge Dependency with different outputs {self.output} and {other.output}"
        intersect = set(self.inputs) & set(other.inputs)
        assert not intersect, f"intersection of inputs is not empty {intersect}"

        cpd = None
        if self.cpd is not None and other.cpd is not None:
            cpd = get_combined_cpd_pair(self.cpd, other.cpd, smoothing=cpd_smoothing)
        elif self.cpd is not None or other.cpd is not None:
            assert (
                False
            ), f"cpds to be merged must either be both np.ndarrays, or both None; got {self.cpd} and {other.cpd}"

        ret = Dependency(
            self.name + "+" + other.name,
            self.inputs + other.inputs,
            self.output,
            cpd=cpd,
        )
        return ret

    def __str__(self) -> str:
        ret = f"{self.name} : {tuple(str(v) for v in self.inputs)} -> {self.output}"

        return ret

    def __repr__(self) -> str:
        return str(self)
