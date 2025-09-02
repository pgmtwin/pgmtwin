"""
Specialized Probabilistic Graphical Model for Structural Health Monitoring
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

from pgmtwin.core.dependency import Dependency
from pgmtwin.core.digital_asset import BaseDigitalAsset
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.action import BaseAction
from pgmtwin.core.pgm import dbn_inference, get_dbn
from pgmtwin.core.utils import (
    normalize_cpd,
    pgmpy_suppress_cpd_sum_warning,
)
from pgmtwin.core.variable import Variable


class PGMHelper:
    """
    Helper class to manage the Probabilistic Graphical Model for Structural Health Monitoring
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        inv_problem_confusion_matrix: np.ndarray,
        policy: Optional[np.ndarray] = None,
    ):
        """
        Implements the utilities to employ the specific PGM for Structural Health monitoring
        Assumes that the policy, if given, is a matrix

        Args:
            state_domain (DiscreteDomain): the digital state domain
            actions (List[BaseAction]): the list of actions
            inv_problem_confusion_matrix (np.ndarray): the inverse problem CPD, i.e. the confusion matrix for the assimilation
            policy (Optional[np.ndarray], optional): the matrix policy. Defaults to None.
        """
        self.state_domain = state_domain
        self.actions = actions
        self.action_domain = DiscreteDomain({"action": np.arange(len(self.actions))})

        self.inv_problem_confusion_matrix = inv_problem_confusion_matrix
        self.policy = policy

        # variables
        S = Variable(
            "S", distribution=False, discrete=True, domain=np.array([[-np.inf, np.inf]])
        )
        O = Variable(
            "O",
            distribution=False,
            discrete=False,
            domain=np.array([[-np.inf, np.inf]]),
        )

        D_ip = Variable(
            "D_ip", distribution=True, discrete=True, domain=self.state_domain
        )
        D = Variable("D", distribution=True, discrete=True, domain=self.state_domain)

        U = Variable("U", distribution=False, discrete=True, domain=self.action_domain)
        U_prev = Variable(
            "U_prev", distribution=False, discrete=True, domain=self.action_domain
        )

        R = Variable(
            "R", distribution=False, discrete=False, domain=[[-np.inf, np.inf]]
        )
        Q = Variable(
            "Q", distribution=False, discrete=False, domain=[[-np.inf, np.inf]]
        )

        self.variables = [S, O, D_ip, D, U, U_prev, R, Q]

        # dependencies
        S2O = Dependency("observe", [(S, 0)], (O, 0))
        O2Dip = Dependency("inverse_problem", [(O, 0)], (D_ip, 0))

        smoothing = 1e-6

        D_ip2D = Dependency(
            "confusion",
            [(D_ip, 0)],
            (D, 0),
            cpd=normalize_cpd(self.inv_problem_confusion_matrix),
        )
        D2U = Dependency("policy", [(D, 0)], (U, 0))
        if self.policy is not None:
            D2U.cpd = normalize_cpd(self.policy.copy())

        SU_prev2S = Dependency("physical_update", [(S, 0), (U_prev, 1)], (S, 1))

        u_priori_cpd = np.stack(
            [a.get_transition_probabilities() for a in self.actions]
        )
        u_priori_cpd = normalize_cpd(
            np.sum(u_priori_cpd, axis=2).T, smoothing=smoothing
        )

        U_prev2D_0 = Dependency(
            "digital_update",
            [(U_prev, 0)],
            (D, 0),
            cpd=u_priori_cpd,
        )

        tmp = np.hstack([a.get_transition_probabilities() for a in self.actions])
        DU_prev2D_1 = Dependency(
            "digital_update",
            [
                (U_prev, 1),
                (D, 0),
            ],
            (D, 1),
            cpd=normalize_cpd(tmp, smoothing=smoothing),
        )
        U2U_prev = Dependency(
            "action_observation",
            [(U, 0)],
            (U_prev, 1),
            cpd=np.eye(len(self.actions)),
        )
        D2Q = Dependency("qois", [(D, 0)], (Q, 0))
        DU_prev2R = Dependency("reward", [(D, 0), (U_prev, 0)], (R, 0))

        self.dependencies = [
            S2O,
            O2Dip,
            D_ip2D,
            D2U,
            SU_prev2S,
            U_prev2D_0,
            DU_prev2D_1,
            U2U_prev,
            D2Q,
            DU_prev2R,
        ]

        self.dependencies_assimilation = [D_ip2D, U_prev2D_0, DU_prev2D_1]
        if self.policy is not None:
            self.dependencies_assimilation.append(D2U)

        # the evolution dbn has only one action node, uses a different name
        DU2D_1 = Dependency(
            "digital_update",
            [
                (U, 0),
                (D, 0),
            ],
            (D, 1),
            cpd=DU_prev2D_1.cpd,
        )

        self.dependencies_evolution = [DU2D_1]
        if self.policy is not None:
            self.dependencies_evolution.append(D2U)

    def init_dbn_full(self, verbose: bool = False) -> DynamicBayesianNetwork:
        """
        Initializes the full dbn, containing also the physical state, observations, reward, and quantities of interest nodes.
        Might not support usage for inference, but can be plotted

        Args:
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Returns:
            DynamicBayesianNetwork: a (possibly invalid) dbn
        """
        ret, _ = get_dbn(self.dependencies, check_model=False, verbose=verbose)
        return ret

    def get_dbn_full_variable_pos(self) -> Dict[str, Tuple[float, float]]:
        """
        Utility to retrieve the metadata for plotting the full dbn

        Returns:
            Dict[str, Tuple[float, float]]: specifies the matplotlib color to paint the nodes of the corresponding variable
        """
        return {
            "S": (0, 0),
            "O": (1, -1),
            "D_ip": (2, -2),
            "D": (3, -3),
            "U": (6, -4),
            "U_prev": (0, -4),
            "R": (4.5, -5),
            "Q": (6, -5),
        }

    def init_dbn_assimilation(self, verbose: bool = False) -> DynamicBayesianNetwork:
        """
        Initializes a reduced dbn, to support the assimilation of the physical asset's observations

        Args:
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Returns:
            DynamicBayesianNetwork: a dbn for assimilation
        """
        ret, _ = get_dbn(self.dependencies_assimilation, verbose=verbose)
        return ret

    def init_dbn_evolution(self, verbose: bool = False) -> DynamicBayesianNetwork:
        ret, _ = get_dbn(self.dependencies_evolution, verbose=verbose)
        return ret

    def assimilate(
        self,
        dbn_infer: DBNInference,
        action_history: List[int],
        sensor_readings: np.ndarray,
        digital_asset: BaseDigitalAsset,
        n_samples: int,
        n_workers: int = 1,
        rng: np.random.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform data assimilation with the use of the given DBN, sensor readings and current (pre-assimilation) digital_asset
        The sensors_readings are assimilated through the digital asset and the DBN is inferenced n_samples times to obtain the updated distribution
        The digital_asset state distribution is updated with the computed values

        Args:
            dbn_infer (DBNInference): the DBN to use for assimilation
            action_history (List[int]): the history of actions taken
            sensor_readings (np.ndarray): the sensor readings to assimilate
            digital_asset (DigitalAsset): the digital asset to update
            n_samples (int): number of samples to use for the PGM inference
            n_workers (int, optional): number of parallel jobs to use for sampling. Defaults to 1.
            rng (np.random.Generator, optional): pseudo-random generator. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the updated digital state distribution and, if a matrix policy was given, the action distribution
        """
        if rng is None:
            rng = np.random.default_rng()

        # retrieve the values from the previous frame
        action = action_history[-1]
        dstate_distro = digital_asset.state_distro.copy()

        # observe and perform inverse problem
        dinv_distro = digital_asset.get_assimilation_distribution(sensor_readings)
        dinv_map_idx = np.argmax(dinv_distro)

        evidence = {
            ("U_prev", 1): int(action),
            ("D_ip", 1): int(dinv_map_idx),
        }
        inference_keys = [
            ("D", 1),
        ]
        if self.policy is not None:
            inference_keys += [("U", 1)]

        inference_results = dbn_inference(
            dbn_infer,
            inference_keys=inference_keys,
            evidence=evidence,
            evidence_sampled_distros={
                ("D", 0): dstate_distro,
                # ("D_ip", 1): dinv_distro,
            },
            n_samples=n_samples,
            rng=rng,
            n_workers=n_workers,
        )

        digital_asset.state_distro[:] = inference_results[("D", 1)]

        return inference_results[("D", 1)], (
            None if self.policy is None else inference_results[("U", 1)]
        )

    def evolve(
        self,
        dbn: DynamicBayesianNetwork,
        dstate_distro: np.ndarray,
        n_samples: int,
        n_steps: int,
        rng: np.random.Generator = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the evolution of the DBN, using the current cpds and the latest digital_state
        Computes the state distributions and actions distributions for the next n_steps, aggregating n_samples trajectories

        Args:
            dbn (DynamicBayesianNetwork): the DBN to use for evolution
            dstate_distro (np.ndarray): the current belief for the digital state distribution
            n_samples (int): number of samples to use for the inference
            n_steps (int): number of steps to simulate
            rng (np.random.Generator, optional): pseudo-random generator. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: the history of digital state distributions and action distributions
        """
        if rng is None:
            rng = np.random.default_rng()

        n_states = len(self.state_domain)
        n_actions = len(self.actions)

        dstate_distro_evolution = np.zeros((n_steps, n_states))
        action_distro_evolution = np.zeros((n_steps, n_actions))

        # simulation with virtual/soft evidence
        with pgmpy_suppress_cpd_sum_warning():
            sim: pd.DataFrame = dbn.simulate(
                n_time_slices=n_steps + 1,
                n_samples=n_samples,
                virtual_evidence=[
                    TabularCPD(("D", 0), n_states, dstate_distro.reshape(-1, 1))
                ],
                seed=rng.choice(2**31),
                show_progress=False,
            )
        # extract the columns of the dataframe
        dstates = sim[[("D", i) for i in range(n_steps + 1)]].to_numpy()
        us = sim[[("U", i) for i in range(n_steps + 1)]].to_numpy()

        for i in range(n_steps):
            for dstate_idx, action_idx in zip(dstates[:, i], us[:, i]):
                dstate_distro_evolution[i, dstate_idx] += 1
                action_distro_evolution[i, action_idx] += 1

        dstate_distro_evolution /= np.sum(
            dstate_distro_evolution, axis=-1, keepdims=True
        )
        action_distro_evolution /= np.sum(
            action_distro_evolution, axis=-1, keepdims=True
        )

        return dstate_distro_evolution, action_distro_evolution
