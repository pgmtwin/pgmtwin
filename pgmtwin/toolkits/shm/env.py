"""
Digital Twin Environment for Structural Health Monitoring, with support for assimilation using a specialized Probabilistic Graphical Model
"""

from typing import Callable, List, Optional

import numpy as np

import gymnasium as gym

from pgmpy.inference import DBNInference

from pgmtwin.core.action import BaseAction
from pgmtwin.core.digital_asset import InverseProblemConfusionDigitalAsset
from pgmtwin.core.env import BaseDigitalTwinEnv
from pgmtwin.core.physical_asset import BasePhysicalAsset
from pgmtwin.core.domain import DiscreteDomain

from pgmtwin.core.utils import pgmpy_suppress_cpd_replacement_warning
from pgmtwin.toolkits.shm.pgm_helper import PGMHelper


class DigitalTwinEnv(BaseDigitalTwinEnv):
    """
    Digital Twin Environment for Structural Health Monitoring, but not limited to SingleDamageDomain and MaintenanceAction.
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        physical_asset: BasePhysicalAsset,
        digital_asset: InverseProblemConfusionDigitalAsset,
        reward: Callable[[BaseAction, np.ndarray], float],
        n_samples_assimilation: int = 1,
        pgm_n_samples_assimilation: int = 0,
        pgm_n_workers: int = 1,
        pgm_policy_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initializes the Digital Twin Environment for Structural Health Monitoring.

        Args:
            state_domain (DiscreteDomain): a description of the digital state domain
            actions (List[BaseAction]): list of actions that can be taken in the environment
            physical_asset (BasePhysicalAsset): description of the physical asset
            digital_asset (DigitalAsset): description of the digital asset
            reward (Callable[[BaseAction, np.ndarray], float]): reward function that takes an action and a state
            n_samples_assimilation (int, optional): number of samples for assimilation of sensor readings. Defaults to 1.
            pgm_n_samples_assimilation (int, optional): number of digital state samples for assimilation via Probabilistic Graphical Model. Defaults to 0.
            pgm_policy_matrix (Optional[np.ndarray], optional): policy matrix. Defaults to None.
        """
        super().__init__(
            state_domain=state_domain,
            actions=actions,
            physical_asset=physical_asset,
            digital_asset=digital_asset,
            reward=reward,
            n_samples_assimilation=n_samples_assimilation,
        )

        self.pgm_policy_matrix = None
        if pgm_policy_matrix is not None:
            self.pgm_policy_matrix = pgm_policy_matrix.copy()

        self.pgm_helper = None
        self.dbn = None
        self.dbn_infer = None
        self.pgm_n_samples_assimilation = pgm_n_samples_assimilation
        self.pgm_n_workers = pgm_n_workers

        if pgm_n_samples_assimilation > 0:
            self.init_pgm_objects()

    def init_pgm_objects(self):
        pgm_helper = PGMHelper(
            self._state_domain,
            self._actions,
            inv_problem_confusion_matrix=self._digital_asset.get_assimilation_cpd(),
            policy=self.pgm_policy_matrix,
        )

        dbn = pgm_helper.init_dbn_assimilation()
        dbn.initialize_initial_state()

        with pgmpy_suppress_cpd_replacement_warning():
            dbn_infer = DBNInference(dbn)

        self.pgm_helper = pgm_helper
        self.dbn = dbn
        self.dbn_infer = dbn_infer

    def _step_assimilate(self, action: Optional[int], sensor_readings: np.ndarray):
        if not self.pgm_helper:
            super()._step_assimilate(action, sensor_readings)
        else:
            self.pgm_helper.assimilate(
                dbn_infer=self.dbn_infer,
                action_history=[action if action is not None else 0],
                sensor_readings=sensor_readings,
                digital_asset=self._digital_asset,
                n_samples=self.pgm_n_samples_assimilation,
                n_workers=self.pgm_n_workers,
                rng=self.np_random,
            )


if "shm" not in gym.registry:
    gym.register("shm", DigitalTwinEnv)
