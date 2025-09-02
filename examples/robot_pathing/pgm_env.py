from typing import Optional

import numpy as np

import gymnasium as gym

from pgmpy.inference import DBNInference

from pgmtwin.core.utils import pgmpy_suppress_cpd_replacement_warning
from examples.robot_pathing.common import (
    RobotPathingEnv,
    RobotPathingPGMHelper,
    RobotPathingSetup,
)


class RobotPathingPGMEnv(RobotPathingEnv):
    def __init__(
        self,
        noise_type: str,
        state_update_type: str,
        n_obs_assimilation: int = 1,
        pgm_n_samples_assimilation: int = 1,
        pgm_n_workers: Optional[int] = None,
        inv_solver_confusion_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(
            noise_type=noise_type,
            state_update_type=state_update_type,
            n_obs_assimilation=n_obs_assimilation,
            inv_solver_confusion_matrix=inv_solver_confusion_matrix,
        )

        setup = RobotPathingSetup()

        pgm_helper = None
        dbn = None
        dbn_infer = None
        if pgm_n_samples_assimilation > 0:
            pgm_helper = RobotPathingPGMHelper(
                self._state_domain,
                self._actions,
                n_beacons=len(setup.beacons_coords),
            )
            dbn = pgm_helper.init_dbn_assimilation()
            dbn.initialize_initial_state()

            with pgmpy_suppress_cpd_replacement_warning():
                dbn_infer = DBNInference(dbn)

        self.pgm_helper = pgm_helper
        self.dbn = dbn
        self.dbn_infer = dbn_infer
        self.pgm_n_samples_assimilation = pgm_n_samples_assimilation
        self.pgm_n_workers = pgm_n_workers

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


if "robot_pathing_pgm" not in gym.registry:
    gym.register("robot_pathing_pgm", RobotPathingPGMEnv)
