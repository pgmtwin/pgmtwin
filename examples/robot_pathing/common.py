import copy
from typing import Dict, List, Optional, Tuple, Union
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import gymnasium as gym

from pgmpy.models import DynamicBayesianNetwork
from pgmpy.inference import DBNInference

from pgmtwin.core.action import BaseAction, ActionCPD
from pgmtwin.core.asset_components.observation_noise import (
    SNRGaussianNoiseComponent,
)
from pgmtwin.core.asset_components.observer import ObserverComponent
from pgmtwin.core.asset_components.state_update import (
    ActionUpdateComponent,
)
from pgmtwin.core.dependency import Dependency
from pgmtwin.core.digital_asset import (
    BaseDigitalAsset,
    InverseProblemConfusionDigitalAsset,
)
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.env import BaseDigitalTwinEnv
from pgmtwin.core.pgm import dbn_inference, get_dbn
from pgmtwin.core.physical_asset import ComposablePhysicalAsset
from pgmtwin.core.utils import CMAP_BLUES, SingletonMeta, normalize_cpd, softmax
from pgmtwin.core.variable import Variable


from stable_baselines3.common.type_aliases import PolicyPredictor


# ==== utils
class RobotPathingSetup(metaclass=SingletonMeta):
    """
    Singleton class to set up the robot pathing environment, including world dimensions,
    beacon and goal coordinates, state domain, and available actions and variables.
    """

    def __init__(
        self,
        n_world_xsteps: int = 5,
        n_world_ysteps: int = 5,
    ):
        self.n_world_xsteps = n_world_xsteps
        self.n_world_ysteps = n_world_ysteps

        self.beacons_coords = np.array(
            [
                [self.n_world_xsteps * 0.25, self.n_world_ysteps * 0.25],
                [self.n_world_xsteps * 0.25, self.n_world_ysteps * 0.75],
                [self.n_world_xsteps * 0.75, self.n_world_ysteps * 0.25],
                [self.n_world_xsteps * 0.75, self.n_world_ysteps * 0.75],
            ]
        )
        self.goals_coords = np.random.default_rng(123456).random(size=(1000, 2)) * [
            [self.n_world_xsteps, self.n_world_ysteps]
        ]

        self.x_steps = np.arange(self.n_world_xsteps)
        self.y_steps = np.arange(self.n_world_ysteps)

        self.cov_measurement = None
        self.cov_movement = None

        self.state_domain = DiscreteDomain(
            {
                "x": self.x_steps,
                "y": self.y_steps,
            }
        )

        self.goals_coords = self.state_domain.values2values(self.goals_coords)

        self.init_actions()
        self.init_variables()

    def init_actions(
        self,
    ):
        """
        Initialize the available actions and their transition probability matrices.
        """
        n_states = len(self.state_domain)

        dirs = ["north", "south", "west", "east"]
        cpds = dict((d, np.zeros((n_states, n_states))) for d in dirs)

        # noiseless cpds
        for x in range(self.n_world_xsteps):
            for y in range(self.n_world_ysteps):
                s_from = self.state_domain.multi_index2index([x, y])

                s_to = self.state_domain.multi_index2index(
                    [x, np.clip(y + 1, 0, self.n_world_ysteps - 1)]
                )
                cpds["north"][s_to, s_from] = 1

                s_to = self.state_domain.multi_index2index(
                    [x, np.clip(y - 1, 0, self.n_world_ysteps - 1)]
                )
                cpds["south"][s_to, s_from] = 1

                s_to = self.state_domain.multi_index2index(
                    [np.clip(x - 1, 0, self.n_world_xsteps - 1), y]
                )
                cpds["west"][s_to, s_from] = 1

                s_to = self.state_domain.multi_index2index(
                    [np.clip(x + 1, 0, self.n_world_xsteps - 1), y]
                )
                cpds["east"][s_to, s_from] = 1

        self.stay = ActionCPD(
            "stay", self.state_domain, transition_probabilities=np.eye(n_states)
        )

        self.go_north = ActionCPD(
            "north",
            self.state_domain,
            transition_probabilities=cpds["north"],
        )

        self.go_south = ActionCPD(
            "south",
            self.state_domain,
            transition_probabilities=cpds["south"],
        )

        self.go_west = ActionCPD(
            "west",
            self.state_domain,
            transition_probabilities=cpds["west"],
        )

        self.go_east = ActionCPD(
            "east",
            self.state_domain,
            transition_probabilities=cpds["east"],
        )

        self.actions: List[BaseAction] = [
            self.stay,
            self.go_north,
            self.go_south,
            self.go_west,
            self.go_east,
        ]

        self.action_domain = DiscreteDomain({"action": np.arange(len(self.actions))})

    def init_variables(self):
        """
        Initialize the state and action variables for the robot pathing environment.
        """
        self.dstate_ip = Variable(
            "D_{IP}",
            distribution=True,
            discrete=True,
            domain=self.state_domain,
        )
        self.dstate = Variable(
            "D",
            distribution=True,
            discrete=True,
            domain=self.state_domain,
        )

        self.prev_action = Variable(
            "U^{A}_{-1}",
            distribution=False,
            discrete=True,
            domain=self.action_domain,
        )

        self.next_action = Variable(
            "U^{P}",
            distribution=False,
            discrete=True,
            domain=self.action_domain,
        )

        self.variables = [
            self.dstate_ip,
            self.dstate,
            self.prev_action,
            self.next_action,
        ]


class RobotPathingBeaconObserver(ObserverComponent):
    """
    Observer component that simulates beacon-based observations for a robot,
    providing distances and angles to each beacon, with optional Gaussian noise.
    """

    def __init__(self, beacons_coords: np.ndarray, mean: float = 0, std: float = 0):
        """
        Initialize the observer with beacon coordinates and optional noise parameters.

        Args:
            beacons_coords (np.ndarray): coordinates of the beacons.
            mean (float): mean of the Gaussian noise.
            std (float): standard deviation of the Gaussian noise.
        """
        super().__init__()

        self.beacons_coords = beacons_coords

        self.mean = mean
        self.std = std

    def get_observation(
        self, state: np.ndarray, rng: np.random.Generator = None
    ) -> np.ndarray:
        """
        Compute the observation (distance and angle) from the robot's state to each beacon.

        Args:
            state (np.ndarray): the current state of the robot.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.

        Returns:
            np.ndarray: array of distances and angles to each beacon.
        """
        if rng is None:
            rng = np.random.default_rng()

        diffs = state - self.beacons_coords
        if not np.isclose(self.std, 0):
            diffs += rng.normal(self.mean, self.std, size=self.beacons_coords.shape)

        # distances and angles
        ret = np.array(
            [np.linalg.norm(diffs, axis=1), np.atan2(diffs[:, 1], diffs[:, 0])]
        ).T

        return ret


class RobotPathingInverseSolverMean:
    """
    Inverse solver that estimates the robot's position as the mean of positions
    inferred from beacon observations.
    """

    def __init__(self, beacons_coords: np.ndarray):
        """
        Initialize the solver with beacon coordinates.

        Args:
            beacons_coords (np.ndarray): coordinates of the beacons.
        """
        self.beacons_coords = beacons_coords

    def mean_pos_single(self, observations: np.ndarray) -> np.ndarray:
        """
        Estimate the robot's position from beacon observations.

        Args:
            observations (np.ndarray): observed distances and angles to beacons.

        Returns:
            np.ndarray: estimated position.
        """
        observations = np.asarray(observations)
        assert observations.ndim == 1
        observations = observations.reshape(len(self.beacons_coords), 2)

        states = (
            self.beacons_coords
            + observations[:, 0][:, np.newaxis]
            * np.array([np.cos(observations[:, 1]), np.sin(observations[:, 1])]).T
        )

        return np.mean(states, axis=0)

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """
        Estimate the robot's position from beacon observations.

        Args:
            observations (np.ndarray): array of multiple observed distances and angles to beacons.

        Returns:
            np.ndarray: estimated position.
        """
        observations = np.asarray(observations)
        squeeze = observations.ndim < 2
        observations = observations.reshape(-1, len(self.beacons_coords) * 2)

        ret = np.array([self.mean_pos_single(o) for o in observations])

        return ret if not squeeze else np.squeeze(ret, axis=0)


class RobotPathingInverseSolverEKF:
    """
    Inverse solver using an Extended Kalman Filter (EKF) to estimate the robot's position
    from beacon observations.
    """

    def __init__(self, beacons_coords: np.ndarray):
        """
        Initialize the EKF solver with beacon coordinates.

        Args:
            beacons_coords (np.ndarray): coordinates of the beacons.
        """
        self.beacons_coords = beacons_coords
        self.beacon_observer = RobotPathingBeaconObserver(beacons_coords)

        self.state_estimate = None
        self.state_covariance = None
        self.observations = None
        self.measurement_noise = None

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """
        Update the state estimate using EKF based on new observations.

        Args:
            observations (np.ndarray): observed distances and angles to beacons.

        Returns:
            np.ndarray: updated state estimate.
        """
        observations = np.asarray(observations)
        assert len(observations.shape) == 2

        # kalman filtering of the observed position process

        def dhdx(x_hat):
            H = np.asmatrix(np.zeros([2, len(self.beacons_coords) * 2]))
            for i, landmark in enumerate(self.beacons_coords):
                H[0, 2 * i] = (x_hat[0, 0] - landmark[0]) / (
                    2 * np.linalg.norm(x_hat - landmark, 2)
                )
                H[1, 2 * i] = (x_hat[1, 0] - landmark[1]) / (
                    2 * np.linalg.norm(x_hat - landmark, 2)
                )
                H[0, 2 * i + 1] = -(x_hat[1, 0] - landmark[1]) / (
                    np.linalg.norm(x_hat - landmark, 2) ** 2
                )
                H[1, 2 * i + 1] = (x_hat[0, 0] - landmark[0]) / (
                    np.linalg.norm(x_hat - landmark, 2) ** 2
                )
            return H.T

        H = dhdx(self.state_estimate)
        denom = H * self.state_covariance * H.T + self.measurement_noise
        K = self.state_covariance * H.T * denom.I

        yhat = self.beacon_observer.get_observation(self.state_estimate)

        updated_state_estimate = self.state_estimate + K * (observations - yhat)
        updated_covariance = (np.eye(2) - K * H) * self.state_covariance

        self.state_estimate = updated_state_estimate
        self.state_covariance = updated_covariance

        return self.state_estimate


class RobotPathingPolicy(PolicyPredictor):
    """
    Policy for selecting actions to move the robot towards a sequence of goals
    in the discrete state domain.
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        goals: np.ndarray,
        goal_reached_tol: float = 0.5,
        verbose: bool = False,
        rng: np.random.Generator = None,
    ):
        """
        Initialize the policy with state domain, actions, goals, and parameters.

        Args:
            state_domain (DiscreteDomain): the discrete state space.
            actions (List[BaseAction]): list of possible actions.
            goals (np.ndarray): array of goal coordinates.
            goal_reached_tol (float): tolerance for considering a goal reached.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.state_domain = state_domain
        self.actions = actions

        assert len(goals)
        self.goals = goals.copy()
        self.goal_reached_tol = goal_reached_tol

        self.goal_idx = 0
        self.goal_distances = self.get_state_distances(self.goals[self.goal_idx])

        self.verbose = verbose

    def get_current_goal(self) -> np.ndarray:
        """
        Retrieves the coordinates of the current goal

        Returns:
            np.ndarray: current goal coordinates
        """
        return self.goals[self.goal_idx]

    def get_state_distances(self, goal: np.ndarray):
        """
        Compute distances from all states to the given goal.

        Args:
            goal (np.ndarray): goal coordinates.

        Returns:
            np.ndarray: distances from each state to the goal.
        """
        states = self.state_domain.index2values(np.arange(len(self.state_domain)))
        return np.linalg.norm(goal - states, axis=1)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Select an action based on the current observation and policy.

        Args:
            observation (np.ndarray or dict): current observation.
            state (tuple, optional): hidden state for recurrent policies.
            episode_start (np.ndarray, optional): mask for episode start.
            deterministic (bool): whether to select actions deterministically.

        Returns:
            tuple: selected action and next hidden state.
        """

        # check if arrived at current goal - if so, update goal and distances
        # the first version used to weight the distance from goal, according to the digital state distribution
        # goal_state_dist = self.goal_distances @ observation
        dstate_idx = np.argmax(observation)
        goal_state_dist = self.goal_distances[dstate_idx]
        if self.verbose:
            print(f"map distance from goal {goal_state_dist}")

        if (
            goal_state_dist < self.goal_reached_tol
            and self.goal_idx < len(self.goals) - 1
        ):
            goal_prev = self.goals[self.goal_idx]
            self.goal_idx = min(self.goal_idx + 1, len(self.goals) - 1)
            self.goal_distances = self.get_state_distances(self.goals[self.goal_idx])

            if self.verbose:
                print(f"updated goal {goal_prev} to {self.goals[self.goal_idx]}")

        # check which action gets nearest to the current goal
        action_results = [
            # self.goal_distances @ a.get_transition_probabilities() @ observation
            self.goal_distances @ a.get_transition_probabilities()[:, dstate_idx]
            for a in self.actions
        ]
        if self.verbose:
            print(f"expected actions distance results {np.array(action_results)}")

        if deterministic:
            ret = np.argmin(action_results, axis=0)
        else:
            action_results = softmax(-np.array(action_results))
            ret = self.rng.choice(len(self.actions), p=action_results)

        return ret, None


class RobotPathingPGMHelper:
    """
    Helper class for initializing the Probabilistigc Graphical Model
    for robot pathing.
    """

    def __init__(
        self,
        state_domain: DiscreteDomain,
        actions: List[BaseAction],
        n_beacons: int,
    ):
        self.state_domain = state_domain
        self.actions = actions
        self.action_domain = DiscreteDomain({"action": np.arange(len(self.actions))})

        S = Variable("S", distribution=False, discrete=True, domain=self.state_domain)

        obs_domain = np.array([[0, np.inf], [-np.pi, np.pi]] * n_beacons)
        O = Variable("O", distribution=False, discrete=False, domain=obs_domain)

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

        S2O = Dependency("observe", [(S, 0)], (O, 0))
        O2D_ip = Dependency("inverse_problem", [(O, 0)], (D_ip, 0))

        n_states = len(self.state_domain)
        smoothing = 1e-6

        confusion = np.eye(n_states)
        confusion = normalize_cpd(confusion)

        D_ip2D = Dependency("confusion", [(D_ip, 0)], (D, 0), cpd=confusion)
        D2U = Dependency("policy", [(D, 0)], (U, 0))
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
        U2U_prev = Dependency("action_observation", [(U, 0)], (U_prev, 1))
        D2Q = Dependency("qois", [(D, 0)], (Q, 0))
        SU2R = Dependency("reward", [(D, 0), (U_prev, 0)], (R, 0))

        self.dependencies = [
            S2O,
            O2D_ip,
            D_ip2D,
            D2U,
            SU_prev2S,
            U_prev2D_0,
            DU_prev2D_1,
            U2U_prev,
            D2Q,
            SU2R,
        ]

        self.dependencies_assimilation = [D_ip2D, U_prev2D_0, DU_prev2D_1]

    def init_dbn_full(self, verbose: bool = False) -> DynamicBayesianNetwork:
        ret, _ = get_dbn(self.dependencies, check_model=False, verbose=verbose)
        return ret

    def get_dbn_full_variable_pos(self) -> Dict[str, Tuple[float, float]]:
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
        Initialize a Dynamic Bayesian Network for state assimilation.
        """
        ret, _ = get_dbn(self.dependencies_assimilation, verbose=verbose)
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

        evidence = {
            ("U_prev", 1): action,
        }
        inference_keys = [
            ("D", 1),
        ]

        inference_results = dbn_inference(
            dbn_infer,
            inference_keys=inference_keys,
            evidence=evidence,
            evidence_sampled_distros={
                ("D", 0): dstate_distro,
                ("D_ip", 1): dinv_distro,
            },
            n_samples=n_samples,
            rng=rng,
            n_workers=n_workers,
        )

        digital_asset.state_distro[:] = inference_results[("D", 1)]

        return inference_results[("D", 1)]


class RobotPathingEnv(BaseDigitalTwinEnv):
    """
    Digital twin environment for robot pathing, supporting different noise and
    state update models.
    """

    def __init__(
        self,
        noise_type: str,
        state_update_type: str,
        n_obs_assimilation: int = 1,
        inv_solver_confusion_matrix: Optional[np.ndarray] = None,
    ):
        """
        Initialize the robot pathing environment.

        Args:
            noise_type (str): type of noise model ('gaussian', 'snr', etc.).
            state_update_type (str): sState update model ('deterministic', 'stochastic').
            n_obs_assimilation (int): number of observations for assimilation.
            inv_solver_confusion_matrix (np.ndarray, optional): confusion matrix of the inverse problem in the digital asset.
        """
        setup = RobotPathingSetup()

        n_states = len(setup.state_domain)

        state_update_model = None
        if state_update_type == "deterministic":
            state_update_model = ActionUpdateComponent(deterministic=True)
        elif state_update_type == "stochastic":
            state_update_model = ActionUpdateComponent(deterministic=False)

        observer = RobotPathingBeaconObserver(setup.beacons_coords)
        inv_solver = RobotPathingInverseSolverMean(setup.beacons_coords)
        if inv_solver_confusion_matrix is None:
            inv_solver_confusion_matrix = np.eye(n_states)

        noise_model = None
        if noise_type == "gaussian":
            # notice how the gaussian noise is added at the observer, as it's just easier to model their uncertainty
            observer = RobotPathingBeaconObserver(
                setup.beacons_coords,
                mean=0,
                std=1,
            )
        elif noise_type == "snr":
            noise_model = SNRGaussianNoiseComponent(100)

        physical_asset = ComposablePhysicalAsset(
            initial_state=setup.state_domain.index2values(0),
            state_update_component=state_update_model,
            sensor_component=observer,
            noise_component=noise_model,
        )

        digital_asset = InverseProblemConfusionDigitalAsset(
            state_domain=setup.state_domain,
            inv_solver=inv_solver,
            inv_solver_confusion_matrix=inv_solver_confusion_matrix,
        )

        super().__init__(
            state_domain=setup.state_domain,
            actions=setup.actions,
            physical_asset=physical_asset,
            digital_asset=digital_asset,
            reward=lambda x, y: 0,
            n_samples_assimilation=n_obs_assimilation,
        )


if "robot_pathing" not in gym.registry:
    gym.register("robot_pathing", RobotPathingEnv)


class RobotPathingPredictor:
    """
    Simulator for running policy-driven episodes in the robot pathing environment,
    collecting episode history and results.
    """

    def __init__(
        self,
        env: RobotPathingEnv,
        policy: RobotPathingPolicy,
    ):
        """
        Initialize the predictor with environment and policy.

        Args:
            env (RobotPathingEnv): the robot pathing environment.
            policy (RobotPathingPolicy): the policy to use for action selection.
        """
        self.env: RobotPathingEnv = env.unwrapped
        assert isinstance(self.env, RobotPathingEnv)

        self.policy = policy

    def simulate(
        self,
        n_iters: int,
        action_deterministic: bool = True,
        reset_options: Optional[dict] = None,
        rng: np.random.Generator = None,
        verbose: bool = False,
    ):
        """
        Run a simulation episode using the policy in the environment.

        Args:
            n_iters (int): maximum number of iterations.
            action_deterministic (bool): whether to use deterministic actions.
            reset_options (dict, optional): options for environment reset.
            rng (np.random.Generator, optional): the pseudo-random generator, or None to create a new one. Defaults to None.
            verbose (bool, optional): whether to print debug messages. Defaults to False.

        Returns:
            pd.DataFrame: dataFrame containing the episode history.
        """
        if rng is None:
            rng = np.random.default_rng()

        # region history
        cols = [  # physical state
            "physical_state",
            "digital_state_true",
            "digital_state_true_idx",
            # digital state
            "digital_state_distro",
            "digital_state_map",
            "digital_state_map_idx",
            # policy
            "action_idx",
            "action_true_idx",
            # goal at the start of the frame
            "goal_state",
            "goal_state_idx",
        ]
        episode_dfdict = {c: [] for c in cols}

        def history_append():
            # physical state
            episode_dfdict[f"physical_state"].append(info["physical_state"])

            pstate_dstate_idx = self.env._state_domain.values2index(
                info["physical_state"]
            )
            pstate_dstate = self.env._state_domain.index2values(pstate_dstate_idx)
            episode_dfdict[f"digital_state_true"].append(pstate_dstate)
            episode_dfdict[f"digital_state_true_idx"].append(pstate_dstate_idx)

            # digital state
            episode_dfdict[f"digital_state_distro"].append(obs)
            dstate_idx = np.argmax(obs)
            dstate = self.env._state_domain.index2values(dstate_idx)
            episode_dfdict[f"digital_state_map"].append(dstate)
            episode_dfdict[f"digital_state_map_idx"].append(dstate_idx)

            # policy
            episode_dfdict[f"action_idx"].append(action)

            pstate2dstate_obs = np.zeros_like(obs)
            pstate2dstate_obs[pstate_dstate_idx] = 1
            pstate2dstate_action, _ = copy.deepcopy(self.policy).predict(
                pstate2dstate_obs, deterministic=action_deterministic
            )
            episode_dfdict[f"action_true_idx"].append(pstate2dstate_action)

            # goal at the start of the frame
            goal_state = self.policy.get_current_goal()
            episode_dfdict[f"goal_state"].append(goal_state)
            episode_dfdict[f"goal_state_idx"].append(
                self.env._state_domain.values2index(goal_state)
            )

            if verbose:
                print(f'frame {len(episode_dfdict[f"physical_state"])-1}')

                print(f'    physical_state {episode_dfdict["physical_state"][-1]}')
                print(
                    f'    physical_state to digital {episode_dfdict["digital_state_true"][-1]}'
                )
                print(
                    f'    digital_state map {episode_dfdict["digital_state_map"][-1]}'
                )
                print(f'    goal {episode_dfdict["goal_state"][-1]}')
                print(
                    f'    digital_state_distro action {episode_dfdict["action_idx"][-1]}'
                )
                print(
                    f'    physical_state to digital action {episode_dfdict["action_true_idx"][-1]}'
                )

        # endregion

        if verbose:
            print(f"env reset")

        # frame 0
        obs, info = self.env.reset(seed=int(rng.integers(2**32)), options=reset_options)

        action, _ = self.policy.predict(obs, deterministic=action_deterministic)

        history_append()

        for i in range(n_iters):
            # physical state update and assimilation
            obs, _, terminated, truncated, info = self.env.step(action)

            # policy invocation
            action, _ = self.policy.predict(obs, deterministic=action_deterministic)

            history_append()

            if terminated or truncated:
                break

        # region history
        # expand physical_state columns
        pstate_cols = [
            f"physical_state[{d}]" for d in self.env._state_domain.var2values
        ]
        pstate_vals = episode_dfdict.pop("physical_state")

        # expand digital_state_distro columns
        dstate_distro_cols = [
            f"digital_state_distro[{self.env._state_domain.index2values(i)}]"
            for i in range(len(self.env._state_domain))
        ]
        dstate_distro_vals = episode_dfdict.pop("digital_state_distro")

        episode_df = pd.DataFrame(episode_dfdict)

        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        episode_df[pstate_cols] = np.array(pstate_vals)
        episode_df[dstate_distro_cols] = np.array(dstate_distro_vals)
        # endregion

        return episode_df


def plot_map(state_distro: np.ndarray, setup: RobotPathingSetup):
    plt.imshow(
        state_distro.reshape(setup.n_world_xsteps, setup.n_world_ysteps).T,
        cmap=CMAP_BLUES,
        origin="lower",
        vmin=0,
        vmax=1,
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.colorbar()
