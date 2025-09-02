from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import sklearn
import sklearn.preprocessing

import gymnasium as gym

from pgmtwin.core.action import BaseAction, ActionCPD
from pgmtwin.core.asset_components.observation_noise import (
    BaseObservationNoiseComponent,
    GaussianNoiseComponent,
    SNRGaussianNoiseComponent,
)
from pgmtwin.core.asset_components.observer import ObserverComponent
from pgmtwin.core.asset_components.state_update import (
    ActionUpdateComponent,
    StateUpdateComponent,
)
from pgmtwin.core.digital_asset import InverseProblemConfusionDigitalAsset
from pgmtwin.core.distributions.beta_bernoulli import BetaBernoulli
from pgmtwin.core.domain import DiscreteDomain
from pgmtwin.core.physical_asset import ComposablePhysicalAsset

from pgmtwin.core.utils import SingletonMeta
from pgmtwin.toolkits.shm.env import DigitalTwinEnv
from pgmtwin.toolkits.shm.action import MaintenanceAction
from pgmtwin.toolkits.shm.domain import SingleDamageDomain


# ==== utils
class BasicCantileverSetup(metaclass=SingletonMeta):
    def __init__(self):
        self.damage_locations = np.arange(3)
        self.digital_damage_levels = np.array([0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        self.hifi_params_damage_level_range = (0.25, 0.85)
        self.hifi_params_pressure_range = (60 * 1e3, 60 * 1e3)

        self.lofi_params_pressure_range = (60 * 1e3, 60 * 1e3)

        self.state_domain = SingleDamageDomain(
            self.damage_locations, self.digital_damage_levels
        )

        self.init_actions()

    def init_actions(
        self,
    ):
        actions: List[BaseAction] = []

        self.do_nothing = MaintenanceAction(
            "do_nothing",
            self.state_domain,
            transition_prior=BetaBernoulli(5, mode=0.5),
            degradation=True,
            dmg_lvl_jump_probability=np.array([0.5, 0.5]),
        )
        actions.append(self.do_nothing)

        self.minor_imperfect = MaintenanceAction(
            "minor_imperfect",
            self.state_domain,
            transition_prior=BetaBernoulli(5, mode=0.75),
            degradation=False,
            dmg_lvl_jump_probability=np.array([0.25, 0.75]),
        )
        actions.append(self.minor_imperfect)

        # a perfect repair transitions any state to undamaged
        cpd = np.zeros((len(self.state_domain), len(self.state_domain)))
        cpd[0, :] = 1

        self.perfect_repair = ActionCPD(
            "perfect_repair",
            self.state_domain,
            transition_probabilities=cpd,
        )
        actions.append(self.perfect_repair)

        self.actions = actions

        self.action_domain = DiscreteDomain({"action": np.arange(len(self.actions))})

    def make_noise_component(self, noise_type: str) -> BaseObservationNoiseComponent:
        if noise_type == "snr":
            return SNRGaussianNoiseComponent(100)
        elif noise_type == "gaussian":
            return GaussianNoiseComponent(0, 0.025)

        return BaseObservationNoiseComponent()

    @staticmethod
    def action2reward(action: BaseAction):
        ret = 0
        if action.name in ["do_nothing"]:
            ret = 10
        elif action.name in ["minor_imperfect"]:
            ret = -10
        elif action.name in ["major_imperfect"]:
            ret = -30
        elif action.name in ["perfect_repair"]:
            ret = -100

        return ret

    @staticmethod
    def state2reward(state: np.ndarray):
        dmg_loc, dmg_lvl = state.flatten()

        factor = 10

        if np.isclose(dmg_loc, 0) or dmg_lvl < 0.15:
            ret = 1
        else:
            ret = -np.exp(factor * 0.7 * dmg_lvl)

        return ret

    @staticmethod
    def reward(action: BaseAction, state: np.ndarray):
        ret = BasicCantileverSetup.action2reward(
            action
        ) + BasicCantileverSetup.state2reward(state)

        return ret


class BasicCantileverROM:
    """
    Reduced Order Model of the high-fidelity solver
    """

    def __init__(self):
        self.sensor_type: str = None
        self.modes: np.ndarray = None
        self.dmg_loc2coeffs_pred: Dict[int, sklearn.base.RegressorMixin] = None

    def __call__(self, dmg_loc: int, dmg_loc_dmg: float, pressure: float) -> np.ndarray:
        x = np.array([[dmg_loc_dmg, pressure]])
        coeffs: np.ndarray = self.dmg_loc2coeffs_pred[dmg_loc].predict(x)
        return self.modes @ coeffs.flatten()


class BasicCantileverInverseSolver:
    """
    Retrieval of the digital state from sensor observation
    """

    def __init__(self):
        self.dmg_loc_classifier: sklearn.base.ClassifierMixin = None
        self.dmg_loc2dmg_handler: Dict[
            int, Union[sklearn.base.ClassifierMixin, sklearn.base.RegressorMixin]
        ] = None
        self.dmg_loc2label_encoder: Dict[int, sklearn.preprocessing.LabelEncoder] = None

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """
        Use the most common location from the classifier
        The damage level is the average when using regressors, or the most common when using
        classifiers

        Args:
            observations (np.ndarray): observations to classify

        Returns:
            np.ndarray: digital state, with shape (2,) - [damage_location, damage_level]
        """
        observations = np.asarray(observations)
        observations = np.atleast_2d(observations)

        setup = BasicCantileverSetup()

        dmg_loc_idxs = self.dmg_loc_classifier.predict(observations).astype(int)
        # most frequent dmg_loc
        vs, cs = np.unique(dmg_loc_idxs, return_counts=True)
        dmg_loc_idx = vs[np.argmax(cs)]

        if not dmg_loc_idx:
            dmg_lvl_idx = 0
        else:
            dmg_lvls = self.dmg_loc2dmg_handler[dmg_loc_idx].predict(observations)

            # handle the case where the model needs an external LabelEncoder -> this was a classifier
            if self.dmg_loc2label_encoder[dmg_loc_idx]:
                # most frequent damage level
                vs, cs = np.unique(dmg_lvls, return_counts=True)
                dmg_lvl_idx = vs[np.argmax(cs)]
                dmg_lvl_idx = self.dmg_loc2label_encoder[dmg_loc_idx].inverse_transform(
                    [dmg_lvl_idx]
                )[0]

            if isinstance(
                self.dmg_loc2dmg_handler[dmg_loc_idx], sklearn.base.RegressorMixin
            ):
                # the dmg_lvl model is a regressor - use the mean
                dmg_lvl = np.mean(dmg_lvls)
                dmg_lvl_idx = setup.state_domain.values2multi_index(
                    [setup.state_domain.damage_locations[dmg_loc_idx], dmg_lvl]
                )[1]

        return setup.state_domain.multi_index2values([dmg_loc_idx, dmg_lvl_idx])


class BasicCantileverObserver(ObserverComponent):
    """
    Wrapper functor which complements a state with random pressure
    """

    def __init__(
        self,
        rom: BasicCantileverROM,
        pressure_range: Tuple[float, float],
    ):
        super().__init__()

        self.rom = rom
        self.pressure_range = pressure_range

    def get_observation(
        self,
        state: np.ndarray,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        p = rng.uniform(*self.pressure_range)
        return self.rom(int(state[0]), state[1], p)


class PhysicalDamageModel(StateUpdateComponent):
    def __init__(
        self,
        state_domain: SingleDamageDomain,
    ):
        self.state_domain = state_domain

        self.do_nothing_dmg_delta_mean = 0.015
        self.do_nothing_dmg_delta_std = 0.01

        self.restricted_op_dmg_delta_mean = 0.0095
        self.restricted_op_dmg_delta_std = 0.005

        self.minor_imperfect_dmg_delta_mean = 0.03
        self.minor_imperfect_dmg_delta_std = 0.006

        self.major_imperfect_dmg_delta_mean = 0.1
        self.major_imperfect_dmg_delta_std = 0.02

    def step(
        self,
        state: np.ndarray,
        action: BaseAction,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        ret = state.copy()

        undamaged = np.allclose(ret, 0)
        damage_min = self.state_domain.damage_levels[1]

        if action.name == "do_nothing":
            if undamaged:
                # .5 prob of entering a low damage level
                if rng.random() < 0.5:
                    ret[0] = rng.integers(1, len(self.state_domain.damage_locations))
                    ret[1] = rng.uniform(
                        damage_min,
                        (damage_min + self.state_domain.damage_levels[2]) / 2,
                    )
                    # print(
                    #     f"rng.uniform({damage_min},{(damage_min + self.state_domain.damage_levels[2]) / 2}) -> {ret[1]}"
                    # )
            else:
                dmg_delta = rng.normal(
                    self.do_nothing_dmg_delta_mean, self.do_nothing_dmg_delta_std
                )
                dmg_delta = max(0, dmg_delta)
                ret[1] += dmg_delta
        elif action.name == "restrict_operations":
            if undamaged:
                # .25 prob of entering a low damage level
                if rng.random() < 0.25:
                    ret[0] = rng.integers(1, len(self.state_domain.damage_locations))
                    ret[1] = rng.uniform(
                        damage_min,
                        (damage_min + self.state_domain.damage_levels[2]) / 2,
                    )
            else:
                dmg_delta = rng.normal(
                    self.restricted_op_dmg_delta_mean, self.restricted_op_dmg_delta_std
                )
                dmg_delta = max(0, dmg_delta)
                ret[1] += dmg_delta
        elif action.name == "minor_imperfect":
            if not undamaged:
                # reduce the damage level by a delta, but cannot fully heal
                dmg_delta = rng.normal(
                    self.minor_imperfect_dmg_delta_mean,
                    self.minor_imperfect_dmg_delta_std,
                )
                dmg_delta = max(0, dmg_delta)
                ret[1] = max(damage_min, ret[1] - dmg_delta)
        elif action.name == "major_imperfect":
            if not undamaged:
                # reduce the damage level by a delta, but cannot fully heal
                dmg_delta = rng.normal(
                    self.major_imperfect_dmg_delta_mean,
                    self.major_imperfect_dmg_delta_std,
                )
                dmg_delta = max(0, dmg_delta)
                ret[1] = max(damage_min, ret[1] - dmg_delta)
        elif action.name == "perfect_repair":
            ret = self.state_domain.index2values(0)

        return ret


class BasicCantileverEnv(DigitalTwinEnv):
    def __init__(
        self,
        rom: BasicCantileverROM,
        inv_solver: BasicCantileverInverseSolver,
        inv_solver_confusion_matrix: np.ndarray,
        noise_type: str,
        state_update_type: str,
        n_obs_assimilation: int = 1,
        pgm_n_samples_assimilation: int = 0,
        pgm_n_workers: int = 1,
        pgm_policy_matrix: Optional[np.ndarray] = None,
    ):
        setup = BasicCantileverSetup()

        n_states = len(setup.state_domain)

        rom_model = BasicCantileverObserver(
            rom,
            pressure_range=setup.lofi_params_pressure_range,
        )

        noise_model = setup.make_noise_component(noise_type)

        state_update_model = None
        if state_update_type == "deterministic":
            state_update_model = ActionUpdateComponent(deterministic=True)
        elif state_update_type == "stochastic":
            state_update_model = ActionUpdateComponent(deterministic=False)
        elif state_update_type == "damage_model":
            state_update_model = PhysicalDamageModel(
                setup.state_domain,
            )

        physical_asset = ComposablePhysicalAsset(
            initial_state=setup.state_domain.index2values(0),
            state_update_component=state_update_model,
            sensor_component=rom_model,
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
            reward=BasicCantileverSetup.reward,
            n_samples_assimilation=n_obs_assimilation,
            pgm_n_samples_assimilation=pgm_n_samples_assimilation,
            pgm_n_workers=pgm_n_workers,
            pgm_policy_matrix=pgm_policy_matrix,
        )

    # def _terminated(self):
    #     ret = np.all(
    #         self._physical_asset.state[1]
    #         >= self._state_domain[SingleDamageDomain.damage_level_id][-1] - 0.01
    #     )
    #     # explicit cast due to stable-baselines complaints
    #     return bool(ret)


if "basic_cantilever" not in gym.registry:
    gym.register("basic_cantilever", BasicCantileverEnv)


# ==== simulations hifi utils
def build_mesh_and_selectors(
    dim_lengths: np.ndarray,
    dim_cells: np.ndarray,
    n_dmg_domains: int,
):
    from dolfinx import mesh
    from mpi4py import MPI

    mesh_cantilever = mesh.create_box(
        MPI.COMM_WORLD,
        [np.zeros_like(dim_lengths), dim_lengths],
        dim_cells.tolist(),
        cell_type=mesh.CellType.hexahedron,
    )

    def bc_clamped_selector(x):
        return np.isclose(x[0], 0)

    def bc_forced_selector(x):
        eps = 1e-5
        aabb_lb = np.array([dim_lengths[0] - dim_lengths[1], 0, dim_lengths[2]])
        aabb_ub = np.array([dim_lengths[0], dim_lengths[1], dim_lengths[2]])

        return np.all(
            (x >= aabb_lb[:, None] - eps) & (x <= aabb_ub[:, None] + eps), axis=0
        )

    def x_selector(x, a, b):
        return (x[0] >= a) & (x[0] <= b)

    x_cuts = np.linspace(0, dim_lengths[0], n_dmg_domains + 1)
    subdomain_selectors = []
    for a, b in zip(x_cuts, x_cuts[1:]):
        subdomain_selectors.append(partial(x_selector, a=a, b=b))

    return (
        mesh_cantilever,
        [bc_clamped_selector, bc_forced_selector],
        subdomain_selectors,
    )
