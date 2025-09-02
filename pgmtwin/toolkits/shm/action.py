"""
Specialized action for the Structural Health Monitoring digital twins.
"""

from typing import Optional

import numpy as np

from pgmtwin.core.action import ActionCPD
from pgmtwin.core.distributions.base import BayesModel

from .domain import SingleDamageDomain


class MaintenanceAction(ActionCPD):
    """
    Simple degradation/repair stochastic transition for a SingleDamageDomain
    Can only handle a domain with dimensions (n_damage_locations, n_damage_levels)
    Holds a distribution to generate damage level transition probabilities
    Able to recompute the transitions, and update the relevant tables, as data become available
    """

    def __init__(
        self,
        name: str,
        state_domain: SingleDamageDomain,
        transition_prior: BayesModel,
        degradation: bool,
        dmg_lvl_jump_probability: np.ndarray,
        risk_aversion: Optional[float] = None,
    ):
        """
        Initializes the MaintenanceAction with the given parameters.

        Args:
            name (str): the name of the action
            state_domain (SingleDamageDomain): the domain of the action, must be a SingleDamageDomain
            transition_prior (BayesModel): the prior distribution for the transition probabilities
            degradation (bool): whether the action is a degradation (True) or a repair (False)
            dmg_lvl_jump_probability (np.ndarray): an array of probabilities for each jump in damage level
            risk_aversion (Optional[float], optional): if given, will be used as the threshold of the
                conditional value at risk used when fitting the action's transition. Defaults to None.

        Raises:
            ValueError: if the state domain does not have the expected labels or dimensions.
        """
        super().__init__(name, state_domain)

        self.transition_prior = transition_prior
        self.degradation = degradation
        self.risk_aversion = risk_aversion

        if (
            SingleDamageDomain.damage_location_id not in self.state_domain.labels
            or SingleDamageDomain.damage_level_id not in self.state_domain.labels
            or len(self.state_domain.labels) != 2
        ):
            raise ValueError(
                f"invalid domain, expected a 2d domain with labels ['{SingleDamageDomain.damage_location_id}', '{SingleDamageDomain.damage_level_id}']"
            )

        self.update_from_dmg_lvl_jumps(dmg_lvl_jump_probability)

    def update(self, transition_counts: np.ndarray):
        self.transition_prior.update(transition_counts)

        dmg_lvl_jump_probability = 1
        if not self.risk_aversion:
            dmg_lvl_jump_probability = self.transition_prior.mode
        else:
            risk_aversion = max(0, min(self.risk_aversion, 1))
            dmg_lvl_jump_probability = self.transition_prior.conditional_value_at_risk(
                risk_aversion
            )
        dmg_lvl_jump_probability = np.array(
            [1 - dmg_lvl_jump_probability, dmg_lvl_jump_probability]
        )

        self.update_from_dmg_lvl_jumps(dmg_lvl_jump_probability)

    def update_from_dmg_lvl_jumps(self, dmg_lvl_jump_probability: np.ndarray):
        """
        Compute the transition probabilities from the given damage level jump probabilities

        Args:
            dmg_lvl_jump_probability (np.ndarray): an array of probabilities for each jump in damage level
        """
        # dmg_advance_probability[i] is the probability of augmenting the dmg by i steps
        self.transition_probabilities[:] = np.eye(len(self.state_domain))

        dmg_locs = self.state_domain[SingleDamageDomain.damage_location_id]
        dmg_lvls = self.state_domain[SingleDamageDomain.damage_level_id]

        assert len(dmg_lvl_jump_probability) <= len(dmg_lvls)
        if self.degradation:
            # valorize cpd with dmg increases only

            # compile the transition from (0, 0) to every (>=1, >=1)
            self.transition_probabilities[0, 0] = dmg_lvl_jump_probability[0]

            # for every nonzero loc, copy the jump probabilities adjusting for the replication on all dmg_locs
            from_state_idx = 0
            i_dmg = 1
            for i_loc in range(1, len(dmg_locs)):
                to_state_idx = self.state_domain.multi_index2index([i_loc, i_dmg])

                range_start = to_state_idx
                range_end = to_state_idx + len(dmg_lvl_jump_probability) - 1
                self.transition_probabilities[range_start:range_end, from_state_idx] = (
                    dmg_lvl_jump_probability[1:] / (len(dmg_locs) - 1)
                )

            # now, process the degradation from every (>=1, dmg_lvl) to the same location's higher levels
            for j_lvl in range(1, len(dmg_lvls)):
                n_transitions = len(dmg_lvls) - j_lvl
                if n_transitions < len(dmg_lvl_jump_probability):
                    # all jumps above the limit are conpressed in the last one
                    dmg_lvl_jump_probability = np.concatenate(
                        (
                            dmg_lvl_jump_probability[: n_transitions - 1],
                            [np.sum(dmg_lvl_jump_probability[n_transitions - 1 :])],
                        )
                    )

                # write the transition probs for all nonzero locs
                for j_loc in range(1, len(dmg_locs)):
                    from_state_idx = self.state_domain.multi_index2index([j_loc, j_lvl])

                    range_start = from_state_idx
                    range_end = from_state_idx + len(dmg_lvl_jump_probability)
                    self.transition_probabilities[
                        range_start:range_end, from_state_idx
                    ] = dmg_lvl_jump_probability
        else:
            # valorize cpd with dmg lvl decreases only
            # similar to the previous case, but the jumps are instead written from bottom up

            # the no dmg state is terminal
            self.transition_probabilities[0, 0] = 1.0

            # for each damaged lvl, write the damage decreases
            for j_lvl in range(len(dmg_lvls) - 1, 0, -1):
                n_transitions = j_lvl + 1
                if n_transitions < len(dmg_lvl_jump_probability):
                    # all jumps under the no damage are compressed in the last one
                    dmg_lvl_jump_probability = np.concatenate(
                        (
                            dmg_lvl_jump_probability[: n_transitions - 1],
                            [np.sum(dmg_lvl_jump_probability[n_transitions - 1 :])],
                        )
                    )

                goes_to_zero = n_transitions == len(dmg_lvl_jump_probability)

                for j_loc in range(1, len(dmg_locs)):
                    from_state_idx = self.state_domain.multi_index2index([j_loc, j_lvl])

                    range_start = (
                        from_state_idx
                        + 1
                        - len(dmg_lvl_jump_probability)
                        + goes_to_zero
                    )
                    range_end = from_state_idx + 1
                    self.transition_probabilities[
                        range_start:range_end, from_state_idx
                    ] = dmg_lvl_jump_probability[::-1][goes_to_zero:]
                    if goes_to_zero:
                        self.transition_probabilities[0, from_state_idx] = (
                            dmg_lvl_jump_probability[-1]
                        )

        assert np.allclose(
            np.sum(self.transition_probabilities, axis=0), 1
        ), "computed CPD columns do not sum to 1"
