"""
Specialized domains for the Structural Health Monitoring digital twins.
"""

import numpy as np

from pgmtwin.core.domain import DiscreteDomain


class SingleDamageDomain(DiscreteDomain):
    """
    A DiscreteDomain, but all combinations of location==0 or damage==0 get conflated
    """

    damage_location_id = "damage_location"
    damage_level_id = "damage_level"

    def __init__(self, damage_locations: np.ndarray, damage_levels: np.ndarray):
        """
        Initializes the SingleDamageDomain with the given damage locations and levels.

        Args:
            damage_locations (np.ndarray): array of damage locations, the first corresponds to the undamaged state
            damage_levels (np.ndarray): array of damage levels, the first corresponds to the undamaged state
        """
        self.damage_locations = damage_locations.copy()
        self.damage_levels = damage_levels.copy()

        super().__init__(
            {
                SingleDamageDomain.damage_location_id: self.damage_locations,
                SingleDamageDomain.damage_level_id: self.damage_levels,
            }
        )

    def __len__(self):
        return 1 + (self.damage_locations.size - 1) * (self.damage_levels.size - 1)

    def index2multi_index(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        ret_squeeze = indices.ndim < 1
        if ret_squeeze:
            indices = np.atleast_1d(indices)

        ret = np.empty((len(indices), len(self.var2values)), dtype=int)
        for i, idx in enumerate(indices):
            if not idx:
                ret[i, :] = 0
            else:
                ret[i, :] = 1 + np.array(
                    np.unravel_index(idx - 1, np.array(self.shape) - 1)
                )

        return ret.squeeze(0) if ret_squeeze else ret

    def multi_index2index(self, m_indices: np.ndarray) -> np.ndarray:
        m_indices = np.asarray(m_indices)
        ret_squeeze = m_indices.ndim < 2
        if ret_squeeze:
            m_indices = m_indices.reshape(1, -1)

        ret = np.empty(len(m_indices), dtype=int)
        for i, idxs in enumerate(m_indices):
            if np.any(np.isclose(idxs, 0)):
                ret[i] = 0
            else:
                ret[i] = 1 + np.ravel_multi_index(idxs - 1, np.array(self.shape) - 1)

        return ret.squeeze(0) if ret_squeeze else ret
