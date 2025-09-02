import os
import argparse

import numpy as np
import pandas as pd

from common import BasicCantileverSetup


def sample_params(n_samples: int, rng: np.random.Generator = None):
    setup = BasicCantileverSetup()

    dmg_lb, dmg_ub = setup.hifi_params_damage_level_range
    forcing_pressure_lb, forcing_pressure_ub = setup.lofi_params_pressure_range

    params_ids = ["damage_location", "youngs_modulus_damage", "forcing_pressure"]
    params_values = np.empty((n_samples, len(params_ids)))

    if rng is None:
        rng = np.random.default_rng()

    # assign state
    idxs = setup.state_domain.sample_indices(n_samples, rng=rng)
    m_idxs = setup.state_domain.index2multi_index(idxs)
    no_dmg_mask = m_idxs[:, 0] == 0

    # set dmg locs
    params_values[:, 0] = m_idxs[:, 0]
    # draw pressures from uniform
    params_values[:, 2] = forcing_pressure_lb + (
        forcing_pressure_ub - forcing_pressure_lb
    ) * rng.random(n_samples)

    # all the other: draw uniform random dmg lvl
    params_values[~no_dmg_mask, 1] = dmg_lb + (dmg_ub - dmg_lb) * rng.random(
        n_samples - np.sum(no_dmg_mask)
    )

    return pd.DataFrame(params_values, columns=params_ids)


if __name__ == "__main__":
    # region argparse
    parser = argparse.ArgumentParser(
        description="Generate a csv sampling the parametric domain"
    )

    parser.add_argument("--db-dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        required=True,
        help="number of parameter sets to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, required=False, help="random generator seed"
    )

    args = parser.parse_args()
    print(args)

    database_dir = args.db_dir
    n_samples = args.n_samples
    seed = args.seed
    # endregion

    os.makedirs(database_dir, exist_ok=True)

    params_df = sample_params(n_samples, np.random.default_rng(seed))

    fname = os.path.join(database_dir, "parameters.csv")
    params_df.to_csv(fname, index=None)
