import glob
import os
import argparse
import pickle
import yaml

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt

from pgmtwin.core.utils import (
    get_number_from_text,
    plot_discrete_domain_confusion_matrix,
)

from common import (
    BasicCantileverSetup,
    BasicCantileverInverseSolver,
)

if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    noise_types = ["", "gaussian", "snr"]

    parser = argparse.ArgumentParser(
        description="Compute confusion matrix for the inverse problem"
    )

    parser.add_argument(
        "--assimilation-dir", type=str, required=True, help="inverse problem directory"
    )
    parser.add_argument(
        "--db-dir", type=str, required=True, help="simulations db directory"
    )
    parser.add_argument("--roms-dir", type=str, required=True, help="roms db directory")
    parser.add_argument(
        "--noise-type",
        choices=noise_types,
        default=noise_types[0],
        help="noise model",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="number of repeated noisy samples for each original snapshot",
    )

    args = parser.parse_args()
    print(args)

    assimilation_dir = args.assimilation_dir
    database_dir = args.db_dir
    roms_dir = args.roms_dir

    noise_type = args.noise_type
    n_repeats = args.n_repeats
    # endregion

    snapshot_dirs = list(glob.glob(os.path.join(database_dir, "*/")))
    snapshot_dirs.sort(key=get_number_from_text)

    params_df = pd.concat(
        [
            pd.read_csv(os.path.join(d, "parameters.csv"), index_col=None)
            for d in snapshot_dirs
        ],
        axis=0,
        ignore_index=True,
    )
    print(f"params_df shape {params_df.shape}")

    with open(os.path.join(roms_dir, "config.yaml"), "r") as fin:
        config_data = yaml.safe_load(fin)

    sensor_type = config_data["sensor_type"]
    sensor_node_selector = np.array(config_data["sensor_nodes"])
    sensor_raw_selector = np.array(config_data["sensor_raw"])
    n_sensors = len(sensor_node_selector)

    data_sensors = np.load(os.path.join(roms_dir, "data_sensors.npy")).T
    data_sensors = np.repeat(data_sensors, n_repeats, axis=0)

    print(f"data_sensors shape {data_sensors.shape}")

    # noiser = SNRGaussianNoiseComponent(noise_snr)
    noiser = setup.make_noise_component(noise_type)
    data_sensors = noiser.apply_noise(None, data_sensors)

    inv_solver_fname = os.path.join(assimilation_dir, "inv_solver.pkl")
    with open(inv_solver_fname, "rb") as fin:
        inv_solver: BasicCantileverInverseSolver = pickle.load(fin)

    X = data_sensors

    Y_true = params_df[["damage_location", "youngs_modulus_damage"]].values
    Y_true_idxs = setup.state_domain.values2index(Y_true)
    Y_true_idxs = np.repeat(Y_true_idxs, n_repeats, axis=0)

    Y_pred = np.empty_like(Y_true)
    Y_pred = np.array([inv_solver(x) for x in X])
    Y_pred_idxs = setup.state_domain.values2index(Y_pred)

    n_elems = len(setup.state_domain)
    confusion_matrix = np.zeros((n_elems, n_elems))
    for i, j in zip(Y_true_idxs, Y_pred_idxs):
        confusion_matrix[i, j] += 1

    np.save(os.path.join(assimilation_dir, "conf_matrix.npy"), confusion_matrix)

    print(
        f"assimilation accuracy {np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)}"
    )

    plot_discrete_domain_confusion_matrix(setup.state_domain, confusion_matrix)

    plt.savefig(os.path.join(assimilation_dir, "conf_matrix.png"))
    plt.savefig(os.path.join(assimilation_dir, "conf_matrix.pdf"))

    plt.close()
