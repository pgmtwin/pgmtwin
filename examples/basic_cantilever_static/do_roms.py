import os
import glob
import argparse
import pickle
import warnings
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn
import sklearn.gaussian_process
import sklearn.model_selection
from sklearn.exceptions import ConvergenceWarning

import pyvista as pv

from pgmtwin.core.utils import get_number_from_text

from common import BasicCantileverSetup, BasicCantileverROM

if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    parser = argparse.ArgumentParser(
        description="Perform analysis and implementation of the surrogate model for sensors measurements"
    )

    parser.add_argument("--roms-dir", type=str, required=True, help="roms db directory")
    parser.add_argument(
        "--db-dir", type=str, required=True, help="simulations db directory"
    )
    parser.add_argument(
        "--truncation-rank",
        type=int,
        default=-1,
        required=False,
        help="truncation rank of sensors modes",
    )

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    database_dir = args.db_dir
    sensors_truncation_rank = args.truncation_rank
    # endregion

    with open(os.path.join(roms_dir, "config.yaml"), "r") as fin:
        config_data = yaml.safe_load(fin)

    sensor_type = config_data["sensor_type"]
    sensor_node_selector = np.array(config_data["sensor_nodes"])
    sensor_raw_selector = np.array(config_data["sensor_raw"])
    n_sensors = len(sensor_node_selector)

    os.makedirs(roms_dir, exist_ok=True)

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

    grid = pv.read(os.path.join(database_dir, "0", "results.vtk"))

    coords = grid.points
    print(f"grid n_dofs {len(coords)}")

    # snapshot matrix (n_samples, n_dofs, n_comps, )
    data_raw = np.array(
        [np.load(os.path.join(d, f"{sensor_type}.npy")) for d in snapshot_dirs]
    )
    n_samples, n_dofs, n_comps = data_raw.shape
    print(f"loaded n_samples {n_samples} n_dofs {n_dofs} n_comps {n_comps}")
    assert n_samples == len(snapshot_dirs)
    assert n_dofs == len(coords)
    assert n_comps == 3 if sensor_type == "displacement" else 6

    # snapshot matrix (n_dofs, n_comps, n_samples,)
    data_raw = data_raw.transpose([1, 2, 0])

    # hifi_sensors analysis
    data_sensors = data_raw[sensor_raw_selector[0], sensor_raw_selector[1], :]
    data_sensors = data_sensors.reshape(-1, len(snapshot_dirs))

    np.save(os.path.join(roms_dir, "data_sensors.npy"), data_sensors)

    U_sensors, s_sensors, VT_sensors = np.linalg.svd(data_sensors, full_matrices=False)
    coeffs_sensors = (np.diag(s_sensors) @ VT_sensors).T

    np.save(os.path.join(roms_dir, "modes_sensors.npy"), U_sensors)
    np.save(os.path.join(roms_dir, "singular_values_sensors.npy"), s_sensors)
    np.save(os.path.join(roms_dir, "V_sensors.npy"), VT_sensors.T)
    np.save(os.path.join(roms_dir, "coeffs_sensors.npy"), coeffs_sensors)

    plt.semilogy(s_sensors / s_sensors[0])
    plt.grid()

    plt.savefig(os.path.join(roms_dir, "singular_values_sensors.png"))
    plt.close()

    # rom_sensors analysis
    if sensors_truncation_rank < 1:
        sensors_truncation_rank = len(s_sensors)
    else:
        sensors_truncation_rank = min(sensors_truncation_rank, len(s_sensors))

    U_sensors_reduced = U_sensors[:, :sensors_truncation_rank]
    s_sensors_reduced = s_sensors[:sensors_truncation_rank]
    VT_sensors_reduced = VT_sensors[:sensors_truncation_rank]
    coeffs_sensors_reduced = (np.diag(s_sensors_reduced) @ VT_sensors_reduced).T

    np.save(os.path.join(roms_dir, "modes_sensors_reduced.npy"), U_sensors_reduced)
    np.save(
        os.path.join(roms_dir, "singular_values_sensors_reduced.npy"), s_sensors_reduced
    )
    np.save(os.path.join(roms_dir, "V_sensors_reduced.npy"), VT_sensors_reduced.T)
    np.save(
        os.path.join(roms_dir, "coeffs_sensors_reduced.npy"), coeffs_sensors_reduced
    )

    # train rom for sensors
    # dict of GPs
    dmg_loc2coeffs_pred = {}
    for dmg_loc in setup.state_domain.damage_locations:
        sample_mask = params_df["damage_location"].astype(int) == dmg_loc
        sample_selector = np.where(sample_mask)[0]
        assert len(sample_selector)
        print(f"train dmg_loc {dmg_loc} on {len(sample_selector)} samples")

        X = params_df.loc[
            sample_mask, ["youngs_modulus_damage", "forcing_pressure"]
        ].values
        Y = coeffs_sensors_reduced[sample_mask]
        (X_train, X_test, Y_train, Y_test, data_train, data_test) = (
            sklearn.model_selection.train_test_split(
                X, Y, data_sensors.T[sample_selector], test_size=0.2
            )
        )

        assert len(X_train)
        assert len(X_test)
        assert len(Y_train)
        assert len(Y_test)

        print(f"    train size {X_train.shape} test size {X_test.shape}")

        kernel = (
            sklearn.gaussian_process.kernels.RBF(length_scale=np.ones(X.shape[-1]))
            # * sklearn.gaussian_process.kernels.ConstantKernel()
            # + sklearn.gaussian_process.kernels.WhiteKernel(1e-2)
        )
        pred = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=100,
            normalize_y=False,
            copy_X_train=True,
            n_targets=Y.shape[-1],
            random_state=None,
        )

        # latent space errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pred.fit(X_train, Y_train)

        Y_train_pred = pred.predict(X_train)
        Y_test_pred = pred.predict(X_test)

        Y_train_norm = np.linalg.norm(Y_train, axis=1)
        Y_test_norm = np.linalg.norm(Y_test, axis=1)

        err_train_norm = np.linalg.norm(Y_train - Y_train_pred, axis=1)
        err_test_norm = np.linalg.norm(Y_test - Y_test_pred, axis=1)

        print(f"    coeffs space")
        print(f"    err_train norm mean {np.mean(err_train_norm)}")
        print(f"    rel err_train norm mean {np.mean(err_train_norm / Y_train_norm)}")

        print(f"    err_test norm mean {np.mean(err_test_norm)}")
        print(f"    rel err_test norm mean {np.mean(err_test_norm / Y_test_norm)}")

        # sensors space errors
        data_train_pred = Y_train_pred @ U_sensors_reduced.T
        data_test_pred = Y_test_pred @ U_sensors_reduced.T

        data_train_norm = np.linalg.norm(data_train, axis=1)
        data_test_norm = np.linalg.norm(data_test, axis=1)

        err_train_norm = np.linalg.norm(data_train - data_train_pred, axis=1)
        err_test_norm = np.linalg.norm(data_test - data_test_pred, axis=1)

        print(f"    sensors space")
        print(f"    err_train norm mean {np.mean(err_train_norm)}")
        print(
            f"    rel err_train norm mean {np.mean(err_train_norm / data_train_norm)}"
        )

        print(f"    err_test norm mean {np.mean(err_test_norm)}")
        print(f"    rel err_test norm mean {np.mean(err_test_norm / data_test_norm)}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pred.fit(X, Y)

        dmg_loc2coeffs_pred[dmg_loc] = pred

        os.makedirs(os.path.join(roms_dir, "figures"), exist_ok=True)
        for i in range(n_sensors):
            for j in range(n_comps):
                plt.scatter(
                    X[:, 0],
                    X[:, 1],
                    c=data_sensors[n_comps * i + j, sample_selector],
                    cmap="viridis",
                    edgecolors="k",
                    s=100,
                )
                plt.colorbar()

                plt.xlabel("youngs_modulus_damage")
                plt.ylabel("forcing_pressure")

                plt.title(f"{dmg_loc} sensor[{i}, {j}]")

                plt.savefig(
                    os.path.join(
                        roms_dir, "figures", f"{dmg_loc}_sensor_{i}_comp_{j}.png"
                    )
                )

                plt.close()

    rom = BasicCantileverROM()
    rom.sensor_type = sensor_type
    rom.dmg_loc2coeffs_pred = dmg_loc2coeffs_pred
    rom.modes = U_sensors_reduced

    rom_fname = os.path.join(roms_dir, "rom_sensors.pkl")
    with open(rom_fname, "wb") as fout:
        pickle.dump(rom, fout)

    print(f"saved ROM to {rom_fname}")
