import os
import glob
import argparse
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import scipy

import pyvista as pv

from pgmtwin.core.utils import get_number_from_text

if __name__ == "__main__":
    # region argparse
    parser = argparse.ArgumentParser(
        description="Perform analysis for the surrogate model for sensors measurements"
    )

    parser.add_argument("--roms-dir", type=str, required=True, help="roms db directory")
    parser.add_argument(
        "--db-dir", type=str, required=True, help="simulations db directory"
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        choices=["displacement", "strain"],
        default="displacement",
        required=False,
        help="type of sensor to be used",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=None,
        required=True,
        help="number of sensors that can be placed",
    )

    args = parser.parse_args()
    print(args)

    roms_dir = args.roms_dir
    database_dir = args.db_dir
    sensor_type = args.sensor_type
    n_sensors = args.n_sensors
    # endregion

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

    # sensors can be placed on the surface, or just on the bottom surface
    # also, symmetry would impose that y = 0.15
    slot_nodes_mask = np.full(len(coords), fill_value=False)

    # anywhere on surface
    # slot_nodes_mask[grid.surface_indices()] = True

    # only on bottom
    slot_nodes_mask = np.isclose(coords[:, 2], 0)

    # assume y symmetry
    slot_nodes_mask &= np.isclose(coords[:, 1], 0.15)

    n_slot_nodes = np.sum(slot_nodes_mask)
    print(f"n_node_slots {n_slot_nodes}")

    slot_nodes_selector = np.where(slot_nodes_mask)[0]

    # snapshot matrix (n_samples, n_nodes, n_comps, )
    data_raw = np.array(
        [np.load(os.path.join(d, f"{sensor_type}.npy")) for d in snapshot_dirs]
    )
    n_samples, n_nodes, n_comps = data_raw.shape
    print(f"loaded n_samples {n_samples} n_nodes {n_nodes} n_comps {n_comps}")

    assert n_samples == len(snapshot_dirs)
    assert n_nodes == len(coords)
    assert n_comps == (3 if sensor_type == "displacement" else 6)

    # snapshot matrix (n_nodes, n_comps, n_samples,)
    data_raw = data_raw.transpose([1, 2, 0])

    # hifi_full analysis, with flattened dofs
    U_raw, s_raw, VT_raw = np.linalg.svd(
        data_raw.reshape(-1, len(snapshot_dirs)), full_matrices=False
    )

    np.save(os.path.join(roms_dir, "singular_values_raw.npy"), s_raw)

    plt.semilogy(s_raw / s_raw[0])
    plt.grid()

    plt.savefig(os.path.join(roms_dir, "singular_values_raw.png"))

    plt.close()

    # hifi sensor dof analysis
    data_slot_dofs = data_raw[slot_nodes_mask].reshape(-1, len(snapshot_dirs))

    U_slot_dofs, s_slot_dofs, VT_slot_dofs = np.linalg.svd(
        data_slot_dofs, full_matrices=False
    )

    np.save(os.path.join(roms_dir, "singular_values_slot_dofs.npy"), s_slot_dofs)

    plt.semilogy(s_slot_dofs / s_slot_dofs[0])
    plt.grid()

    plt.savefig(os.path.join(roms_dir, "singular_values_slot_dofs.png"))

    plt.close()

    # choose slot nodes with QR on data_slot_dofs, then bringing each dof to its node id
    _, _, p_slot_dofs = scipy.linalg.qr(data_slot_dofs.T, pivoting=True)
    p_slot_node_dofs = np.mod(p_slot_dofs, n_slot_nodes)

    # select node sensors from the most significant dof sensors
    p_slot_nodes_selected = []
    used = set()
    for i in p_slot_node_dofs:
        if i not in used:
            p_slot_nodes_selected.append(i)
            used.add(i)
        if len(p_slot_nodes_selected) >= n_sensors:
            break

    p_slot_nodes = np.array(p_slot_nodes_selected)
    print(f"p_slot_nodes {p_slot_nodes}")

    sensor_node_selector = slot_nodes_selector[p_slot_nodes]
    print(f"sensor_node_selector {sensor_node_selector}")

    print(f"placing sensors at coords")
    print(coords[sensor_node_selector])

    sensor_dofs_selector = sensor_node_selector.reshape(-1, 1).repeat(
        3, axis=1
    ) + np.arange(n_comps)
    sensor_raw_selector = (
        sensor_node_selector.repeat(n_comps),
        np.tile(np.arange(n_comps), len(sensor_node_selector)),
    )

    config_data = {
        "sensor_type": sensor_type,
        "sensor_nodes": sensor_node_selector.tolist(),
        "sensor_dofs": sensor_dofs_selector.flatten().tolist(),
        "sensor_raw": list(x.tolist() for x in sensor_raw_selector),
    }

    with open(os.path.join(roms_dir, "config.yaml"), "w") as fout:
        yaml.dump(config_data, fout, default_flow_style=False)
