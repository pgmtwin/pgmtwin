import os
import pickle
import re
import argparse
import shutil

import numpy as np
import pandas as pd

from common import BasicCantileverROM


def simulate_rom(
    output_dir: str,
    rom: BasicCantileverROM,
    damage_location: int = 0,
    youngs_modulus_damage: float = 0.0,
    forcing_pressure: float = 20 * 1e3,
):
    damage_location = int(damage_location)

    print(
        f"simulate (ROM) damage location {damage_location} with youngs_modulus_damage {youngs_modulus_damage} and forcing_pressure {forcing_pressure:.2e}"
    )

    ret = rom(damage_location, youngs_modulus_damage, forcing_pressure)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, rom.sensor_type + ".npy"), ret)

    return ret


if __name__ == "__main__":
    # region argparse
    parser = argparse.ArgumentParser(
        description="Perform the simulations from a given csv of parametric samples"
    )

    parser.add_argument("--db-dir", type=str, required=True, help="output directory")
    parser.add_argument("--roms-dir", type=str, required=True, help="roms directory")
    parser.add_argument(
        "--params-csv",
        type=str,
        default=None,
        required=False,
        help="simulations parameters",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to delete and recreate an existing simulation",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="whether to delete simulation directories outside the given parameter set",
    )

    args = parser.parse_args()
    print(args)

    database_dir = args.db_dir
    roms_dir = args.roms_dir

    params_csv = args.params_csv

    overwrite = args.overwrite
    prune = args.prune
    # endregion

    os.makedirs(database_dir, exist_ok=True)

    rom_fname = os.path.join(roms_dir, "rom_sensors.pkl")
    with open(rom_fname, "rb") as fin:
        rom = pickle.load(fin)

    params_csv_default = os.path.join(database_dir, "parameters.csv")
    if params_csv is None:
        params_csv = params_csv_default

    params_df = pd.read_csv(params_csv, index_col=None)
    if os.path.abspath(params_csv) != os.path.abspath(params_csv_default):
        params_df.to_csv(params_csv_default, index=False)
    print(f"params_df shape {params_df.shape}")

    simulation_dirs = set(
        d
        for d in os.listdir(database_dir)
        if os.path.isdir(os.path.join(database_dir, d)) and re.fullmatch(r"\d+", d)
    )

    for idx, params_dict in enumerate(params_df.to_dict(orient="records")):
        print(idx, params_dict)
        output_dir = os.path.join(database_dir, str(idx))

        if overwrite:
            shutil.rmtree(output_dir, ignore_errors=True)

        if not os.path.isdir(output_dir):
            simulate_rom(output_dir, rom, **params_dict)

        params_simulation_df = pd.DataFrame.from_records(params_dict, index=[0])
        fname = os.path.join(output_dir, "parameters.csv")
        params_simulation_df.to_csv(fname, index=False)

        simulation_dirs.discard(str(idx))

    if prune:
        for d in simulation_dirs:
            dname = os.path.join(database_dir, d)
            print(f"pruning directory {dname}")
            shutil.rmtree(dname)
