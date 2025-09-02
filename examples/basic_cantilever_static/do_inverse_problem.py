import glob
import os
import argparse
import pickle
import warnings
import yaml

import numpy as np
import pandas as pd

import sklearn

import sklearn.gaussian_process
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.exceptions import ConvergenceWarning

import xgboost as xgb

from pgmtwin.core.utils import get_number_from_text

from common import (
    BasicCantileverSetup,
    BasicCantileverInverseSolver,
)

if __name__ == "__main__":
    setup = BasicCantileverSetup()

    # region argparse
    noise_types = ["", "gaussian", "snr"]
    classifier_models = ["xgboost", "random-forest", "decision-tree"]
    regressor_models = ["gaussian-process"]

    parser = argparse.ArgumentParser(
        description="Perform analysis and implementation of the inverse problem"
    )

    parser.add_argument(
        "--assimilation-dir", type=str, required=True, help="assimilation directory"
    )
    parser.add_argument(
        "--db-dir", type=str, required=True, help="simulations db directory"
    )
    parser.add_argument("--roms-dir", type=str, required=True, help="roms directory")
    parser.add_argument(
        "--dmg-loc-model",
        choices=classifier_models,
        default=classifier_models[0],
        help="type of classifier for the damage location",
    )
    parser.add_argument(
        "--dmg-lvl-model",
        choices=classifier_models + regressor_models,
        default=classifier_models[0],
        help="type of model for the damage level",
    )
    # parser.add_argument(
    #     "--noise-snr",
    #     default=100,
    #     help="signal-to-noise ratio",
    # )
    parser.add_argument(
        "--noise-type",
        choices=noise_types,
        default=noise_types[0],
        help="noise model",
    )

    args = parser.parse_args()
    print(args)

    assimilation_dir = args.assimilation_dir
    database_dir = args.db_dir
    roms_dir = args.roms_dir

    dmg_loc_model = args.dmg_loc_model
    dmg_lvl_model = args.dmg_lvl_model

    # noise_snr = args.noise_snr
    noise_type = args.noise_type
    # endregion

    os.makedirs(assimilation_dir, exist_ok=True)

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

    # classifier from sensors to dmg_loc
    data_sensors = np.concatenate(
        [
            np.load(os.path.join(d, f"{sensor_type}.npy")).reshape(1, -1)
            for d in snapshot_dirs
        ],
        axis=0,
    )

    noiser = setup.make_noise_component(noise_type)
    data_sensors = noiser.apply_noise(None, data_sensors)

    X = data_sensors
    Y = params_df["damage_location"].astype(int).values

    (X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2
    )

    assert len(X_train)
    assert len(X_test)
    assert len(Y_train)
    assert len(Y_test)

    print(f"train dmg_loc classification on {len(X)} samples")
    print(f"    train size {X_train.shape} test size {X_test.shape}")

    dmg_loc_handler = None
    if dmg_loc_model == "xgboost":
        dmg_loc_handler = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=50,
            max_leaves=0,
            # max_bin
            # grow_policy
            # learning_rate=.1,
            tree_method="hist",  # exact, approx and hist
            n_jobs=int(os.environ["OMP_NUM_THREADS"]),
            # gamma
            # min_child_weight=50,
            # max_delta_step
            # subsample
            # sampling_method
            colsample_bytree=0.8,
            # colsample_bylevel=.8,
            colsample_bynode=0.2,
            # reg_alpha
            # reg_lambda
            # scale_pos_weight
            base_score=0.5,
        )
    elif dmg_loc_model == "random-forest":
        dmg_loc_handler = sklearn.ensemble.RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            max_depth=50,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=int(os.environ["OMP_NUM_THREADS"]),
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
        )
    elif dmg_loc_model == "decision-tree":
        dmg_loc_handler = sklearn.tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
            monotonic_cst=None,
        )
    else:
        raise ValueError(f"unsupported model {dmg_loc_model} for damage location")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        dmg_loc_handler.fit(X_train, Y_train)

    Y_train_pred = dmg_loc_handler.predict(X_train)
    Y_test_pred = dmg_loc_handler.predict(X_test)

    accuracy_train = sklearn.metrics.accuracy_score(Y_train, Y_train_pred)
    accuracy_test = sklearn.metrics.accuracy_score(Y_test, Y_test_pred)

    print(f"    accuracy_train {accuracy_train}")
    print(f"    accuracy_test {accuracy_test}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        dmg_loc_handler.fit(X, Y)

    # classifier/regressor from sensors, dmg_loc to dmg
    dmg_loc2dmg_handler = {}
    dmg_loc2label_encoder = {}
    for dmg_loc in setup.state_domain.damage_locations:
        sample_mask = params_df["damage_location"].astype(int) == dmg_loc
        sample_selector = np.where(sample_mask)[0]
        assert len(sample_selector)
        print(
            f"train dmg_lvl classification for dmg_loc {dmg_loc} on {len(sample_selector)} samples"
        )

        X = data_sensors[sample_mask]

        if dmg_lvl_model in regressor_models:
            Y = params_df.loc[sample_mask, ["youngs_modulus_damage"]].values
        elif dmg_lvl_model in classifier_models:
            Y = params_df.loc[
                sample_mask, ["damage_location", "youngs_modulus_damage"]
            ].values
            Y = setup.state_domain.values2multi_index(Y)[:, 1]
        else:
            raise ValueError(f"unsupported model {dmg_lvl_model} for damage level")

        (X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(
            X, Y, test_size=0.2
        )

        assert len(X_train)
        assert len(X_test)
        assert len(Y_train)
        assert len(Y_test)

        print(f"    train size {X_train.shape} test size {X_test.shape}")

        label_encoder = None

        dmg_lvl_handler = None
        if dmg_lvl_model == "gaussian-process":
            kernel = (
                sklearn.gaussian_process.kernels.RBF(length_scale=np.ones(X.shape[-1]))
                #   * sklearn.gaussian_process.kernels.ConstantKernel() +
                #   sklearn.gaussian_process.kernels.WhiteKernel()
            )
            dmg_lvl_handler = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=100,
                normalize_y=True,
                copy_X_train=True,
                n_targets=Y.shape[-1],
                random_state=None,
            )
        elif dmg_lvl_model == "xgboost":
            dmg_lvl_handler = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=50,
                max_leaves=0,
                # max_bin
                # grow_policy
                # learning_rate=.1,
                tree_method="hist",  # exact, approx and hist
                n_jobs=int(os.environ["OMP_NUM_THREADS"]),
                # gamma
                # min_child_weight=50,
                # max_delta_step
                # subsample
                # sampling_method
                colsample_bytree=0.8,
                # colsample_bylevel=.8,
                colsample_bynode=0.2,
                # reg_alpha
                # reg_lambda
                # scale_pos_weight
                base_score=0.5,
            )

            label_encoder = sklearn.preprocessing.LabelEncoder()
            label_encoder.fit(Y)
        elif dmg_lvl_model == "random-forest":
            dmg_lvl_handler = sklearn.ensemble.RandomForestClassifier(
                n_estimators=2000,
                criterion="entropy",
                max_depth=50,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="sqrt",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=int(os.environ["OMP_NUM_THREADS"]),
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
                monotonic_cst=None,
            )
        elif dmg_lvl_model == "decision-tree":
            dmg_lvl_handler = sklearn.tree.DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight=None,
                ccp_alpha=0.0,
                monotonic_cst=None,
            )
        else:
            raise ValueError(f"unsupported model {dmg_lvl_model} for damage level")

        if label_encoder:
            Y_train = label_encoder.transform(Y_train)
            Y_test = label_encoder.transform(Y_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            dmg_lvl_handler.fit(X_train, Y_train)

        Y_train_pred = dmg_lvl_handler.predict(X_train)
        Y_test_pred = dmg_lvl_handler.predict(X_test)

        if dmg_lvl_model in regressor_models:
            print(f"    dmg level regression")

            r2_train = sklearn.metrics.r2_score(Y_train, Y_train_pred)
            r2_test = sklearn.metrics.r2_score(Y_test, Y_test_pred)
            print(f"    r2_train {r2_train}")
            print(f"    r2_test {r2_test}")

            # from regressed values to labels
            shape = Y_train.shape
            locs = np.full(len(Y_train_pred), fill_value=dmg_loc)
            values = np.stack((locs, Y_train_pred.flatten()), axis=1)
            Y_train_pred = setup.state_domain.values2multi_index(values)[:, 1]
            Y_train_pred = Y_train_pred.reshape(shape)

            values = np.stack((locs, Y_train.flatten()), axis=1)
            Y_train = setup.state_domain.values2multi_index(values)[:, 1]
            Y_train = Y_train.reshape(shape)

            shape = Y_test.shape
            locs = np.full(len(Y_test_pred), fill_value=dmg_loc)
            values = np.stack((locs, Y_test_pred.flatten()), axis=1)
            Y_test_pred = setup.state_domain.values2multi_index(values)[:, 1]
            Y_test_pred = Y_test_pred.reshape(shape)

            values = np.stack((locs, Y_test.flatten()), axis=1)
            Y_test = setup.state_domain.values2multi_index(values)[:, 1]
            Y_test = Y_test.reshape(shape)

        print(f"    dmg level accuracy")
        accuracy_train = sklearn.metrics.accuracy_score(Y_train, Y_train_pred)
        accuracy_test = sklearn.metrics.accuracy_score(Y_test, Y_test_pred)

        print(f"    accuracy_train {accuracy_train}")
        print(f"    accuracy_test {accuracy_test}")

        if label_encoder:
            Y = label_encoder.transform(Y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            dmg_lvl_handler.fit(X, Y)

        dmg_loc2dmg_handler[dmg_loc] = dmg_lvl_handler
        dmg_loc2label_encoder[dmg_loc] = label_encoder

    inv_solver = BasicCantileverInverseSolver()
    inv_solver.dmg_loc_classifier = dmg_loc_handler
    inv_solver.dmg_loc2dmg_handler = dmg_loc2dmg_handler
    inv_solver.dmg_loc2label_encoder = dmg_loc2label_encoder

    inv_solver_fname = os.path.join(assimilation_dir, "inv_solver.pkl")
    with open(inv_solver_fname, "wb") as fout:
        pickle.dump(inv_solver, fout)

    print(f"saved inv_solver to {inv_solver_fname}")
