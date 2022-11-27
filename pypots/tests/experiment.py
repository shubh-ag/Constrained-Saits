import warnings

warnings.filterwarnings('ignore')

import pickle
from tqdm import tqdm
import numpy as np

from pypots.imputation import (
    SAITS,
    BRITS,
    LOCF
)
from pypots.data import mcar_sample_all, mcar_sample_feature
from pypots.utils.metrics import cal_mae, cal_mre, cal_rmse
from unified_data_for_test import finance_data


def run_saits(EPOCHS, DATA):
    train_X = DATA["train_X"]
    val_X = DATA["val_X"]
    test_X = DATA["test_X"]
    test_X_intact = DATA["test_X_intact"]
    test_X_indicating_mask = DATA["test_X_indicating_mask"]
    # print("Running test cases for SAITS...")
    saits = SAITS(
        DATA["n_steps"],
        DATA["n_features"],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        epochs=EPOCHS,
    )
    saits.fit(train_X, val_X)

    imputed_X = saits.impute(test_X)
    assert not np.isnan(
        imputed_X
    ).any(), "Output still has missing values after running impute()."
    test_MAE = cal_mae(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"SAITS test_MAE: {test_MAE}")
    test_rmse = cal_rmse(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"SAITS test_MAE: {test_rmse}")
    test_MRE = cal_mre(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"SAITS test_MAE: {test_MRE}")

    return [test_MAE, test_rmse, test_MRE]


def run_brits(EPOCHS, DATA):
    train_X = DATA["train_X"]
    val_X = DATA["val_X"]
    test_X = DATA["test_X"]
    test_X_intact = DATA["test_X_intact"]
    test_X_indicating_mask = DATA["test_X_indicating_mask"]
    # print("Running test cases for BRITS...")
    brits = BRITS(DATA["n_steps"], DATA["n_features"], 256, epochs=EPOCHS)
    brits.fit(train_X, val_X)

    imputed_X = brits.impute(test_X)
    assert not np.isnan(
        imputed_X
    ).any(), "Output still has missing values after running impute()."
    test_MAE = cal_mae(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"BRITS test_MAE: {test_MAE}")
    test_rmse = cal_rmse(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"BRITS test_MAE: {test_rmse}")
    test_MRE = cal_mre(imputed_X, test_X_intact, test_X_indicating_mask)
    # print(f"BRITS test_MAE: {test_MRE}")

    return [test_MAE, test_rmse, test_MRE]


def run_locf(EPOCHS, DATA):
    train_X = DATA["train_X"]
    val_X = DATA["val_X"]
    test_X = DATA["test_X"]
    test_X_intact = DATA["test_X_intact"]
    test_X_indicating_mask = DATA["test_X_indicating_mask"]
    locf = LOCF(nan=0)

    test_X_imputed = locf.impute(test_X)
    assert not np.isnan(
        test_X_imputed
    ).any(), "Output still has missing values after running impute()."
    test_MAE = cal_mae(test_X_imputed, test_X_intact, test_X_indicating_mask)
    test_rmse = cal_rmse(test_X_imputed, test_X_intact, test_X_indicating_mask)
    test_MRE = cal_mre(test_X_imputed, test_X_intact, test_X_indicating_mask)

    return [test_MAE, test_rmse, test_MRE]


def run(func, feature_index, rate):
    all_results_saits = []
    all_results_brits = []
    all_results_locf = []
    for _ in tqdm(range(2)):
        data = finance_data(func, feature_index, rate)
        all_results_saits.append(run_saits(5, data.copy()))
        all_results_brits.append(run_brits(5, data.copy()))
        all_results_locf.append(run_locf(5, data.copy()))

    return all_results_saits, all_results_brits, all_results_locf


mcar_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

feature_indices = [(mcar_sample_feature, 0),
                   (mcar_sample_feature, 1),
                   (mcar_sample_feature, 2),
                   (mcar_sample_feature, 3),
                   (mcar_sample_feature, 4),
                   (mcar_sample_all, 'all')]

for rate in tqdm(mcar_rates):

    f = mcar_sample_feature
    for i in tqdm(feature_indices):
        results = {}
        f = i[0]
        index = i[1]
        all_results_saits, all_results_brits, all_results_locf = run(f, index, rate)

        results['SAITS'] = np.mean(all_results_saits, axis=0)
        results['BRITS'] = np.mean(all_results_brits, axis=0)
        results['LOCF'] = np.mean(all_results_locf, axis=0)

        with open('experiment_' + str(index) + '_' + str(rate) + '.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
