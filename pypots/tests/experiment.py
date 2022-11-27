import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from pypots.imputation import (
    SAITS,
    BRITS
)
from pypots.utils.metrics import cal_mae, cal_mre, cal_rmse
from unified_data_for_test import finance_data

mcar_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results = {
    'SAITS': {

    },
    'BRITS': {

    }
}
models = [BRITS, SAITS]


def run_saits(EPOCHS, DATA):
    train_X = DATA["train_X"]
    val_X = DATA["val_X"]
    test_X = DATA["test_X"]
    test_X_intact = DATA["test_X_intact"]
    test_X_indicating_mask = DATA["test_X_indicating_mask"]
    print("Running test cases for SAITS...")
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
    print(f"SAITS test_MAE: {test_MAE}")
    test_rmse = cal_rmse(imputed_X, test_X_intact, test_X_indicating_mask)
    print(f"SAITS test_MAE: {test_rmse}")
    test_MRE = cal_mre(imputed_X, test_X_intact, test_X_indicating_mask)
    print(f"SAITS test_MAE: {test_MRE}")

    return [test_MAE, test_rmse, test_MRE]


def run_brits(EPOCHS, DATA):
    train_X = DATA["train_X"]
    val_X = DATA["val_X"]
    test_X = DATA["test_X"]
    test_X_intact = DATA["test_X_intact"]
    test_X_indicating_mask = DATA["test_X_indicating_mask"]
    print("Running test cases for BRITS...")
    brits = BRITS(DATA["n_steps"], DATA["n_features"], 256, epochs=EPOCHS)
    brits.fit(train_X, val_X)

    imputed_X = brits.impute(test_X)
    assert not np.isnan(
        imputed_X
    ).any(), "Output still has missing values after running impute()."
    test_MAE = cal_mae(imputed_X, test_X_intact, test_X_indicating_mask)
    print(f"BRITS test_MAE: {test_MAE}")
    test_rmse = cal_rmse(imputed_X, test_X_intact, test_X_indicating_mask)
    print(f"BRITS test_MAE: {test_rmse}")
    test_MRE = cal_mre(imputed_X, test_X_intact, test_X_indicating_mask)
    print(f"BRITS test_MAE: {test_MRE}")

    return [test_MAE, test_rmse, test_MRE]


for rate in mcar_rates:
    all_results_saits = []
    all_results_brits = []
    print("Running for Rate:", rate)
    for i in range(100):
        print("Running for", i)
        data = finance_data(rate)
        all_results_saits.append(run_saits(5, data.copy()))
        all_results_brits.append(run_brits(5, data.copy()))

    results['SAITS'][str(rate)] = np.mean(all_results_saits, axis=0)
    results['BRITS'][str(rate)] = np.mean(all_results_brits, axis=0)

pd.DataFrame.from_dict(results).to_csv('results.csv')

