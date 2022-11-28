import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pypots.data.generate_data as data_generator

def convert_to_numpy(df, key, X_cols):
    df_X = df[X_cols+[key]]

    d = {}

    for index, row in df_X.iterrows():
        if row[key] not in d:
            d[row[key]] = []
        d[row[key]].append(row[X_cols].values.squeeze())
    ans = []

    for k in sorted(d.keys()):
        ans.append(np.array(d[k]))

    return np.array(ans, dtype='float32')

def data(provider, filename):
    df = data_generator.get_data(provider, filename)
    df = convert_to_numpy(df, 'date', ['close', 'high', 'low', 'open', 'price'])
    return df
