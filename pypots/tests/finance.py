# import warnings
#
# warnings.filterwarnings('ignore')
#
# import numpy as np
# import pypots.data.generate_data as data_generator
#
#
# def convert_to_numpy(df, key, X_cols):
#     df_X = df[X_cols + [key]]
#
#     d = {}
#
#     for index, row in df_X.iterrows():
#         if row[key] not in d:
#             d[row[key]] = []
#         d[row[key]].append(row[X_cols].values.squeeze())
#     ans = []
#
#     for k in sorted(d.keys()):
#         ans.append(np.array(d[k]))
#
#     return np.array(ans, dtype='float32')
#
#
# def data_save(provider, filename):
#     df = create_data(provider, filename)
#     with open('../data/df_' + provider + '_' + filename + '.npy', 'wb') as f:
#         np.save(f, df)
#     return df
#
#
# def data_load(provider, filename):
#     with open('../data/df_' + provider + '_' + filename + '.npy', 'rb') as f:
#         df = np.load(f)
#     return df
#
#
# def create_data(provider, filename):
#     df = data_generator.get_data(provider, filename)
#     df = convert_to_numpy(df, 'date', ['close', 'high', 'low', 'open', 'price'])
#     return df
#
