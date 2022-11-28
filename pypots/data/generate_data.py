import warnings

warnings.filterwarnings('ignore')

import random

import numpy as np
import pyfinancialdata

SETTINGS = {
    'I_5': ('I', 5),
    'I_10': ('I', 10),
    'I_20': ('I', 20),
    'I_30': ('I', 30),
    'I_40': ('I', 40),
    'I_50': ('I', 50),
    'R_5': ('R', 5),
    'R_10': ('R', 10),
    'R_20': ('R', 20),
    'R_30': ('R', 30),
    'R_40': ('R', 40),
    'R_50': ('R', 50)
}


def mask_data(size, setting_percentage, setting_type):
    count = int((setting_percentage * size) / 100)

    if setting_type == "I":
        interval = size / count
        index_numbers = np.arange(0, size, interval)
        index_numbers = index_numbers.astype(np.int32)
    else:
        index_numbers = random.sample(range(0, size - 1), count)
    return index_numbers


def get_data(provider='histdata', filename='SPXUSD'):
    data = pyfinancialdata.get_multi_year(provider=provider,
                                          instrument=filename,
                                          years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
                                          time_group='30min')
    data = data.reset_index()
    data['time'] = data['date'].apply(lambda x: x.time())
    data['date'] = data['date'].apply(lambda x: x.date())

    counts = data.groupby('date').count().reset_index()
    counts = counts[counts['time'] == 47]['date']

    data = data[data['date'].isin(counts)]
    #
    # train_date = counts[:int(0.80*len(counts))]
    # val_date = counts[int(0.80*len(counts)):int(0.90*len(counts))]
    # test_date = counts[int(0.90 * len(counts)):]
    #
    # train_data = data[data['date'].isin(train_date)]
    # val_data = data[data['date'].isin(val_date)]
    # test_data = data[data['date'].isin(test_date)]
    #
    # # val_data["mask"] = 0
    # # indexes = mask_data(len(val_data), SETTINGS[setting_id][1], SETTINGS[setting_id][0])
    # # val_data.iloc[indexes, -1] = 1
    #
    # test_data["mask"] = 0
    # indexes = mask_data(len(test_data), SETTINGS[setting_id][1], SETTINGS[setting_id][0])
    # test_data.iloc[indexes, -1] = 1

    # return train_data, val_data, test_data
    return data
