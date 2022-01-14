import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

def create_nested_dict(df, nth):
    nth_values = df.explode("missing_or_unknown").groupby(["attribute"])["missing_or_unknown"].nth(nth)
    nan_array = np.empty(nth_values.values.shape)
    nan_array[:] = np.NaN
    dict_ = {}
    for attribute, missing_value_code, nans in zip(nth_values.index, nth_values.values, nan_array):
        dict_[attribute] = {missing_value_code: nans}
    return dict_

def my_scree(pca_obj):
    cum_sum = np.cumsum(pca_obj.explained_variance_ratio_)
    x = np.arange(0, cum_sum.shape[0])
    return plt.plot(x, cum_sum)

def new_func(x):
    return x