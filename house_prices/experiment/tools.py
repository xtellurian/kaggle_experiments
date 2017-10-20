import pandas as pd
import numpy as np

def summarise_data(df):
    for name in df.columns:
        s = df[name]
        print("{0} has type {1}, {2} unique values, max: {3}, min: {4}".format(name, s.dtype, len(s.unique()), s.max(), s.min()))
        #print("{0} has vals: {1}".format(name, df[name].unique()))

def years_since_1900(x):
    if np.isnan(x):
        return np.nan
    else:
        return x - 1900
