import datetime
import pandas as pd
import numpy as np
from util import log, timeit

def parse_time(xtime: pd.Series):
    result = pd.DataFrame()

    dtcol = pd.to_datetime(xtime, unit='s')

    result[f'{xtime.name}'] = dtcol.astype('int64') // 10**9
    result[f'{xtime.name}_year'] = dtcol.dt.year.astype('category')
    result[f'{xtime.name}_month'] = dtcol.dt.month.astype('category')
    result[f'{xtime.name}_day'] = dtcol.dt.day.astype('category')
    result[f'{xtime.name}_weekday'] = dtcol.dt.weekday.astype('category')
    result[f'{xtime.name}_hour'] = dtcol.dt.hour.astype('category')
    result[f'{xtime.name}_dayofyear'] = dtcol.dt.dayofyear.astype('category')
    result[f'{xtime.name}_quarter'] = dtcol.dt.quarter.astype('category')

    return result


class TypeAdapter:
    def __init__(self, primitive_cat, primary_timestamp, y, label, info):
        self.adapt_cols = primitive_cat.copy()
        self.time = primary_timestamp
        self.y = y
        self.label= label
        self.info = info

    @timeit
    def fit_transform(self, X):
        cols_dtype = dict(zip(X.columns, X.dtypes))
        self.fill_na(X)
        for key, dtype in cols_dtype.items():
            if dtype == np.dtype('object'):
                self.adapt_cols.append(key)
            if key in self.adapt_cols:
                X[key] = X[key].astype('category')
                #X[key+'hash'] = X[key].apply(hash_m)
                X_copy = X.copy()
                X_copy[self.time] = pd.to_datetime(X_copy[self.time], unit='s')
                X_copy =pd.concat([X_copy, self.y], axis=1)
                X['timediff'] = X_copy.sort_values([key, self.time]).groupby(key)[self.time].diff().dt.days
                #for i in X[key].unique():
                X[key + 'Unique_Count'] = len(X[key].unique())
                X[key+'t-1'] = X_copy.sort_values(self.time).groupby([key])[self.label].shift(1)
                X[key+'t-2'] = X_copy.sort_values(self.time).groupby([key])[self.label].shift(2)
                X[key+'t-7'] = X_copy.sort_values(self.time).groupby([key])[self.label].shift(7)
                X[key+'t-30'] = X_copy.sort_values(self.time).groupby([key])[self.label].shift(30)

                #X['timediff']= pd.to_datetime(X['timediff'],unit='s')
                X['timediff'].fillna(datetime.datetime(1970, 1, 1))
                #X['timediff'].dt.days.astype('str')


        return X


    @timeit
    def transpose_matrix(self, X):
        c = (X.groupby(['ID', 'col']).cumcount() + 1).astype(str)

        # remove col2
        X = X.set_index(['ID', 'col', c]).unstack()
        # flatten Multiindex
        X.columns = X.columns.map('_'.join)
        X = X.reset_index()


    @timeit
    def transform(self, X):
        for key in X.columns:
            if key in self.adapt_cols:
                X[key] = X[key].astype('category')
                #X[key+'hash'] = X[key].apply(hash_m)

        return X

    @timeit
    def fill_na(self, df):
        schema = self.info['schema']
        num_cols = [col for col, types in schema.items() if types == 'num']
        num_cols.remove(self.label)
        # cat_cols = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
        m_cat_cols = [col for col, types in schema.items() if types == 'str']
        time_cols = [col for col, types in schema.items() if types == 'timestamp']

        for c in [num_cols]:
            df.groupby(m_cat_cols)[c].fillna(method='ffill', inplace=True)
        for c in [m_cat_cols]:
            df[c].fillna("Unknown", inplace=True)
        for c in [time_cols]:
            df[c].fillna(method='ffill', inplace=True)


def hash_m(x):
    return hash(x) % 1048575
