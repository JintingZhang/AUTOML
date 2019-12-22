import time

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class LGBMRegressor:
    def __init__(self, params=None):
        self.model = None
        self.test_model = None

    def fit(self, X_train, y_train, hyperparams, Time_data_info, categorical_feature=None, X_eval=None, y_eval=None):
        self.feature_name = list(X_train.columns)

        categorical_feature = X_train.select_dtypes(include=['object'])
        if X_eval is None or y_eval is None:
            X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)

        params = {"objective": "regression", "metric": "rmse", 'verbosity': -1, "seed": 0}

        train_time_start = time.time()

        self.test_model = lgb.train({**params, **hyperparams},
                                    lgb_train,
                                    30,
                                    lgb_eval,
                                    early_stopping_rounds=30)
        train_time_end = time.time()

        _30_boost_rounds_for_train_time = train_time_end - train_time_start
        if Time_data_info['time_ramain_so_far'] - _30_boost_rounds_for_train_time <= Time_data_info['For_safe']:
            self.model = self.test_model
        else:
            num_boost_rounds = int(0.9 * (30 * ((Time_data_info[
                                                           'time_ramain_so_far'] - _30_boost_rounds_for_train_time) / _30_boost_rounds_for_train_time)))
            print("leave_num_boost_rounds", num_boost_rounds)

            self.model = lgb.train({**params, **hyperparams}, train_set=lgb_train,

                                   num_boost_round=num_boost_rounds,
                                   valid_sets=lgb_eval,
                                   valid_names='eval',
                                   early_stopping_rounds=200)

        return self

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("You must fit first!")

        return self.model.predict(X_test)

    def score(self):
        return dict(zip(self.feature_name, self.model.feature_importance('gain')))
