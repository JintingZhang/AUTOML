import gc
import pickle
import random
from hyperopt_class import train_hyperopt
import pandas as pd
from models import LGBMRegressor
from preprocessing import parse_time, TypeAdapter
import os
from util import Config, log, show_dataframe, timeit
import time


os.system("pip3 install hyperopt")

class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.primary_timestamp = info['primary_timestamp']
        self.primary_id = info['primary_id']
        self.label = info['label']
        self.schema = info['schema']

        print(f"\ninfo: {self.info}")

        self.dtype_cols = {}
        self.dtype_cols['cat'] = [col for col, types in self.schema.items() if types == 'str']
        self.dtype_cols['num'] = [col for col, types in self.schema.items() if types == 'num']
        self.dtype_cols['time'] = [col for col, types in self.schema.items() if types == 'timestamp']

        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp

        self.n_test_timestamp = len(pred_timestamp)
        self.update_interval = int(self.n_test_timestamp / 5)

        print(f"sample of test record: {len(test_timestamp)}")
        print(f"number of pred timestamp: {len(pred_timestamp)}")

        self.lgb_model = LGBMRegressor()
        self.n_predict = 0

        self.config = Config(info)
        self.tables = None
        self.targets = None

        self.Time_data_info = {
            # time
            'total_time': 0,
            'time_ramain_so_far': 0,
            'time_for_feature_engineering': 0,
            'time_for_hyperparams_searching': 0,
            'time_for_model_train': 0,
            'time_for_model_prediction': 0,

            # size
            'feature_engineering_input_size': 0,
            'data_rows_for_hp': 0,
            'data_cols_for_hp': 0,
            'test_data_rows': 0,
            'test_data_columns': 0,
            'For_safe': 50,
        }
        self.randomintvalue = random.randint(1, 100)

        print(f"Finish init\n")

    @timeit
    def train(self, train_data, time_info):
        print(f"\nTrain time budget: {time_info['train']}s")
        self.Time_data_info['total_time'] = time_info['train']
        self.Time_data_info['For_safe'] = (self.Time_data_info['total_time'] / 10)
        self.Time_data_info['time_ramain_so_far'] = time_info['train']

        X = train_data
        y = train_data.pop(self.label)



        log(f"Feature engineering...")

        # type adapter
        start_feature = time.time()
        self.type_adapter = TypeAdapter(self.dtype_cols['cat'], y = y , label = self.label,
                                        primary_timestamp= self.primary_timestamp,
                                        info = self.info)
        X = self.type_adapter.fit_transform(X)

        # parse time feature
        time_fea = parse_time(X[self.primary_timestamp])
        #time_fea2 = parse_time(X['timediff'])

        X.drop(self.primary_timestamp, axis=1, inplace=True)
        X = pd.concat([X, time_fea], axis=1)

        X.to_csv('feature_automl.csv')
        #clean_tables(X, self.info)
        #X = clean_df(X, self.info)
        end_feature = time.time()
        self.Time_data_info['time_for_feature_engineering'] = (end_feature - start_feature)

        self.Time_data_info['time_ramain_so_far'] = self.Time_data_info['time_ramain_so_far'] - self.Time_data_info[
            'time_for_feature_engineering']

        print(f"TIME info:", self.Time_data_info)

        log(f"Hyperparams searching ...")
        start_hyper_searching = time.time()
        new_y = train_hyperopt(self.Time_data_info)
        time_limitation_for_hp = (self.Time_data_info['time_ramain_so_far'] - self.Time_data_info['For_safe']) / 3

        params_hyp = {"objective": "regression", "metric": "rmse", 'verbosity': -1, "num_threads": 4, "seed": 0}
        hyperparams = new_y.hyperopt_lightgbm(X, y, params=params_hyp, time_limitation=time_limitation_for_hp)

        end_hyper_searching = time.time()
        self.Time_data_info['time_for_hyperparams_searching'] = (end_hyper_searching - start_hyper_searching)

        self.Time_data_info['time_ramain_so_far'] = self.Time_data_info['time_ramain_so_far'] - self.Time_data_info[
            'time_for_hyperparams_searching']
        print(f"TIME info:", self.Time_data_info)

        # lightgbm model use parse time feature
        log(f"Training...")
        train_start = time.time()

        self.lgb_model.fit(X, y, hyperparams,self.Time_data_info)
        gc.collect()
        print(f"Feature importance: {self.lgb_model.score()}")

        print("Finish train\n")

        train_end = time.time()
        self.Time_data_info['time_ramain_so_far'] = self.Time_data_info['time_ramain_so_far'] - (
                    train_end - train_start)
        self.Time_data_info['time_for_model_train'] = (train_end - train_start)

        print("TIME info:", self.Time_data_info)

        next_step = 'predict'
        return next_step

    @timeit
    def predict(self, new_history, pred_record, time_info):
        log(f"Predicting...")
        predict_start = time.time()

        if self.n_predict % 100 == 0:
            print(f"\nPredict time budget: {time_info['predict']}s")
        self.n_predict += 1

        # type adapter
        pred_record = pred_record.append(new_history)
        pred_record = self.type_adapter.transform(pred_record)

        # parse time feature
        time_fea = parse_time(pred_record[self.primary_timestamp])

        pred_record.drop(self.primary_timestamp, axis=1, inplace=True)
        pred_record = pd.concat([pred_record, time_fea], axis=1)
        pd.set_option('display.max_columns', 15)
        print(pred_record)
        predictions = self.lgb_model.predict(pred_record)

        predict_end = time.time()
        self.Time_data_info['time_for_model_prediction'] = (predict_end - predict_start)

        if self.n_predict > self.update_interval:
            next_step = 'update'
            self.n_predict = 0
        else:
            next_step = 'predict'

        return list(predictions), next_step

    @timeit
    def update(self, train_data, test_history_data, time_info):
        print(f"\nUpdate time budget: {time_info['update']}s")

        total_data = pd.concat([train_data, test_history_data])

        self.train(total_data, time_info)

        print("Finish update\n")

        next_step = 'predict'
        return next_step

    @timeit
    def save(self, model_dir, time_info):
        print(f"\nSave time budget: {time_info['save']}s")

        pkl_list = []

        for attr in dir(self):
            if attr.startswith('__') or attr in ['train', 'predict', 'update', 'save', 'load']:
                continue

            pkl_list.append(attr)
            pickle.dump(getattr(self, attr), open(os.path.join(model_dir, f'{attr}.pkl'), 'wb'))

        pickle.dump(pkl_list, open(os.path.join(model_dir, f'pkl_list.pkl'), 'wb'))

        print("Finish save\n")

    @timeit
    def load(self, model_dir, time_info):
        print(f"\nLoad time budget: {time_info['load']}s")

        pkl_list = pickle.load(open(os.path.join(model_dir, 'pkl_list.pkl'), 'rb'))

        for attr in pkl_list:
            setattr(self, attr, pickle.load(open(os.path.join(model_dir, f'{attr}.pkl'), 'rb')))

        print("Finish load\n")
