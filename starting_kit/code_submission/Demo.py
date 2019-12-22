import datetime

from dataset import Dataset
from  model import Model

from sklearn import metrics
from preprocessing import TypeAdapter, parse_time

dataset_dir = "C:/Users/a628123/Desktop/Git_projects/autoseries2020-master/public_data/public/data/15"


Dataset = Dataset(dataset_dir)
train_data = Dataset.get_train()
test_data = Dataset.get_test()
info = Dataset.metadata_

"""
import  pandas as pd
X = pd.read_csv('feature_automl.csv')
c = (X.groupby(['A1']).cumcount()+1 ).astype(str)
print (c)
# remove col2

Z = X.set_index(['A1', c])['A2t-1'].unstack()
# flatten Multiindex
Z.columns = Z.columns.map('_'.join)
Z = Z.reset_index()

X.join(Z, on= "A1", how= "left")

print(X)
print(Z)


"""
test_timestamp = Dataset.get_test_timestamp()
pred_timestamp = Dataset.get_pred_timestamp()
time_info = info['time_budget']
print(test_timestamp)
print(pred_timestamp)

model_test = Model(info, test_timestamp,pred_timestamp)
next_step_1 = model_test.train(train_data, time_info)
print(next_step_1)
new_history = Dataset.get_all_history(0)
prediction, next_step_2 = model_test.predict(new_history, test_data, time_info)
print(prediction)
"""
for i in range (len(pred_timestamp)):
    pred_record = Dataset.get_next_pred(i)
    new_history = Dataset.get_all_history(i)
    predictions = []
    prediction, next_step_2 = model_test.predict(new_history, pred_record, time_info)
    predictions.append(prediction)

print(predictions)"""

#rmse = metrics.mean_squared_error(test_data['A3'],predictions)
#print(rmse)
