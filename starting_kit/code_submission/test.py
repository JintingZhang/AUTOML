
from dataset import Dataset
from  model import Model
from preprocessing import TypeAdapter, parse_time

dataset_dir = "C:/Users/a628123/Desktop/Git_projects/autoseries2020-master/public_data/public/data/14"


Dataset = Dataset(dataset_dir)
train_data = Dataset.get_train()
test_data = Dataset.get_test()
info = Dataset.metadata_


train_data['timediff'] = train_data.sort_values(['A2', 'A1']).groupby('A2')['A1'].diff().days
train_data['timediff'] = train_data.sort_values(['A2', 'A1']).groupby('A2')['A1'].diff().days

print(train_data)

