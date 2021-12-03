import numpy as np
import pandas as pd
import gc
import time

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import warnings

train = pd.read_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/test.csv')

train.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True) 
test.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

features = train.columns[1:-1]

perm_dict = {1:2, 2:1, 3:3, 4:6, 7:4, 6:5, 5:7,}

train['Cover_Type'].replace(perm_dict, inplace=True)

train = train[train.Cover_Type !=7]

target = train[['Cover_Type']].values
train.drop(['Cover_Type', 'Id'], axis=1, inplace=True)
test.drop(['Id', ], axis=1, inplace=True)

train_test = pd.concat([train, test], axis =0)
RS = RobustScaler()
RS.fit(train_test)
del train_test
gc.collect()
gc.collect()
train[features] = RS.transform(train)
test[features] = RS.transform(test)
del RS
gc.collect()
gc.collect()

xgtrain, xgval, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)

xgtrain['target'] = y_train
xgval['target'] = y_val

xgtrain.to_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgtrain.csv', index=False)
xgval.to_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgval.csv', index=False)
