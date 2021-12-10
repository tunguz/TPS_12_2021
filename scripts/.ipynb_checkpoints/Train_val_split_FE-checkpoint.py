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

train.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'], inplace=True) 
test.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'], inplace=True)

new_names = {
    "Horizontal_Distance_To_Hydrology": "x_dist_hydrlgy",
    "Vertical_Distance_To_Hydrology": "y_dist_hydrlgy",
    "Horizontal_Distance_To_Roadways": "x_dist_rdwys",
    "Horizontal_Distance_To_Fire_Points": "x_dist_firepts"
}

train.rename(new_names, axis=1, inplace=True)
test.rename(new_names, axis=1, inplace=True)

train["Aspect"][train["Aspect"] < 0] += 360
train["Aspect"][train["Aspect"] > 359] -= 360

test["Aspect"][test["Aspect"] < 0] += 360
test["Aspect"][test["Aspect"] > 359] -= 360

# Manhhattan distance to Hydrology
train["mnhttn_dist_hydrlgy"] = np.abs(train["x_dist_hydrlgy"]) + np.abs(train["y_dist_hydrlgy"])
test["mnhttn_dist_hydrlgy"] = np.abs(test["x_dist_hydrlgy"]) + np.abs(test["y_dist_hydrlgy"])

# Euclidean distance to Hydrology
train["ecldn_dist_hydrlgy"] = (train["x_dist_hydrlgy"]**2 + train["y_dist_hydrlgy"]**2)**0.5
test["ecldn_dist_hydrlgy"] = (test["x_dist_hydrlgy"]**2 + test["y_dist_hydrlgy"]**2)**0.5

train.loc[train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
test.loc[test["Hillshade_9am"] < 0, "Hillshade_9am"] = 0

train.loc[train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
test.loc[test["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0

train.loc[train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
test.loc[test["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

train.loc[train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
test.loc[test["Hillshade_9am"] > 255, "Hillshade_9am"] = 255

train.loc[train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
test.loc[test["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

train.loc[train["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
test.loc[test["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255

features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
soil_features = [x for x in train.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in train.columns if x.startswith("Wilderness_Area")]

def addFeature(X):
    # Thanks @mpwolke : https://www.kaggle.com/mpwolke/tooezy-where-are-you-no-camping-here
    X["Soil_Count"] = X[soil_features].apply(sum, axis=1)

    # Thanks @yannbarthelemy : https://www.kaggle.com/yannbarthelemy/tps-december-first-simple-feature-engineering
    X["Wilderness_Area_Count"] = X[wilderness_features].apply(sum, axis=1)
    X["Hillshade_mean"] = X[features_Hillshade].mean(axis=1)
    X['amp_Hillshade'] = X[features_Hillshade].max(axis=1) - X[features_Hillshade].min(axis=1)
    
addFeature(train)
addFeature(test)

cols = [
    "Elevation",
    "Aspect",
    "mnhttn_dist_hydrlgy",
    "ecldn_dist_hydrlgy",
    "Slope",
    "x_dist_hydrlgy",
    "y_dist_hydrlgy",
    "x_dist_rdwys",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "x_dist_firepts",
    
    "Soil_Count","Wilderness_Area_Count","Hillshade_mean","amp_Hillshade"
]

scaler = RobustScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])

features = test.columns

perm_dict = {1:2, 2:1, 3:3, 4:6, 7:4, 6:5, 5:7,}

train['Cover_Type'].replace(perm_dict, inplace=True)

train = train[train.Cover_Type !=7]

def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

target = train[['Cover_Type']].values
train.drop(['Cover_Type'], axis=1, inplace=True)

xgtrain, xgval, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)

xgtrain['target'] = y_train
xgval['target'] = y_val

xgtrain.to_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgtrain_fe.csv', index=False)
xgval.to_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgval_fe.csv', index=False)
