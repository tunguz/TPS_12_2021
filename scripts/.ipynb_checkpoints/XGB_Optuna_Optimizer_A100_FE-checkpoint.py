import os
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask import dataframe as dd
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import optuna
import gc
import logging

num_round = 1000

def objective(client, dtrain, dtest, test_y, trial):
        
    params = {
        'objective': trial.suggest_categorical('objective',['multi:softprob']), 
        'num_class': trial.suggest_categorical('num_class',[6]), 
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'
        'lambda': trial.suggest_loguniform('lambda',1e-3,10.0),
        'alpha': trial.suggest_loguniform('alpha',1e-3,10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001,0.1),
        #'n_estimators': trial.suggest_categorical('n_estimators', [1000]),
        'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17,20]),
        #'random_state': trial.suggest_categorical('random_state', [24,48,2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1,300),
        'eval_metric': trial.suggest_categorical('eval_metric',['mlogloss']),

    }

    output = xgb.dask.train(client, params, dtrain, num_round)
    
    booster = output['booster']  # booster is the trained model
    booster.set_param({'predictor': 'gpu_predictor'})

    predictions = xgb.dask.predict(client, booster, dtest)
    
    predictions = np.argmax(predictions.compute(), axis=1)

    acc = accuracy_score(test_y, predictions)
    
    return acc

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


def main():
    train_x = dd.read_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgtrain_fe.csv')
    test_x = dd.read_csv('../../TPS_2021/input/tabular-playground-series-dec-2021/xgval_fe.csv')

    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    train_y = train_x['target'] - 1
    train_x = train_x[train_x.columns.difference(['target'])]

    test_x = test_x.replace([np.inf, -np.inf], np.nan)
    test_y = test_x['target'] - 1
    test_x = test_x[test_x.columns.difference(['target'])]
    
    #train_x = reduce_mem_usage(train_x)
    #test_x = reduce_mem_usage(test_x)

    with LocalCUDACluster(n_workers=4) as cluster:
        client = Client(cluster)
        dtrain = xgb.dask.DaskDMatrix(client, train_x, train_y)
        dtest = xgb.dask.DaskDMatrix(client, test_x, test_y)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler("optuna_xgb_output_fe.log", mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        study = optuna.load_study(storage="sqlite:///xgb_optuna_tests_fe.db", study_name="dec_2021_fe_test_0")
        logger.info("Start optimization.")
        study.optimize(lambda trial: objective(client, dtrain, dtest, test_y, trial), n_trials=150)
        
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df.to_csv('optuna_xgb_fe_output.csv', index=False)

if __name__ == "__main__":
    main()
