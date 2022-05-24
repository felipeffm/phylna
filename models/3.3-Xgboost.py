import src
import pandas as pd
from tqdm import tqdm
import mlflow
import git
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from src.etl import utils

# paths
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)
repo_name = os.path.basename(repo.working_dir)

pf = Path(__file__)
script_name = pf.name
parent = pf.resolve().parent.name

tags={'commit':hash, 'script':script_name, 'folder':parent, 'repo_name':repo_name, 'strategy':'XGboost'}

X =  utils.data_loader('X')
Y = ( utils.data_loader('rf_base')
    .dropna().loc[:,['dt','price','id_station']]
    .set_index('dt'))


dataset = Y.join(X, how='left')

ids_station = dataset['id_station'].unique()

for id_station in tqdm(ids_station):
    dataset_ = (dataset
                .query(f'id_station=={id_station}')
                .sample(frac=1, random_state=0))

    len_ds = dataset_.shape[0]
    test_size = 0.4

    train = dataset_.iloc[int(len_ds*0.4):].sort_index()
    test = dataset_.iloc[:int(len_ds*0.4)].sort_index()

    X_train, y_train  =  train.drop(columns=['price']), train['price']
    X_test, y_test = test.drop(columns=['price']), test['price']

    with mlflow.start_run(experiment_id='3',
                        tags = tags,
                        run_name=id_station):
        
        xgb.set_config(verbosity=0)
        
        model = xgb.XGBRegressor(n_estimators=100, 
                                max_depth=6, 
                                learning_rate=0.05,
                                scoring='neg_mean_absolute_error',
                                verbosity = 0,
                                random_state=0
                                )
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], 
            early_stopping_rounds=20,
            verbose=False) 
        
        # Model metrics
        y_hat = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_hat)

        # Logging params and metrics to MLFlow
        mlflow.log_param('scoring', 'neg_mean_absolute_error')
        mlflow.log_metric('MAE', mae)
        mlflow.log_text('id_station', str(id_station))

        # Logging model to MLFlow
        mlflow.sklearn.log_model(model, 'model')