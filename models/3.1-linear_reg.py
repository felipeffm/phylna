from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import pandas as pd
from tqdm import tqdm
import mlflow
import git
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import src
from src.etl import utils

# paths
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)
repo_name = os.path.basename(repo.working_dir)

pf = Path(__file__)
script_name = pf.name
parent = pf.resolve().parent.name

tags={'commit':hash, 'script':script_name, 'folder':parent, 'repo_name':repo_name, 'strategy':'Linear Regression'}

X =  utils.data_loader('X')
Y = utils.data_loader('rf_base').dropna().loc[:,['dt','price','id_station']].set_index('dt')

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

        clf = LinearRegression()

        # Build step forward feature selection

        sfs1 = sfs(clf,
                k_features = (3,12),
                forward=True,
                floating=False,
                scoring='neg_mean_absolute_error',
                cv=5,
                n_jobs=1)

        # Perform SFFS
        sfs1 = sfs1.fit(X_train, y_train)
        cols_subset = list(sfs1.k_feature_names_)
        
        X_train  =  train[cols_subset]
        X_test  = test[cols_subset] 
        
        # Model metrics
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        mae = mean_absolute_error(y_test, y_hat)

        # Train model all base
        model = clf.fit(dataset_[cols_subset], dataset_['price'])

        # Logging params and metrics to MLFlow
        mlflow.log_param('scoring', 'neg_mean_absolute_error')
        mlflow.log_metric('MAE', mae)
        mlflow.log_text('n_features', str(len(sfs1.k_feature_names_)))
        mlflow.log_text('id_station', str(id_station))

        # Logging model to MLFlow
        mlflow.sklearn.log_model(model, 'model')