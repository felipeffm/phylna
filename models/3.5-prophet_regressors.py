from statistics import median
from webbrowser import UnixBrowser
import pandas as pd
from tqdm import tqdm
import mlflow
import git
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import src
from src.etl import utils
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import add_changepoints_to_plot
import itertools
import holidays

# paths
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)
repo_name = os.path.basename(repo.working_dir)

pf = Path(__file__)
script_name = pf.name
parent = pf.resolve().parent.name

tags={'commit':hash, 'script':script_name, 'folder':parent, 'repo_name':repo_name, 'strategy':'Prophet with Regressors'}

X =  utils.data_loader('X')


Y = ( utils.data_loader('rf_base')
     .loc[:,['dt','price','id_station','ffilled_price']]
     .set_index('dt'))

dataset = (Y
           .dropna()
           .join(X, how='left')
           .reset_index()
           .rename(columns={'price':'y', 'dt':'ds'})
           )


# Dates when the price change was too high, so probably are related to government regulation
change_points = (Y
               .pivot( columns = 'id_station', 
                    values = 'ffilled_price')
          
               .pipe(pd.DataFrame.median, axis=1)
               .pipe(pd.Series.to_frame, name='median')
               .assign(
               ma = lambda x: x['median'].rolling(10).median(),
               detrended = lambda x: x['median'] - x['ma'])
               .dropna()
               .assign(ub = lambda x: x['detrended'].mean() + 2*x['detrended'].std(),
                    lb = lambda x: x['detrended'].mean() - 2*x['detrended'].std(),
                    bol_outlier = lambda x: (x['detrended'] > x['ub']) | (x['detrended'] < x['lb']))
               .query('bol_outlier==True')
               .reset_index()
               .loc[:,['dt']]
               .rename(columns={'dt':'ds'})
               )

ids_station = dataset['id_station'].unique()

#Holidays
h = holidays.Brazil()
start_dt = min(dataset['ds'])
end_dt = max(dataset['ds'])
df_h = (pd.DataFrame(data = h[start_dt:end_dt],columns = ['ds'])
          .assign(holiday='holiday'))

for id_station in tqdm(ids_station):
    dataset_ = (dataset
                .query(f'id_station=={id_station}')
                .sample(frac=1, random_state=0)
                .sort_values(by=['ds'])
                )
    
    len_ds = dataset_.shape[0]

    change_points_ =     (dataset_[['ds']]
                         .merge(change_points, on=['ds'], how='inner')
                         .loc[:,'ds']
                         .to_list())
    
    
    with mlflow.start_run(experiment_id='3',
                        tags = tags,
                        run_name=id_station):
        
        # Use cross validation to evaluate all parameters

        m = Prophet(growth='linear',
               changepoint_range=1,
               changepoints=change_points_,
               yearly_seasonality=False,
               seasonality_mode='additive',
               weekly_seasonality=7,
               daily_seasonality=False)
        
        regressors = dataset_.columns[dataset_.columns.str.startswith('p_')].to_list()

        for regressor in regressors:
            m.add_regressor(regressor)
        columns = regressors + ['y','ds']

        m.fit(dataset_[columns])  # Fit model with given params
        y_hat = m.predict(dataset_[columns])['yhat']
        mae = mean_absolute_error(dataset_['y'], y_hat)

        # Logging params and metrics to MLFlow
        mlflow.log_param('scoring', 'neg_mean_absolute_error')
        mlflow.log_metric('MAE', mae)
        mlflow.log_text('id_station', str(id_station))

        # Logging model to MLFlow
        mlflow.sklearn.log_model(m, 'model')