from msilib import schema
import pandas as pd
from mlflow.tracking import MlflowClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import numpy as np

def runs_to_pandas(experiment):
    """
    Consolidate semi structured log data in structured data 
    with columns ['commit', 'id_station', 'repo', 'script', 'MAE', 'run_id']

    Args:
        experiment (string): experiment id from MLFlow

    Returns:
        pd.DataFrame
    """

    client = MlflowClient()
    client.list_experiments()
    ls_run_info = client.list_run_infos(experiment,max_results=50000)

    ls_data = []
    run_ids_erros = []
    
    schema = {
        'commit': str,
        'id_station': int,
        'repo': str,
        'script': str,
        'strategy':str,
        'MAE': float,
        'end_time':'int64',
        'run_id': str,
        
    }

    for run_info in tqdm(ls_run_info):
        data = client.get_run(run_info.run_id).data

        try:
            ls_data.append(
                            (data.tags['commit'],
                            data.tags['mlflow.runName'],
                            data.tags['repo_name'],
                            data.tags['script'],
                            data.tags['strategy'],
                            data.metrics['MAE'],
                            run_info.end_time,
                            run_info.run_id))
        except KeyError:
            run_ids_erros.append(run_info.run_id)

    df = (pd.DataFrame(data = ls_data, columns = schema.keys())
            .dropna(subset=['end_time'])
            .astype(schema))

    # Convert unix timestamp to datetime 
    df['end_time'] = df['end_time']/1000
    df['datetime'] = df['end_time'].map(datetime.datetime.fromtimestamp)

    # Create an boolean for stations presents in all models
    sets=[]
    for strategy in df['strategy'].unique():
        sets.append(set(df.query(f"strategy=='{strategy}'").loc[:,'id_station']))

    all_models_stations = sets[0]

    for set_stations in sets[1:]:
        all_models_stations = set_stations.intersection(all_models_stations)

    df['isin_allmodels'] = False
    df.loc[df['id_station'].isin(all_models_stations), 'isin_allmodels'] = True 
    df = df.dropna(subset=['MAE'])
    df['MAE'] = df['MAE'].round(3)

    # In case of tie the simpler model should have the lower MAE (Just for ranking with MAE)
    df.loc[df['strategy']=='Linear Regression', 'MAE'] = df.loc[df['strategy']=='Linear Regression', 'MAE'] - 0.0002
    df.loc[df['strategy']=='Prophet', 'MAE'] = df.loc[df['strategy']=='Prophet', 'MAE'] - 0.0001
    df.loc[df['strategy']=='Prophet with Regressors', 'MAE'] = df.loc[df['strategy']=='Prophet with Regressors', 'MAE'] - 0.00005
    df.loc[df['strategy']=='XGboost', 'MAE'] = df.loc[df['strategy']=='XGboost', 'MAE'] + 0.0001
    
    return  df, run_ids_erros

df_reg, _ = runs_to_pandas(experiment='2')


df_rf= (df_reg
      .query('(MAE**2)<1')
      .query('isin_allmodels==True')
      .sort_values(by=['id_station', 'strategy','datetime'])
      .groupby(['id_station', 'strategy'], as_index=False)[['MAE']]
      .first()
      .assign(MAE=lambda x: abs(x['MAE']))
      )

df_rf.to_parquet('data/metrics_mlflow.parquet')