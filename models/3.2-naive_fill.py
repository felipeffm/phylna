import pandas as pd
from tqdm import tqdm
import mlflow
import git
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from src.etl import utils

# log info
repo = git.Repo(search_parent_directories=True)
hash = repo.git.rev_parse(repo.head, short=True)
repo_name = os.path.basename(repo.working_dir)
script_name = Path(__file__).name
parent = Path(__file__).resolve().parent.name
tags={'commit':hash, 'script':script_name, 'folder':parent, 'repo_name':repo_name}

# load data
dataset = utils.data_loader('rf_base')

considered_nulls = 30

datas = []
#calculate metrics per station - consecutive nulls - filling type
for id_station, df in tqdm(dataset.groupby('id_station')):    

    price = df[['price']]

    for nulls in range(1,considered_nulls):
         
        s = pd.Series(dtype=float)
        win_size = nulls+2
        
        for window in df['price'].rolling(window=win_size):
            if ~window.isnull().values.any() and win_size==len(window):
                
                # Create virtual nulls 
                window.iloc[1:nulls+1] = np.nan
                s = s.append(window)

        if len(s)>3:

            # Filled virtual nulls and real values
            price_ = price.loc[s.index]

            
            mae_bfill = mae(price_,s.fillna(method='bfill'))
            mae_ffill = mae(price_,s.fillna(method='ffill'))
            mae_linear = mae(price_,s.interpolate(method='linear'))
            mae_nearest = mae(price_,s.interpolate(method='nearest'))
            
            data = {'MAE bfill': mae_bfill,
                    'MAE ffill': mae_ffill,
                    'MAE linear': mae_linear,
                    'MAE nearest': mae_nearest,
                    'consecutive_nulls': nulls,
                    'id_station': str(id_station)}

            datas.append(data)

pd.DataFrame(datas).to_parquet('data/naive_approach_metrics.parquet')

