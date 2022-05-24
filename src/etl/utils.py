from logging import raiseExceptions
import os
import functools
import pandas as pd
import numpy as np
from ipp_ds.data_cleaning.feature_engineering import new_date_encoder
from ipp_ds.io import io
import src.etl.constants as c

def data_loader(metadata_name='precos_rj_gas_urbano_semoutliers'):
    """
    Load dataset from local, if does not exist then download it from Azure blob.
    All metadata is saved at constants.py 
    If you try to download innexistent dataset there's a suggestion about what to do.

    Args:
        path_base (string): blob uri

    Returns:
        pd.DataFrame: DataFrame with price data preprocessed in databricks
    """

    path_local = c.METADATA[metadata_name]['local']
    path_source = c.METADATA[metadata_name]['source']

    if os.path.exists(path_local):
        df = pd.read_parquet(path_local)

    #.env file is used to download from azure blob. That's a private file, is not necessary.  
    elif os.path.exists('.env'):
        path_cloud = c.METADATA[metadata_name]['uri']
        file_paths = io.glob(path_cloud)
        if len(file_paths>0):
            df = pd.concat([io.read_parquet(f) for f in file_paths], sort=True)
            df.to_parquet(path_local)
        else:
            print('No file local or in blob.')
            
    elif not os.path.exists('.env'):
        raise FileNotFoundError(
            f"""You are trying to load inexistent files! 
            \n Maybe you should generate this dataset before call it. 
            \n Run {path_source} to generate it 
            """
        )

    else:
        print('Sorry, thing have gone wrong. Send a message to the programmer who made that problematic code.')
    return df


def treat_dataset(df):
    """Encode date and create interpolated columns

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: Same dataframe with new columns
    """
    start = df['dt'].min()

    df = (df
          .assign(date_index = new_date_encoder(df['dt'], freq='D', date_start=start))
          .sort_values(by=['id_station','date_index']))

    groupedf = df.groupby(['id_station'])['price']

    df['lin_price_interp'] = groupedf.apply(pd.Series.interpolate, args = 'linear')
    df['near_price_interp'] = groupedf.apply(pd.Series.interpolate, args = 'near')

    return df