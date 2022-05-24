import pandas as pd
import numpy as np
import src
from src.etl import utils
import src.etl.constants as c

df = utils.data_loader('base_clus')

# Generate many aggregated values for each cluster
percentiles = [5,10,25,50,75,90,95]
alias_perc = ['p_'+ str(p) for p in percentiles]

def agg_perc(r, percentiles=percentiles, alias_perc=alias_perc):
    
    # Wide format
    wide = (r
            .pivot(index = 'dt', 
                columns = 'id_station', 
                values = 'ffilled_price'))

    dates = wide.index

    ar_agg = np.percentile(wide.values,
                           q=percentiles,
                           axis =1).T
             
    df_agg = pd.DataFrame(data = ar_agg, 
                          columns = alias_perc,
                          index = dates)
    
    return df_agg 

df = (df
      .sort_values(by=['id_cluster','id_station', 'dt'])
      .groupby('id_cluster', as_index=False)
      .apply(agg_perc)
      .reset_index()
      .rename(columns={'level_0':'id_cluster'})
      .pivot(index = 'dt',
              columns = 'id_cluster',
              values = alias_perc))

# As a business rule, variations lower than 0.05 are not very significant
# Also equivalent columns generate numerical instability
keep_index1 = (df).round(1).T.drop_duplicates().T.columns

# Delete columns that have irrelevant variation
keep_index2 = (df
              .std()
              .mask(df.std()<0.1)
              .dropna()
              .index)


keep_index = keep_index1.intersection(keep_index2)

df = df.loc[:,keep_index]

# Remove multicolinearity with Variance Inflation Factor
def calculate_vif(df):
    
    # Calculate R²
    df_corr = df.corr(method='pearson')

    # Invert de R² , pinv avoid numerical instability
    df_vif = pd.DataFrame(data = np.linalg.pinv(a = df_corr.to_numpy()).diagonal(),
                        index = df_corr.columns,
                        columns=['VIF'])

    return df_vif

# Iteratively remove columns with VIF gt 10
df_vif = (calculate_vif(df)
          .query('VIF > 10'))

# loop
while len(df_vif)!=0:

    col_remove = (df_vif
                .sort_values(by='VIF')
                .iloc[-1]
                .name)

    df = df.drop(columns=[col_remove])

    df_vif = (calculate_vif(df)
        .query('VIF > 10'))

#Multiindex can't be saved, so the percentil and cluster index are concatenated
new_columns = [(col[0]) + '_clus_' + str(col[1]) for col in df.columns.values]
df.columns = new_columns
df.to_parquet('data/X.parquet')