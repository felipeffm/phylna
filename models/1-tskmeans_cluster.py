import os
import pandas as pd
import tqdm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series
import src
from src.etl import utils
import src.etl.constants as c
import numpy as np
from scipy.optimize import curve_fit

# Load data
df = utils.data_loader('rf_base')

# Change table format from long to wide
df2 = (df
        .sort_values(by='dt')
        .pivot(index = 'dt', 
               columns = 'id_station', 
               values = 'ffilled_price')
        )

#Postos de baixo desvio padrão foram descartados
df2 = df2[(df2.std()>0.05).index]

# Elbow method in K-means for time series
def hiperparam_tsk(df2, max_clus=30):
    
    X = to_time_series(df2.round(2).values.T)
    max_clus = int(df2.shape[1]/10) if max_clus==None else max_clus
    data=[]
    print("Elbow method")

    #Calculate inertia for different numbers of clusters
    for n_clus in tqdm.tqdm(range(3,max_clus,1)):

        print(f'Calculate inertia for {n_clus} clusters')
        
        tsk = TimeSeriesKMeans(n_clusters=n_clus,
                        metric="euclidean",
                        max_iter=5,
                        n_init=2,
                        n_jobs=-1)
        
        tsk.fit(X)
        
        data.append({'inertia':tsk.inertia_, 
                      'labels':[tsk.labels_],
                      'n_clus':n_clus})

    df_clus = pd.DataFrame.from_dict(data)

    x = df_clus.index.to_numpy()
    y = df_clus['inertia'].values
    
    def model_func(x, a, k, b):
        return a * np.exp(-k*x) + b
    
    # curve fit
    p0 = (1.,1.e-5,1.) # starting search koefs
    opt, pcov = curve_fit(model_func, x, y, p0)
    a, k, b = opt

    # second derivative of k*e**ax+b
    # from calculos, d²/dx² k*e**ax = a²*k*e^(ax+b)
    # inflexion point when its equal to zero...
    # a²*k*e^(ax+b) = 0 

    
    df_clus['inertia_curvefitting'] = model_func(x, a, k, b)
        # just to check if it was really exponential
    
    from matplotlib import pyplot as plt
    x2 = np.linspace(int(min(y)-4), int(max(y)+5))
    y2 = model_func(x2, a, k, b)
    fig, ax = plt.subplots()
    ax.plot(x2, y2, color='r', label='Fit. func: $f(x) = %.3f e^{%.3f x} %+.3f$' % (a,k,b))
    ax.plot(x, y, 'bo', label='data with noise')
    ax.legend(loc='best')
    plt.show()
    

    #flag optimal number of clusters
    df_clus = (df_clus
                .assign(
                    dif2 = (df_clus
                        .loc[:,'inertia_curvefitting']
                        .diff() # "First derivative"
                        .diff()), # "Second derivative"
                    
                 best_nclus = lambda x: x['dif2'].min() == x['dif2'],)
                .drop(columns = ['dif2','inertia_curvefitting']))

    return df_clus

df_clus = hiperparam_tsk(df2)

# Assign cluster label to price registers based in the best number of clusters
base_clus = (df_clus
            .query('best_nclus == True')
            .loc[:,'labels']
            .explode()
            .explode()
            .pipe(pd.DataFrame)
            .rename(columns = {'labels':'id_cluster'}) 
            .assign(id_station = df2.columns)
            .reset_index(drop=True)
            .merge(df, on='id_station', how='right') # Assign cluster id to price register based on station
            ) 

# Write results
df_clus.to_parquet(c.METADATA['hiperparam_clus']['local'])
base_clus.to_parquet(c.METADATA['base_clus']['local'])