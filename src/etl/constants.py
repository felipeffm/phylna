METADATA = {
     "outliers_base": {
        "uri": None,
        "source": None,
        "local":'data/outliers_base.parquet'
        },
    
    "rf_base": {
        "uri": None,
        "source": None, 
        "local":'data/rf_base.parquet'
    },
    
    "hiperparam_clus": {
        "source": "models\\1-tskmeans_cluster.py",
        "local": "data/hiperparam_clus.parquet"
    },
    "base_clus": {
        "source": "models\\1-tskmeans_cluster.py",
        "local": 'data/base_clus.parquet'
    },
    "X": {
    "source": "models\\2-feature_engineering.py",
    "local": 'data/X.parquet'
    },
    "naive_approach_metrics": {
    "source": "models\\3.2-naive_fill.py",
    "local": 'data/naive_approach_metrics.parquet'
    },
    
    "metrics_mlflow": {
    "source": "models\\4-performance_metrics.py",
    "local": 'data/metrics_mlflow.parquet'
    }

}