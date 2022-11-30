import mlflow
import pandas as pd
import numpy as np

glpe_runs = mlflow.search_runs(experiment_ids=['3']).fillna(value=np.nan)

query = glpe_runs['params.method'] == 'GE'

ge_runs = glpe_runs[query]



for _, ge_run in ge_runs.iterrows():
    print(ge_run)
    ranks = pd.read_csv(ge_run['artifact_uri']+'/feature_ranks.csv', index_col = 0)
    ranks_query = ranks['batch:Selected'] == 1
    best_features = list(ranks[ranks_query].index)

    feature_df = pd.DataFrame(columns = ['ProbeID'], data = best_features)
     
    experiment = ge_run['params.experiment']
    feature_df.to_csv(f'../features/{experiment}.csv')
