import mlflow
import numpy as np
import pandas as pd
from plotly import express as px




def get_max_config(df, method):
    query = df['params.method'] == method
    method_runs = df[query]

    best_runs = pd.DataFrame(columns = method_runs.columns)

    for experiment in np.unique(method_runs['params.experiment']):
        query = method_runs['params.experiment'] == experiment
        exp_method_runs = method_runs[query]
        max_idx = exp_method_runs['metrics.train_test.Test.Mean.BSR'].idxmax()    
        best_run = exp_method_runs.loc[max_idx]
        best_runs = best_runs.append(best_run, ignore_index = True)

    return best_runs

def get_precomputed_directed_pagerank(df):
    query = (df['params.method'] == 'CPE') &\
            (df['params.centrality_measure'] == 'page_rank') &\
            (df['params.similarity_measure'] == 'precomputed') &\
            (df['params.directed'] == 'True')
    good_cpe_runs = df[query]

    return good_cpe_runs

def calc_mean_acc(df):
    test_fold_columns = [col for col in df.columns if 'Test.batch' in col]
    test_fold_df = df[test_fold_columns]
    test_fold_df.groupby(lambda x: x.split(".")[-1], axis=1).mean()
    df['ACC Mean'] = test_fold_df.groupby(lambda x: x.split(".")[-1], axis=1).mean()['ACC']
    df['ACC Std'] = test_fold_df.groupby(lambda x: x.split(".")[-1], axis=1).std()['ACC'] 

    return df  


if __name__ == '__main__':

    runs_PE = mlflow.search_runs(experiment_ids=['2']).fillna(value=np.nan)
    runs_GE = mlflow.search_runs(experiment_ids=['3']).fillna(value=np.nan)

    # best_runs = pd.DataFrame(columns = runs_PE.columns)
    # best_runs = best_runs.append(get_max_config(runs_PE, 'CPE'), ignore_index = True)
    # print(len(best_runs))
    # best_runs = best_runs.append(get_max_config(runs_PE, 'LPE'), ignore_index = True)
    # print(len(best_runs))
    # best_runs = best_runs.append(runs_GE[runs_GE['params.method'] == 'GE'], ignore_index = True)
    # print(len(best_runs))

    # best_runs_small = best_runs[['params.experiment', 'metrics.train_test.Test.Mean.BSR', 'params.method']]

    # best_runs_small = best_runs.rename(columns = {'params.experiment':'Experiment', 'metrics.train_test.Test.Mean.BSR':'BSR', 'params.method':'Method'})

    # exps = list(best_runs_small['Experiment'])
    # exps = [' '.join(e.split('_')[1:]) for e in exps]
    # best_runs_small['Experiment'] = exps


    # best_runs_results = best_runs_small[['Experiment', 'Method', 'BSR']]

    # best_runs_results.to_csv('./results/best_classifier_results.csv')



    # new_query = best_runs['params.method'] == 'CPE'

    # cpe_runs = best_runs[new_query]

    # best_cpe_params = pd.DataFrame(columns = ['Experiment', 'Centrality', 'Similarity', 'Directed'])

    # for _, run in cpe_runs.iterrows():
    #     exp = run['params.experiment']
    #     print(exp)
    #     experiment = ' '.join(exp.split('_')[1:])
    #     centrality = run['params.centrality_measure']
    #     similarity = run['params.similarity_measure']
    #     directed = run['params.directed']
    #     row = pd.DataFrame(columns = ['Experiment', 'Centrality', 'Similarity', 'Directed'], data = [[experiment, centrality, similarity, directed]])
    #     best_cpe_params = best_cpe_params.append(row, ignore_index = True)


    # best_cpe_params.to_csv('./results/best_cpe_params.csv')




    best_runs = pd.DataFrame(columns = runs_PE.columns)
    best_runs = best_runs.append(get_precomputed_directed_pagerank(runs_PE), ignore_index = True)
    print(len(best_runs))
    best_runs = best_runs.append(get_max_config(runs_PE, 'LPE'), ignore_index = True)
    print(len(best_runs))
    best_runs = best_runs.append(runs_GE[runs_GE['params.method'] == 'GE'], ignore_index = True)
    print(len(best_runs))


    best_runs = calc_mean_acc(best_runs)

    best_runs_small = best_runs[['params.experiment', 'metrics.train_test.Test.Mean.BSR', 'metrics.train_test.Test.Mean.TPR', 'metrics.train_test.Test.Mean.PPV', 'ACC Mean', 'ACC Std', 'params.method']]

    # best_runs_small = best_runs[['params.experiment', 'metrics.train_test.Test.Mean.BSR', 'params.method']]

    best_runs_small = best_runs.rename(columns = {'params.experiment':'Experiment', 'metrics.train_test.Test.Mean.BSR':'BSR', 'metrics.train_test.Test.Mean.TPR':'Recall', 'metrics.train_test.Test.Mean.PPV':'Precision', 'ACC Mean': 'ACC Mean', 'ACC Std': 'ACC Std','params.method':'Method'})

    exps = list(best_runs_small['Experiment'])
    exps = [' '.join(e.split('_')[1:]) for e in exps]
    best_runs_small['Experiment'] = exps


    best_runs_results = best_runs_small[['Experiment', 'Method', 'BSR', 'Precision', 'Recall', 'ACC Mean', 'ACC Std']]




    best_runs_results.to_csv('./results/new_best_classifier_results.csv')

    res = best_runs_results.copy()

    idx = ['limma' in r for r in res['Experiment']]
    limma_res = res[idx]

    limma_res['Experiment'] = [' '.join(exp.split(' ')[:3]) for exp in limma_res['Experiment']]

    limma_res['Split'] =[exp.split(' ')[0] for exp in limma_res['Experiment']]

    limma_res['Time Bin'] = [' to '.join(str(x) for x in exp.split(' ')[1:]) for exp in limma_res['Experiment']]

    for col_name in ['BSR', 'Precision', 'Recall', 'ACC Mean', 'ACC Std']:
        limma_res[col_name] = limma_res[col_name].astype(float).round(2)

    limma_res = limma_res.sort_values("Method").sort_values("Experiment")

    limma_res = limma_res[['Time Bin', 'Split', 'Method', 'BSR', 'Precision', 'Recall', 'ACC Mean', 'ACC Std']]

    limma_res.to_csv('./results/new_best_classifier_results_limma.csv')